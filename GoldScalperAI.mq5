//+------------------------------------------------------------------+
//| GoldScalperAI.mq5                                                |
//+------------------------------------------------------------------+
#property version "2.10"
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

input double InpRiskPct        = 0.005;
input double InpDailyLossLimit = 0.02;
input double InpMaxDrawdown    = 0.15;
input int    InpMaxConsecLoss  = 4;
input double InpMaxLots        = 2.0;
input int    InpMagicNumber    = 202506;
input double InpMinConfidence  = 0.62;
input int    InpDataBars       = 500;
input int    InpSignalStaleMin = 15;
input int    InpMaxPositions   = 3;      // max simultaneous positions

CTrade        Trade;
CPositionInfo PosInfo;

double   g_InitialBalance, g_Balance, g_DailyPnL;
int      g_ConsecLoss;
bool     g_IsActive;
datetime g_LastDay, g_LastBar;

int OnInit()
{
   Trade.SetExpertMagicNumber(InpMagicNumber);
   Trade.SetDeviationInPoints(30);
   g_InitialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_Balance        = g_InitialBalance;
   g_DailyPnL       = 0;
   g_ConsecLoss     = 0;
   g_IsActive       = true;
   g_LastDay        = 0;
   g_LastBar        = 0;
   Print("GoldScalperAI v2.10 started | Balance: ", g_InitialBalance);
   return INIT_SUCCEEDED;
}

void ExportDataToCSV()
{
   int handle = FileOpen("mt5_data.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("Cannot write mt5_data.csv error: ", GetLastError());
      return;
   }
   FileWrite(handle, "time","open","high","low","close","tick_volume");
   MqlRates rates[];
   int copied = CopyRates(_Symbol, PERIOD_M5, 0, InpDataBars, rates);
   if(copied > 0)
      for(int i = copied-1; i >= 0; i--)
         FileWrite(handle,
            TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES),
            DoubleToString(rates[i].open,  5),
            DoubleToString(rates[i].high,  5),
            DoubleToString(rates[i].low,   5),
            DoubleToString(rates[i].close, 5),
            IntegerToString(rates[i].tick_volume));
   FileClose(handle);
}

// signal,confidence,entry,sl,tp,atr,regime,timestamp
// BUY,0.6823,4495.73,4488.55,4507.70,4.72,TRENDING,2025-01-15 08:15:00+00:00
string ReadSignal(double &confidence, double &entry_price, double &sl, double &tp, datetime &sig_time)
{
   confidence  = 0;
   entry_price = 0;
   sl          = 0;
   tp          = 0;
   sig_time    = 0;

   int handle = FileOpen("signal.csv", FILE_READ|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE) return "FLAT";

   // Skip header line
   while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
      FileReadString(handle);

   if(FileIsEnding(handle)) { FileClose(handle); return "FLAT"; }

   // Read data row: signal,confidence,entry,sl,tp,atr,regime,timestamp
   string signal    = FileReadString(handle);  // col 0: signal
   confidence       = StringToDouble(FileReadString(handle));  // col 1
   entry_price      = StringToDouble(FileReadString(handle));  // col 2
   sl               = StringToDouble(FileReadString(handle));  // col 3
   tp               = StringToDouble(FileReadString(handle));  // col 4
   FileReadString(handle);  // col 5: atr (skip)
   FileReadString(handle);  // col 6: regime (skip)
   string ts        = FileReadString(handle);  // col 7: timestamp

   FileClose(handle);

   // Parse timestamp — format: "2025-01-15 08:15:00+00:00"
   // Strip timezone suffix for StringToTime
   int plus_pos = StringFind(ts, "+", 10);
   if(plus_pos > 0) ts = StringSubstr(ts, 0, plus_pos);
   StringReplace(ts, "T", " ");
   sig_time = StringToTime(ts);

   return signal;
}

int CountMyPositions()
{
   int n = 0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
      if(PosInfo.SelectByIndex(i))
         if(PosInfo.Symbol()==_Symbol && PosInfo.Magic()==InpMagicNumber)
            n++;
   return n;
}

double CalcLots(double entry, double sl_price)
{
   double pip_risk = MathAbs(entry - sl_price);
   if(pip_risk < 0.0001) return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double risk = g_Balance * InpRiskPct;
   double lots = risk / (pip_risk * 100.0);
   lots = MathMin(lots, InpMaxLots);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minL = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(step <= 0) step = 0.01;
   lots = MathMax(minL, MathFloor(lots/step)*step);
   return NormalizeDouble(lots, 2);
}

bool RiskOK()
{
   if(!g_IsActive) return false;
   if(g_InitialBalance <= 0) return false;
   double dd = (g_InitialBalance - g_Balance) / g_InitialBalance;
   if(dd >= InpMaxDrawdown)    { g_IsActive = false; return false; }
   if(g_DailyPnL <= -InpDailyLossLimit * g_InitialBalance) return false;
   if(g_ConsecLoss >= InpMaxConsecLoss) return false;
   return true;
}

void OnTick()
{
   datetime curBar = iTime(_Symbol, PERIOD_M5, 0);
   if(curBar == g_LastBar) return;
   g_LastBar = curBar;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));
   if(today != g_LastDay)
   {
      g_DailyPnL = 0;
      g_Balance  = AccountInfoDouble(ACCOUNT_BALANCE);
      if(g_ConsecLoss >= InpMaxConsecLoss) g_ConsecLoss = 0;
      g_LastDay  = today;
   }

   g_Balance = AccountInfoDouble(ACCOUNT_BALANCE);
   ExportDataToCSV();

   if(!RiskOK())                              { Comment("Risk limit hit"); return; }
   if(CountMyPositions() >= InpMaxPositions)  { Comment("Max positions reached"); return; }

   double   confidence, entry_price, sl, tp;
   datetime sig_time;
   string   signal = ReadSignal(confidence, entry_price, sl, tp, sig_time);

   // Stale signal check — ignore if signal is older than InpSignalStaleMin
   int age_min = (int)((TimeCurrent() - sig_time) / 60);
   if(sig_time > 0 && age_min > InpSignalStaleMin)
   {
      Comment(StringFormat("Signal STALE (%d min old) | Last: %s", age_min, signal));
      return;
   }

   Comment(StringFormat("Signal:%s Conf:%.3f Age:%dmin Balance:$%.2f DailyPnL:$%.2f",
           signal, confidence, age_min, g_Balance, g_DailyPnL));

   if(signal != "BUY" && signal != "SELL") return;
   if(confidence < InpMinConfidence)        return;
   if(sl == 0 || tp == 0)                  return;

   double liveEntry = (signal == "BUY") ?
                      SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                      SymbolInfoDouble(_Symbol, SYMBOL_BID);

   double lots = CalcLots(liveEntry, sl);
   if(lots <= 0) return;

   bool ok;
   if(signal == "BUY")
      ok = Trade.Buy(lots, _Symbol, liveEntry, sl, tp,
                     StringFormat("AI|%.3f", confidence));
   else
      ok = Trade.Sell(lots, _Symbol, liveEntry, sl, tp,
                      StringFormat("AI|%.3f", confidence));

   if(ok)
      Print(StringFormat("TRADE %s Lots:%.2f Entry:%.2f SL:%.2f TP:%.2f Conf:%.3f",
            signal, lots, liveEntry, sl, tp, confidence));
   else
      Print("Order failed: ", Trade.ResultRetcodeDescription());
}

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest     &req,
                        const MqlTradeResult      &res)
{
   // Only count when a position is closed (DEAL_OUT or DEAL_INOUT)
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      ulong deal_ticket = trans.deal;
      if(HistoryDealSelect(deal_ticket))
      {
         ENUM_DEAL_ENTRY entry_type = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
         if(entry_type == DEAL_ENTRY_OUT || entry_type == DEAL_ENTRY_INOUT)
         {
            double newBal = AccountInfoDouble(ACCOUNT_BALANCE);
            double pnl    = newBal - g_Balance;
            g_Balance     = newBal;
            g_DailyPnL   += pnl;
            g_ConsecLoss  = (pnl > 0) ? 0 : g_ConsecLoss + 1;
         }
      }
   }
}
