# ============================================================
#  GoldBot Pro — Main Bot Loop (Paper Mode)
# ============================================================
import time
import os
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import config
import notifier
from risk_engine import RiskEngine
from news_filter import is_blackout, next_event
from retrain_scheduler import load_model, start_scheduler
from features import get_signal_row


class PaperPosition:
    def __init__(self, direction, entry, sl, tp, lots, confidence, regime):
        self.direction    = direction
        self.entry        = entry
        self.sl           = sl
        self.tp           = tp
        self.lots         = lots
        self.confidence   = confidence
        self.regime       = regime
        self.open_time    = datetime.now(timezone.utc)
        self.is_open      = True
        self.trail_sl     = sl
        self.partial_done = False  # partial close at 1R already taken

    def update_trail(self, current_price, atr):
        if self.direction == "BUY":
            new_sl = current_price - atr * config.SL_ATR_MULT
            if new_sl > self.trail_sl:
                self.trail_sl = new_sl
        else:
            new_sl = current_price + atr * config.SL_ATR_MULT
            if new_sl < self.trail_sl:
                self.trail_sl = new_sl

    def bars_open(self, bar_seconds=300):
        elapsed = (datetime.now(timezone.utc) - self.open_time).total_seconds()
        return int(elapsed / bar_seconds)

    def check_partial(self, current_price):
        """Returns True if price hit 1R and partial hasn't been taken yet."""
        if self.partial_done:
            return False
        one_r = abs(self.entry - self.sl)
        if self.direction == "BUY" and current_price >= self.entry + one_r:
            return True
        if self.direction == "SELL" and current_price <= self.entry - one_r:
            return True
        return False

    def check_exit(self, current_price, max_bars=20):
        if not self.is_open:
            return False, 0.0, ""
        # Time-based exit
        if self.bars_open() >= max_bars:
            pnl_per_lot = (current_price - self.entry) if self.direction == "BUY" else (self.entry - current_price)
            return True, pnl_per_lot * self.lots * 100, "TIMEOUT"
        if self.direction == "BUY":
            if current_price <= self.trail_sl:
                return True, (self.trail_sl - self.entry) * self.lots * 100, "SL"
            if current_price >= self.tp:
                return True, (self.tp - self.entry) * self.lots * 100, "TP"
        else:  # SELL
            if current_price >= self.trail_sl:
                return True, (self.entry - self.trail_sl) * self.lots * 100, "SL"
            if current_price <= self.tp:
                return True, (self.entry - self.tp) * self.lots * 100, "TP"
        return False, 0.0, ""


class GoldBot:
    def __init__(self, initial_balance=10_000.0, paper=True):
        self.paper          = paper
        self.positions      = []          # list of open PaperPositions
        self.risk           = RiskEngine(initial_balance)
        self.model          = None
        self.feat_cols      = None
        self.running        = False
        self._last_atr      = 0.0
        self._last_h1_trend = -1
        self._last_h4_trend = -1
        self._dyn_conf      = config.MIN_CONFIDENCE

    def load(self):
        self.model, self.feat_cols = load_model()
        print("[Bot] Model loaded OK")

    def get_rates(self):
        def read(path):
            df = pd.read_csv(path, sep="\t", header=None,
                             names=["time","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
            df = df.apply(pd.to_numeric, errors="coerce")
            return df.dropna().tail(500)
        m1 = read("data/XAUUSD1.csv").to_records()
        m5 = read("data/XAUUSD5.csv").to_records()
        htf_paths = {}
        for prefix, fname in [("h1","data/XAUUSD60.csv"),("h4","data/XAUUSD240.csv")]:
            if os.path.exists(fname):
                htf_paths[prefix] = fname
        return m1, m5, htf_paths

    def get_current_price(self):
        df = pd.read_csv("data/XAUUSD1.csv")
        return float(df['close'].iloc[-1])

    def _open_trade(self, signal, confidence, regime, atr14, price):
        direction   = "BUY" if signal == 2 else "SELL"
        sl_dist     = atr14 * config.SL_ATR_MULT
        tp_dist     = atr14 * config.TP_ATR_MULT
        sl          = (price - sl_dist) if direction == "BUY" else (price + sl_dist)
        tp          = (price + tp_dist) if direction == "BUY" else (price - tp_dist)
        h1_bull     = self._last_h1_trend
        h4_bull     = self._last_h4_trend
        with_trend  = True
        if h1_bull != -1 and h4_bull != -1:
            with_trend = (signal == 2 and h1_bull == 1 and h4_bull == 1) or \
                         (signal == 0 and h1_bull == 0 and h4_bull == 0)
        lots = self.risk.lot_size(price, sl, confidence, with_trend, len(self.positions))
        if lots <= 0:
            return
        pos = PaperPosition(direction, price, sl, tp, lots, confidence, regime)
        self.positions.append(pos)
        print(f"[Bot] {direction} #{len(self.positions)} | Entry:{price:.2f} SL:{sl:.2f} TP:{tp:.2f} | "
              f"Lots:{lots} Conf:{confidence:.2f} Regime:{regime} Trend:{'✓' if with_trend else '✗'}")
        notifier.trade_opened(direction, lots, price, sl, tp, confidence, regime)

    def _close_trade(self, pos, reason, exit_price, pnl):
        self.risk.update(pnl)
        pnl_pct = pnl / self.risk.balance * 100
        if pnl < 0:
            self._dyn_conf = min(self._dyn_conf + 0.02, 0.75)
        else:
            self._dyn_conf = max(self._dyn_conf - 0.01, config.MIN_CONFIDENCE)
        print(f"[Bot] CLOSED {pos.direction} | Exit:{exit_price:.2f} | PnL:${pnl:+.2f} | {reason} | MinConf:{self._dyn_conf:.2f}")
        notifier.trade_closed(pos.direction, pos.entry, exit_price, pnl, pnl_pct)
        self._log_trade(pos, exit_price, pnl, reason)
        self.positions.remove(pos)

    def _log_trade(self, pos, exit_price, pnl, reason):
        import csv, os
        os.makedirs("data", exist_ok=True)
        path   = config.TRADE_LOG_PATH
        is_new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(['time','direction','entry','exit',
                            'sl','tp','lots','pnl','reason',
                            'confidence','regime'])
            w.writerow([
                datetime.now(timezone.utc).isoformat(),
                pos.direction, pos.entry, exit_price,
                pos.sl, pos.tp, pos.lots,
                round(pnl, 2), reason,
                round(pos.confidence, 3), pos.regime
            ])

    def tick(self):
        now   = datetime.now(timezone.utc)
        price = self.get_current_price()

        # Manage all open positions
        for pos in list(self.positions):
            if self._last_atr > 0:
                pos.update_trail(price, self._last_atr)
            if pos.check_partial(price):
                pos.partial_done = True
                pos.lots         = round(pos.lots / 2, 2)
                pos.trail_sl     = pos.entry
                print(f"[Bot] PARTIAL CLOSE | SL→BE | Lots now:{pos.lots}")
            should_exit, pnl, reason = pos.check_exit(price)
            if should_exit:
                self._close_trade(pos, reason, price, pnl)

        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            print(f"[Bot] Paused: {reason}")
            return

        blocked, event_name = is_blackout()
        if blocked:
            print(f"[Bot] News blackout: {event_name}")
            return

        nxt = next_event(within_hours=1)
        if nxt:
            mins = int((nxt['time'] - now).total_seconds() / 60)
            if mins <= config.NEWS_BLACKOUT_MINUTES:
                notifier.news_blackout(nxt['title'], mins)
                return

        # Check max simultaneous positions + correlation
        max_pos = getattr(config, 'MAX_POSITIONS', 3)
        if len(self.positions) >= max_pos:
            return

        try:
            rates_m1, rates_m5, htf_paths = self.get_rates()
            signal, confidence, regime, atr14, latest = get_signal_row(
                rates_m1, rates_m5, self.model, self.feat_cols, htf_paths
            )
            self._last_atr      = atr14
            self._last_h1_trend = int(latest.get("h1_trend_8_21", -1)) if hasattr(latest, 'get') else -1
            self._last_h4_trend = int(latest.get("h4_trend_8_21", -1)) if hasattr(latest, 'get') else -1
        except Exception as e:
            print(f"[Bot] Signal error: {e}")
            return

        if signal == 1:
            return
        if confidence < self._dyn_conf:
            return

        # Correlation check — max 2 positions same direction
        direction  = "BUY" if signal == 2 else "SELL"
        same_dir   = sum(1 for p in self.positions if p.direction == direction)
        if same_dir >= 2:
            print(f"[Bot] Skipping {direction} — already {same_dir} open same direction")
            return

        print(f"[Bot] {now.strftime('%H:%M')} | "
              f"Signal:{direction} | Conf:{confidence:.2f} | "
              f"Regime:{regime} | Positions:{len(self.positions)}/{max_pos}")

        self._open_trade(signal, confidence, regime, atr14, price)

    def run(self):
        self.running = True
        print(f"[Bot] GoldBot Pro started | Paper={self.paper} | "
              f"Balance:${self.risk.balance:.2f} | MaxPositions:{getattr(config,'MAX_POSITIONS',3)}")
        notifier.send("GoldBot Pro started\n"
                      f"Mode: {'Paper' if self.paper else 'LIVE'}\n"
                      f"Balance: ${self.risk.balance:.2f}")
        retrain_thread = threading.Thread(target=start_scheduler, daemon=True)
        retrain_thread.start()
        notifier.start_command_listener(self.risk)
        notifier.start_heartbeat(lambda: {
            "balance":    self.risk.balance,
            "signal":     "N/A",
            "confidence": self._dyn_conf,
            "regime":     "N/A"
        })
        while self.running:
            try:
                self.tick()
            except Exception as e:
                print(f"[Bot] Tick error: {e}")
            time.sleep(60)

    def stop(self):
        self.running = False
        s = self.risk.summary()
        print(f"[Bot] Stopped. {s}")
        notifier.daily_summary(
            s['total_trades'],
            int(s['win_rate'] / 100 * s['total_trades']),
            s['daily_pnl'],
            s['balance']
        )


if __name__ == "__main__":
    bot = GoldBot(initial_balance=10_000.0, paper=True)
    bot.load()
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.stop()
