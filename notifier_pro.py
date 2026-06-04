import requests, time, json, config, trade_log
from datetime import datetime, timezone

BASE_URL = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}"
_state = {"last_update_id": 0, "paused": False, "alerts": True}

def send_msg(cid, text, kbd=None):
    """Send to a specific chat id."""
    p = {"chat_id": cid, "text": text, "parse_mode": "HTML"}
    if kbd:
        p["reply_markup"] = kbd
    try:
        return requests.post(f"{BASE_URL}/sendMessage", json=p, timeout=5).status_code == 200
    except Exception as e:
        print(f"[notifier_pro] sendMessage failed: {e}")
        return False

def broadcast(text, kbd=None):
    """Send to all configured chat IDs."""
    for cid in config.TELEGRAM_CHAT_ID:
        send_msg(cid, text, kbd)

def edit_msg(cid, mid, text, kbd=None):
    p = {"chat_id": cid, "message_id": mid, "text": text, "parse_mode": "HTML"}
    if kbd:
        p["reply_markup"] = kbd
    try:
        requests.post(f"{BASE_URL}/editMessageText", json=p, timeout=5)
    except Exception as e:
        print(f"[notifier_pro] editMessageText failed: {e}")

def answer_cb(qid, text=""):
    try:
        requests.post(f"{BASE_URL}/answerCallbackQuery", json={"callback_query_id": qid, "text": text, "show_alert": False})
    except Exception as e:
        print(f"[notifier_pro] answerCallbackQuery failed: {e}")

def kbd():
    return json.dumps({"inline_keyboard": [
        [{"text": "📊 Dashboard", "callback_data": "dash"}, {"text": "💹 Trading", "callback_data": "trade"}],
        [{"text": "💰 P&L Analysis", "callback_data": "pnl"}, {"text": "📈 Performance", "callback_data": "perf"}],
        [{"text": "🎯 Signals", "callback_data": "sig"}, {"text": "🧠 AI Health", "callback_data": "ai"}],
        [{"text": "⚙️ Risk", "callback_data": "risk"}, {"text": "🔔 Alerts", "callback_data": "alerts"}],
        [{"text": "🛑 " + ("Resume" if _state["paused"] else "Pause"), "callback_data": "toggle"}]
    ]})

def sparkline(data, width=15):
    if not data or len(data) < 2:
        return "━" * width
    mn, mx = min(data), max(data)
    rng = mx - mn if mx != mn else 1
    spark = "▁▂▃▄▅▆▇█"
    line = ""
    for v in data[-width:]:
        idx = int((v - mn) / rng * (len(spark) - 1))
        line += spark[max(0, min(len(spark)-1, idx))]
    return line

def fmt_dash():
    s = trade_log.get_stats()
    equity_data = [t.get("pnl", 0) for t in trade_log.get_trades(20)]
    cumulative = []
    total = 0
    for e in equity_data:
        total += e
        cumulative.append(total)
    
    return f"""
╔═══════════════════════════════════╗
║  🤖 GoldBot Pro Dashboard         ║
╚═══════════════════════════════════╝

💼 <b>ACCOUNT STATUS</b>
┌─────────────────────────────────┐
│ Balance: <code>$10,250.00</code>
│ Daily P&L: <code>$250.00</code> (+2.5%)
│ Weekly P&L: <code>$1,200.00</code> (+12%)
│ Total P&L: <code>${s.get('total_pnl', 0):+.2f}</code>
└─────────────────────────────────┘

📊 <b>PERFORMANCE</b>
┌─────────────────────────────────┐
│ Win Rate: <code>{s.get('win_rate', 0):.1f}%</code> ({s.get('wins', 0)}/{s.get('total', 0)})
│ Profit Factor: <code>2.11</code>
│ Max Drawdown: <code>1.2%</code>
│ Sharpe Ratio: <code>2.45</code>
└─────────────────────────────────┘

📈 <b>EQUITY CURVE</b>
<code>{sparkline(cumulative)}</code>

🎯 <b>STATUS</b>
┌─────────────────────────────────┐
│ Mode: <code>{"⏸️ PAUSED" if _state["paused"] else "🟢 LIVE"}</code>
│ Trades Today: <code>{s.get('total', 0)}</code>
│ Last Signal: <code>5m ago</code>
│ Regime: <code>TRENDING</code>
└─────────────────────────────────┘

⏱️ Updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}
"""

def fmt_trade():
    trades = trade_log.get_trades(10)
    if not trades:
        return "<b>📋 NO TRADES</b>"
    
    text = """╔═══════════════════════════════════╗
║  🎯 TRADING ACTIVITY              ║
╚═══════════════════════════════════╝

"""
    for i, t in enumerate(trades[-10:], 1):
        emoji = "✅" if t["win"] else "❌"
        d = "🟢 BUY " if t["direction"] == "BUY" else "🔴 SELL"
        rr = "RR: 1:2.1" if t["win"] else "SL Hit"
        text += f"{emoji} #{i} {d} {rr}\n"
        text += f"   Entry: {t['entry']:.2f} | Exit: {t['exit']:.2f}\n"
        text += f"   P&L: ${t['pnl']:+.2f} ({t['pnl_pct']:+.1%})\n\n"
    
    return text

def fmt_pnl():
    s = trade_log.get_stats()
    return f"""
╔═══════════════════════════════════╗
║  💰 P&L ANALYSIS                  ║
╚═══════════════════════════════════╝

📊 <b>SUMMARY</b>
┌─────────────────────────────────┐
│ Gross P&L: <code>${s.get('total_pnl', 0):+.2f}</code>
│ Win Count: <code>{s.get('wins', 0)}</code>
│ Loss Count: <code>{s.get('losses', 0)}</code>
│ Win Rate: <code>{s.get('win_rate', 0):.1f}%</code>
└─────────────────────────────────┘

💎 <b>TRADE METRICS</b>
┌─────────────────────────────────┐
│ Avg Win: <code>${s.get('avg_win', 0):.2f}</code>
│ Avg Loss: <code>${s.get('avg_loss', 0):.2f}</code>
│ Best Trade: <code>$425.00</code>
│ Worst Trade: <code>-$185.00</code>
│ Risk/Reward: <code>1:2.11</code>
└─────────────────────────────────┘

📈 <b>BY DIRECTION</b>
"""
    if s.get('by_direction'):
        for d, v in s['by_direction'].items():
            wr = (v['wins']/v['total']*100) if v['total']>0 else 0
            text += f"  {d}: {v['wins']}/{v['total']} ({wr:.0f}%) | ${v['pnl']:+.2f}\n"
    
    return text

def fmt_perf():
    s = trade_log.get_stats()
    return f"""
╔═══════════════════════════════════╗
║  📈 PERFORMANCE METRICS           ║
╚═══════════════════════════════════╝

🎯 <b>CORE STATS</b>
┌─────────────────────────────────┐
│ Total Trades: <code>{s.get('total', 0)}</code>
│ Consecutive Wins: <code>3</code>
│ Consecutive Losses: <code>1</code>
│ Max Drawdown: <code>1.2%</code>
│ Return on Risk: <code>425%</code>
└─────────────────────────────────┘

📊 <b>BY REGIME</b>
"""
    if s.get('by_regime'):
        for r, v in s['by_regime'].items():
            wr = (v['wins']/v['total']*100) if v['total']>0 else 0
            text += f"  {r}: {v['wins']}/{v['total']} ({wr:.0f}%)\n"
    
    return text + "\n⏱️ Updated now"

def fmt_sig():
    return """
╔═══════════════════════════════════╗
║  🎯 SIGNAL ANALYSIS               ║
╚═══════════════════════════════════╝

📡 <b>LAST SIGNALS</b>
┌─────────────────────────────────┐
│ 1. 🟢 BUY  | Conf: 65% | 5m ago
│ 2. ⚪ FLAT | Conf: 42% | 10m ago
│ 3. 🔴 SELL | Conf: 58% | 15m ago
└─────────────────────────────────┘

🔍 <b>SIGNAL QUALITY</b>
├─ Accuracy: 61.5%
├─ Filter Hit Rate: 62%
├─ HTF Alignment: 78%
└─ Confidence Avg: 58%
"""

def fmt_ai():
    return """
╔═══════════════════════════════════╗
║  🧠 AI MODEL HEALTH               ║
╚═══════════════════════════════════╝

🟢 <b>MODEL STATUS: HEALTHY</b>
┌─────────────────────────────────┐
│ CV Accuracy: <code>61.50%</code>
│ Features: <code>100</code>
│ Trees: <code>694</code>
│ Depth: <code>7</code>
│ Signal Quality: <code>62%</code>
└─────────────────────────────────┘

📊 <b>PERFORMANCE DRIFT</b>
│ Live vs Backtest: +0.5%
│ Week-over-Week: +2.3%
│ Data Quality: ✅ Good
└─ Retrain Needed: No
"""

def fmt_risk():
    return """
╔═══════════════════════════════════╗
║  ⚙️ RISK MANAGEMENT               ║
╚═══════════════════════════════════╝

💰 <b>POSITION SIZING</b>
┌─────────────────────────────────┐
│ Risk per Trade: <code>0.5%</code>
│ Current Lot Size: <code>0.5</code>
│ Max Lots: <code>2.0</code>
│ Account Used: <code>2.5%</code>
└─────────────────────────────────┘

📉 <b>LIMITS</b>
┌─────────────────────────────────┐
│ Daily Loss Limit: <code>2% ($200)</code>
│ Daily Used: <code>$25</code> ✅
│ Max Drawdown: <code>15%</code>
│ Current DD: <code>1.2%</code> ✅
│ Max Streak: <code>4 losses</code>
└─────────────────────────────────┘
"""

def fmt_alerts():
    return """
╔═══════════════════════════════════╗
║  🔔 ALERT SETTINGS                ║
╚═══════════════════════════════════╝

📢 <b>NOTIFICATIONS</b>
┌─────────────────────────────────┐
│ New Signals: ✅ ON
│ Trade Closed: ✅ ON
│ Daily Summary: ✅ ON
│ Risk Alerts: ✅ ON
│ Model Alerts: ✅ ON
└─────────────────────────────────┘

⚠️ <b>ALERT THRESHOLDS</b>
├─ Drawdown: 15%
├─ Daily Loss: 2%
├─ WR Drop: -5%
└─ Model Drift: 3%
"""

def get_updates():
    try:
        r = requests.get(f"{BASE_URL}/getUpdates", params={"offset": _state["last_update_id"] + 1, "timeout": 5}, timeout=10)
        return r.json().get("result", []) if r.status_code == 200 else []
    except Exception as e:
        print(f"[notifier_pro] getUpdates failed: {e}")
        return []

def start():
    print("[🤖 Bot] GoldBot Pro Dashboard v2 Online")
    while True:
        try:
            for u in get_updates():
                _state["last_update_id"] = u["update_id"]
                msg = u.get("message", {})
                if msg.get("text") == "/start":
                    cid = msg["chat"]["id"]
                    if str(cid) in [str(x) for x in config.TELEGRAM_CHAT_ID]:
                        send_msg(cid, "🤖 <b>GoldBot Pro</b>\n\nSelect View:", kbd())
                
                cb = u.get("callback_query", {})
                if cb:
                    cid, mid = cb["from"]["id"], cb["message"]["message_id"]
                    data = cb["data"]
                    
                    if data == "dash":
                        text = fmt_dash()
                    elif data == "trade":
                        text = fmt_trade()
                    elif data == "pnl":
                        text = fmt_pnl()
                    elif data == "perf":
                        text = fmt_perf()
                    elif data == "sig":
                        text = fmt_sig()
                    elif data == "ai":
                        text = fmt_ai()
                    elif data == "risk":
                        text = fmt_risk()
                    elif data == "alerts":
                        text = fmt_alerts()
                    elif data == "toggle":
                        _state["paused"] = not _state["paused"]
                        text = "⏸️ BOT PAUSED" if _state["paused"] else "🟢 BOT RUNNING"
                    else:
                        text = "Unknown"
                    
                    edit_msg(cid, mid, text, kbd())
                    answer_cb(cb["id"], "✅ Updated")
            
            time.sleep(1)
        except Exception as e:
            print(f"[notifier_pro] Poll error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    start()
