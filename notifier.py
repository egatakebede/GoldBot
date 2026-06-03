"""
Thin notifier wrapper — broadcasts to all configured Telegram chat IDs.
Used by signal_server.py, bot.py, retrain_scheduler.py.
"""
import requests
import config

BASE_URL = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}"

def send(text):
    for cid in config.TELEGRAM_CHAT_ID:
        try:
            requests.post(f"{BASE_URL}/sendMessage",
                          json={"chat_id": cid, "text": text, "parse_mode": "HTML"},
                          timeout=5)
        except Exception as e:
            print(f"[Notifier] Failed to send to {cid}: {e}")

def trade_opened(direction, lots, entry, sl, tp, confidence, regime):
    emoji = "🟢" if direction == "BUY" else "🔴"
    send(f"{emoji} <b>{direction}</b> opened\n"
         f"Entry: <code>{entry:.2f}</code> | SL: <code>{sl:.2f}</code> | TP: <code>{tp:.2f}</code>\n"
         f"Lots: <code>{lots}</code> | Conf: <code>{confidence:.2%}</code> | Regime: <code>{regime}</code>")

def trade_closed(direction, entry, exit_price, pnl, pnl_pct):
    emoji = "✅" if pnl >= 0 else "❌"
    send(f"{emoji} <b>{direction}</b> closed\n"
         f"Entry: <code>{entry:.2f}</code> → Exit: <code>{exit_price:.2f}</code>\n"
         f"P&L: <code>${pnl:+.2f}</code> (<code>{pnl_pct:+.2f}%</code>)")

def news_blackout(event_name, minutes):
    send(f"⏸️ <b>News blackout</b>: {event_name} in {minutes}min")

def retrain_done(accuracy, n_features):
    send(f"🔄 <b>Retrain complete</b>\n"
         f"Accuracy: <code>{accuracy:.1%}</code> | Features: <code>{n_features}</code>")
