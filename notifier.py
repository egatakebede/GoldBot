"""
Thin notifier wrapper — broadcasts to all configured Telegram chat IDs.
Used by signal_server.py, bot.py, retrain_scheduler.py.
"""
import requests
import config

BASE_URL = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}"

def send(text):
    chat_ids = config.TELEGRAM_CHAT_ID
    if isinstance(chat_ids, str):
        chat_ids = [chat_ids]
    for cid in chat_ids:
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

def daily_summary(total_trades, wins, daily_pnl, balance):
    emoji = "✅" if daily_pnl >= 0 else "❌"
    send(f"{emoji} <b>Daily summary</b>\n"
         f"Trades: <code>{total_trades}</code> | Wins: <code>{wins}</code>\n"
         f"Daily P&L: <code>${daily_pnl:+.2f}</code> | Balance: <code>${balance:.2f}</code>")

def start_command_listener(risk_engine):
    """Poll Telegram for /pause /resume /status commands in a background thread."""
    import threading

    def _poll():
        import time
        offset = 0
        while True:
            try:
                r = requests.get(f"{BASE_URL}/getUpdates",
                                 params={"offset": offset, "timeout": 30},
                                 timeout=35)
                if not r.ok:
                    time.sleep(5)
                    continue
                for upd in r.json().get("result", []):
                    offset = upd["update_id"] + 1
                    text   = upd.get("message", {}).get("text", "").strip().lower()
                    cid    = upd.get("message", {}).get("chat", {}).get("id")
                    if text == "/pause":
                        risk_engine.is_active  = False
                        risk_engine.pause_reason = "Paused via Telegram"
                        requests.post(f"{BASE_URL}/sendMessage",
                                      json={"chat_id": cid, "text": "⏸️ Bot paused."}, timeout=5)
                    elif text == "/resume":
                        risk_engine.is_active  = True
                        risk_engine.pause_reason = None
                        requests.post(f"{BASE_URL}/sendMessage",
                                      json={"chat_id": cid, "text": "▶️ Bot resumed."}, timeout=5)
                    elif text == "/status":
                        s = risk_engine.summary()
                        msg = (f"📊 <b>Status</b>\n"
                               f"Balance: <code>${s['balance']}</code>\n"
                               f"Daily P&L: <code>${s['daily_pnl']}</code>\n"
                               f"Win rate: <code>{s['win_rate']}%</code>\n"
                               f"Drawdown: <code>{s['drawdown']}%</code>\n"
                               f"Active: <code>{s['is_active']}</code>")
                        requests.post(f"{BASE_URL}/sendMessage",
                                      json={"chat_id": cid, "text": msg, "parse_mode": "HTML"}, timeout=5)
            except Exception as e:
                print(f"[Notifier] Command poll error: {e}")
                time.sleep(10)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()

def start_heartbeat(stats_fn):
    """Send an hourly heartbeat with current bot stats."""
    import threading, time

    def _beat():
        while True:
            time.sleep(3600)
            try:
                s = stats_fn()
                send(f"💓 <b>Heartbeat</b>\n"
                     f"Balance: <code>${s.get('balance', '?'):.2f}</code>\n"
                     f"Conf: <code>{s.get('confidence', 0):.2f}</code> | "
                     f"Regime: <code>{s.get('regime', '?')}</code>")
            except Exception as e:
                print(f"[Notifier] Heartbeat error: {e}")

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
