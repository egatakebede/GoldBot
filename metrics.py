import trade_log
from datetime import datetime, timezone

def equity_spark(trades, width=20):
    if not trades:
        return "─" * width
    equity = 0
    values = []
    for t in trades:
        equity += t["pnl"]
        values.append(equity)
    if not values:
        return "─" * width
    min_val, max_val = min(values), max(values)
    range_val = max_val - min_val if max_val != min_val else 1
    sparks = "▁▂▃▄▅▆▇█"
    line = ""
    step = len(values) // width if len(values) > width else 1
    for i in range(0, len(values), max(1, step)):
        if len(line) < width:
            idx = int((values[i] - min_val) / range_val * (len(sparks) - 1))
            line += sparks[max(0, min(len(sparks)-1, idx))]
    return line.ljust(width, "─")

def format_status(balance, daily_pnl, win_rate, drawdown, active, paused):
    stats = trade_log.get_stats()
    emoji_status = "🟢" if active and not paused else "🔴"
    emoji_wr = "📈" if win_rate >= 60 else "📊" if win_rate >= 50 else "📉"
    emoji_pnl = "💰" if daily_pnl > 0 else "⚠️"
    return f"""
{emoji_status} <b>STATUS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
💵 Balance: <code>${balance:,.2f}</code>
{emoji_pnl} Daily P&L: <code>${daily_pnl:+,.2f}</code>
{emoji_wr} Win Rate: <code>{win_rate:.1f}%</code> ({stats['wins']}/{stats['total']})
📉 Max DD: <code>{drawdown:.2f}%</code>
🎯 Streak: <code>W:{stats.get('streak_win', 0)} L:{stats.get('streak_loss', 0)}</code>
⏸️ Paused: <code>{"Yes" if paused else "No"}</code>

<b>Equity Curve:</b>
<code>{equity_spark(trade_log.get_trades(20))}</code>
"""

def format_trades(limit=10):
    trades = trade_log.get_trades(limit)
    if not trades:
        return "<b>📋 NO TRADES YET</b>"
    text = "<b>📋 LAST 10 TRADES</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    for i, t in enumerate(trades[-limit:], 1):
        emoji = "✅" if t["win"] else "❌"
        direction_emoji = "🟢" if t["direction"] == "BUY" else "🔴"
        text += f"{emoji} #{i} {direction_emoji} {t['direction']}\n   Entry: {t['entry']:.2f} → Exit: {t['exit']:.2f}\n   P&L: ${t['pnl']:+.2f} ({t['pnl_pct']:+.2%}) | {t['exit_reason']}\n"
    return text

def format_stats():
    s = trade_log.get_stats()
    if s["total"] == 0:
        return "<b>📊 NO DATA YET</b>"
    text = "<b>📊 PERFORMANCE</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    text += f"Total: {s['total']} | ✅ {s['wins']} | ❌ {s['losses']}\n"
    text += f"Win Rate: <b>{s['win_rate']:.1f}%</b>\n"
    text += f"Total P&L: <code>${s['total_pnl']:+,.2f}</code>\n"
    text += f"Avg Win/Loss: ${s['avg_win']:.2f} / ${s['avg_loss']:.2f}\n"
    text += f"Max Win/Loss: ${s['max_win']:.2f} / ${s['max_loss']:.2f}\n\n"
    if s["by_direction"]:
        text += "<b>By Direction:</b>\n"
        for d, v in s["by_direction"].items():
            wr = (v["wins"] / v["total"] * 100) if v["total"] > 0 else 0
            text += f"  {d}: {v['wins']}/{v['total']} ({wr:.0f}%) | ${v['pnl']:+.2f}\n"
    if s["by_regime"]:
        text += "<b>By Regime:</b>\n"
        for r, v in s["by_regime"].items():
            wr = (v["wins"] / v["total"] * 100) if v["total"] > 0 else 0
            text += f"  {r}: {v['wins']}/{v['total']} ({wr:.0f}%) | ${v['pnl']:+.2f}\n"
    return text

def format_health(model_accuracy, signal_quality, last_signal, last_signal_time):
    emoji_health = "🟢" if model_accuracy > 0.55 else "🟡" if model_accuracy > 0.50 else "🔴"
    emoji_signal = "🟢" if signal_quality > 0.6 else "🟡" if signal_quality > 0.5 else "🔴"
    return f"""
{emoji_health} <b>MODEL HEALTH</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 Accuracy: <code>{model_accuracy:.2%}</code>
{emoji_signal} Signal Quality: <code>{signal_quality:.2%}</code>
📡 Last Signal: <code>{last_signal}</code> ({last_signal_time})
⚙️ Features: <code>100</code>
🔄 Last Retrain: <code>2h ago</code>
"""
