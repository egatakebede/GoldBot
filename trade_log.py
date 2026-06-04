import os, json
from datetime import datetime, timezone

LOG_FILE = os.path.expanduser("~/GoldBot/logs/trades.json")

def utcnow(): 
    return datetime.now(timezone.utc).isoformat()

def log_trade(direction, entry, exit_price, sl, tp, lots, pnl, pnl_pct, regime, confidence, duration_seconds, exit_reason):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    trade = {"timestamp": utcnow(), "direction": direction, "entry": round(entry, 2), "exit": round(exit_price, 2), "sl": round(sl, 2), "tp": round(tp, 2), "lots": round(lots, 2), "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 4), "regime": regime, "confidence": round(confidence, 3), "duration_seconds": duration_seconds, "exit_reason": exit_reason, "win": 1 if pnl > 0 else 0}
    trades = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            trades = json.load(f)
    trades.append(trade)
    with open(LOG_FILE, "w") as f:
        json.dump(trades, f, indent=2)

def get_trades(limit=None):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE) as f:
        trades = json.load(f)
    return trades[-limit:] if limit else trades

def get_stats():
    trades = get_trades()
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0, "avg_win": 0, "avg_loss": 0, "max_win": 0, "max_loss": 0, "by_regime": {}, "by_direction": {}}
    total, wins = len(trades), sum(1 for t in trades if t["win"])
    losses, win_rate = total - wins, (wins / total * 100) if total > 0 else 0
    total_pnl = sum(t["pnl"] for t in trades)
    win_pnls, loss_pnls = [t["pnl"] for t in trades if t["win"]], [t["pnl"] for t in trades if not t["win"]]
    avg_win, avg_loss = (sum(win_pnls) / len(win_pnls) if win_pnls else 0), (sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0)
    max_win, max_loss = (max(win_pnls) if win_pnls else 0), (min(loss_pnls) if loss_pnls else 0)
    by_regime, by_direction = {}, {}
    for t in trades:
        r, d = t["regime"], t["direction"]
        if r not in by_regime:
            by_regime[r] = {"wins": 0, "total": 0, "pnl": 0}
        if d not in by_direction:
            by_direction[d] = {"wins": 0, "total": 0, "pnl": 0}
        by_regime[r]["wins"] += t["win"]
        by_regime[r]["total"] += 1
        by_regime[r]["pnl"] += t["pnl"]
        by_direction[d]["wins"] += t["win"]
        by_direction[d]["total"] += 1
        by_direction[d]["pnl"] += t["pnl"]
    return {"total": total, "wins": wins, "losses": losses, "win_rate": win_rate, "total_pnl": total_pnl, "avg_win": avg_win, "avg_loss": avg_loss, "max_win": max_win, "max_loss": max_loss, "by_regime": by_regime, "by_direction": by_direction, "streak_win": get_streaks(trades)[0], "streak_loss": get_streaks(trades)[1]}


def get_streaks(trades):
    """Calculate current win and loss streaks."""
    if not trades:
        return 0, 0
    win_streak = loss_streak = 0
    # Current win streak
    for t in reversed(trades):
        if t["win"]:
            win_streak += 1
        else:
            break
    # Current loss streak
    for t in reversed(trades):
        if not t["win"]:
            loss_streak += 1
        else:
            break
    return win_streak, loss_streak
