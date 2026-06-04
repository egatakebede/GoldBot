import json
import os
import logging
from datetime import datetime, date, timezone
import config

class RiskEngine:
    def __init__(self, initial_balance: float):
        self.initial_balance    = initial_balance
        self.balance            = initial_balance
        self.daily_pnl          = 0.0
        self.daily_trades       = 0
        self.consec_loss        = 0
        self.total_trades       = 0
        self.winning_trades     = 0
        self.is_active          = True
        self.last_day           = datetime.now(timezone.utc).date()
        self.pause_reason       = None
        self._state_file        = "data/risk_state.json"
        self._recent_pnl        = []
        self._equity_curve_weak = False
        self._load_state()

    def check_daily_reset(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_day:
            self.daily_pnl          = 0.0
            self.daily_trades       = 0
            self.consec_loss        = 0
            self.is_active          = True
            self.pause_reason       = None
            self._equity_curve_weak = False
            self.last_day           = today
            self._save_state()

    def can_trade(self):
        self.check_daily_reset()

        if not self.is_active:
            return False, self.pause_reason or "KILLED"

        dd = (self.initial_balance - self.balance) / self.initial_balance
        if dd >= config.MAX_DRAWDOWN:
            self.is_active    = False
            self.pause_reason = f"MAX_DRAWDOWN {dd:.1%}"
            self._save_state()
            return False, self.pause_reason

        if self.daily_pnl <= -(config.DAILY_LOSS_LIMIT * self.initial_balance):
            return False, f"DAILY_LIMIT (P&L: ${self.daily_pnl:.2f})"

        if self.consec_loss >= config.MAX_CONSEC_LOSS:
            return False, f"CONSEC_LOSS ({self.consec_loss} in a row)"

        if self.daily_trades >= config.MAX_DAILY_TRADES:
            return False, f"MAX_DAILY_TRADES ({self.daily_trades})"

        # Equity curve — flag weak but don't block, reduce size instead
        if len(self._recent_pnl) >= 10:
            self._equity_curve_weak = sum(self._recent_pnl[-10:]) < 0
        else:
            self._equity_curve_weak = False

        return True, "OK"

    def lot_size(self, entry: float, stop_loss: float,
                 confidence: float = 1.0, with_trend: bool = True,
                 open_positions: int = 0) -> float:
        pip_risk = abs(entry - stop_loss)
        if pip_risk <= 0:
            return 0.0

        if self.consec_loss > 0:
            streak_scale = max(0.25, 1.0 - self.consec_loss * 0.15)
        else:
            wins_in_row  = self._current_win_streak()
            streak_scale = min(1.5, 1.0 + wins_in_row * 0.10)

        trend_scale = 1.0 if with_trend else 0.5
        curve_scale = 0.5 if self._equity_curve_weak else 1.0
        # Scale down for each additional open position — avoid overexposure
        pos_scale   = 1.0 / (1 + open_positions * 0.5)

        scaled_risk = (config.RISK_PCT
                       * confidence
                       * streak_scale
                       * trend_scale
                       * curve_scale
                       * pos_scale)

        risk_amount = self.balance * scaled_risk
        lots = risk_amount / (pip_risk * 100.0)
        lots = min(lots, config.MAX_LOTS)
        lots = max(0.01, round(lots, 2))
        return lots

    def _current_win_streak(self):
        streak = 0
        for pnl in reversed(self._recent_pnl):
            if pnl > 0:
                streak += 1
            else:
                break
        return streak

    def update(self, pnl: float):
        self.balance         += pnl
        self.daily_pnl       += pnl
        self.daily_trades    += 1
        self.total_trades    += 1
        self._recent_pnl.append(pnl)
        if len(self._recent_pnl) > 20:
            self._recent_pnl.pop(0)
        if pnl > 0:
            self.winning_trades += 1
            self.consec_loss     = 0
        else:
            self.consec_loss += 1
        self._save_state()

    @property
    def win_rate(self):
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def drawdown(self):
        return (self.initial_balance - self.balance) / self.initial_balance

    def summary(self):
        return {
            "balance":            round(self.balance, 2),
            "initial":            round(self.initial_balance, 2),
            "daily_pnl":          round(self.daily_pnl, 2),
            "daily_trades":       self.daily_trades,
            "total_trades":       self.total_trades,
            "win_rate":           round(self.win_rate * 100, 1),
            "drawdown":           round(self.drawdown * 100, 2),
            "consec_loss":        self.consec_loss,
            "is_active":          self.is_active,
            "pause_reason":       self.pause_reason,
            "equity_curve_weak":  self._equity_curve_weak,
        }

    def _save_state(self):
        os.makedirs("data", exist_ok=True)
        with open(self._state_file, "w") as f:
            json.dump({
                "balance":         self.balance,
                "initial_balance": self.initial_balance,
                "daily_pnl":       self.daily_pnl,
                "consec_loss":     self.consec_loss,
                "total_trades":    self.total_trades,
                "winning_trades":  self.winning_trades,
                "is_active":       self.is_active,
                "pause_reason":    self.pause_reason,
                "last_day":        str(self.last_day),
            }, f, indent=2)

    def _load_state(self):
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file) as f:
                s = json.load(f)
            self.balance        = s.get("balance",         self.balance)
            self.daily_pnl      = s.get("daily_pnl",       0.0)
            self.consec_loss    = s.get("consec_loss",      0)
            self.total_trades   = s.get("total_trades",     0)
            self.winning_trades = s.get("winning_trades",   0)
            self.is_active      = s.get("is_active",        True)
            self.pause_reason   = s.get("pause_reason",     None)
            self.last_day       = date.fromisoformat(
                s.get("last_day", str(date.today()))
            )
            print(f"[RiskEngine] State loaded — "
                  f"Balance: ${self.balance:.2f} | "
                  f"Trades: {self.total_trades} | "
                  f"WR: {self.win_rate:.1%}")
        except Exception as e:
            print(f"[RiskEngine] Could not load state: {e}")
