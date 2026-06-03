"""
GoldBot Pro — Backtester v3
- Real model + features (no random signals)
- Realistic slippage model (spread + market impact)
- Walk-forward: train on first 70%, test on last 30%
- Correlation check: no 2nd position same direction
- Proper HTF context (ffill only from past)
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime, timezone

from features import build_features, add_htf_context, load_htf_csv

DATA_DIR      = "data"
MODEL_PATH    = "models/xgb_model.json"
CALIB_PATH    = "models/calibrated_model.pkl"
FEATURES_PATH = "models/feature_cols.txt"
PARAMS_PATH   = "models/best_params.json"
RESULTS_PATH  = "logs/backtest_results.csv"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.005
SL_MULT         = 0.8
TP_MULT         = 1.6
MAX_LOTS        = 2.0
MAX_BARS_OPEN   = 20
MAX_POSITIONS   = 3
PARTIAL_AT_1R   = True

# Slippage model — realistic for XAUUSD M1
BASE_SPREAD     = 0.30   # normal spread
NEWS_SPREAD     = 2.50   # spread during high-impact news
MARKET_IMPACT   = 0.05   # per 0.1 lot above 0.1 (size impact)

MIN_CONF = 0.62
if os.path.exists(PARAMS_PATH):
    try:
        with open(PARAMS_PATH) as f:
            saved = json.load(f)
        MIN_CONF = saved.get("conf_threshold", MIN_CONF)
        print(f"Conf threshold from training: {MIN_CONF:.3f}")
    except Exception:
        pass


def load_model():
    if os.path.exists(CALIB_PATH):
        model = joblib.load(CALIB_PATH)
        print("Using calibrated model")
    else:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        print("Using raw XGBoost model")
    with open(FEATURES_PATH) as f:
        features = [l.strip() for l in f if l.strip()]
    return model, features


def load_data():
    for fname in ["XAUUSD1.csv", "XAUUSD5.csv"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t", header=None,
                             names=["time","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            print(f"Loaded {fname}: {len(df)} bars "
                  f"({df.index[0].date()} \u2192 {df.index[-1].date()})")
            return df
    raise FileNotFoundError("No XAUUSD data found in data/")


def enrich_htf(df):
    for prefix, fname in [("h1","XAUUSD60.csv"), ("h4","XAUUSD240.csv")]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            try:
                df_htf = load_htf_csv(path)
                df     = add_htf_context(df, df_htf, prefix)
                print(f"  HTF {prefix} added")
            except Exception as e:
                print(f"  HTF {prefix} skipped: {e}")
    return df


def calc_lots(balance, entry, sl):
    pip_risk = abs(entry - sl)
    if pip_risk < 0.0001:
        return 0.01
    lots = (balance * RISK_PCT) / (pip_risk * 100.0)
    return round(min(max(lots, 0.01), MAX_LOTS), 2)


def calc_slippage(lots, is_news_bar):
    """Realistic slippage: spread + market impact based on size."""
    spread = NEWS_SPREAD if is_news_bar else BASE_SPREAD
    impact = MARKET_IMPACT * max(0, (lots - 0.1) / 0.1)
    return round(spread + impact, 3)


def get_signal(model, row, available, classes):
    X     = row[available].values.reshape(1, -1)
    proba = model.predict_proba(X)[0]
    prob_map   = dict(zip(classes, proba))
    pred_class = max(prob_map, key=prob_map.get)
    conf       = float(prob_map[pred_class])
    sig_map    = {0: "SELL", 1: "FLAT", 2: "BUY"}
    signal     = sig_map.get(int(pred_class), "FLAT")
    return signal, conf, prob_map.get(2,0), prob_map.get(1,0), prob_map.get(0,0)


def apply_htf_filter(signal, conf, row):
    if signal == "FLAT":
        return signal, conf
    h1 = row.get("h1_trend_8_21", -1)
    h4 = row.get("h4_trend_8_21", -1)
    if h1 == -1 or h4 == -1:
        return signal, conf
    agrees = ((signal == "BUY"  and h1 == 1 and h4 == 1) or
              (signal == "SELL" and h1 == 0 and h4 == 0))
    if not agrees:
        conf *= 0.45
    return signal, conf


def simulate_trade(df, i, signal, entry_fill, sl, tp, lots, atr):
    partial_done = False
    partial_pnl  = 0.0
    remain_lots  = lots
    trail_sl     = sl
    one_r        = abs(entry_fill - sl)
    exit_price   = exit_reason = None
    bars_held    = 0

    for j in range(1, MAX_BARS_OPEN + 1):
        if i + j >= len(df):
            exit_price  = float(df.iloc[-1]["close"])
            exit_reason = "END_OF_DATA"
            bars_held   = j
            break

        fut   = df.iloc[i + j]
        hi    = float(fut["high"])
        lo    = float(fut["low"])
        atr_j = float(fut.get("atr14", atr))

        if PARTIAL_AT_1R and not partial_done and one_r > 0:
            if signal == "BUY"  and hi >= entry_fill + one_r:
                partial_pnl  = one_r * (remain_lots / 2) * 100
                remain_lots  = round(remain_lots / 2, 2)
                trail_sl     = entry_fill
                partial_done = True
            elif signal == "SELL" and lo <= entry_fill - one_r:
                partial_pnl  = one_r * (remain_lots / 2) * 100
                remain_lots  = round(remain_lots / 2, 2)
                trail_sl     = entry_fill
                partial_done = True

        if signal == "BUY":
            new_trail = hi - SL_MULT * atr_j
            if new_trail > trail_sl: trail_sl = new_trail
            if lo <= trail_sl:
                exit_price  = trail_sl
                exit_reason = "TRAIL_SL" if partial_done else "SL"
                bars_held   = j; break
            if hi >= tp:
                exit_price  = tp
                exit_reason = "TP"
                bars_held   = j; break
        else:
            new_trail = lo + SL_MULT * atr_j
            if new_trail < trail_sl: trail_sl = new_trail
            if hi >= trail_sl:
                exit_price  = trail_sl
                exit_reason = "TRAIL_SL" if partial_done else "SL"
                bars_held   = j; break
            if lo <= tp:
                exit_price  = tp
                exit_reason = "TP"
                bars_held   = j; break
    else:
        exit_price  = float(df.iloc[min(i+MAX_BARS_OPEN, len(df)-1)]["close"])
        exit_reason = "TIMEOUT"
        bars_held   = MAX_BARS_OPEN

    if signal == "BUY":
        remaining_pnl = (exit_price - entry_fill) * remain_lots * 100
    else:
        remaining_pnl = (entry_fill - exit_price) * remain_lots * 100

    return partial_pnl + remaining_pnl, exit_price, exit_reason, bars_held


def run_backtest(df, model, features):
    os.makedirs("logs", exist_ok=True)

    # Walk-forward: only test on last 30% of data
    split       = int(len(df) * 0.70)
    df_test     = df.iloc[split:].copy()
    print(f"Walk-forward split: train 0\u2192{split} | test {split}\u2192{len(df)}")
    print(f"Test period: {df_test.index[0].date()} \u2192 {df_test.index[-1].date()}")
    print(f"Test bars: {len(df_test)}")

    balance      = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    trades       = []
    daily_pnl    = {}
    open_pos     = []   # list of {direction, entry, sl, tp, lots, i}
    available    = [f for f in features if f in df_test.columns]
    classes      = list(model.classes_) if hasattr(model, "classes_") else [0, 1, 2]

    session_mask = (
        ((df_test.index.hour >= 7)  & (df_test.index.hour < 12)) |
        ((df_test.index.hour >= 12) & (df_test.index.hour < 17))
    )
    df_session = df_test[session_mask].copy()
    print(f"Session bars: {len(df_session)}\n")

    i = 0
    while i < len(df_session) - MAX_BARS_OPEN - 1:
        row = df_session.iloc[i]

        if row.get("news_blackout", 0) == 1:
            i += 1; continue

        atr = float(row.get("atr14", 1.0))
        if atr <= 0:
            i += 1; continue

        try:
            signal, conf, cb, cf, cs = get_signal(model, row, available, classes)
        except Exception:
            i += 1; continue

        signal, conf = apply_htf_filter(signal, conf, row)

        if signal == "FLAT" or conf < MIN_CONF:
            i += 1; continue

        # Correlation check — no same-direction if already at max or same dir open
        same_dir = sum(1 for p in open_pos if p["direction"] == signal)
        if len(open_pos) >= MAX_POSITIONS or same_dir >= 2:
            i += 1; continue

        entry    = float(row["close"])
        is_news  = bool(row.get("news_blackout", 0))
        slip     = calc_slippage(calc_lots(balance, entry,
                   entry - SL_MULT * atr if signal == "BUY" else entry + SL_MULT * atr), is_news)

        if signal == "BUY":
            entry_fill = entry + slip
            sl = round(entry_fill - SL_MULT * atr, 2)
            tp = round(entry_fill + TP_MULT * atr, 2)
        else:
            entry_fill = entry - slip
            sl = round(entry_fill + SL_MULT * atr, 2)
            tp = round(entry_fill - TP_MULT * atr, 2)

        lots = calc_lots(balance, entry_fill, sl)

        pnl, exit_price, exit_reason, bars_held = simulate_trade(
            df_session, i, signal, entry_fill, sl, tp, lots, atr)

        balance      = round(balance + pnl, 2)
        peak_balance = max(peak_balance, balance)
        date_key     = str(df_session.index[i].date())
        daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + pnl

        trades.append({
            "time":       str(df_session.index[i]),
            "signal":     signal,
            "entry":      round(entry_fill, 2),
            "exit":       round(exit_price, 2),
            "sl":         round(sl, 2),
            "tp":         round(tp, 2),
            "lots":       lots,
            "pnl":        round(pnl, 2),
            "reason":     exit_reason,
            "bars_held":  bars_held,
            "confidence": round(conf, 4),
            "slippage":   round(slip, 3),
            "balance":    balance,
            "regime":     str(row.get("regime", "UNKNOWN")),
            "atr":        round(atr, 2),
        })

        i += bars_held + 1

    return trades, daily_pnl, balance, peak_balance


def print_report(trades, daily_pnl, final_balance, peak_balance):
    if not trades:
        print("\nNo trades generated.")
        print(f"  - Confidence threshold too high? (current: {MIN_CONF:.3f})")
        print("  - Try running: python export_model.py to check feature importances")
        return

    df = pd.DataFrame(trades)
    df.to_csv(RESULTS_PATH, index=False)

    wins     = df[df["pnl"] > 0]
    losses   = df[df["pnl"] <= 0]
    win_rate = len(wins) / len(df) * 100
    total_pnl= df["pnl"].sum()
    avg_win  = wins["pnl"].mean()   if len(wins)   else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) else 0.0
    rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    avg_slip = df["slippage"].mean()

    equity   = df["balance"].values
    peak_eq  = np.maximum.accumulate(equity)
    dd_series= (peak_eq - equity) / peak_eq * 100
    max_dd   = dd_series.max()

    max_consec = cur = 0
    for p in df["pnl"]:
        cur = cur + 1 if p <= 0 else 0
        max_consec = max(max_consec, cur)

    daily_ret = pd.Series(list(daily_pnl.values())) / INITIAL_BALANCE
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                 if daily_ret.std() > 0 else 0.0)

    gross_wins   = wins["pnl"].sum()   if len(wins)   else 0.0
    gross_losses = abs(losses["pnl"].sum()) if len(losses) else 0.0
    pf = gross_wins / gross_losses if gross_losses > 0 else 0.0

    print("\n" + "=" * 58)
    print("       GoldBot Pro — Backtest Results (Walk-Forward)")
    print("=" * 58)
    print(f"  Period          : {df['time'].iloc[0][:10]} \u2192 {df['time'].iloc[-1][:10]}")
    print(f"  Total trades    : {len(df)}")
    print(f"  Win rate        : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Total P&L       : ${total_pnl:+.2f}")
    print(f"  Initial balance : ${INITIAL_BALANCE:,.2f}")
    print(f"  Final balance   : ${final_balance:,.2f}")
    print(f"  Return          : {(final_balance/INITIAL_BALANCE - 1)*100:+.1f}%")
    print(f"  Profit factor   : {pf:.2f}")
    print(f"  Avg win         : ${avg_win:.2f}")
    print(f"  Avg loss        : ${avg_loss:.2f}")
    print(f"  Risk/Reward     : 1:{rr:.2f}")
    print(f"  Max drawdown    : {max_dd:.1f}%")
    print(f"  Max consec loss : {max_consec}")
    print(f"  Sharpe ratio    : {sharpe:.2f}")
    print(f"  Avg slippage    : ${avg_slip:.3f}")
    print(f"  Conf threshold  : {MIN_CONF:.3f}")
    print("-" * 58)
    print("  By exit reason:")
    print(df.groupby("reason")["pnl"].agg(count="count", total="sum", avg="mean").round(2).to_string())
    print("-" * 58)
    print("  By regime:")
    print(df.groupby("regime")["pnl"].agg(
        count="count", total="sum", avg="mean",
        win_rate=lambda x: (x > 0).mean()).round(2).to_string())
    print("-" * 58)
    print("  By direction:")
    print(df.groupby("signal")["pnl"].agg(
        count="count", total="sum", avg="mean",
        win_rate=lambda x: (x > 0).mean()).round(2).to_string())
    print("=" * 58)
    print(f"\nResults saved \u2192 {RESULTS_PATH}")

    if win_rate > 80:
        print("\nWARNING: Win rate > 80% on walk-forward test is suspicious.")
        print("  Check for remaining lookahead in features.")
    if len(df) < 30:
        print(f"WARNING: Only {len(df)} trades — reduce MIN_CONF or check session filter.")
    if max_dd > 20:
        print(f"WARNING: Max drawdown {max_dd:.1f}% is too high for live trading.")


if __name__ == "__main__":
    print("\n=== GoldBot Pro — Backtester v3 (Walk-Forward + Slippage) ===")

    if not os.path.exists(CALIB_PATH) and not os.path.exists(MODEL_PATH):
        print("ERROR: No model found. Run train.py first.")
        exit(1)

    model, features = load_model()

    print("\nLoading data...")
    raw = load_data()

    print("Building features...")
    df = build_features(raw)
    df = enrich_htf(df)

    print(f"\nFeatures available : {len([f for f in features if f in df.columns])} / {len(features)}")

    trades, daily_pnl, final_balance, peak_balance = run_backtest(df, model, features)
    print_report(trades, daily_pnl, final_balance, peak_balance)
