"""
GoldBot Pro — Backtester v2
Replays historical XAUUSD data through the actual XGBoost model.
Run: python backtest.py
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime, timezone

from features import build_features, add_htf_context, load_htf_csv

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
MODEL_PATH      = "models/xgb_model.json"
CALIB_PATH      = "models/calibrated_model.pkl"
FEATURES_PATH   = "models/feature_cols.txt"
PARAMS_PATH     = "models/best_params.json"
RESULTS_PATH    = "logs/backtest_results.csv"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.005
SL_MULT         = 1.0
TP_MULT         = 2.0
SPREAD          = 0.30
MAX_LOTS        = 2.0
PARTIAL_AT_1R   = True
MAX_BARS_OPEN   = 20

# Load confidence threshold from training if available
MIN_CONF = 0.52
if os.path.exists(PARAMS_PATH):
    try:
        with open(PARAMS_PATH) as f:
            saved = json.load(f)
        MIN_CONF = saved.get("conf_threshold", MIN_CONF)
        print(f"Loaded conf threshold from training: {MIN_CONF:.3f}")
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
    # FIX: use M1 for backtest — same TF the model was trained on
    # M5 was used before but model trained on M1 data
    for fname in ["XAUUSD1.csv", "XAUUSD5.csv"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t", header=None,
                             names=["time","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            print(f"Loaded {fname}: {len(df)} bars "
                  f"({df.index[0].date()} → {df.index[-1].date()})")
            return df
    raise FileNotFoundError("No XAUUSD data file found in data/")


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


def get_signal(model, row, available, classes):
    """
    FIX: original used np.argmax(proba) directly — breaks if
    XGBoost reorders classes. Use classes_ map instead.
    """
    X     = row[available].values.reshape(1, -1)
    proba = model.predict_proba(X)[0]

    prob_map  = dict(zip(classes, proba))
    pred_class = max(prob_map, key=prob_map.get)
    conf       = float(prob_map[pred_class])

    sig_map = {0: "SELL", 1: "FLAT", 2: "BUY"}
    signal  = sig_map.get(int(pred_class), "FLAT")

    conf_buy  = prob_map.get(2, 0.0)
    conf_flat = prob_map.get(1, 0.0)
    conf_sell = prob_map.get(0, 0.0)

    return signal, conf, conf_buy, conf_flat, conf_sell


def apply_htf_filter(signal, conf, row):
    """
    FIX: original had no HTF filter in backtest
    — but signal_server applies one. Must match.
    """
    if signal == "FLAT":
        return signal, conf

    h1_bull = row.get("h1_trend_8_21", -1)
    h4_bull = row.get("h4_trend_8_21", -1)

    if h1_bull == -1 or h4_bull == -1:
        return signal, conf

    htf_agrees = (
        (signal == "BUY"  and h1_bull == 1 and h4_bull == 1) or
        (signal == "SELL" and h1_bull == 0 and h4_bull == 0)
    )

    if not htf_agrees:
        conf *= 0.45

    return signal, conf


def simulate_trade(df_session, i, signal, entry_fill, sl, tp, lots, atr):
    """
    FIX 1: original had a bug where partial close set lots=lots/2
    but then used the NEW lots value for full PnL calculation — 
    double-counted partial close PnL.

    FIX 2: trailing stop was updating on every bar using current bar ATR
    but SL trail direction check was wrong for SELL trades.

    FIX 3: time exit used min(i+MAX_BARS_OPEN, len-1) but should
    use last bar of the open window, not the session slice end.
    """
    partial_done = False
    partial_pnl  = 0.0
    full_lots    = lots
    remain_lots  = lots
    trail_sl     = sl
    exit_price   = None
    exit_reason  = None
    bars_held    = 0
    one_r        = abs(entry_fill - sl)

    for j in range(1, MAX_BARS_OPEN + 1):
        if i + j >= len(df_session):
            # End of data
            exit_price  = float(df_session.iloc[-1]["close"])
            exit_reason = "END_OF_DATA"
            bars_held   = j
            break

        future = df_session.iloc[i + j]
        hi     = float(future["high"])
        lo     = float(future["low"])
        atr_j  = float(future.get("atr14", atr))

        # ── Partial close at 1R ──────────────────────────────────────
        if PARTIAL_AT_1R and not partial_done and one_r > 0:
            if signal == "BUY" and hi >= entry_fill + one_r:
                # Close half at 1R
                partial_pnl  = one_r * (remain_lots / 2) * 100
                remain_lots  = round(remain_lots / 2, 2)
                trail_sl     = entry_fill   # move to breakeven
                partial_done = True
            elif signal == "SELL" and lo <= entry_fill - one_r:
                partial_pnl  = one_r * (remain_lots / 2) * 100
                remain_lots  = round(remain_lots / 2, 2)
                trail_sl     = entry_fill
                partial_done = True

        # ── Trailing stop update ─────────────────────────────────────
        if signal == "BUY":
            new_trail = hi - SL_MULT * atr_j
            if new_trail > trail_sl:
                trail_sl = new_trail
        else:
            new_trail = lo + SL_MULT * atr_j
            if new_trail < trail_sl:
                trail_sl = new_trail

        # ── Check exits — SL before TP (first touch wins) ───────────
        if signal == "BUY":
            if lo <= trail_sl:
                exit_price  = trail_sl
                exit_reason = "SL" if not partial_done else "TRAIL_SL"
                bars_held   = j
                break
            if hi >= tp:
                exit_price  = tp
                exit_reason = "TP"
                bars_held   = j
                break
        else:
            if hi >= trail_sl:
                exit_price  = trail_sl
                exit_reason = "SL" if not partial_done else "TRAIL_SL"
                bars_held   = j
                break
            if lo <= tp:
                exit_price  = tp
                exit_reason = "TP"
                bars_held   = j
                break
    else:
        # Time-based exit — close at last bar close
        exit_price  = float(df_session.iloc[
            min(i + MAX_BARS_OPEN, len(df_session) - 1)
        ]["close"])
        exit_reason = "TIMEOUT"
        bars_held   = MAX_BARS_OPEN

    # ── PnL calculation ──────────────────────────────────────────────
    # FIX: partial_pnl already calculated above
    # remaining position closed at exit_price
    if signal == "BUY":
        remaining_pnl = (exit_price - entry_fill) * remain_lots * 100
    else:
        remaining_pnl = (entry_fill - exit_price) * remain_lots * 100

    total_pnl = partial_pnl + remaining_pnl

    return total_pnl, exit_price, exit_reason, bars_held, remain_lots


def run_backtest(df, model, features):
    os.makedirs("logs", exist_ok=True)

    balance      = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    trades       = []
    daily_pnl    = {}
    skipped      = 0

    available = [f for f in features if f in df.columns]
    missing   = set(features) - set(available)
    if missing:
        print(f"  Missing features ({len(missing)}): {sorted(missing)[:5]}...")

    # Get model class order
    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1, 2]

    # Session filter — London + NY only
    session_mask = (
        ((df.index.hour >= 7)  & (df.index.hour < 12)) |
        ((df.index.hour >= 12) & (df.index.hour < 17))
    )
    df_session = df[session_mask].copy()
    print(f"Session-filtered bars: {len(df_session)}")
    print(f"Running backtest (MIN_CONF={MIN_CONF:.3f})...\n")

    i = 0
    while i < len(df_session) - MAX_BARS_OPEN - 1:
        row = df_session.iloc[i]

        # Skip news blackout bars
        if row.get("news_blackout", 0) == 1:
            i += 1
            skipped += 1
            continue

        atr = float(row.get("atr14", 1.0))
        if atr <= 0:
            i += 1
            continue

        # ── Get signal ───────────────────────────────────────────────
        try:
            signal, conf, cb, cf, cs = get_signal(model, row, available, classes)
        except Exception as e:
            i += 1
            continue

        # ── HTF filter ───────────────────────────────────────────────
        signal, conf = apply_htf_filter(signal, conf, row)

        if signal == "FLAT" or conf < MIN_CONF:
            i += 1
            continue

        # ── Entry fill with spread ───────────────────────────────────
        entry = float(row["close"])
        if signal == "BUY":
            entry_fill = entry + SPREAD
            sl = round(entry_fill - SL_MULT * atr, 2)
            tp = round(entry_fill + TP_MULT * atr, 2)
        else:
            entry_fill = entry - SPREAD
            sl = round(entry_fill + SL_MULT * atr, 2)
            tp = round(entry_fill - TP_MULT * atr, 2)

        lots = calc_lots(balance, entry_fill, sl)

        # ── Simulate ─────────────────────────────────────────────────
        pnl, exit_price, exit_reason, bars_held, remain_lots = simulate_trade(
            df_session, i, signal, entry_fill, sl, tp, lots, atr
        )

        balance      = round(balance + pnl, 2)
        peak_balance = max(peak_balance, balance)

        date_key = str(df_session.index[i].date())
        daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + pnl

        regime = str(row.get("regime", "UNKNOWN"))

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
            "conf_buy":   round(cb, 4),
            "conf_sell":  round(cs, 4),
            "balance":    balance,
            "regime":     regime,
            "atr":        round(atr, 2),
        })

        # Skip forward past open trade bars
        i += bars_held + 1

    print(f"Skipped (news/ATR): {skipped} bars")
    return trades, daily_pnl, balance, peak_balance


def print_report(trades, daily_pnl, final_balance, peak_balance):
    if not trades:
        print("\nNo trades generated.")
        print("Possible causes:")
        print(f"  - Confidence threshold too high (current: {MIN_CONF:.3f})")
        print("  - Model never predicts BUY/SELL")
        print("  - All bars in news blackout or wrong session")
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

    # Max drawdown from equity curve
    equity   = df["balance"].values
    peak_eq  = np.maximum.accumulate(equity)
    dd_series= (peak_eq - equity) / peak_eq * 100
    max_dd   = dd_series.max()

    # Consecutive losses
    max_consec = cur_consec = 0
    for p in df["pnl"]:
        cur_consec = cur_consec + 1 if p <= 0 else 0
        max_consec = max(max_consec, cur_consec)

    # Sharpe
    daily_returns = pd.Series(list(daily_pnl.values())) / INITIAL_BALANCE
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0.0)

    # Profit factor
    gross_wins   = wins["pnl"].sum()   if len(wins)   else 0.0
    gross_losses = abs(losses["pnl"].sum()) if len(losses) else 0.0
    pf = gross_wins / gross_losses if gross_losses > 0 else 0.0

    print("\n" + "=" * 55)
    print("        GoldBot Pro — Backtest Results")
    print("=" * 55)
    print(f"  Period          : {df['time'].iloc[0][:10]} → "
          f"{df['time'].iloc[-1][:10]}")
    print(f"  Total trades    : {len(df)}")
    print(f"  Win rate        : {win_rate:.1f}%  "
          f"({len(wins)}W / {len(losses)}L)")
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
    print(f"  Conf threshold  : {MIN_CONF:.3f}")
    print("-" * 55)
    print("  By exit reason:")
    reason_stats = df.groupby("reason")["pnl"].agg(
        count="count", total="sum", avg="mean"
    ).round(2)
    print(reason_stats.to_string())
    print("-" * 55)
    print("  By regime:")
    regime_stats = df.groupby("regime")["pnl"].agg(
        count="count", total="sum", avg="mean", win_rate=lambda x: (x > 0).mean()
    ).round(2)
    print(regime_stats.to_string())
    print("-" * 55)
    print("  By signal direction:")
    dir_stats = df.groupby("signal")["pnl"].agg(
        count="count", total="sum", avg="mean", win_rate=lambda x: (x > 0).mean()
    ).round(2)
    print(dir_stats.to_string())
    print("=" * 55)
    print(f"\nFull results → {RESULTS_PATH}")

    # Warn if results look suspicious
    if win_rate > 85:
        print("\nWARNING: Win rate > 85% — possible lookahead bias in features.")
    if max_dd < 1.0 and len(df) > 50:
        print("WARNING: Max drawdown < 1% — results may be unrealistic.")
    if len(df) < 20:
        print("WARNING: Fewer than 20 trades — reduce MIN_CONF or check data.")


if __name__ == "__main__":
    print("\n=== GoldBot Pro — Backtester v2 ===")

    if not os.path.exists(CALIB_PATH) and not os.path.exists(MODEL_PATH):
        print("ERROR: No model found. Run train.py first.")
        exit(1)

    model, features = load_model()

    print("\nLoading data...")
    raw = load_data()

    print("Building features...")
    df = build_features(raw)
    df = enrich_htf(df)

    print(f"\nFeatures available : {len([f for f in features if f in df.columns])}"
          f" / {len(features)}")

    trades, daily_pnl, final_balance, peak_balance = run_backtest(
        df, model, features
    )
    print_report(trades, daily_pnl, final_balance, peak_balance)
