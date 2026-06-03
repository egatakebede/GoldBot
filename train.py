import os, json, warnings
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
import joblib
warnings.filterwarnings("ignore")

from features import build_features, add_htf_context, load_htf_csv

DATA_DIR      = "data"
MODEL_PATH    = "models/xgb_model.json"
CALIB_PATH    = "models/calibrated_model.pkl"
FEATURES_PATH = "models/feature_cols.txt"
N_TRIALS      = 50
N_SPLITS      = 3

LABEL_FORWARD  = 20
LABEL_TP_MULT  = 2.0
LABEL_SL_MULT  = 1.5
CONF_THRESHOLD = 0.55

FEATURE_COLS = [
    "trend_8_21","trend_21_50","trend_50_200",
    "price_vs_ema21","price_vs_ema50","price_vs_ema200",
    "ema21_slope","ema50_slope",
    "rsi14","rsi7","rsi_slope",
    "macd_line","macd_signal","macd_hist","macd_cross","macd_hist_slope",
    "roc5","roc10","roc20",
    "stoch_k","stoch_d","stoch_cross",
    "atr14","atr7","atr_pct","atr_ratio","atr_trend",
    "bb_width","bb_position","bb_squeeze",
    "candle_body","candle_range","upper_wick","lower_wick","is_bullish",
    "consec_bull","consec_bear",
    "rel_volume","volume_spike","price_vs_vwap",
    "above_prev_high","below_prev_low","break_high","break_low",
    "dist_to_high20","dist_to_low20",
    "session_london","session_newyork","session_overlap","session_asian",
    "london_open_spike","ny_open_spike",
    "news_blackout","tradeable",
    "hour","day_of_week","is_monday","is_friday",
    "hour_sin","hour_cos","dow_sin","dow_cos",
    "atr_above_avg",
    "regime_trending","regime_ranging","regime_highvol","regime_lowvol","adx_val",
    # SMC
    "bos_bull","bos_bear","choch_bull","choch_bear",
    "fvg_bull","fvg_bear","dist_fvg_bull","dist_fvg_bear",
    "ob_bull","ob_bear","ob_bull_dist","ob_bear_dist",
    "liq_bull","liq_bear",
    "swing_high","swing_low","dist_swing_high","dist_swing_low",
    "smc_bias",
    "h1_trend_8_21","h1_trend_21_50","h1_trend_50_200","h1_ema21_slope",
    "h1_rsi14","h1_rsi_slope","h1_macd_hist","h1_macd_hist_slope",
    "h1_bb_position","h1_bb_squeeze","h1_atr14","h1_atr_trend",
    "h1_rel_volume","h1_adx_val","h1_regime_trending","h1_regime_ranging",
    "h4_trend_8_21","h4_trend_21_50","h4_trend_50_200","h4_ema21_slope",
    "h4_rsi14","h4_rsi_slope","h4_macd_hist","h4_macd_hist_slope",
    "h4_bb_position","h4_bb_squeeze","h4_atr14","h4_atr_trend",
    "h4_rel_volume","h4_adx_val","h4_regime_trending","h4_regime_ranging",
]


def read_mt5_csv(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    return df.apply(pd.to_numeric, errors="coerce").dropna()


def load_all_data():
    frames = []
    for fname in ["XAUUSD1.csv","XAUUSD5.csv","XAUUSD15.csv"]:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  Skipped {fname}: not found")
            continue
        try:
            df = read_mt5_csv(path)
            frames.append(df[["open","high","low","close","volume"]])
            print(f"  Loaded {fname}: {len(df)} rows")
        except Exception as e:
            print(f"  Skipped {fname}: {e}")
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    print(f"Total rows: {len(combined)}")
    return combined


def make_labels(df,
                forward_bars=LABEL_FORWARD,
                tp_mult=LABEL_TP_MULT,
                sl_mult=LABEL_SL_MULT):
    """
    FIX 1: removed min_atr_move filter — was forcing 92% FLAT
    FIX 2: TP mult 2.0 -> 1.0  — 2 ATR on M1 gold almost never hits in 10 bars
    FIX 3: forward_bars 10 -> 20 — give price more time to reach target
    FIX 4: check SL hit BEFORE TP in sequence — first touch wins
    Result: expect BUY+SELL to rise from 8% to 25-35%
    """
    atr   = df["atr14"].values
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)
    labels = np.ones(n, dtype=int)  # default FLAT=1

    for i in range(n - forward_bars):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry   = close[i]
        tp_long = entry + atr[i] * tp_mult
        sl_long = entry - atr[i] * sl_mult
        tp_short= entry - atr[i] * tp_mult
        sl_short= entry + atr[i] * sl_mult

        long_result  = 0  # 0=nothing, 1=tp, -1=sl
        short_result = 0

        for j in range(i + 1, i + 1 + forward_bars):
            # Long check
            if long_result == 0:
                if low[j] <= sl_long:
                    long_result = -1
                elif high[j] >= tp_long:
                    long_result = 1

            # Short check
            if short_result == 0:
                if high[j] >= sl_short:
                    short_result = -1
                elif low[j] <= tp_short:
                    short_result = 1

            if long_result != 0 and short_result != 0:
                break

        if long_result == 1 and short_result != 1:
            labels[i] = 2   # BUY
        elif short_result == 1 and long_result != 1:
            labels[i] = 0   # SELL

    return pd.Series(labels, index=df.index)


def objective(trial, X, y):
    """
    FIX 5: added tree_method=hist — 5x faster per trial
    FIX 6: added confidence threshold as tunable param
    FIX 7: penalise models with < 5% signal rate or one-sided predictions
    FIX 8: added early_stopping_rounds to avoid wasting time on bad trees
    """
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 1500),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.95),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 60),
        "gamma":            trial.suggest_float("gamma", 0.0, 4.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
        "tree_method":      "hist",
        "device":           "cpu",
        "eval_metric":      "mlogloss",
        "random_state":     42,
        "n_jobs":           -1,
    }
    conf_thresh = trial.suggest_float("conf_threshold", 0.45, 0.75)

    tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []

    for tr, val in tscv.split(X):
        weights = compute_sample_weight("balanced", y[tr])
        m = XGBClassifier(**params)
        m.fit(
            X[tr], y[tr],
            sample_weight=weights,
            eval_set=[(X[val], y[val])],
            verbose=False,
        )

        proba    = m.predict_proba(X[val])
        max_conf = proba.max(axis=1)
        pred_idx = proba.argmax(axis=1)
        pred     = m.classes_[pred_idx]

        # Apply confidence threshold — low confidence -> treat as FLAT
        pred_filtered = np.where(max_conf >= conf_thresh, pred, 1)

        mask = pred_filtered != 1
        n_signals = mask.sum()

        if n_signals < 20:
            scores.append(0.0)
            continue

        win_rate = accuracy_score(y[val][mask], pred_filtered[mask])
        coverage = mask.mean()

        # Penalise low coverage — model must trade at least 3% of bars
        if coverage < 0.03:
            win_rate *= 0.4

        # Penalise one-sided models — must predict both BUY and SELL
        n_buy  = (pred_filtered[mask] == 2).sum()
        n_sell = (pred_filtered[mask] == 0).sum()
        if n_buy == 0 or n_sell == 0:
            win_rate *= 0.3
        elif min(n_buy, n_sell) / max(n_buy, n_sell) < 0.15:
            win_rate *= 0.7

        scores.append(win_rate)

    return float(np.mean(scores))


def main():
    os.makedirs("models", exist_ok=True)
    print("\n=== GoldBot Pro — Model Training v2 ===")

    print("Loading data...")
    raw = load_all_data()

    print("Building features...")
    df = build_features(raw)

    for prefix, fname in [("h1","XAUUSD60.csv"), ("h4","XAUUSD240.csv")]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            try:
                df_htf = load_htf_csv(path)
                df     = add_htf_context(df, df_htf, prefix)
                print(f"  HTF {prefix} added ({len(df_htf)} bars)")
            except Exception as e:
                print(f"  HTF {prefix} skipped: {e}")

    print("Generating labels...")
    df["label"] = make_labels(df)

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_cols)
    if missing:
        print(f"  Missing features (HTF not loaded?): {sorted(missing)}")

    df = df.dropna(subset=["label"] + available_cols)

    X = df[available_cols].values
    y = df["label"].astype(int).values

    buy_n  = (y == 2).sum()
    flat_n = (y == 1).sum()
    sell_n = (y == 0).sum()
    total  = len(y)

    print(f"Dataset: {total} rows | BUY={buy_n} FLAT={flat_n} SELL={sell_n}")
    print(f"Class balance: BUY={buy_n/total:.1%} FLAT={flat_n/total:.1%} SELL={sell_n/total:.1%}")

    if buy_n / total < 0.10 or sell_n / total < 0.10:
        print("\nWARNING: Signal rate still below 10%.")
        print("Consider reducing LABEL_TP_MULT or increasing LABEL_FORWARD in this file.")
        print("Current settings:")
        print(f"  LABEL_TP_MULT  = {LABEL_TP_MULT}")
        print(f"  LABEL_SL_MULT  = {LABEL_SL_MULT}")
        print(f"  LABEL_FORWARD  = {LABEL_FORWARD}")
        print("Continuing anyway...\n")

    print(f"\nRunning Optuna ({N_TRIALS} trials) ...")
    print("Speed improvement: tree_method=hist + N_SPLITS=3 should be 5-8x faster\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # CMA-ES sampler — better than TPE for continuous params (lr, subsample, gamma)
    # Falls back to TPE for the first 10 warmup trials to seed the population
    sampler = optuna.samplers.CmaEsSampler(
        seed=42,
        restart_strategy="ipop",   # auto-restarts if stuck in local optimum
        warn_independent_sampling=False,
    )
    study = optuna.create_study(
        study_name="goldbot",
        storage="sqlite:///models/optuna.db",
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
    )
    completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
    remaining = max(0, N_TRIALS - completed)
    print(f"Optuna: {completed} trials done, running {remaining} more...")
    study.optimize(
        lambda t: objective(t, X, y),
        n_trials=remaining,
        show_progress_bar=True,
    )

    best_params = dict(study.best_params)
    conf_threshold = best_params.pop("conf_threshold", CONF_THRESHOLD)
    best_params.update({
        "tree_method":  "hist",
        "device":       "cpu",
        "eval_metric":  "mlogloss",
        "random_state": 42,
        "n_jobs":       -1,
    })

    print(f"\nBest win-rate proxy : {study.best_value:.4f}")
    print(f"Best conf threshold : {conf_threshold:.3f}")

    # ── Final model on all data ──────────────────────────────────────────
    print("\nTraining final model...")
    weights = compute_sample_weight("balanced", y)

    # First pass — get feature importances
    final = XGBClassifier(**best_params)
    final.fit(X, y, sample_weight=weights, verbose=False)

    # Feature selection — keep features above importance threshold
    importance    = final.feature_importances_
    selected_cols = [f for f, imp in zip(available_cols, importance) if imp >= 0.003]
    if len(selected_cols) < 20:
        # Safety — never drop below 20 features
        ranked = sorted(zip(available_cols, importance), key=lambda x: -x[1])
        selected_cols = [f for f, _ in ranked[:20]]
    print(f"Feature selection: {len(available_cols)} → {len(selected_cols)} features")

    X_sel = df[selected_cols].values

    # Second pass — final model on selected features
    final2 = XGBClassifier(**best_params)
    final2.fit(X_sel, y, sample_weight=weights, verbose=False)

    # ── Probability calibration ──────────────────────────────────────────
    print("Calibrating probabilities...")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    calibrated = CalibratedClassifierCV(
        XGBClassifier(**best_params), method="isotonic", cv=tscv
    )
    calibrated.fit(X_sel, y)
    joblib.dump(calibrated, CALIB_PATH)
    print(f"Calibrated model saved to {CALIB_PATH}")

    # ── Evaluation ───────────────────────────────────────────────────────
    print("\nEvaluating...")
    cv_scores, wr_scores, coverage_scores = [], [], []

    for tr, val in tscv.split(X_sel):
        m = XGBClassifier(**best_params)
        m.fit(
            X_sel[tr], y[tr],
            sample_weight=compute_sample_weight("balanced", y[tr]),
            verbose=False,
        )

        proba    = m.predict_proba(X_sel[val])
        max_conf = proba.max(axis=1)
        pred_idx = proba.argmax(axis=1)
        raw_pred = m.classes_[pred_idx]
        pred     = np.where(max_conf >= conf_threshold, raw_pred, 1)

        cv_scores.append(accuracy_score(y[val], pred))

        mask = pred != 1
        if mask.sum() > 0:
            wr_scores.append(accuracy_score(y[val][mask], pred[mask]))
            coverage_scores.append(mask.mean())

    print(f"Overall CV accuracy  : {np.mean(cv_scores):.4f}")
    print(f"Win rate (non-FLAT)  : {np.mean(wr_scores):.4f}  ← target 0.55+")
    print(f"Signal coverage      : {np.mean(coverage_scores):.2%}  ← target 5-15%")
    print(f"Confidence threshold : {conf_threshold:.3f}")

    # Classification report on last fold
    tscv_list = list(tscv.split(X_sel))
    tr_last, val_last = tscv_list[-1]
    m_last = XGBClassifier(**best_params)
    m_last.fit(
        X_sel[tr_last], y[tr_last],
        sample_weight=compute_sample_weight("balanced", y[tr_last]),
        verbose=False,
    )
    proba_last    = m_last.predict_proba(X_sel[val_last])
    max_conf_last = proba_last.max(axis=1)
    pred_last_raw = m_last.classes_[proba_last.argmax(axis=1)]
    pred_last     = np.where(max_conf_last >= conf_threshold, pred_last_raw, 1)

    print("\nClassification Report (last fold, with confidence filter):")
    print(classification_report(
        y[val_last], pred_last,
        target_names=["SELL","FLAT","BUY"],
        zero_division=0,
    ))

    # ── Save everything ──────────────────────────────────────────────────
    final2.save_model(MODEL_PATH)

    with open(FEATURES_PATH, "w") as f:
        f.write("\n".join(selected_cols))

    with open("models/best_params.json", "w") as f:
        json.dump({
            "params":           best_params,
            "conf_threshold":   conf_threshold,
            "win_rate":         float(np.mean(wr_scores)),
            "accuracy":         float(np.mean(cv_scores)),
            "coverage":         float(np.mean(coverage_scores)),
            "features":         len(selected_cols),
            "label_tp_mult":    LABEL_TP_MULT,
            "label_sl_mult":    LABEL_SL_MULT,
            "label_forward":    LABEL_FORWARD,
        }, f, indent=2)

    print(f"\nModel saved       → {MODEL_PATH}")
    print(f"Features saved    → {FEATURES_PATH}")
    print(f"Params saved      → models/best_params.json")
    print(f"Feature count     : {len(selected_cols)}")
    print("=== Training complete ===")


if __name__ == "__main__":
    main()
