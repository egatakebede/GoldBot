import os, time, json, schedule, logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV

import config, notifier
from features import build_features, add_htf_context, load_htf_csv

def utcnow(): return datetime.now(timezone.utc)

MODEL_PATH     = config.MODEL_PATH
MODEL_PATH_UBJ = getattr(config, "MODEL_PATH_UBJ", "models/xgb_model.ubj")
CALIB_PATH     = "models/calibrated_model.pkl"
FEATURES_PATH  = "models/feature_cols.txt"
DATA_DIR       = "data"
WIN_RATE_MIN   = 0.60   # trigger emergency retrain if win rate drops below this

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
    "news_blackout","tradeable",
    "hour","day_of_week","is_monday","is_friday",
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
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.startswith("XAUUSD") or not fname.endswith(".csv"):
            continue
        path = os.path.join(DATA_DIR, fname)
        try:
            df = read_mt5_csv(path)
            if len(df) < 100:
                continue
            frames.append(df[["open","high","low","close","volume"]])
            print(f"  Loaded {fname}: {len(df)} rows")
        except Exception as e:
            print(f"  Skipped {fname}: {e}")
    if not frames:
        raise ValueError("No valid XAUUSD CSV files found in data/")
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    print(f"[Retrain] Total rows: {len(combined)}")
    return combined

def make_labels(df, forward_bars=20, tp_mult=2.0, sl_mult=1.5):
    """Matches train.py label logic exactly — sequential SL/TP check, first touch wins."""
    atr   = df["atr14"].values
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)
    labels = np.ones(n, dtype=int)
    for i in range(n - forward_bars):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        entry    = close[i]
        tp_long  = entry + atr[i] * tp_mult
        sl_long  = entry - atr[i] * sl_mult
        tp_short = entry - atr[i] * tp_mult
        sl_short = entry + atr[i] * sl_mult
        long_result = short_result = 0
        for j in range(i + 1, i + 1 + forward_bars):
            if long_result == 0:
                if low[j] <= sl_long:   long_result = -1
                elif high[j] >= tp_long: long_result = 1
            if short_result == 0:
                if high[j] >= sl_short:  short_result = -1
                elif low[j] <= tp_short: short_result = 1
            if long_result != 0 and short_result != 0:
                break
        if long_result == 1 and short_result != 1:
            labels[i] = 2
        elif short_result == 1 and long_result != 1:
            labels[i] = 0
    return pd.Series(labels, index=df.index)


def retrain():
    print(f"[Retrain] Starting at {utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    os.makedirs("models", exist_ok=True)
    try:
        raw = load_all_data()
        df  = build_features(raw)

        for prefix, fname in [("h1","XAUUSD60.csv"), ("h4","XAUUSD240.csv")]:
            path = os.path.join(DATA_DIR, fname)
            if os.path.exists(path):
                try:
                    df = add_htf_context(df, load_htf_csv(path), prefix)
                    print(f"  HTF {prefix} added")
                except Exception as e:
                    print(f"  HTF {prefix} skipped: {e}")

        df_m5         = df[df.index.minute % 5 == 0].copy()
        df_m5["label"] = make_labels(df_m5)

        available = [c for c in FEATURE_COLS if c in df_m5.columns]
        df_m5     = df_m5.dropna(subset=["label"] + available)
        X         = df_m5[available].values
        y         = df_m5["label"].astype(int).values

        print(f"[Retrain] {len(X)} rows | BUY={sum(y==2)} FLAT={sum(y==1)} SELL={sum(y==0)}")

        params = {"n_estimators":300, "max_depth":5, "learning_rate":0.05,
                  "subsample":0.8, "colsample_bytree":0.8, "min_child_weight":10,
                  "gamma":0.5, "reg_alpha":1.0, "reg_lambda":2.0,
                  "eval_metric":"mlogloss", "random_state":42, "n_jobs":-1}

        tscv = TimeSeriesSplit(n_splits=5)
        scores, wr_scores = [], []
        for fold, (tr, val) in enumerate(tscv.split(X)):
            m = XGBClassifier(**params)
            m.fit(X[tr], y[tr],
                  sample_weight=compute_sample_weight("balanced", y[tr]),
                  eval_set=[(X[val], y[val])], verbose=False)
            preds = m.predict(X[val])
            scores.append(accuracy_score(y[val], preds))
            mask = preds != 1
            if mask.sum() > 0:
                wr_scores.append(accuracy_score(y[val][mask], preds[mask]))
            print(f"  Fold {fold+1}: acc={scores[-1]:.4f} wr={wr_scores[-1] if wr_scores else 0:.4f}")

        mean_acc = np.mean(scores)
        mean_wr  = np.mean(wr_scores) if wr_scores else 0.0
        print(f"[Retrain] CV accuracy: {mean_acc:.4f} | Win rate: {mean_wr:.4f}")

        final = XGBClassifier(**params)
        final.fit(X, y, sample_weight=compute_sample_weight("balanced", y), verbose=False)

        # Calibrate
        calibrated = CalibratedClassifierCV(XGBClassifier(**params), method="isotonic", cv=tscv)
        calibrated.fit(X, y)

        final.save_model(MODEL_PATH)
        # Save versioned copy
        ts          = utcnow().strftime("%Y%m%d_%H%M")
        version_path = f"models/xgb_model_{ts}.json"
        final.save_model(version_path)
        # Keep only last N versions
        import glob
        versions = sorted(glob.glob("models/xgb_model_*.json"))
        for old in versions[:-config.MODEL_VERSIONS]:
            os.remove(old)
            print(f"  Removed old model: {old}")
        joblib.dump(calibrated, CALIB_PATH)
        with open(FEATURES_PATH, "w") as f:
            f.write("\n".join(available))
        with open("models/best_params.json", "w") as f:
            json.dump({"params": params, "accuracy": mean_acc, "conf_threshold": config.MIN_CONFIDENCE,
                       "win_rate": mean_wr, "features": len(available)}, f, indent=2)

        print(f"[Retrain] Saved to {MODEL_PATH}")
        notifier.retrain_done(mean_acc, len(available))
        notifier.send(f"Retrain complete\nAccuracy: {mean_acc:.1%}\nWin rate: {mean_wr:.1%}\nFeatures: {len(available)}")
        return final, mean_acc, mean_wr

    except Exception as e:
        print(f"[Retrain] FAILED: {e}")
        notifier.send(f"RETRAIN FAILED: {e}")
        return None, 0.0, 0.0


def check_model_health():
    """Read last N trades from log and check live win rate. Trigger retrain if degraded."""
    log_path = config.TRADE_LOG_PATH
    if not os.path.exists(log_path):
        return
    try:
        trades = pd.read_csv(log_path)
        recent = trades.tail(30)
        if len(recent) < 10:
            return
        wins    = (recent["pnl"].astype(float) > 0).sum()
        win_rate = wins / len(recent)
        print(f"[HealthCheck] Last {len(recent)} trades | Win rate: {win_rate:.1%}")
        if win_rate < WIN_RATE_MIN:
            print(f"[HealthCheck] Win rate {win_rate:.1%} below {WIN_RATE_MIN:.1%} — triggering emergency retrain")
            notifier.send(f"⚠️ Win rate dropped to {win_rate:.1%} — emergency retrain triggered")
            retrain()
    except Exception as e:
        logging.exception("[HealthCheck] Error: %s", e)


def load_model():
    if not os.path.exists(MODEL_PATH) and not os.path.exists(MODEL_PATH_UBJ) and not os.path.exists(CALIB_PATH):
        print("[Retrain] No model found — running initial train...")
        model_obj, acc, wr = retrain()
        if model_obj is None:
            raise RuntimeError("[Retrain] Initial training failed — cannot load model.")
    if os.path.exists(CALIB_PATH):
        model = joblib.load(CALIB_PATH)
    else:
        model = XGBClassifier()
        model_file = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_PATH_UBJ
        model.load_model(model_file)
        print(f"[Retrain] Loaded raw model from {model_file}")
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            cols = [l.strip() for l in f if l.strip()]
    else:
        cols = FEATURE_COLS
    print(f"[Retrain] Model loaded ({len(cols)} features)")
    return model, cols


def start_scheduler():
    schedule.every().sunday.at("02:00").do(retrain)
    schedule.every(6).hours.do(check_model_health)
    print("[Retrain] Scheduler running — retrains Sunday 02:00 UTC, health check every 6h")
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    retrain()
