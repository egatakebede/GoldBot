
import time, os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime, timezone

from features import build_features, add_htf_context, load_htf_csv
import config

# ── Telegram Notifier ──────────────────────────────
try:
    import notifier
    NOTIFIER_READY = True
except ImportError:
    NOTIFIER_READY = False
    print("[Warning] Notifier not available")

def utcnow(): return datetime.now(timezone.utc)

MODEL_PATH   = os.path.expanduser("~/GoldBot/models/xgb_model.json")
CALIB_PATH   = os.path.expanduser("~/GoldBot/models/calibrated_model.pkl")
FEATURE_PATH = os.path.expanduser("~/GoldBot/models/feature_cols.txt")
PARAMS_PATH  = os.path.expanduser("~/GoldBot/models/best_params.json")
DATA_PATH    = "/home/e/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/mt5_data.csv"
SIGNAL_PATH  = "/home/e/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/signal.csv"
LOG_PATH     = os.path.expanduser("~/GoldBot/logs/signal_log.csv")
HTF_PATHS    = {
    "h1": os.path.expanduser("~/GoldBot/data/XAUUSD60.csv"),
    "h4": os.path.expanduser("~/GoldBot/data/XAUUSD240.csv"),
}

os.makedirs(os.path.expanduser("~/GoldBot/logs"), exist_ok=True)

def load_model():
    if os.path.exists(CALIB_PATH):
        model = joblib.load(CALIB_PATH)
        print("Loaded calibrated model")
    else:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        print("Loaded raw XGBoost model")
    with open(FEATURE_PATH) as f:
        features = [l.strip() for l in f if l.strip()]
    conf_threshold = config.MIN_CONFIDENCE
    if os.path.exists(PARAMS_PATH):
        import json
        with open(PARAMS_PATH) as f:
            saved = json.load(f)
        conf_threshold = saved.get("conf_threshold", config.MIN_CONFIDENCE)
        print(f"Confidence threshold: {conf_threshold:.3f}")
    mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    return model, features, conf_threshold, mtime

def write_signal(signal, confidence, entry, sl, tp, atr_val, regime):
    tmp = SIGNAL_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write("signal,confidence,entry,sl,tp,atr,regime,timestamp\n")
        f.write(f"{signal},{confidence:.4f},{entry:.2f},{sl:.2f},{tp:.2f},{atr_val:.2f},{regime},{utcnow().isoformat()}\n")
    os.replace(tmp, SIGNAL_PATH)

def log_signal(signal, confidence, entry, sl, tp, regime, conf_b, conf_f, conf_s):
    # FIX: always write header if file is empty or missing — was missing 'timestamp' column
    write_header = (not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0)
    with open(LOG_PATH, "a") as f:
        if write_header:
            f.write("timestamp,signal,confidence,entry,sl,tp,regime,conf_buy,conf_flat,conf_sell\n")
        f.write(f"{utcnow().isoformat()},{signal},{confidence:.4f},{entry:.2f},{sl:.2f},{tp:.2f},{regime},{conf_b:.4f},{conf_f:.4f},{conf_s:.4f}\n")

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    time_col = next((c for c in ["time","date","datetime","timestamp"] if c in df.columns), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()
    if "tick_volume" in df.columns:
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
    required = ["open","high","low","close","volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[required]

def check_spread(feat):
    last_range = float(feat["high"].iloc[-1] - feat["low"].iloc[-1])
    atr_val    = float(feat["atr14"].iloc[-1])
    if atr_val <= 0:
        return False, 0.0
    spread_ratio = last_range / atr_val
    return spread_ratio > 3.0, spread_ratio

def compute_sl_tp(signal, entry, atr_val):
    spread_pts = getattr(config, "SPREAD", 0.30)
    if signal == "BUY":
        sl = round(entry - config.SL_ATR_MULT * atr_val - spread_pts, 2)
        tp = round(entry + config.TP_ATR_MULT * atr_val, 2)
    elif signal == "SELL":
        sl = round(entry + config.SL_ATR_MULT * atr_val + spread_pts, 2)
        tp = round(entry - config.TP_ATR_MULT * atr_val, 2)
    else:
        sl = tp = 0.0
    return sl, tp

def apply_htf_filter(signal, conf, latest):
    if signal == "FLAT":
        return signal, conf
    h1_bull = latest.get("h1_trend_8_21", -1)
    h4_bull = latest.get("h4_trend_8_21", -1)
    if h1_bull == -1 or h4_bull == -1:
        return signal, conf
    htf_agrees = ((signal == "BUY" and h1_bull == 1 and h4_bull == 1) or (signal == "SELL" and h1_bull == 0 and h4_bull == 0))
    if not htf_agrees:
        conf *= 0.45
    return signal, conf

def main():
    print("=" * 60)
    print("GoldBot Signal Server")
    print("=" * 60)
    print("Loading model...")
    model, FEATURES, CONF_THRESHOLD, model_mtime = load_model()
    print(f"Features      : {len(FEATURES)}")
    print(f"Conf threshold: {CONF_THRESHOLD:.3f}")
    print(f"Notifier      : {'✅ Ready' if NOTIFIER_READY else '❌ Offline'}")
    print(f"Data file     : {DATA_PATH}")
    print(f"Signal file   : {SIGNAL_PATH}")
    print("Running... (Ctrl+C to stop)\n")

    last_bar       = None
    bars_processed = 0
    signals_fired  = {"BUY": 0, "SELL": 0, "FLAT": 0}

    while True:
        try:
            if os.path.exists(MODEL_PATH):
                mtime = os.path.getmtime(MODEL_PATH)
                if mtime > model_mtime:
                    print("[HotReload] New model detected — reloading...")
                    model, FEATURES, CONF_THRESHOLD, model_mtime = load_model()

            if not os.path.exists(DATA_PATH):
                print(f"[{utcnow().strftime('%H:%M:%S')}] Waiting for data file...")
                time.sleep(5)
                continue

            try:
                df = load_data(DATA_PATH)
            except Exception as e:
                print(f"[DataError] {e}")
                time.sleep(10)
                continue

            if len(df) < 200:
                print(f"[DataError] Only {len(df)} rows")
                time.sleep(10)
                continue

            current_bar = df.index[-1]
            if current_bar == last_bar:
                time.sleep(10)
                continue

            age_minutes = (datetime.now(timezone.utc) - current_bar).total_seconds() / 60
            if age_minutes > 15:
                print(f"[{utcnow().strftime('%H:%M:%S')}] WARNING: Data is {age_minutes:.0f}min stale — is MT5 running?")
                time.sleep(30)
                continue

            last_bar = current_bar
            bars_processed += 1

            try:
                feat = build_features(df.copy())
            except Exception as e:
                print(f"[FeatureError] {e}")
                time.sleep(10)
                continue

            if len(feat) < 50:
                print(f"[FeatureError] Too few rows")
                time.sleep(10)
                continue

            for prefix, path in HTF_PATHS.items():
                if os.path.exists(path):
                    try:
                        df_htf = load_htf_csv(path)
                        feat = add_htf_context(feat, df_htf, prefix)
                    except Exception as e:
                        print(f"[HTF:{prefix}] Skipped: {e}")

            wide_spread, spread_ratio = check_spread(feat)
            if wide_spread:
                print(f"[{utcnow().strftime('%H:%M:%S')}] SKIP — wide spread")
                entry   = float(feat["close"].iloc[-1])
                atr_now = float(feat["atr14"].iloc[-1])
                write_signal("FLAT", 0.0, entry, 0.0, 0.0, atr_now, "WIDE_SPREAD")
                time.sleep(10)
                continue

            latest  = feat.iloc[-1]
            entry   = float(latest["close"])
            atr_now = float(latest.get("atr14", 0))
            regime  = str(latest.get("regime", "UNKNOWN"))

            available = [f for f in FEATURES if f in latest.index]
            X = latest[available].values.reshape(1, -1)

            try:
                proba = model.predict_proba(X)[0]
            except Exception as e:
                print(f"[ModelError] {e}")
                time.sleep(10)
                continue

            classes   = list(model.classes_) if hasattr(model, "classes_") else [0, 1, 2]
            prob_map  = dict(zip(classes, proba))
            conf_sell = prob_map.get(0, 0.0)
            conf_flat = prob_map.get(1, 0.0)
            conf_buy  = prob_map.get(2, 0.0)

            pred_class = int(max(prob_map, key=prob_map.get))
            conf       = float(prob_map[pred_class])
            sig_map    = {0: "SELL", 1: "FLAT", 2: "BUY"}
            raw_signal = sig_map.get(pred_class, "FLAT")

            raw_signal, conf = apply_htf_filter(raw_signal, conf, latest)
            signal = raw_signal if conf >= CONF_THRESHOLD else "FLAT"

            sl, tp = compute_sl_tp(signal, entry, atr_now)

            write_signal(signal, conf, entry, sl, tp, atr_now, regime)
            log_signal(signal, conf, entry, sl, tp, regime, conf_buy, conf_flat, conf_sell)

            if NOTIFIER_READY and signal in ("BUY", "SELL"):
                notifier.trade_opened(signal, 1.0, entry, sl, tp, conf, regime)

            signals_fired[signal] = signals_fired.get(signal, 0) + 1

            print(f"[{utcnow().strftime('%H:%M:%S')}] "
                  f"Bar#{bars_processed:4d} | {signal:4s} conf:{conf:.3f} | "
                  f"B:{conf_buy:.2f} F:{conf_flat:.2f} S:{conf_sell:.2f} | "
                  f"entry:{entry:.2f} sl:{sl:.2f} tp:{tp:.2f} | {regime:10s} | "
                  f"BUY:{signals_fired['BUY']} SELL:{signals_fired['SELL']} FLAT:{signals_fired['FLAT']}")

            time.sleep(10)

        except KeyboardInterrupt:
            print(f"\nStopped after {bars_processed} bars.")
            print(f"Signals: {signals_fired}")
            break
        except Exception as e:
            import traceback
            print(f"[Error] {e}")
            traceback.print_exc()
            time.sleep(10)


if __name__ == "__main__":
    main()
