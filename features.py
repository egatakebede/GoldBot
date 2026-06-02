import pandas as pd
import numpy as np
from indicators import (ema, rsi, atr, macd, bollinger_bands,
                        stochastic, roc, vwap_session,
                        add_session, add_news_filter, detect_regime)

def build_features(df, blackout_events=None):
    d = df.copy()
    c = d["close"]
    h = d["high"]
    l = d["low"]
    v = d["volume"]

    d["ema8"]   = ema(c, 8)
    d["ema21"]  = ema(c, 21)
    d["ema50"]  = ema(c, 50)
    d["ema200"] = ema(c, 200)

    d["trend_8_21"]   = (d["ema8"]  > d["ema21"]).astype(int)
    d["trend_21_50"]  = (d["ema21"] > d["ema50"]).astype(int)
    d["trend_50_200"] = (d["ema50"] > d["ema200"]).astype(int)

    d["price_vs_ema21"]  = (c - d["ema21"])  / d["ema21"]  * 100
    d["price_vs_ema50"]  = (c - d["ema50"])  / d["ema50"]  * 100
    d["price_vs_ema200"] = (c - d["ema200"]) / d["ema200"] * 100

    # EMA slope — model learns trend acceleration
    d["ema21_slope"] = d["ema21"].diff(3) / d["ema21"] * 100
    d["ema50_slope"] = d["ema50"].diff(3) / d["ema50"] * 100

    d["rsi14"] = rsi(c, 14)
    d["rsi7"]  = rsi(c, 7)
    # RSI momentum — rate of change of RSI
    d["rsi_slope"] = d["rsi14"].diff(3)

    d["macd_line"], d["macd_signal"], d["macd_hist"] = macd(c)
    d["macd_cross"] = (d["macd_line"] > d["macd_signal"]).astype(int)
    d["macd_hist_slope"] = d["macd_hist"].diff(2)

    d["roc5"]  = roc(c, 5)
    d["roc10"] = roc(c, 10)
    d["roc20"] = roc(c, 20)

    d["stoch_k"], d["stoch_d"] = stochastic(h, l, c)
    d["stoch_cross"] = (d["stoch_k"] > d["stoch_d"]).astype(int)

    d["atr14"]     = atr(h, l, c, 14)
    d["atr7"]      = atr(h, l, c, 7)
    d["atr_pct"]   = d["atr14"] / c * 100
    d["atr_ratio"] = d["atr7"]  / d["atr14"]
    # ATR trend — is volatility expanding or contracting?
    d["atr_trend"] = d["atr14"].diff(5) / d["atr14"]

    d["bb_upper"], d["bb_mid"], d["bb_lower"] = bollinger_bands(c)
    d["bb_width"]    = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"] * 100
    d["bb_position"] = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])
    d["bb_squeeze"]  = (d["bb_width"] < d["bb_width"].rolling(20).mean()).astype(int)

    d["candle_body"]  = (c - d["open"]).abs() / d["atr14"]
    d["candle_range"] = (h - l) / d["atr14"]
    d["upper_wick"]   = (h - d[["open","close"]].max(axis=1)) / d["atr14"]
    d["lower_wick"]   = (d[["open","close"]].min(axis=1) - l) / d["atr14"]
    d["is_bullish"]   = (c > d["open"]).astype(int)
    # Consecutive bullish/bearish bars — momentum context
    d["consec_bull"]  = d["is_bullish"].groupby(
        (d["is_bullish"] != d["is_bullish"].shift()).cumsum()).cumcount() + 1
    d["consec_bull"]  = d["consec_bull"] * d["is_bullish"]
    d["consec_bear"]  = (1 - d["is_bullish"]).groupby(
        (d["is_bullish"] != d["is_bullish"].shift()).cumsum()).cumcount() + 1
    d["consec_bear"]  = d["consec_bear"] * (1 - d["is_bullish"])

    d["rel_volume"]    = v / v.rolling(20).mean()
    d["volume_spike"]  = (d["rel_volume"] > 1.5).astype(int)
    d["vwap_val"]      = vwap_session(h, l, c, v)
    d["price_vs_vwap"] = (c - d["vwap_val"]) / d["vwap_val"] * 100

    d["prev_high"]       = h.shift(1)
    d["prev_low"]        = l.shift(1)
    d["above_prev_high"] = (c > d["prev_high"]).astype(int)
    d["below_prev_low"]  = (c < d["prev_low"]).astype(int)
    d["high_20"]         = h.rolling(20).max().shift(1)
    d["low_20"]          = l.rolling(20).min().shift(1)
    d["break_high"]      = (c > d["high_20"]).astype(int)
    d["break_low"]       = (c < d["low_20"]).astype(int)
    # Distance to key levels — model learns proximity matters
    d["dist_to_high20"]  = (d["high_20"] - c) / d["atr14"]
    d["dist_to_low20"]   = (c - d["low_20"]) / d["atr14"]

    d = add_session(d)
    d = add_news_filter(d, blackout_events)
    d["hour"]        = d.index.hour
    d["day_of_week"] = d.index.dayofweek
    d["is_monday"]   = (d["day_of_week"] == 0).astype(int)
    d["is_friday"]   = (d["day_of_week"] == 4).astype(int)

    # Cyclic time encoding — better than raw hour for tree models
    d["hour_sin"]    = np.sin(d.index.hour * 2 * np.pi / 24)
    d["hour_cos"]    = np.cos(d.index.hour * 2 * np.pi / 24)
    d["dow_sin"]     = np.sin(d.index.dayofweek * 2 * np.pi / 5)
    d["dow_cos"]     = np.cos(d.index.dayofweek * 2 * np.pi / 5)

    # Session open filter — avoid first 15 min of London/NY (chaotic)
    d["london_open_spike"]  = ((d.index.hour == 7)  & (d.index.minute < 15)).astype(int)
    d["ny_open_spike"]      = ((d.index.hour == 12) & (d.index.minute < 15)).astype(int)

    # ATR quality filter — is current ATR above average (tradeable volatility)
    d["atr_above_avg"] = (d["atr14"] > d["atr14"].rolling(20).mean()).astype(int)

    # Regime as numeric features — model learns which regime favours which signal
    d["regime"], d["adx"], _ = detect_regime(d)
    d["regime_trending"] = (d["regime"] == "TRENDING").astype(int)
    d["regime_ranging"]  = (d["regime"] == "RANGING").astype(int)
    d["regime_highvol"]  = (d["regime"] == "HIGH_VOL").astype(int)
    d["regime_lowvol"]   = (d["regime"] == "LOW_VOL").astype(int)
    d["adx_val"]         = d["adx"]  # numeric ADX value

    return d.dropna()


HTF_COLS = ["trend_8_21", "trend_21_50", "trend_50_200", "ema21_slope",
            "rsi14", "rsi_slope", "macd_hist", "macd_hist_slope",
            "bb_position", "bb_squeeze", "atr14", "atr_trend",
            "rel_volume", "adx_val", "regime_trending", "regime_ranging"]

def add_htf_context(df_ltf, df_htf, prefix):
    available   = [c for c in HTF_COLS if c in df_htf.columns]
    ctx         = df_htf[available].copy()
    ctx.columns = [f"{prefix}_{c}" for c in available]
    ctx         = ctx.reindex(df_ltf.index, method="ffill")
    return pd.concat([df_ltf, ctx], axis=1)

def load_htf_csv(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return build_features(df)

def enrich_with_htf(df_base, htf_paths):
    for prefix, path in htf_paths.items():
        try:
            df_htf  = load_htf_csv(path)
            df_base = add_htf_context(df_base, df_htf, prefix)
        except Exception as e:
            print(f"[HTF] Could not load {prefix} from {path}: {e}")
    return df_base

def get_live_df(rates_m1, rates_m5, htf_paths=None):
    def to_df(rates):
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["open","high","low","close","volume"]]
    df_m1      = to_df(rates_m1)
    df_m5      = to_df(rates_m5)
    df_m5_feat = build_features(df_m5)
    df_m1_feat = build_features(df_m1)
    df_m1_feat = add_htf_context(df_m1_feat, df_m5_feat, "m5")
    if htf_paths:
        df_m1_feat = enrich_with_htf(df_m1_feat, htf_paths)
    latest = df_m1_feat.iloc[-2]
    return df_m1_feat, df_m5_feat, latest

def get_signal_row(rates_m1, rates_m5, model, feature_cols, htf_paths=None):
    df_m1_feat, _, latest = get_live_df(rates_m1, rates_m5, htf_paths)
    available  = [f for f in feature_cols if f in latest.index]
    X          = latest[available].values.reshape(1, -1)
    proba      = model.predict_proba(X)[0]
    signal     = int(np.argmax(proba))
    confidence = float(proba[signal])

    # HTF alignment — counter-trend signals get 50% confidence penalty
    if signal != 1:
        h1_bull = latest.get("h1_trend_8_21", -1)
        h4_bull = latest.get("h4_trend_8_21", -1)
        if h1_bull != -1 and h4_bull != -1:
            htf_agrees = (signal == 2 and h1_bull == 1 and h4_bull == 1) or \
                         (signal == 0 and h1_bull == 0 and h4_bull == 0)
            if not htf_agrees:
                confidence *= 0.5  # counter-trend penalty

    return signal, confidence, latest.get("regime", "UNKNOWN"), latest.get("atr14", 0), latest
