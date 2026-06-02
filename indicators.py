import pandas as pd
import numpy as np

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(series, fast=12, slow=26, signal=9):
    fast_ema    = ema(series, fast)
    slow_ema    = ema(series, slow)
    macd_line   = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series, period=20, std=2):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid + std * sigma, mid, mid - std * sigma

def stochastic(high, low, close, k=14, d=3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    K  = 100 * (close - ll) / (hh - ll)
    D  = K.rolling(d).mean()
    return K, D

def roc(series, period=10):
    return ((series - series.shift(period)) / series.shift(period)) * 100

def vwap_session(high, low, close, volume):
    tp  = (high + low + close) / 3
    df  = pd.DataFrame({"tp": tp, "vol": volume})
    df["date"] = df.index.date
    cum_tpv = []
    cum_vol = []
    for date, group in df.groupby("date"):
        tpv = (group["tp"] * group["vol"]).cumsum()
        vol = group["vol"].cumsum()
        cum_tpv.extend(tpv.tolist())
        cum_vol.extend(vol.tolist())
    df["cum_tpv"] = cum_tpv
    df["cum_vol"] = cum_vol
    result = df["cum_tpv"] / df["cum_vol"]
    result.index = tp.index
    return result

def add_session(df):
    hour = df.index.hour
    df["session_london"]  = ((hour >= 7)  & (hour < 12)).astype(int)
    df["session_newyork"] = ((hour >= 12) & (hour < 17)).astype(int)
    df["session_overlap"] = ((hour >= 12) & (hour < 14)).astype(int)
    df["session_asian"]   = ((hour >= 0)  & (hour < 7)).astype(int)
    return df

def add_news_filter(df, blackout_events=None):
    if blackout_events is None:
        hour   = df.index.hour
        minute = df.index.minute
        blackout = (
            ((hour == 8)  & (minute >= 25) & (minute <= 35)) |
            ((hour == 13) & (minute >= 25) & (minute <= 35)) |
            ((hour == 20) & (minute >= 55)) |
            ((hour == 21) & (minute <= 5))
        )
    else:
        import pandas as pd
        blackout = pd.Series(False, index=df.index)
        for event_time in blackout_events:
            window = (
                (df.index >= event_time - pd.Timedelta(minutes=30)) &
                (df.index <= event_time + pd.Timedelta(minutes=30))
            )
            blackout |= window
    df["news_blackout"] = blackout.astype(int)
    df["tradeable"]     = (~blackout).astype(int)
    return df

def detect_regime(df, lookback=50):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    atr_val  = atr(h, l, c, 14)
    atr_mean = atr_val.rolling(lookback).mean()
    atr_std  = atr_val.rolling(lookback).std()
    up_move   = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di  = pd.Series(plus_dm,  index=df.index).rolling(14).mean() / atr_val * 100
    minus_di = pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr_val * 100
    dx       = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx      = dx.rolling(14).mean()
    regime = pd.Series("RANGING", index=df.index)
    regime[adx > 25]                     = "TRENDING"
    regime[atr_val > atr_mean + atr_std] = "HIGH_VOL"
    regime[atr_val < atr_mean - atr_std] = "LOW_VOL"
    return regime, adx, atr_val