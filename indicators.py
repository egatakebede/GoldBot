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


# ── SMC Indicators (fully vectorized) ───────────────────────────────────────

def swing_points(high, low, left=3, right=3):
    """
    Swing high/low using only past bars (no lookahead).
    A swing high at bar i requires: high[i] == max(high[i-left-right : i])
    We confirm it right=bars AFTER it forms, so the signal fires at bar i+right.
    This is the correct real-time approach: we know bar i was a swing high
    only after right bars have passed.
    """
    h = high.values
    l = low.values
    n = len(h)
    sh = np.zeros(n)
    sl = np.zeros(n)
    window = left + right + 1
    for i in range(window - 1, n):
        # Window is purely backward: bars [i-window+1 .. i]
        w_h = h[i - window + 1 : i + 1]
        w_l = l[i - window + 1 : i + 1]
        mid = right  # center of window relative to start = left + right positions back from i = index `right` from end
        # The candidate bar is `right` bars ago
        cand = i - right
        if h[cand] == w_h.max():
            sh[cand] = 1
        if l[cand] == w_l.min():
            sl[cand] = 1
    return pd.Series(sh, index=high.index), pd.Series(sl, index=low.index)


def bos_choch(close, high, low, swing_left=3, swing_right=3):
    """
    BOS/CHoCH using confirmed swing points (no lookahead).
    Tracks last confirmed swing high AND swing low independently.
    BOS bull  = close breaks above last swing high, last move was up (HH)
    CHoCH bull = close breaks above last swing high, last move was down (reversal)
    BOS bear  = close breaks below last swing low, last move was down (LL)
    CHoCH bear = close breaks below last swing low, last move was up (reversal)
    """
    sh, sl = swing_points(high, low, swing_left, swing_right)
    c    = close.values
    h    = high.values
    l    = low.values
    sh_v = sh.values
    sl_v = sl.values
    n    = len(c)

    bos_bull = np.zeros(n)
    bos_bear = np.zeros(n)
    choch_bull = np.zeros(n)
    choch_bear = np.zeros(n)

    last_sh    = np.nan
    last_sl    = np.nan
    last_sh_i  = -1
    last_sl_i  = -1

    for i in range(1, n):
        # Update last confirmed swing points
        if sh_v[i-1]:
            last_sh   = h[i-1]
            last_sh_i = i - 1
        if sl_v[i-1]:
            last_sl   = l[i-1]
            last_sl_i = i - 1

        # BOS/CHoCH bull: close breaks above last swing high
        if not np.isnan(last_sh) and c[i] > last_sh:
            # CHoCH if last swing low was more recent (downtrend being broken)
            if last_sl_i > last_sh_i:
                choch_bull[i] = 1
            else:
                bos_bull[i] = 1
            last_sh = np.nan  # consumed

        # BOS/CHoCH bear: close breaks below last swing low
        if not np.isnan(last_sl) and c[i] < last_sl:
            # CHoCH if last swing high was more recent (uptrend being broken)
            if last_sh_i > last_sl_i:
                choch_bear[i] = 1
            else:
                bos_bear[i] = 1
            last_sl = np.nan  # consumed

    idx = close.index
    return (pd.Series(bos_bull, index=idx), pd.Series(bos_bear, index=idx),
            pd.Series(choch_bull, index=idx), pd.Series(choch_bear, index=idx))


def fair_value_gaps(high, low, close, min_gap_atr=0.3):
    h = high.values
    l = low.values
    c = close.values
    atr_v = atr(high, low, close, 14).values
    n = len(c)

    # Bullish FVG: low[i] > high[i-2] with gap >= min_gap_atr * atr
    fvg_bull = np.zeros(n)
    fvg_bear = np.zeros(n)
    fvg_bull[2:] = np.where(
        (l[2:] > h[:-2]) & ((l[2:] - h[:-2]) >= atr_v[2:] * min_gap_atr), 1, 0)
    fvg_bear[2:] = np.where(
        (h[2:] < l[:-2]) & ((l[:-2] - h[2:]) >= atr_v[2:] * min_gap_atr), 1, 0)

    # Distance: forward-fill last FVG reference price
    bull_ref = np.where(fvg_bull, l, np.nan)   # top of bull gap
    bear_ref = np.where(fvg_bear, h, np.nan)   # bottom of bear gap
    bull_ref = pd.Series(bull_ref).ffill().values
    bear_ref = pd.Series(bear_ref).ffill().values
    safe_atr = np.where(atr_v > 0, atr_v, 1)
    dist_bull = np.where(~np.isnan(bull_ref), (c - bull_ref) / safe_atr, 0)
    dist_bear = np.where(~np.isnan(bear_ref), (bear_ref - c) / safe_atr, 0)

    idx = close.index
    return (pd.Series(fvg_bull, index=idx), pd.Series(fvg_bear, index=idx),
            pd.Series(dist_bull, index=idx), pd.Series(dist_bear, index=idx))


def order_blocks(high, low, close, open_, atr_val, lookback=10, strength=1.5):
    """
    Vectorized OB — no lookahead.
    Bull OB: bearish candle where the prior N-bar max close move BACKWARD was bullish impulse.
    Uses only past data: rolling backward max/min with shift(1).
    """
    h = high.values
    l = low.values
    c = close.values
    o = open_.values
    a = atr_val.values

    bearish = (c < o).astype(float)
    bullish = (c > o).astype(float)

    # Backward-looking: max close over last lookback bars (no future)
    past_max = pd.Series(c).shift(1).rolling(lookback).max().values
    past_min = pd.Series(c).shift(1).rolling(lookback).min().values

    safe_a = np.where(a > 0, a, np.nan)
    # Bull OB: current bar is bearish AND price has risen strongly from the lookback low
    bull_impulse = ((c - past_min) >= strength * safe_a) & (bearish == 1)
    # Bear OB: current bar is bullish AND price has fallen strongly from the lookback high
    bear_impulse = ((past_max - c) >= strength * safe_a) & (bullish == 1)

    ob_bull_top = np.where(bull_impulse, h, np.nan)
    ob_bull_bot = np.where(bull_impulse, l, np.nan)
    ob_bear_top = np.where(bear_impulse, h, np.nan)
    ob_bear_bot = np.where(bear_impulse, l, np.nan)

    ob_bull_top = pd.Series(ob_bull_top).ffill().values
    ob_bull_bot = pd.Series(ob_bull_bot).ffill().values
    ob_bear_top = pd.Series(ob_bear_top).ffill().values
    ob_bear_bot = pd.Series(ob_bear_bot).ffill().values

    in_bull_ob = (c >= ob_bull_bot) & (c <= ob_bull_top) & ~np.isnan(ob_bull_top)
    in_bear_ob = (c >= ob_bear_bot) & (c <= ob_bear_top) & ~np.isnan(ob_bear_top)

    ob_bull_mid  = (ob_bull_top + ob_bull_bot) / 2
    ob_bear_mid  = (ob_bear_top + ob_bear_bot) / 2
    ob_bull_dist = np.where(in_bull_ob, (c - ob_bull_mid) / safe_a, 0)
    ob_bear_dist = np.where(in_bear_ob, (ob_bear_mid - c) / safe_a, 0)

    idx = close.index
    return (pd.Series(in_bull_ob.astype(float), index=idx),
            pd.Series(in_bear_ob.astype(float), index=idx),
            pd.Series(ob_bull_dist, index=idx),
            pd.Series(ob_bear_dist, index=idx))


def liquidity_sweep(high, low, close, swing_left=3, swing_right=3):
    sh, sl = swing_points(high, low, swing_left, swing_right)
    h = high.values
    l = low.values
    c = close.values
    sh_v = sh.values
    sl_v = sl.values
    n = len(c)

    # Forward-fill last swing high/low price
    last_sh = np.where(sh_v, h, np.nan)
    last_sl = np.where(sl_v, l, np.nan)
    last_sh = pd.Series(last_sh).shift(1).ffill().values
    last_sl = pd.Series(last_sl).shift(1).ffill().values

    liq_bull = ((l < last_sl) & (c > last_sl) & ~np.isnan(last_sl)).astype(float)
    liq_bear = ((h > last_sh) & (c < last_sh) & ~np.isnan(last_sh)).astype(float)

    return pd.Series(liq_bull, index=close.index), pd.Series(liq_bear, index=close.index)

