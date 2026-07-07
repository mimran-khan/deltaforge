"""Comprehensive technical indicator library -- 40+ indicators.

Every indicator used by professional intraday traders, implemented
in pure pandas/numpy for speed and zero external dependencies.

Categories:
  1. Trend        : EMA, SMA, DEMA, TEMA, WMA, HMA, KAMA, ZLEMA
  2. Momentum     : RSI, Stochastic, MACD, CCI, Williams %R, MFI, ROC, TSI, UO
  3. Volatility   : Bollinger, Keltner, Donchian, ATR, Std Dev
  4. Volume       : OBV, VWAP, A/D Line, CMF, Volume SMA
  5. Trend Strength: ADX, Aroon, SuperTrend, Parabolic SAR, Ichimoku
  6. Support/Res  : Pivot Points, Fibonacci Levels
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
#  1. TREND INDICATORS
# ═══════════════════════════════════════════════════════════════════

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average -- recent prices get more weight."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period, min_periods=1).apply(
        lambda x: np.dot(x[-len(weights):], weights[:len(x)]) / weights[:len(x)].sum(),
        raw=True
    )


def dema(series: pd.Series, period: int) -> pd.Series:
    """Double EMA -- faster response to price changes."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


def tema(series: pd.Series, period: int) -> pd.Series:
    """Triple EMA -- even faster trend following."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average -- very responsive, minimal lag."""
    half = int(period / 2)
    sqrt_p = int(np.sqrt(period))
    half_wma = wma(series, half)
    full_wma = wma(series, period)
    diff = 2 * half_wma - full_wma
    return wma(diff, sqrt_p)


def zlema(series: pd.Series, period: int) -> pd.Series:
    """Zero-Lag EMA -- compensates for EMA lag."""
    lag = (period - 1) // 2
    adjusted = 2 * series - series.shift(lag).fillna(series)
    return ema(adjusted, period)


def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average -- adapts to volatility."""
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    vals = series.values.copy().astype(float)
    result = np.full_like(vals, np.nan)
    if len(vals) < period + 1:
        return pd.Series(result, index=series.index)

    result[period] = vals[period]
    for i in range(period + 1, len(vals)):
        direction = abs(vals[i] - vals[i - period])
        volatility = sum(abs(vals[j] - vals[j - 1]) for j in range(i - period + 1, i + 1))
        if volatility == 0:
            er = 0
        else:
            er = direction / volatility
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        result[i] = result[i - 1] + sc * (vals[i] - result[i - 1])
    return pd.Series(result, index=series.index)


# ═══════════════════════════════════════════════════════════════════
#  2. MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════════════════════

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator -- %K and %D lines."""
    lowest = low.rolling(k_period, min_periods=1).min()
    highest = high.rolling(k_period, min_periods=1).max()
    k = ((close - lowest) / (highest - lowest).replace(0, np.nan)) * 100
    d = k.rolling(d_period, min_periods=1).mean()
    return k.fillna(50), d.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD -- Moving Average Convergence Divergence."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def cci(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    tp_sma = sma(tp, period)
    mean_dev = tp.rolling(period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    """Williams %R -- momentum oscillator."""
    highest = high.rolling(period, min_periods=1).max()
    lowest = low.rolling(period, min_periods=1).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change."""
    prev = series.shift(period)
    return ((series - prev) / prev.replace(0, np.nan)) * 100


def tsi(series: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    """True Strength Index."""
    diff = series.diff()
    smooth1 = ema(ema(diff, long), short)
    smooth2 = ema(ema(diff.abs(), long), short)
    return (smooth1 / smooth2.replace(0, np.nan)) * 100


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                         p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
    """Ultimate Oscillator -- multi-timeframe momentum."""
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)

    avg1 = bp.rolling(p1, min_periods=1).sum() / tr.rolling(p1, min_periods=1).sum().replace(0, np.nan)
    avg2 = bp.rolling(p2, min_periods=1).sum() / tr.rolling(p2, min_periods=1).sum().replace(0, np.nan)
    avg3 = bp.rolling(p3, min_periods=1).sum() / tr.rolling(p3, min_periods=1).sum().replace(0, np.nan)

    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index -- volume-weighted RSI."""
    tp = (high + low + close) / 3
    mf = tp * volume
    diff = tp.diff()
    pos_mf = mf.where(diff > 0, 0).rolling(period, min_periods=1).sum()
    neg_mf = mf.where(diff < 0, 0).rolling(period, min_periods=1).sum()
    mr = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + mr))


# ═══════════════════════════════════════════════════════════════════
#  3. VOLATILITY INDICATORS
# ═══════════════════════════════════════════════════════════════════

def atr(df_or_high, low=None, close=None, period: int = 14) -> pd.Series:
    """Average True Range. Accepts DataFrame or separate series."""
    if isinstance(df_or_high, pd.DataFrame):
        h, l, c = df_or_high["high"], df_or_high["low"], df_or_high["close"]
    else:
        h, l, c = df_or_high, low, close
    prev_close = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    """Bollinger Bands -- upper, middle, lower, %B, bandwidth."""
    middle = sma(series, period)
    std = series.rolling(period, min_periods=1).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / middle.replace(0, np.nan) * 100
    return upper, middle, lower, pct_b, bandwidth


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                      ema_period: int = 20, atr_period: int = 10,
                      atr_mult: float = 2.0):
    """Keltner Channels."""
    mid = ema(close, ema_period)
    df_temp = pd.DataFrame({"high": high, "low": low, "close": close})
    atr_val = atr(df_temp, atr_period)
    upper = mid + atr_mult * atr_val
    lower = mid - atr_mult * atr_val
    return upper, mid, lower


def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20):
    """Donchian Channels -- highest high, lowest low."""
    upper = high.rolling(period, min_periods=1).max()
    lower = low.rolling(period, min_periods=1).min()
    mid = (upper + lower) / 2
    return upper, mid, lower


# ═══════════════════════════════════════════════════════════════════
#  4. VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════════════

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def vwap_intraday(df: pd.DataFrame) -> pd.Series:
    """VWAP that resets at the start of each trading day."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]
    dates = df.index.date if hasattr(df.index, 'date') else pd.Series(df.index).dt.date.values
    result = pd.Series(np.nan, index=df.index)
    for date in pd.unique(dates):
        mask = dates == date
        cum_tp = tp_vol[mask].cumsum()
        cum_v = df["volume"][mask].cumsum()
        result[mask] = cum_tp / cum_v.replace(0, np.nan)
    return result


def ad_line(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series) -> pd.Series:
    """Accumulation/Distribution Line."""
    hl_range = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    return (clv * volume).cumsum()


def cmf(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    hl_range = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    return (clv * volume).rolling(period, min_periods=1).sum() / \
           volume.rolling(period, min_periods=1).sum().replace(0, np.nan)


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume / average volume ratio."""
    avg = volume.rolling(period, min_periods=1).mean()
    return volume / avg.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════
#  5. TREND STRENGTH INDICATORS
# ═══════════════════════════════════════════════════════════════════

def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14):
    """Average Directional Index -- trend strength. Returns ADX, +DI, -DI."""
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    df_temp = pd.DataFrame({"high": high, "low": low, "close": close})
    atr_val = atr(df_temp, period)

    plus_di = 100 * ema(plus_dm, period) / atr_val.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / atr_val.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = ema(dx, period)

    return adx_val, plus_di, minus_di


def aroon(high: pd.Series, low: pd.Series, period: int = 25):
    """Aroon indicator -- trend detection. Returns aroon_up, aroon_down, oscillator."""
    def _period_since(s, func, p):
        result = pd.Series(np.nan, index=s.index)
        for i in range(p, len(s)):
            window = s.iloc[i - p:i + 1]
            idx = func(window)
            result.iloc[i] = (p - (i - idx)) / p * 100
        return result

    aroon_up = pd.Series(np.nan, index=high.index)
    aroon_down = pd.Series(np.nan, index=low.index)

    for i in range(period, len(high)):
        hi_window = high.iloc[max(0, i - period):i + 1]
        lo_window = low.iloc[max(0, i - period):i + 1]
        days_since_hi = i - hi_window.idxmax()
        days_since_lo = i - lo_window.idxmin()
        if isinstance(days_since_hi, (pd.Timestamp,)):
            pos = high.index.get_loc(hi_window.idxmax())
            days_since_hi = i - pos
            pos2 = low.index.get_loc(lo_window.idxmin())
            days_since_lo = i - pos2
        aroon_up.iloc[i] = (period - days_since_hi) / period * 100
        aroon_down.iloc[i] = (period - days_since_lo) / period * 100

    oscillator = aroon_up - aroon_down
    return aroon_up, aroon_down, oscillator


def supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 2.5):
    """Returns (supertrend_line, direction) where direction=1 is bullish."""
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val

    close = df["close"].values
    up = upper.values.copy()
    dn = lower.values.copy()
    direction = np.ones(len(df), dtype=int)
    st = np.full(len(df), np.nan)

    for i in range(1, len(df)):
        if dn[i] < dn[i - 1]:
            dn[i] = dn[i - 1]
        if up[i] > up[i - 1]:
            up[i] = up[i - 1]

        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < dn[i] else 1
        else:
            direction[i] = 1 if close[i] > up[i] else -1

        st[i] = dn[i] if direction[i] == 1 else up[i]

    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)


supertrend_fast = supertrend


def parabolic_sar(high: pd.Series, low: pd.Series,
                   af_start: float = 0.02, af_step: float = 0.02,
                   af_max: float = 0.2):
    """Parabolic SAR. Returns (sar_values, direction)."""
    n = len(high)
    sar = np.zeros(n)
    direction = np.ones(n, dtype=int)
    af = af_start
    ep = low.iloc[0]
    sar[0] = high.iloc[0]

    for i in range(1, n):
        if direction[i - 1] == 1:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = min(sar[i], low.iloc[i - 1])
            if i >= 2:
                sar[i] = min(sar[i], low.iloc[i - 2])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_step, af_max)
            if low.iloc[i] < sar[i]:
                direction[i] = -1
                sar[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                direction[i] = 1
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = max(sar[i], high.iloc[i - 1])
            if i >= 2:
                sar[i] = max(sar[i], high.iloc[i - 2])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_step, af_max)
            if high.iloc[i] > sar[i]:
                direction[i] = 1
                sar[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                direction[i] = -1

    return pd.Series(sar, index=high.index), pd.Series(direction, index=high.index)


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    """Ichimoku Cloud. Returns tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou."""
    t_high = high.rolling(tenkan, min_periods=1).max()
    t_low = low.rolling(tenkan, min_periods=1).min()
    tenkan_sen = (t_high + t_low) / 2

    k_high = high.rolling(kijun, min_periods=1).max()
    k_low = low.rolling(kijun, min_periods=1).min()
    kijun_sen = (k_high + k_low) / 2

    s_a = (tenkan_sen + kijun_sen) / 2
    s_b_high = high.rolling(senkou_b, min_periods=1).max()
    s_b_low = low.rolling(senkou_b, min_periods=1).min()
    s_b = (s_b_high + s_b_low) / 2

    chikou = close.shift(-kijun)

    return tenkan_sen, kijun_sen, s_a, s_b, chikou


# ═══════════════════════════════════════════════════════════════════
#  6. SUPPORT / RESISTANCE
# ═══════════════════════════════════════════════════════════════════

def pivot_points(high_val: float, low_val: float, close_val: float):
    """Classic Pivot Points from prior period. Returns dict of levels."""
    pp = (high_val + low_val + close_val) / 3
    r1 = 2 * pp - low_val
    s1 = 2 * pp - high_val
    r2 = pp + (high_val - low_val)
    s2 = pp - (high_val - low_val)
    r3 = high_val + 2 * (pp - low_val)
    s3 = low_val - 2 * (high_val - pp)
    return {"PP": pp, "R1": r1, "R2": r2, "R3": r3,
            "S1": s1, "S2": s2, "S3": s3}


def fibonacci_levels(high_val: float, low_val: float):
    """Fibonacci retracement levels."""
    diff = high_val - low_val
    return {
        "0.0": high_val,
        "0.236": high_val - 0.236 * diff,
        "0.382": high_val - 0.382 * diff,
        "0.500": high_val - 0.500 * diff,
        "0.618": high_val - 0.618 * diff,
        "0.786": high_val - 0.786 * diff,
        "1.0": low_val,
    }


# ═══════════════════════════════════════════════════════════════════
#  7. COMPOSITE / HELPER
# ═══════════════════════════════════════════════════════════════════

def add_all_indicators(df: pd.DataFrame,
                       ema_fast: int = 9,
                       ema_slow: int = 20,
                       rsi_period: int = 14,
                       atr_period: int = 14,
                       st_period: int = 7,
                       st_mult: float = 2.5) -> pd.DataFrame:
    """Add all trading indicators to a OHLCV DataFrame."""
    df = df.copy()
    df["ema_fast"] = ema(df["close"], ema_fast)
    df["ema_slow"] = ema(df["close"], ema_slow)
    df["rsi"] = rsi(df["close"], rsi_period)
    df["atr"] = atr(df, atr_period)
    df["vwap"] = vwap_intraday(df)
    df["volume_sma"] = sma(df["volume"], 20)
    st_line, st_dir = supertrend(df, st_period, st_mult)
    df["supertrend"] = st_line
    df["supertrend_dir"] = st_dir
    return df
