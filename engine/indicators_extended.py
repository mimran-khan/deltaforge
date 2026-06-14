"""Extended indicator library -- candlestick patterns, statistical models,
divergence detection, and derivative indicators.

Adds ~100 additional signals on top of indicators.py to bring the total
signal count above 200 per candle.

Categories added:
  8.  Candlestick Patterns  (20+ patterns)
  9.  Statistical Models     (regression, z-score, skew, etc.)
  10. Divergence Detection   (RSI/MACD/OBV vs price divergence)
  11. Multi-Timeframe        (aggregate 5m into 15m, 1h)
  12. Derivative Indicators  (indicators of indicators)
  13. Candle Structure       (body ratio, wick ratio, range expansion)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from engine import indicators as ind


# ═══════════════════════════════════════════════════════════════════
#  8. CANDLESTICK PATTERNS
# ═══════════════════════════════════════════════════════════════════

def detect_patterns(o: pd.Series, h: pd.Series, l: pd.Series,
                    c: pd.Series) -> dict[str, pd.Series]:
    """Detect 20+ candlestick patterns. Returns dict of pattern series
    where +1 = bullish, -1 = bearish, 0 = no pattern."""

    body = c - o
    body_abs = body.abs()
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    total_range = (h - l).replace(0, np.nan)
    body_ratio = body_abs / total_range
    prev_body = body.shift(1)

    patterns = {}

    # 1. Doji -- tiny body, long wicks
    patterns["DOJI"] = pd.Series(0, index=o.index)
    doji_mask = body_ratio < 0.1
    patterns["DOJI"] = doji_mask.astype(int) * 0  # neutral signal

    # 2. Hammer -- small body at top, long lower wick (bullish reversal)
    hammer = (lower_wick > body_abs * 2) & (upper_wick < body_abs * 0.5) & (body_ratio < 0.4)
    patterns["HAMMER"] = hammer.astype(int)

    # 3. Inverted Hammer (bearish reversal at top)
    inv_hammer = (upper_wick > body_abs * 2) & (lower_wick < body_abs * 0.5) & (body_ratio < 0.4)
    patterns["INV_HAMMER"] = -inv_hammer.astype(int)

    # 4. Bullish Engulfing
    bull_eng = (body > 0) & (prev_body < 0) & (body_abs > prev_body.abs())
    patterns["BULL_ENGULF"] = bull_eng.astype(int)

    # 5. Bearish Engulfing
    bear_eng = (body < 0) & (prev_body > 0) & (body_abs > prev_body.abs())
    patterns["BEAR_ENGULF"] = -bear_eng.astype(int)

    # 6. Morning Star (3-candle bullish reversal)
    prev2_body = body.shift(2)
    morning = (prev2_body < 0) & (prev_body.abs() < body_abs * 0.3) & (body > 0)
    patterns["MORNING_STAR"] = morning.astype(int)

    # 7. Evening Star (3-candle bearish reversal)
    evening = (prev2_body > 0) & (prev_body.abs() < body_abs * 0.3) & (body < 0)
    patterns["EVENING_STAR"] = -evening.astype(int)

    # 8. Bullish Harami
    bull_harami = (prev_body < 0) & (body > 0) & (body_abs < prev_body.abs()) & \
                  (c < o.shift(1)) & (o > c.shift(1))
    patterns["BULL_HARAMI"] = bull_harami.astype(int)

    # 9. Bearish Harami
    bear_harami = (prev_body > 0) & (body < 0) & (body_abs < prev_body.abs()) & \
                  (c > o.shift(1)) & (o < c.shift(1))
    patterns["BEAR_HARAMI"] = -bear_harami.astype(int)

    # 10. Shooting Star
    shooting = (upper_wick > body_abs * 2) & (lower_wick < body_abs * 0.3) & (body < 0)
    patterns["SHOOTING_STAR"] = -shooting.astype(int)

    # 11. Three White Soldiers
    three_white = (body > 0) & (prev_body > 0) & (body.shift(2) > 0) & \
                  (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    patterns["THREE_WHITE"] = three_white.astype(int)

    # 12. Three Black Crows
    three_black = (body < 0) & (prev_body < 0) & (body.shift(2) < 0) & \
                  (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    patterns["THREE_BLACK"] = -three_black.astype(int)

    # 13. Marubozu (full body, no wicks)
    marubozu_bull = (body > 0) & (upper_wick < body_abs * 0.05) & (lower_wick < body_abs * 0.05)
    marubozu_bear = (body < 0) & (upper_wick < body_abs * 0.05) & (lower_wick < body_abs * 0.05)
    patterns["MARUBOZU"] = marubozu_bull.astype(int) - marubozu_bear.astype(int)

    # 14. Spinning Top (small body, long wicks both sides)
    spinning = (body_ratio < 0.3) & (upper_wick > body_abs) & (lower_wick > body_abs)
    patterns["SPINNING_TOP"] = pd.Series(0, index=o.index)  # neutral

    # 15. Tweezer Top
    tweezer_top = (h.round(0) == h.shift(1).round(0)) & (body < 0) & (prev_body > 0)
    patterns["TWEEZER_TOP"] = -tweezer_top.astype(int)

    # 16. Tweezer Bottom
    tweezer_bot = (l.round(0) == l.shift(1).round(0)) & (body > 0) & (prev_body < 0)
    patterns["TWEEZER_BOT"] = tweezer_bot.astype(int)

    # 17. Piercing Line
    piercing = (prev_body < 0) & (body > 0) & (o < c.shift(1)) & \
               (c > (o.shift(1) + c.shift(1)) / 2) & (c < o.shift(1))
    patterns["PIERCING"] = piercing.astype(int)

    # 18. Dark Cloud Cover
    dark_cloud = (prev_body > 0) & (body < 0) & (o > c.shift(1)) & \
                 (c < (o.shift(1) + c.shift(1)) / 2) & (c > o.shift(1))
    patterns["DARK_CLOUD"] = -dark_cloud.astype(int)

    # 19. Dragonfly Doji (bullish)
    dragonfly = (body_ratio < 0.05) & (lower_wick > total_range * 0.6) & \
                (upper_wick < total_range * 0.1)
    patterns["DRAGONFLY"] = dragonfly.astype(int)

    # 20. Gravestone Doji (bearish)
    gravestone = (body_ratio < 0.05) & (upper_wick > total_range * 0.6) & \
                 (lower_wick < total_range * 0.1)
    patterns["GRAVESTONE"] = -gravestone.astype(int)

    return patterns


# ═══════════════════════════════════════════════════════════════════
#  9. STATISTICAL MODELS
# ═══════════════════════════════════════════════════════════════════

def linear_regression_slope(series: pd.Series, period: int = 20) -> pd.Series:
    """Slope of linear regression line over period."""
    result = pd.Series(np.nan, index=series.index)
    vals = series.values
    for i in range(period, len(vals)):
        window = vals[i - period + 1:i + 1]
        x = np.arange(period)
        if np.any(np.isnan(window)):
            continue
        slope = np.polyfit(x, window, 1)[0]
        result.iloc[i] = slope
    return result


def linear_regression_r2(series: pd.Series, period: int = 20) -> pd.Series:
    """R-squared of linear regression -- trend strength."""
    result = pd.Series(np.nan, index=series.index)
    vals = series.values
    for i in range(period, len(vals)):
        window = vals[i - period + 1:i + 1]
        x = np.arange(period)
        if np.any(np.isnan(window)):
            continue
        corr = np.corrcoef(x, window)[0, 1]
        result.iloc[i] = corr ** 2
    return result


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Z-score: how many std devs from mean."""
    mean = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std().replace(0, np.nan)
    return (series - mean) / std


def price_percentile(series: pd.Series, period: int = 50) -> pd.Series:
    """Where current price sits in recent range (0-100)."""
    roll_max = series.rolling(period, min_periods=1).max()
    roll_min = series.rolling(period, min_periods=1).min()
    rng = (roll_max - roll_min).replace(0, np.nan)
    return (series - roll_min) / rng * 100


def hurst_exponent(series: pd.Series, period: int = 50) -> pd.Series:
    """Simplified Hurst exponent. H>0.5 = trending, H<0.5 = mean-reverting."""
    result = pd.Series(np.nan, index=series.index)
    vals = series.values
    for i in range(period, len(vals)):
        window = vals[i - period + 1:i + 1]
        if np.any(np.isnan(window)) or len(window) < 10:
            continue
        lags = range(2, min(20, len(window) // 2))
        tau = []
        for lag in lags:
            diffs = window[lag:] - window[:-lag]
            tau.append(np.sqrt(np.mean(diffs ** 2)))
        if len(tau) < 2 or any(t <= 0 for t in tau):
            continue
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)
        try:
            h = np.polyfit(log_lags, log_tau, 1)[0]
            result.iloc[i] = h
        except (np.linalg.LinAlgError, ValueError):
            pass
    return result


def efficiency_ratio(series: pd.Series, period: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio: direction / volatility. High = trending."""
    direction = (series - series.shift(period)).abs()
    volatility = series.diff().abs().rolling(period, min_periods=1).sum()
    return direction / volatility.replace(0, np.nan)


def mean_reversion_score(series: pd.Series, period: int = 20) -> pd.Series:
    """How far price has deviated from mean, normalized."""
    mean = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std().replace(0, np.nan)
    deviation = (series - mean) / std
    return deviation


# ═══════════════════════════════════════════════════════════════════
#  10. DIVERGENCE DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_divergence(price: pd.Series, indicator: pd.Series,
                       lookback: int = 10) -> pd.Series:
    """Detect bullish/bearish divergence between price and indicator.

    Bullish divergence: price makes lower low, indicator makes higher low (+1)
    Bearish divergence: price makes higher high, indicator makes lower high (-1)
    """
    result = pd.Series(0, index=price.index)
    p = price.values
    ind_v = indicator.values

    for i in range(lookback * 2, len(p)):
        p_window = p[i - lookback:i + 1]
        i_window = ind_v[i - lookback:i + 1]

        if np.any(np.isnan(p_window)) or np.any(np.isnan(i_window)):
            continue

        p_min_idx = np.argmin(p_window)
        p_prev_min = np.min(p[max(0, i - lookback * 2):i - lookback + 1])
        i_at_low = i_window[p_min_idx]
        i_prev_at_low = np.min(ind_v[max(0, i - lookback * 2):i - lookback + 1])

        if p_window[p_min_idx] < p_prev_min and i_at_low > i_prev_at_low:
            result.iloc[i] = 1  # bullish divergence

        p_max_idx = np.argmax(p_window)
        p_prev_max = np.max(p[max(0, i - lookback * 2):i - lookback + 1])
        i_at_high = i_window[p_max_idx]
        i_prev_at_high = np.max(ind_v[max(0, i - lookback * 2):i - lookback + 1])

        if p_window[p_max_idx] > p_prev_max and i_at_high < i_prev_at_high:
            result.iloc[i] = -1  # bearish divergence

    return result


# ═══════════════════════════════════════════════════════════════════
#  11. MULTI-TIMEFRAME AGGREGATION
# ═══════════════════════════════════════════════════════════════════

def aggregate_timeframe(df: pd.DataFrame, period: int = 3) -> pd.DataFrame:
    """Aggregate N candles into higher timeframe using time-based resampling.

    Uses pandas resample to create clock-aligned buckets, preventing
    session boundary mixing when seeding multi-day data.
    """
    if len(df) == 0:
        return df.copy()

    freq = f"{period * 5}min"
    try:
        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open"])
        if len(resampled) == 0:
            return df.copy()
        return resampled
    except Exception:
        n = len(df)
        groups = n // period
        if groups == 0:
            return df.copy()
        indices, opens, highs, lows, closes, volumes = [], [], [], [], [], []
        for g in range(groups):
            start = g * period
            end = min(start + period, n)
            chunk = df.iloc[start:end]
            indices.append(chunk.index[-1])
            opens.append(chunk["open"].iloc[0])
            highs.append(chunk["high"].max())
            lows.append(chunk["low"].min())
            closes.append(chunk["close"].iloc[-1])
            volumes.append(chunk["volume"].sum())
        return pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        }, index=indices)


def compute_htf_indicators(df_5m: pd.DataFrame):
    """Compute indicators on 15m and 1h timeframes from 5m data."""
    results = {}

    # 15-minute
    df_15m = aggregate_timeframe(df_5m, 3)
    if len(df_15m) > 5:
        results["htf_15m_ema9"] = ind.ema(df_15m["close"], 9)
        results["htf_15m_ema21"] = ind.ema(df_15m["close"], 21)
        results["htf_15m_rsi"] = ind.rsi(df_15m["close"], 14)
        ml, ms, mh = ind.macd(df_15m["close"])
        results["htf_15m_macd_hist"] = mh
        st, st_d = ind.supertrend(df_15m, 7, 2.5)
        results["htf_15m_st_dir"] = st_d
        results["htf_15m_index"] = df_15m.index

    # Hourly (12 x 5min)
    df_1h = aggregate_timeframe(df_5m, 12)
    if len(df_1h) > 3:
        results["htf_1h_ema9"] = ind.ema(df_1h["close"], 9)
        results["htf_1h_ema21"] = ind.ema(df_1h["close"], 21)
        results["htf_1h_rsi"] = ind.rsi(df_1h["close"], 14)
        ml, ms, mh = ind.macd(df_1h["close"])
        results["htf_1h_macd_hist"] = mh
        st, st_d = ind.supertrend(df_1h, 7, 2.5)
        results["htf_1h_st_dir"] = st_d
        results["htf_1h_index"] = df_1h.index

    return results


# ═══════════════════════════════════════════════════════════════════
#  12. DERIVATIVE INDICATORS (indicators of indicators)
# ═══════════════════════════════════════════════════════════════════

def rsi_of_macd(close: pd.Series, macd_period_fast: int = 12,
                macd_period_slow: int = 26, rsi_period: int = 14) -> pd.Series:
    """RSI applied to MACD histogram -- momentum of momentum."""
    _, _, hist = ind.macd(close, macd_period_fast, macd_period_slow)
    return ind.rsi(hist, rsi_period)


def ema_of_rsi(close: pd.Series, rsi_period: int = 14,
               ema_period: int = 9) -> pd.Series:
    """Smoothed RSI for cleaner signals."""
    r = ind.rsi(close, rsi_period)
    return ind.ema(r, ema_period)


def bollinger_of_rsi(close: pd.Series, rsi_period: int = 14,
                      bb_period: int = 20) -> tuple:
    """Bollinger Bands applied to RSI -- detect RSI extremes dynamically."""
    r = ind.rsi(close, rsi_period)
    upper, mid, lower, pct_b, bw = ind.bollinger_bands(r, bb_period, 2.0)
    return pct_b


def stoch_rsi(close: pd.Series, rsi_period: int = 14,
               stoch_period: int = 14) -> pd.Series:
    """Stochastic RSI -- RSI treated as stochastic input."""
    r = ind.rsi(close, rsi_period)
    lowest = r.rolling(stoch_period, min_periods=1).min()
    highest = r.rolling(stoch_period, min_periods=1).max()
    return (r - lowest) / (highest - lowest).replace(0, np.nan) * 100


def obv_momentum(close: pd.Series, volume: pd.Series,
                  period: int = 10) -> pd.Series:
    """Rate of change of OBV -- volume momentum."""
    o = ind.obv(close, volume)
    return ind.roc(o, period)


# ═══════════════════════════════════════════════════════════════════
#  13. CANDLE STRUCTURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def body_ratio_series(o: pd.Series, h: pd.Series, l: pd.Series,
                      c: pd.Series) -> pd.Series:
    """Body size as ratio of total range. High = directional, Low = indecision."""
    body = (c - o).abs()
    total = (h - l).replace(0, np.nan)
    return body / total


def upper_wick_ratio(o: pd.Series, h: pd.Series, l: pd.Series,
                     c: pd.Series) -> pd.Series:
    top = pd.concat([o, c], axis=1).max(axis=1)
    total = (h - l).replace(0, np.nan)
    return (h - top) / total


def lower_wick_ratio(o: pd.Series, h: pd.Series, l: pd.Series,
                     c: pd.Series) -> pd.Series:
    bot = pd.concat([o, c], axis=1).min(axis=1)
    total = (h - l).replace(0, np.nan)
    return (bot - l) / total


def range_expansion(h: pd.Series, l: pd.Series, period: int = 5) -> pd.Series:
    """Current range vs average range. >1 = expanding, <1 = contracting."""
    current_range = h - l
    avg_range = current_range.rolling(period, min_periods=1).mean()
    return current_range / avg_range.replace(0, np.nan)


def consecutive_direction(c: pd.Series) -> pd.Series:
    """Count consecutive up/down closes. Positive = up streak, negative = down."""
    diff = c.diff()
    result = pd.Series(0, index=c.index, dtype=float)
    for i in range(1, len(c)):
        if diff.iloc[i] > 0:
            result.iloc[i] = max(result.iloc[i - 1], 0) + 1
        elif diff.iloc[i] < 0:
            result.iloc[i] = min(result.iloc[i - 1], 0) - 1
    return result


def gap_analysis(o: pd.Series, prev_c: pd.Series) -> pd.Series:
    """Gap between open and previous close. Positive = gap up."""
    return o - prev_c.shift(1)
