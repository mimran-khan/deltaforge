"""Pure-numpy/pandas technical indicators (no external TA library needed)."""

from __future__ import annotations
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP reset each day (expects 'high','low','close','volume')."""
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


def supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 2.5):
    """Returns (supertrend_line, direction) where direction=1 is bullish."""
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2

    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    st = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        prev_upper = upper_band.iloc[i - 1] if not np.isnan(upper_band.iloc[i - 1]) else upper_band.iloc[i]
        prev_lower = lower_band.iloc[i - 1] if not np.isnan(lower_band.iloc[i - 1]) else lower_band.iloc[i]

        if lower_band.iloc[i] > prev_lower:
            lower_band.iloc[i] = lower_band.iloc[i]
        else:
            lower_band.iloc[i] = prev_lower

        if upper_band.iloc[i] < prev_upper:
            upper_band.iloc[i] = upper_band.iloc[i]
        else:
            upper_band.iloc[i] = prev_upper

        if np.isnan(st.iloc[i - 1]):
            direction.iloc[i] = 1
        elif st.iloc[i - 1] == upper_band.iloc[i - 1]:
            direction.iloc[i] = -1 if df["close"].iloc[i] > upper_band.iloc[i] else 1
            if direction.iloc[i] == -1:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1
        else:
            direction.iloc[i] = 1 if df["close"].iloc[i] < lower_band.iloc[i] else -1
            if direction.iloc[i] == 1:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1

        # Simplified: bullish when close > upper, bearish when close < lower
        if df["close"].iloc[i] > upper_band.iloc[i - 1] if not np.isnan(upper_band.iloc[i - 1]) else False:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1] if not np.isnan(lower_band.iloc[i - 1]) else False:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        st.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return st, direction


def supertrend_fast(df: pd.DataFrame, period: int = 7, multiplier: float = 2.5):
    """Vectorised supertrend (faster for backtesting)."""
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
            if close[i] < dn[i]:
                direction[i] = -1
            else:
                direction[i] = 1
        else:
            if close[i] > up[i]:
                direction[i] = 1
            else:
                direction[i] = -1

        st[i] = dn[i] if direction[i] == 1 else up[i]

    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)


def add_all_indicators(df: pd.DataFrame,
                       ema_fast: int = 9,
                       ema_slow: int = 20,
                       rsi_period: int = 14,
                       atr_period: int = 14,
                       st_period: int = 7,
                       st_mult: float = 2.5) -> pd.DataFrame:
    """Add all trading indicators to a OHLCV DataFrame in-place."""
    df = df.copy()
    df["ema_fast"] = ema(df["close"], ema_fast)
    df["ema_slow"] = ema(df["close"], ema_slow)
    df["rsi"] = rsi(df["close"], rsi_period)
    df["atr"] = atr(df, atr_period)
    df["vwap"] = vwap_intraday(df)
    df["volume_sma"] = sma(df["volume"], 20)
    st_line, st_dir = supertrend_fast(df, st_period, st_mult)
    df["supertrend"] = st_line
    df["supertrend_dir"] = st_dir
    return df
