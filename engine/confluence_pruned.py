"""Pruned Confluence Engine -- only validated indicators.

Based on walk-forward IC analysis across 58 trading days:
  - 200+ indicators tested
  - Only 5 survived as predictive across both train/test periods
  - 8 identified as contrarian (signal inverted)
  - 66+ were pure noise (removed)

Stable predictive indicators (IC > 0.03 in both train and test):
  1. VOL_SPIKE_10  (volume confirmation)       IC: +0.09 / +0.09
  2. TSI           (true strength index)        IC: +0.06 / +0.07
  3. RSI_21        (long-period momentum)       IC: +0.13 / +0.05
  4. HTF_15M_RSI   (higher-timeframe momentum)  IC: +0.43 / +0.04
  5. LR_R2_TREND   (trend quality/R²)          IC: +0.15 / +0.03

Stable contrarian indicators (IC < -0.03 in both periods):
  Signals from these are INVERTED before use.

Architecture: same precompute/score interface as full ConfluenceEngine
for drop-in replacement. IC tracking built in for ongoing validation.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine import indicators as ind
from engine import indicators_extended as indx


@dataclass
class PrunedVote:
    name: str
    signal: int       # +1, 0, -1 (already inverted if contrarian)
    raw_signal: int   # original signal before inversion
    weight: float
    value: float
    is_contrarian: bool = False


@dataclass
class PrunedResult:
    score: float
    direction: str
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_indicators: int
    votes: list = field(default_factory=list)
    strength: str = "NONE"

    def summary(self) -> str:
        return (f"Pruned: {self.score:+.1f} ({self.strength} {self.direction}) "
                f"[{self.bullish_count}B/{self.bearish_count}S/{self.neutral_count}N "
                f"of {self.total_indicators}]")


# Weights based on train IC magnitude
PREDICTIVE_WEIGHTS = {
    "HTF_15M_RSI":  2.5,    # highest IC
    "RSI_21":       1.5,
    "LR_R2_TREND":  1.5,
    "VOL_SPIKE_10": 1.2,    # most stable IC
    "TSI":          1.0,
}

CONTRARIAN = {
    "AD_LINE", "BODY_RATIO", "RSI_OF_MACD", "MACD_8_17",
    "HTF_1H_EMA", "HTF_1H_MACD", "STOCH_RSI", "DIV_MFI",
}


class PrunedConfluenceEngine:
    """Pruned engine using only IC-validated indicators.

    Same interface as ConfluenceEngine for drop-in replacement.
    """

    def __init__(self):
        self._ic_buffer: list[dict] = []

    def precompute(self, df: pd.DataFrame) -> dict:
        """Precompute only the indicators we actually use."""
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]
        o = df["open"]
        data = {"close": c, "high": h, "low": l, "volume": v, "open": o}

        # ── Predictive indicators ──

        # RSI_21
        data["rsi_21"] = ind.rsi(c, 21)

        # TSI (True Strength Index)
        data["tsi"] = ind.tsi(c)

        # LR_R2_TREND
        data["lr_r2"] = indx.linear_regression_slope(c, 20)

        # VOL_SPIKE_10
        avg_vol_10 = v.rolling(10).mean()
        data["vol_spike_10"] = v / avg_vol_10

        # HTF 15M indicators
        try:
            htf_data = indx.compute_htf_indicators(df)
            data["htf_15m_rsi"] = htf_data.get("htf_15m_rsi", pd.Series(np.nan, index=c.index))
            data["htf_15m_ema_9"] = htf_data.get("htf_15m_ema_9", pd.Series(np.nan, index=c.index))
            data["htf_15m_ema_21"] = htf_data.get("htf_15m_ema_21", pd.Series(np.nan, index=c.index))
            data["htf_15m_macd"] = htf_data.get("htf_15m_macd", pd.Series(np.nan, index=c.index))
            data["htf_15m_macd_signal"] = htf_data.get("htf_15m_macd_signal", pd.Series(np.nan, index=c.index))
        except Exception:
            for key in ["htf_15m_rsi", "htf_15m_ema_9", "htf_15m_ema_21",
                         "htf_15m_macd", "htf_15m_macd_signal"]:
                data[key] = pd.Series(np.nan, index=c.index)

        # HTF 1H indicators (contrarian)
        try:
            data["htf_1h_ema_9"] = htf_data.get("htf_1h_ema_9", pd.Series(np.nan, index=c.index))
            data["htf_1h_ema_21"] = htf_data.get("htf_1h_ema_21", pd.Series(np.nan, index=c.index))
            data["htf_1h_macd"] = htf_data.get("htf_1h_macd", pd.Series(np.nan, index=c.index))
            data["htf_1h_macd_signal"] = htf_data.get("htf_1h_macd_signal", pd.Series(np.nan, index=c.index))
        except Exception:
            for key in ["htf_1h_ema_9", "htf_1h_ema_21", "htf_1h_macd", "htf_1h_macd_signal"]:
                data[key] = pd.Series(np.nan, index=c.index)

        # ── Contrarian indicators ──

        # MACD_8_17
        data["macd_8_17"] = ind.macd(c, 8, 17)[0]
        data["macd_8_17_signal"] = ind.macd(c, 8, 17)[1]

        # AD_LINE
        data["ad_line"] = ind.ad_line(h, l, c, v)

        # StochRSI
        data["stoch_rsi"] = indx.stoch_rsi(c, 14, 14)

        # RSI of MACD
        try:
            data["rsi_of_macd"] = indx.rsi_of_macd(c)
        except Exception:
            data["rsi_of_macd"] = pd.Series(np.nan, index=c.index)

        # BODY_RATIO
        try:
            data["body_ratio"] = indx.body_ratio_series(o, h, l, c)
        except Exception:
            data["body_ratio"] = pd.Series(np.nan, index=c.index)

        # DIV_MFI
        try:
            data["mfi"] = ind.mfi(h, l, c, v, 14)
        except Exception:
            data["mfi"] = pd.Series(np.nan, index=c.index)

        return data

    def score(self, data: dict, i: int) -> PrunedResult:
        """Score using only IC-validated indicators."""

        def g(key):
            s = data.get(key)
            if s is None:
                return np.nan
            try:
                return float(s.iloc[i])
            except (IndexError, ValueError):
                return np.nan

        def gp(key, idx, default=np.nan):
            s = data.get(key)
            if s is None:
                return default
            try:
                return float(s.iloc[idx])
            except (IndexError, ValueError):
                return default

        c_now = g("close")
        votes = []

        # ═══════════════════════════════════════════════════
        #  PREDICTIVE INDICATORS (5)
        # ═══════════════════════════════════════════════════

        # 1. HTF_15M_RSI
        htf_rsi = g("htf_15m_rsi")
        if not np.isnan(htf_rsi):
            sig = 1 if htf_rsi > 55 else (-1 if htf_rsi < 45 else 0)
            votes.append(PrunedVote("HTF_15M_RSI", sig, sig,
                                     PREDICTIVE_WEIGHTS["HTF_15M_RSI"], htf_rsi))

        # 2. RSI_21
        rsi21 = g("rsi_21")
        if not np.isnan(rsi21):
            sig = 1 if rsi21 > 55 else (-1 if rsi21 < 45 else 0)
            votes.append(PrunedVote("RSI_21", sig, sig,
                                     PREDICTIVE_WEIGHTS["RSI_21"], rsi21))

        # 3. LR_R2_TREND
        lr = g("lr_r2")
        if not np.isnan(lr):
            sig = 1 if lr > 0.5 else (-1 if lr < -0.5 else 0)
            votes.append(PrunedVote("LR_R2_TREND", sig, sig,
                                     PREDICTIVE_WEIGHTS["LR_R2_TREND"], lr))

        # 4. VOL_SPIKE_10
        vs = g("vol_spike_10")
        if not np.isnan(vs) and vs > 1.5:
            body = c_now - gp("open", i, c_now)
            sig = 1 if body > 0 else -1
            votes.append(PrunedVote("VOL_SPIKE_10", sig, sig,
                                     PREDICTIVE_WEIGHTS["VOL_SPIKE_10"], vs))

        # 5. TSI
        tsi = g("tsi")
        if not np.isnan(tsi):
            sig = 1 if tsi > 5 else (-1 if tsi < -5 else 0)
            votes.append(PrunedVote("TSI", sig, sig,
                                     PREDICTIVE_WEIGHTS["TSI"], tsi))

        # ═══════════════════════════════════════════════════
        #  CONTRARIAN INDICATORS (inverted signals)
        # ═══════════════════════════════════════════════════

        # AD_LINE
        ad = g("ad_line")
        if not np.isnan(ad):
            ad_prev = gp("ad_line", max(0, i - 5), ad)
            raw = 1 if ad > ad_prev else -1
            votes.append(PrunedVote("AD_LINE", -raw, raw, 0.8, ad, True))

        # MACD_8_17
        macd_val = g("macd_8_17")
        macd_sig = g("macd_8_17_signal")
        if not np.isnan(macd_val) and not np.isnan(macd_sig):
            raw = 1 if macd_val > macd_sig else -1
            votes.append(PrunedVote("MACD_8_17", -raw, raw, 0.7, macd_val, True))

        # STOCH_RSI
        sr = g("stoch_rsi")
        if not np.isnan(sr):
            raw = 1 if sr > 70 else (-1 if sr < 30 else 0)
            votes.append(PrunedVote("STOCH_RSI", -raw, raw, 0.6, sr, True))

        # RSI_OF_MACD
        rom = g("rsi_of_macd")
        if not np.isnan(rom):
            raw = 1 if rom > 60 else (-1 if rom < 40 else 0)
            votes.append(PrunedVote("RSI_OF_MACD", -raw, raw, 0.7, rom, True))

        # BODY_RATIO
        br = g("body_ratio")
        if not np.isnan(br) and br > 0.7:
            body = c_now - gp("open", i, c_now)
            raw = 1 if body > 0 else -1
            votes.append(PrunedVote("BODY_RATIO", -raw, raw, 0.5, br, True))

        # HTF_1H_EMA
        h1_e9 = g("htf_1h_ema_9")
        h1_e21 = g("htf_1h_ema_21")
        if not np.isnan(h1_e9) and not np.isnan(h1_e21):
            raw = 1 if h1_e9 > h1_e21 else -1
            votes.append(PrunedVote("HTF_1H_EMA", -raw, raw, 0.6, h1_e9, True))

        # HTF_1H_MACD
        h1_m = g("htf_1h_macd")
        h1_ms = g("htf_1h_macd_signal")
        if not np.isnan(h1_m) and not np.isnan(h1_ms):
            raw = 1 if h1_m > h1_ms else -1
            votes.append(PrunedVote("HTF_1H_MACD", -raw, raw, 0.6, h1_m, True))

        # DIV_MFI
        mfi = g("mfi")
        if not np.isnan(mfi):
            raw = 1 if mfi > 60 else (-1 if mfi < 40 else 0)
            votes.append(PrunedVote("DIV_MFI", -raw, raw, 0.8, mfi, True))

        # ═══════════════════════════════════════════════════
        #  TALLY
        # ═══════════════════════════════════════════════════

        if not votes:
            return PrunedResult(0, "NEUTRAL", 0, 0, 0, 0)

        bullish = sum(1 for v in votes if v.signal > 0)
        bearish = sum(1 for v in votes if v.signal < 0)
        neutral = sum(1 for v in votes if v.signal == 0)

        weighted_sum = sum(v.signal * v.weight for v in votes)
        max_possible = sum(abs(v.weight) for v in votes)
        score = (weighted_sum / max_possible * 100) if max_possible > 0 else 0

        direction = "LONG" if score > 10 else ("SHORT" if score < -10 else "NEUTRAL")

        abs_score = abs(score)
        if abs_score >= 70:
            strength = "EXTREME"
        elif abs_score >= 50:
            strength = "STRONG"
        elif abs_score >= 30:
            strength = "MODERATE"
        elif abs_score >= 15:
            strength = "WEAK"
        else:
            strength = "NONE"

        return PrunedResult(
            score=round(score, 1), direction=direction,
            bullish_count=bullish, bearish_count=bearish,
            neutral_count=neutral, total_indicators=len(votes),
            votes=votes, strength=strength,
        )

    def track_ic(self, predicted_direction: str, predicted_score: float,
                  actual_return: float):
        """Track IC for ongoing validation. Call after each trade."""
        self._ic_buffer.append({
            "score": predicted_score,
            "return": actual_return if predicted_direction == "LONG" else -actual_return,
        })

    def get_ic(self) -> float:
        """Get current IC from buffer."""
        if len(self._ic_buffer) < 10:
            return 0
        scores = [d["score"] for d in self._ic_buffer[-50:]]
        rets = [d["return"] for d in self._ic_buffer[-50:]]
        if len(set(scores)) <= 1:
            return 0
        ic = np.corrcoef(scores, rets)[0, 1]
        return ic if not np.isnan(ic) else 0
