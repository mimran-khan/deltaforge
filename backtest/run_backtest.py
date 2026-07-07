"""Backtest with true daily compounding -- lot size scales with equity every day.

Usage:
    python -m backtest.run_backtest [--days 100] [--capital 10000]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from engine.premium_model import STRATEGY_SL_PCT, create_premium_state


def _compute_dte_for_date(d: date) -> float:
    """Days to next Nifty weekly expiry (Tuesday) from a given date."""
    expiry_weekday = getattr(settings, 'NIFTY_EXPIRY_DAY', 1)  # Tuesday=1
    days_ahead = (expiry_weekday - d.weekday()) % 7
    return float(max(days_ahead, 0))


DATA_DIR = PROJECT_ROOT / "data"

_5M_FILES = [
    DATA_DIR / "nifty_5m_real.csv",
    DATA_DIR / "nifty_5m_oos_2015_2024.csv",
    DATA_DIR / "nifty_5m_all_merged.csv",
]

_1M_RAW = DATA_DIR / "nifty50_1min_github_raw.csv"
_COMBINED_CACHE = DATA_DIR / "nifty_5m_combined.csv"


def _load_5m_csv(path: Path) -> pd.DataFrame:
    """Load a standard 5m CSV with date index + OHLCV columns."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.sort_index()
    return df


def _resample_1m_to_5m(path: Path) -> pd.DataFrame:
    """Resample the 1-minute raw CSV (Instrument,Date,Time,O,H,L,C) to 5m OHLCV."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%d-%m-%Y %H:%M:%S"
    )
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
    if "volume" not in df.columns:
        df["volume"] = 5000

    resampled = df[["open", "high", "low", "close", "volume"]].resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    return resampled


def _build_combined() -> pd.DataFrame:
    """Combine ALL Nifty data sources into one deduplicated 5m DataFrame."""
    frames = []

    for path in _5M_FILES:
        if path.exists():
            try:
                frames.append(_load_5m_csv(path))
                logger.info("Loaded {} from {}", len(frames[-1]), path.name)
            except Exception as e:
                logger.debug("Skip {}: {}", path.name, e)

    if _1M_RAW.exists():
        try:
            resampled = _resample_1m_to_5m(_1M_RAW)
            frames.append(resampled)
            logger.info("Resampled 1m->5m: {} candles from {}", len(resampled), _1M_RAW.name)
        except Exception as e:
            logger.debug("Skip 1m resample: {}", e)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return combined


def load_real_data(days: int = 60) -> pd.DataFrame:
    """Load real Nifty 5m data from ALL sources on disk and return the most recent N trading days.

    Sources combined (deduplicated, most-recent wins):
      - data/nifty_5m_real.csv        (recent 2026 data)
      - data/nifty_5m_oos_2015_2024.csv (10 years of 5m data)
      - data/nifty50_1min_github_raw.csv (resampled 1m->5m, fills volume gaps)
    """
    if _COMBINED_CACHE.exists():
        try:
            df = _load_5m_csv(_COMBINED_CACHE)
            unique_days = sorted(set(df.index.date))
            if len(unique_days) >= days:
                selected = set(unique_days[-days:])
                return df[df.index.map(lambda t: t.date() in selected)]
            return df
        except Exception:
            pass

    df = _build_combined()

    if df.empty:
        logger.warning("No real data found -- falling back to synthetic data")
        return generate_nifty_data(days=days)

    try:
        df.to_csv(_COMBINED_CACHE)
        logger.info("Cached combined dataset: {} candles -> {}", len(df), _COMBINED_CACHE.name)
    except Exception:
        pass

    unique_days = sorted(set(df.index.date))
    if len(unique_days) >= days:
        selected = set(unique_days[-days:])
        return df[df.index.map(lambda t: t.date() in selected)]
    return df


def generate_nifty_data(days: int = 100, interval: int = 5,
                         base: float = 24000) -> pd.DataFrame:
    """Generate realistic Nifty 5-min data with proper ORB breakout patterns."""
    np.random.seed(42)
    rows = []
    price = base
    trading_days = 0
    cur_date = date.today() - timedelta(days=days + 80)

    while trading_days < days:
        cur_date += timedelta(days=1)
        if cur_date.weekday() >= 5:
            continue
        trading_days += 1

        day_type = np.random.choice(
            ["strong_up", "strong_down", "mild_up", "mild_down", "range"],
            p=[0.15, 0.12, 0.18, 0.15, 0.40])

        gap = np.random.normal(0, 30)
        day_open = price + gap

        candles = int((6 * 60 + 15) / interval)
        p = day_open
        day_prices = []

        for j in range(candles):
            mins = j * interval
            hour = 9 + (15 + mins) // 60
            minute = (15 + mins) % 60
            if hour > 15 or (hour == 15 and minute > 30):
                break
            ts = datetime(cur_date.year, cur_date.month, cur_date.day, hour, minute)

            frac = j / candles

            if day_type == "strong_up":
                drift = np.random.uniform(1.5, 4.0)
                if j < 3:
                    drift = np.random.uniform(-2, 2)
            elif day_type == "strong_down":
                drift = np.random.uniform(-4.0, -1.5)
                if j < 3:
                    drift = np.random.uniform(-2, 2)
            elif day_type == "mild_up":
                drift = np.random.uniform(0.3, 2.0)
            elif day_type == "mild_down":
                drift = np.random.uniform(-2.0, -0.3)
            else:
                drift = np.random.normal(0, 1.0)
                drift += (day_open - p) * 0.03

            noise = np.random.normal(0, 3.0)
            p = p + drift + noise

            spread = abs(np.random.normal(0, 5))
            o = p + np.random.normal(0, 2)
            c = p + drift + np.random.normal(0, 2)
            h = max(o, c) + spread
            l = min(o, c) - spread

            if frac < 0.05 or frac > 0.9:
                vol = int(np.random.uniform(200000, 500000))
            elif 0.15 < frac < 0.25:
                vol = int(np.random.uniform(120000, 350000))
            else:
                vol = int(np.random.uniform(50000, 150000))

            rows.append({"timestamp": ts, "open": round(o, 2), "high": round(h, 2),
                         "low": round(l, 2), "close": round(c, 2), "volume": vol})
            day_prices.append(c)

        if day_prices:
            price = day_prices[-1]

    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    return df


def _calc_realistic_costs(entry_prem: float, exit_prem: float,
                           qty: int, lots: int) -> float:
    """Industry-standard execution cost model for NFO options.

    Components (per round trip):
      1. Brokerage: Rs 20/order x 2 = Rs 40
      2. Bid-ask spread: Rs 0.30/unit x qty x 2 sides
      3. STT: 0.05% of sell-side premium turnover
      4. Exchange + SEBI + stamp + GST: 0.05% of total turnover
      5. Market impact: 0.1% of premium for 5+ lots
    """
    brokerage = getattr(settings, 'BROKERAGE_PER_ORDER', 20) * 2

    spread_per_unit = getattr(settings, 'BID_ASK_SPREAD', 0.30)
    spread_cost = spread_per_unit * qty * 2

    sell_turnover = exit_prem * qty
    stt = sell_turnover * getattr(settings, 'STT_SELL_PCT', 0.05) / 100

    total_turnover = (entry_prem + exit_prem) * qty
    exchange_costs = total_turnover * getattr(settings, 'EXCHANGE_TXN_PCT', 0.05) / 100

    impact = 0.0
    if lots >= 5:
        impact_pct = getattr(settings, 'MARKET_IMPACT_PCT', 0.10)
        impact = total_turnover * impact_pct / 100

    return brokerage + spread_cost + stt + exchange_costs + impact


def run_compound_backtest(df: pd.DataFrame,
                           starting_capital: float = 10000,
                           lot_size: int = 65,
                           deploy_pct: float = 80.0,
                           engine_override=None,
                           use_adaptive: bool = True,
                           use_risk_gates: bool = True) -> dict:
    """Full compound backtest using the production MultiStrategyEngine.

    Uses the same signal generation, premium model, and risk gates
    that run in live/paper trading -- results match real behaviour.
    Costs modeled per industry standard (spread + STT + exchange + impact).
    """
    from engine.multi_strategy_engine import MultiStrategyEngine
    from risk.adaptive_mode import AdaptiveModeController

    engine = engine_override if engine_override is not None else MultiStrategyEngine()
    adaptive = AdaptiveModeController() if use_adaptive else None

    capital = starting_capital
    peak = capital
    trades = []
    equity_curve = []

    unique_days = sorted(set(df.index.date))
    WARMUP_DAYS = 5  # Match check_today.py and live candle_builder

    for day_idx, day in enumerate(unique_days):
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 10:
            equity_curve.append({"date": day, "capital": capital, "daily_pnl": 0,
                                  "trades": 0, "lots": 0})
            continue

        prev_day_data = None
        if day_idx > 0:
            prev_day = unique_days[day_idx - 1]
            prev_df = df[df.index.date == prev_day]
            if len(prev_df) > 0:
                prev_day_data = {
                    "high": prev_df["high"].max(),
                    "low": prev_df["low"].min(),
                    "close": prev_df["close"].iloc[-1],
                }
        engine.reset_day(prev_day_data)
        if adaptive:
            adaptive.reset()

        day_start_cap = capital
        day_pnl = 0
        day_trades = 0
        day_wins = 0
        day_losses = 0
        consec_loss = 0
        daily_loss_limit = day_start_cap * (settings.DAILY_LOSS_LIMIT_PCT / 100)
        daily_profit_target = day_start_cap * (getattr(settings, 'DAILY_PROFIT_TARGET_PCT', 35) / 100) if use_risk_gates else float('inf')

        per_lot = getattr(settings, 'CAPITAL_PER_LOT', 10_000)
        deployable = capital * (deploy_pct / 100)
        day_lots = max(1, int(deployable / per_lot))
        day_lots = min(day_lots, getattr(settings, 'MAX_LOTS_CAP', 20))

        # Multi-day warmup (like check_today.py and live engine)
        warmup_start = max(0, day_idx - WARMUP_DAYS)
        warmup_days = unique_days[warmup_start:day_idx + 1]
        warmup_day_set = set(warmup_days)
        warmup_df = df[df.index.map(lambda t: t.date() in warmup_day_set)]
        indicators = engine.precompute(warmup_df)
        today_indices = [i for i, ts in enumerate(warmup_df.index) if ts.date() == day]

        open_positions = []

        for i in today_indices:
            if i < 10:
                continue
            ts = warmup_df.index[i]
            time_str = ts.strftime("%H:%M")
            nifty_price = warmup_df["close"].iloc[i]

            # Adaptive mode bar tick
            if adaptive:
                adaptive.on_bar()

            ap_min_conf = 65 if (adaptive and adaptive.mode.value != "AGGRESSIVE") else 60
            ap_lot_mult = 1.0
            ap_sl_mult = 1.0
            ap_target_mult = 1.0
            ap_max_sim = getattr(settings, 'MAX_SIMULTANEOUS_POSITIONS', 2)
            ap_max_trades = 8
            ap_trail_trigger = settings.TRAIL_TRIGGER_PCT
            ap_trail_pct = settings.TRAIL_PCT
            if adaptive:
                ap = adaptive.profile
                ap_min_conf = ap.min_confidence
                ap_lot_mult = ap.lot_multiplier
                ap_sl_mult = ap.sl_multiplier
                ap_target_mult = ap.target_multiplier
                ap_max_sim = ap.max_simultaneous
                ap_max_trades = ap.max_trades_per_day
                ap_trail_trigger = ap.trail_trigger_pct
                ap_trail_pct = ap.trail_pct

            closed_this_bar = []
            for pos in open_positions:
                pos["candles_held"] += 1
                cur_prem = pos["prem_state"].current_premium(
                    nifty_price, pos["candles_held"])

                if cur_prem > pos["peak_premium"]:
                    pos["peak_premium"] = cur_prem

                trail_floor = pos["prem_state"].update_trail(
                    cur_prem,
                    trigger_pct=ap_trail_trigger,
                    trail_pct=ap_trail_pct)

                partial_pct = getattr(settings, 'PARTIAL_PROFIT_PCT', 25)
                if (not pos.get("partial_booked")
                        and pos["entry_premium"] > 0
                        and (cur_prem - pos["entry_premium"]) / pos["entry_premium"] * 100 >= partial_pct
                        and pos["qty"] > lot_size):
                    half_qty = (pos["qty"] // (2 * lot_size)) * lot_size
                    if half_qty >= lot_size:
                        partial_pnl = (cur_prem - pos["entry_premium"]) * half_qty
                        partial_costs = _calc_realistic_costs(
                            pos["entry_premium"], cur_prem, half_qty, day_lots)
                        net_partial = partial_pnl - partial_costs
                        capital += net_partial
                        day_pnl += net_partial
                        pos["qty"] -= half_qty
                        pos["partial_booked"] = True
                        pos["sl_premium"] = pos["entry_premium"]
                        trades.append({
                            "strategy": pos["signal_type"], "signal": pos["direction"],
                            "entry_time": pos["entry_time"], "exit_time": ts,
                            "entry_premium": round(pos["entry_premium"], 2),
                            "exit_premium": round(cur_prem, 2),
                            "peak_premium": round(pos["peak_premium"], 2),
                            "peak_gain_pct": round(partial_pct, 2),
                            "qty": half_qty, "lots": day_lots,
                            "pnl": round(net_partial, 0), "reason": "PARTIAL",
                            "capital_after": round(capital, 0),
                        })
                        day_trades += 1
                        day_wins += 1
                        consec_loss = 0

                exit_reason = None
                exit_prem = cur_prem

                if cur_prem <= pos["sl_premium"]:
                    exit_reason = "SL"
                    exit_prem = pos["sl_premium"]
                elif cur_prem >= pos["prem_state"].target_premium:
                    exit_reason = "TGT"
                    exit_prem = pos["prem_state"].target_premium
                elif trail_floor is not None and cur_prem <= trail_floor:
                    exit_reason = "TRAIL"
                    exit_prem = trail_floor
                elif pos["candles_held"] >= settings.PULLBACK_HOLD_CANDLES:
                    exit_reason = "TIME"
                elif time_str >= settings.SQUARE_OFF_TIME:
                    exit_reason = "EOD"

                if exit_reason:
                    if exit_reason == "SL":
                        engine.record_sl_exit(pos["signal_type"], i)

                    costs = _calc_realistic_costs(
                        pos["entry_premium"], exit_prem,
                        pos["qty"], day_lots)

                    raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
                    net_pnl = raw_pnl - costs

                    capital += net_pnl
                    day_pnl += net_pnl
                    day_trades += 1
                    won = net_pnl > 0
                    if won:
                        day_wins += 1
                        consec_loss = 0
                    else:
                        day_losses += 1
                        consec_loss += 1

                    if adaptive:
                        daily_pnl_pct = (day_pnl / day_start_cap * 100) if day_start_cap > 0 else 0
                        adaptive.update(
                            daily_pnl_pct=daily_pnl_pct,
                            wins=day_wins, losses=day_losses,
                            consecutive_losses=consec_loss,
                            trades=day_trades, last_trade_won=won,
                        )

                    peak_gain = (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100

                    trades.append({
                        "strategy": pos["signal_type"], "signal": pos["direction"],
                        "entry_time": pos["entry_time"], "exit_time": ts,
                        "entry_premium": round(pos["entry_premium"], 2),
                        "exit_premium": round(exit_prem, 2),
                        "peak_premium": round(pos["peak_premium"], 2),
                        "peak_gain_pct": round(peak_gain, 2),
                        "qty": pos["qty"], "lots": day_lots,
                        "pnl": round(net_pnl, 0), "reason": exit_reason,
                        "capital_after": round(capital, 0),
                    })
                    closed_this_bar.append(pos)

            for pos in closed_this_bar:
                open_positions.remove(pos)

            if len(open_positions) >= ap_max_sim:
                continue

            if consec_loss >= settings.MAX_CONSECUTIVE_LOSSES:
                continue
            if day_pnl < 0 and abs(day_pnl) >= daily_loss_limit:
                continue
            if use_risk_gates and day_pnl >= daily_profit_target:
                continue
            if adaptive and adaptive.mode.value == "HALT":
                continue
            if day_trades >= ap_max_trades:
                continue

            open_dirs = {p["direction"] for p in open_positions}
            signals = engine.scan(indicators, i, time_str,
                                  max_total_override=ap_max_trades)

            for signal in signals:
                if signal.direction in open_dirs:
                    continue
                if signal.confidence < ap_min_conf:
                    continue

                theta = settings.get_scaled_theta(nifty_price)
                dte = _compute_dte_for_date(day)
                prem_state = create_premium_state(
                    entry_index_price=nifty_price,
                    direction=signal.direction,
                    base_premium=settings.PREMIUM_BASE,
                    delta=settings.PREMIUM_DELTA,
                    theta_per_candle=theta,
                    sl_pct=settings.PREMIUM_SL_PCT,
                    confluence_score=signal.confidence,
                    signal_type=signal.signal_type,
                    dte=dte,
                )

                if ap_target_mult != 1.0:
                    prem_state.target_premium = (
                        prem_state.entry_premium
                        + (prem_state.target_premium - prem_state.entry_premium) * ap_target_mult
                    )

                spread = getattr(settings, 'BID_ASK_SPREAD', 0.30)
                entry_premium = prem_state.entry_premium + spread
                vol_ratio = getattr(signal, 'vol_ratio', 1.0)
                eff_sl = STRATEGY_SL_PCT.get(signal.signal_type, settings.PREMIUM_SL_PCT) * ap_sl_mult * vol_ratio
                sl_premium = entry_premium * (1 - eff_sl / 100)
                eff_lots = max(1, int(day_lots * ap_lot_mult))
                qty = eff_lots * lot_size

                open_positions.append({
                    "direction": signal.direction,
                    "signal_type": signal.signal_type,
                    "entry_time": ts,
                    "entry_premium": entry_premium,
                    "sl_premium": sl_premium,
                    "qty": qty,
                    "prem_state": prem_state,
                    "candles_held": 0,
                    "peak_premium": entry_premium,
                })
                break

            if capital <= 0:
                break

        for pos in open_positions:
            last_bar_idx = today_indices[-1]
            nifty_price = warmup_df["close"].iloc[last_bar_idx]
            exit_prem = pos["prem_state"].current_premium(
                nifty_price, pos["candles_held"])
            brokerage = getattr(settings, 'BROKERAGE_PER_ORDER', 20) * 2
            stt = exit_prem * pos["qty"] * getattr(settings, 'STT_SELL_PCT', 0.05) / 100
            slippage = getattr(settings, 'SLIPPAGE_POINTS', 0.30) * pos["qty"]
            costs = brokerage + stt + slippage
            raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
            net_pnl = raw_pnl - costs
            capital += net_pnl
            day_pnl += net_pnl
            day_trades += 1
            peak_gain = (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100

            trades.append({
                "strategy": pos["signal_type"], "signal": pos["direction"],
                "entry_time": pos["entry_time"], "exit_time": warmup_df.index[last_bar_idx],
                "entry_premium": round(pos["entry_premium"], 2),
                "exit_premium": round(exit_prem, 2),
                "peak_premium": round(pos["peak_premium"], 2),
                "peak_gain_pct": round(peak_gain, 2),
                "qty": pos["qty"], "lots": day_lots,
                "pnl": round(net_pnl, 0), "reason": "EOD",
                "capital_after": round(capital, 0),
            })

        if capital > peak:
            peak = capital

        equity_curve.append({
            "date": day, "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "trades": day_trades, "lots": day_lots,
        })

        if capital <= 0:
            break

    # ── Stats ───────────────────────────────────────────────────
    total_pnl = capital - starting_capital
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    eq_vals = [e["capital"] for e in equity_curve]
    peak_arr = np.maximum.accumulate(eq_vals) if eq_vals else [starting_capital]
    dd_arr = [(p - v) / p * 100 for p, v in zip(peak_arr, eq_vals)]
    max_dd = max(dd_arr) if dd_arr else 0

    active_days = [e for e in equity_curve if e["trades"] > 0]
    daily_rets = [(e["daily_pnl"] / max(e["capital"] - e["daily_pnl"], 1)) * 100
                  for e in active_days]
    avg_daily_ret = np.mean(daily_rets) if daily_rets else 0

    profitable_days = sum(1 for e in equity_curve if e["daily_pnl"] > 0)
    loss_days = sum(1 for e in equity_curve if e["daily_pnl"] < 0)
    flat_days = sum(1 for e in equity_curve if e["daily_pnl"] == 0)

    return {
        "starting_capital": starting_capital,
        "final_capital": round(capital, 0),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / starting_capital * 100, 1),
        "total_trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 0), "avg_loss": round(avg_loss, 0),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "profitable_days": profitable_days,
        "loss_days": loss_days, "flat_days": flat_days,
        "trading_days": len(equity_curve),
        "active_trading_days": len(active_days),
        "avg_daily_return_pct": round(avg_daily_ret, 1),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def print_results(results: dict):
    ec = results["equity_curve"]

    print("\n" + "=" * 65)
    print("     BACKTEST: DAILY COMPOUNDING RESULTS")
    print("=" * 65)
    print(f"  Starting Capital      : Rs {results['starting_capital']:>12,.0f}")
    print(f"  Final Capital         : Rs {results['final_capital']:>12,.0f}")
    pnl = results['total_pnl']
    s = "+" if pnl >= 0 else ""
    print(f"  Total P&L             : Rs {s}{pnl:>11,.0f} ({results['return_pct']}%)")
    print(f"  Avg Daily Return      : {results['avg_daily_return_pct']}% (on active days)")
    print("-" * 65)
    print(f"  Calendar Days         : {results['trading_days']}")
    print(f"  Active Trading Days   : {results['active_trading_days']}")
    print(f"  Profitable Days       : {results['profitable_days']}")
    print(f"  Loss Days             : {results['loss_days']}")
    print(f"  No-Trade Days         : {results['flat_days']}")
    print("-" * 65)
    print(f"  Total Trades          : {results['total_trades']}")
    print(f"  Wins                  : {results['wins']}")
    print(f"  Losses                : {results['losses']}")
    print(f"  Win Rate              : {results['win_rate']}%")
    print(f"  Avg Win               : Rs {results['avg_win']:>10,.0f}")
    print(f"  Avg Loss              : Rs {results['avg_loss']:>10,.0f}")
    print(f"  Profit Factor         : {results['profit_factor']}")
    print(f"  Max Drawdown          : {results['max_drawdown_pct']}%")
    print("=" * 65)

    # Strategy breakdown
    strat_stats = {}
    for t in results["trades"]:
        s = t["strategy"]
        if s not in strat_stats:
            strat_stats[s] = {"w": 0, "l": 0, "pnl": 0, "n": 0}
        strat_stats[s]["n"] += 1
        if t["pnl"] > 0:
            strat_stats[s]["w"] += 1
        else:
            strat_stats[s]["l"] += 1
        strat_stats[s]["pnl"] += t["pnl"]

    print("\n  Strategy Breakdown:")
    for name, st in sorted(strat_stats.items()):
        wr = st["w"] / st["n"] * 100 if st["n"] > 0 else 0
        print(f"    {name:12s}: {st['n']:>3d} trades | "
              f"{wr:>4.0f}% win | Rs {st['pnl']:>10,.0f}")

    # Capital journey
    if ec:
        print("\n  Capital Growth (Compound Journey):")
        print(f"    {'Day':>5s}  {'Date':>12s}  {'Capital':>12s}  "
              f"{'Day PnL':>10s}  {'Lots':>5s}  {'Trades':>6s}")
        print("    " + "-" * 58)

        show = set([0, len(ec) - 1])
        for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            show.add(min(int(len(ec) * pct / 100), len(ec) - 1))
        for i, e in enumerate(ec):
            if e["daily_pnl"] != 0 and len(show) < 25:
                show.add(i)

        for i in sorted(show):
            e = ec[i]
            dpnl = e["daily_pnl"]
            s = "+" if dpnl >= 0 else ""
            print(f"    {i+1:>5d}  {e['date']}  Rs {e['capital']:>10,.0f}  "
                  f"Rs {s}{dpnl:>8,.0f}  {e['lots']:>5d}  {e['trades']:>6d}")

    # Theoretical 10% daily compound comparison
    print("\n  10% Daily Compound Target (theoretical):")
    for d in [10, 20, 30, 50, 75, 100]:
        if d <= results["trading_days"]:
            theoretical = results["starting_capital"] * (1.10 ** d)
            print(f"    Day {d:>3d}: Rs {theoretical:>12,.0f}")

    print()
    if results["profit_factor"] > 1.5 and results["win_rate"] > 45:
        print("  VERDICT: STRONG edge -- ready for paper trading")
    elif results["profit_factor"] > 1.2:
        print("  VERDICT: Moderate edge -- paper trade to confirm")
    elif results["profit_factor"] > 1.0:
        print("  VERDICT: Marginal -- needs tuning")
    else:
        print("  VERDICT: Negative expectancy -- do NOT trade live")
    print()


def _load_futures_data(instrument_name: str, days: int) -> pd.DataFrame:
    """Load historical data for a futures instrument.

    Looks for files named <instrument>_5m.csv in the data directory.
    Falls back to Angel One historical API if available.
    """
    data_file = DATA_DIR / f"{instrument_name.lower()}_5m.csv"
    combined_file = DATA_DIR / f"{instrument_name.lower()}_5m_combined.csv"

    for f in [combined_file, data_file]:
        if f.exists():
            df = _load_5m_csv(f)
            unique_days = sorted(set(df.index.date))
            if len(unique_days) > days:
                cutoff_days = unique_days[-days:]
                df = df[df.index.date >= cutoff_days[0]]
            return df

    print(f"  No data file found for {instrument_name}.")
    print(f"  Expected: {data_file} or {combined_file}")
    print("  Download MCX/CDS historical data and save as 5m OHLCV CSV.")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--deploy-pct", type=float, default=100)
    parser.add_argument("--instrument", type=str, default="NIFTY",
                        help="NIFTY, GOLD_PETAL, CRUDEOILM, or USDINR")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    instrument_name = args.instrument.upper()

    if instrument_name != "NIFTY":
        from config.instruments import get_instrument
        inst = get_instrument(instrument_name)
        if not inst:
            print(f"Unknown instrument: {instrument_name}")
            print("Available: NIFTY, GOLD_PETAL, CRUDEOILM, USDINR")
            return

        print(f"\nLoading {args.days} trading days of {inst.display_name} data...")
        df = _load_futures_data(instrument_name, args.days)
        if df.empty:
            return
        unique_days = sorted(set(df.index.date))
        print(f"Loaded {len(df)} candles across {len(unique_days)} days")
        lot_size = inst.lot_size

        print(f"\n  ── FULL PERIOD ({len(unique_days)} days) ──")
        print(f"  Instrument: {inst.display_name} ({inst.exchange})")
        print(f"  Lot size: {lot_size} | Asset type: {inst.asset_type}")
        results = run_compound_backtest(
            df, starting_capital=args.capital,
            lot_size=lot_size, deploy_pct=args.deploy_pct,
        )
        print_results(results)

        pd.DataFrame(results["trades"]).to_csv(
            settings.DATA_DIR / f"backtest_{instrument_name.lower()}_trades.csv", index=False)
        print(f"  Saved: data/backtest_{instrument_name.lower()}_trades.csv")
        return results

    print(f"\nLoading {args.days} trading days of Nifty data...")
    df = load_real_data(days=args.days)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} days")

    lot_size = settings.NIFTY_LOT_SIZE

    # ── Walk-forward split: 70% train / 30% out-of-sample ──
    split_idx = int(len(unique_days) * 0.70)
    train_days = set(unique_days[:split_idx])
    test_days = set(unique_days[split_idx:])
    df_train = df[df.index.map(lambda t: t.date() in train_days)]
    df_test = df[df.index.map(lambda t: t.date() in test_days)]

    print(f"\n{'=' * 65}")
    print("  WALK-FORWARD VALIDATION")
    print(f"  Train: {len(train_days)} days ({min(train_days)} → {max(train_days)})")
    print(f"  Test:  {len(test_days)} days ({min(test_days)} → {max(test_days)})")
    print(f"  Lot size: {lot_size} | Cost model: spread + STT + exchange + impact")
    print(f"{'=' * 65}")

    # ── IN-SAMPLE (train) ──
    print(f"\n  ── IN-SAMPLE ({len(train_days)} days) ──")
    results_train = run_compound_backtest(
        df_train, starting_capital=args.capital,
        lot_size=lot_size, deploy_pct=args.deploy_pct,
    )
    print_results(results_train)

    # ── OUT-OF-SAMPLE (test) ──
    print(f"\n  ── OUT-OF-SAMPLE ({len(test_days)} days) ──")
    results_test = run_compound_backtest(
        df_test, starting_capital=args.capital,
        lot_size=lot_size, deploy_pct=args.deploy_pct,
    )
    print_results(results_test)

    # ── Full period for reference ──
    print(f"\n  ── FULL PERIOD ({len(unique_days)} days) ──")
    results = run_compound_backtest(
        df, starting_capital=args.capital,
        lot_size=lot_size, deploy_pct=args.deploy_pct,
    )
    print_results(results)

    pd.DataFrame(results["trades"]).to_csv(
        settings.DATA_DIR / "backtest_trades.csv", index=False)
    pd.DataFrame(results["equity_curve"]).to_csv(
        settings.DATA_DIR / "equity_curve.csv", index=False)
    print("  Saved: data/backtest_trades.csv, data/equity_curve.csv")

    return results


if __name__ == "__main__":
    main()
