"""Parameter sensitivity analysis -- isolate settings.py impact on 100-day backtest."""

import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")

from backtest.run_backtest import load_real_data, run_compound_backtest
from config import settings


def run_test(label, df, starting_capital=10000):
    result = run_compound_backtest(
        df,
        starting_capital=starting_capital,
        use_adaptive=False,
        use_risk_gates=False,
    )
    td = result["trading_days"]
    cdgr = ((result["final_capital"] / starting_capital) ** (1 / max(td, 1)) - 1) * 100
    print(
        f"{label:45s} | {result['total_trades']:>4d} trades | "
        f"{result['win_rate']:>5.1f}% WR | Rs {result['total_pnl']:>10,.0f} P&L | "
        f"PF {result['profit_factor']:.2f} | DD {result['max_drawdown_pct']:.1f}% | "
        f"CDGR {cdgr:.2f}%"
    )
    return result


def main():
    df_100 = load_real_data(days=100)

    print("=" * 120)
    print("PARAMETER SENSITIVITY ANALYSIS -- 100-DAY BACKTEST")
    print("=" * 120)
    run_test("CURRENT (all new values)", df_100)

    # Test 1: Revert CAPITAL_PER_LOT to 6000
    old = settings.CAPITAL_PER_LOT
    settings.CAPITAL_PER_LOT = 6000
    run_test("Revert CAPITAL_PER_LOT to 6000", df_100)
    settings.CAPITAL_PER_LOT = old

    # Test 2: Revert DAILY_LOSS_LIMIT_PCT to 15
    old = settings.DAILY_LOSS_LIMIT_PCT
    settings.DAILY_LOSS_LIMIT_PCT = 15
    run_test("Revert DAILY_LOSS_LIMIT_PCT to 15", df_100)
    settings.DAILY_LOSS_LIMIT_PCT = old

    # Test 3: Revert PULLBACK_HOLD_CANDLES to 24
    old = settings.PULLBACK_HOLD_CANDLES
    settings.PULLBACK_HOLD_CANDLES = 24
    run_test("Revert PULLBACK_HOLD_CANDLES to 24", df_100)
    settings.PULLBACK_HOLD_CANDLES = old

    # Test 4: Try CAPITAL_PER_LOT = 8000 (middle ground)
    old = settings.CAPITAL_PER_LOT
    settings.CAPITAL_PER_LOT = 8000
    run_test("CAPITAL_PER_LOT = 8000 (middle ground)", df_100)
    settings.CAPITAL_PER_LOT = old

    # Test 5: Try CAPITAL_PER_LOT = 7000
    old = settings.CAPITAL_PER_LOT
    settings.CAPITAL_PER_LOT = 7000
    run_test("CAPITAL_PER_LOT = 7000", df_100)
    settings.CAPITAL_PER_LOT = old

    # Test 6: Revert CAPITAL_PER_LOT to 6000 AND DAILY_LOSS_LIMIT_PCT to 15
    old_cpl = settings.CAPITAL_PER_LOT
    old_dll = settings.DAILY_LOSS_LIMIT_PCT
    settings.CAPITAL_PER_LOT = 6000
    settings.DAILY_LOSS_LIMIT_PCT = 15
    run_test("Revert BOTH CPL=6k + DLL=15%", df_100)
    settings.CAPITAL_PER_LOT = old_cpl
    settings.DAILY_LOSS_LIMIT_PCT = old_dll

    # Test 7: Revert ALL params (CPL=6k, DLL=15, HOLD=24)
    old_cpl = settings.CAPITAL_PER_LOT
    old_dll = settings.DAILY_LOSS_LIMIT_PCT
    old_hold = settings.PULLBACK_HOLD_CANDLES
    settings.CAPITAL_PER_LOT = 6000
    settings.DAILY_LOSS_LIMIT_PCT = 15
    settings.PULLBACK_HOLD_CANDLES = 24
    all_reverted = run_test("Revert ALL params to original", df_100)
    settings.CAPITAL_PER_LOT = old_cpl
    settings.DAILY_LOSS_LIMIT_PCT = old_dll
    settings.PULLBACK_HOLD_CANDLES = old_hold

    print("\n--- Baseline target: 516 trades | 56.0% WR | Rs 33,20,000 P&L | CDGR 6.0% ---")

    # Jun 9-12 window: last 4 trading days in 30-day load
    print("\n" + "=" * 120)
    print("JUN 9-12 WINDOW (last 4 trading days)")
    print("=" * 120)

    df_30 = load_real_data(days=30)
    unique_days = sorted(set(df_30.index.date))
    last_4 = set(unique_days[-4:])
    df_jun = df_30[df_30.index.map(lambda t: t.date() in last_4)]
    print(f"Dates: {sorted(last_4)}")

    run_test("CURRENT (all new values)", df_jun)

    old_cpl = settings.CAPITAL_PER_LOT
    old_dll = settings.DAILY_LOSS_LIMIT_PCT
    old_hold = settings.PULLBACK_HOLD_CANDLES
    settings.CAPITAL_PER_LOT = 6000
    settings.DAILY_LOSS_LIMIT_PCT = 15
    settings.PULLBACK_HOLD_CANDLES = 24
    run_test("Revert ALL params to original", df_jun)
    settings.CAPITAL_PER_LOT = old_cpl
    settings.DAILY_LOSS_LIMIT_PCT = old_dll
    settings.PULLBACK_HOLD_CANDLES = old_hold

    # Also test CPL=6k alone (often closest single-param revert)
    old = settings.CAPITAL_PER_LOT
    settings.CAPITAL_PER_LOT = 6000
    run_test("Revert CAPITAL_PER_LOT to 6000", df_jun)
    settings.CAPITAL_PER_LOT = old

    print("\n--- Baseline target: 18 trades | 72.2% WR | Rs 10k -> Rs 36,386 ---")


if __name__ == "__main__":
    main()
