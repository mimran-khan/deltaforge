import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level='ERROR')

from backtest.run_backtest import load_real_data, run_compound_backtest
from collections import defaultdict

def report(label, result, start_cap=10000):
    td = result['trading_days']
    cdgr = ((result['final_capital'] / start_cap) ** (1/max(td,1)) - 1) * 100 if td > 0 else 0
    win_loss = abs(result['avg_win'] / result['avg_loss']) if result['avg_loss'] != 0 else 0
    
    print()
    print("=" * 65)
    print(f"     {label}")
    print(f"     (Rs {start_cap:,} start, compounding, {td} trading days)")
    print("=" * 65)
    print(f"  Starting Capital      : Rs {start_cap:>12,}")
    print(f"  Final Capital         : Rs {result['final_capital']:>12,.0f}")
    print(f"  Total P&L             : Rs {result['total_pnl']:>+12,.0f}")
    print(f"  Return                : {result['return_pct']:.1f}%")
    print(f"  CDGR                  : {cdgr:.1f}%")
    print(f"  Avg Daily Return      : {result['avg_daily_return_pct']:.1f}%")
    print("-" * 65)
    print(f"  Total Trades          : {result['total_trades']}")
    print(f"  Winners / Losers      : {result['wins']} / {result['losses']}")
    print(f"  Win Rate              : {result['win_rate']:.1f}%")
    print(f"  Profit Factor         : {result['profit_factor']:.2f}")
    print(f"  Avg Win               : Rs {result['avg_win']:>+,.0f}")
    print(f"  Avg Loss              : Rs {result['avg_loss']:>+,.0f}")
    print(f"  Win/Loss Ratio        : {win_loss:.2f}x")
    print(f"  Avg Trades/Day        : {result['total_trades'] / max(td,1):.1f}")
    print("-" * 65)
    print(f"  Max Drawdown          : {result['max_drawdown_pct']:.1f}%")
    print(f"  Profitable Days       : {result['profitable_days']}/{td} ({result['profitable_days']/max(td,1)*100:.0f}%)")
    print(f"  Loss Days             : {result['loss_days']}")
    print("=" * 65)
    
    # Strategy breakdown
    trades = result['trades']
    strat_stats = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0, "best": -999999})
    for t in trades:
        s = t['strategy']
        strat_stats[s]['count'] += 1
        if t['pnl'] > 0:
            strat_stats[s]['wins'] += 1
        strat_stats[s]['pnl'] += t['pnl']
        strat_stats[s]['best'] = max(strat_stats[s]['best'], t['pnl'])
    
    print(f"\n  {'Strategy':<18} {'Trades':>7} {'WR':>7} {'P&L':>14} {'Best':>10}")
    print("-" * 65)
    for s in sorted(strat_stats, key=lambda x: strat_stats[x]['pnl'], reverse=True):
        d = strat_stats[s]
        wr = d['wins'] / d['count'] * 100 if d['count'] else 0
        print(f"  {s:<18} {d['count']:>7} {wr:>6.1f}% Rs {d['pnl']:>+10,.0f} Rs {d['best']:>+8,.0f}")
    
    # Exit breakdown
    exit_stats = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0})
    for t in trades:
        reason = t.get('reason', 'unknown')
        if 'trail' in reason.lower(): exit_type = 'TRAIL'
        elif 'target' in reason.lower() or 'tgt' in reason.lower(): exit_type = 'TGT'
        elif 'sl' in reason.lower() or 'stop' in reason.lower(): exit_type = 'SL'
        elif 'time' in reason.lower() or 'eod' in reason.lower() or 'hold' in reason.lower(): exit_type = 'TIME/EOD'
        else: exit_type = reason[:15]
        exit_stats[exit_type]['count'] += 1
        if t['pnl'] > 0: exit_stats[exit_type]['wins'] += 1
        exit_stats[exit_type]['pnl'] += t['pnl']
    
    print(f"\n  {'Exit Type':<18} {'Count':>7} {'WR':>7} {'Total P&L':>14} {'Avg P&L':>10}")
    print("-" * 65)
    for e in sorted(exit_stats, key=lambda x: exit_stats[x]['pnl'], reverse=True):
        d = exit_stats[e]
        wr = d['wins'] / d['count'] * 100 if d['count'] else 0
        avg = d['pnl'] / d['count'] if d['count'] else 0
        print(f"  {e:<18} {d['count']:>7} {wr:>6.1f}% Rs {d['pnl']:>+10,.0f} Rs {avg:>+8,.0f}")
    
    # Per-trade detail for short periods
    if td <= 10:
        print(f"\n  {'Time':<20} {'Strategy':<15} {'Exit':>8} {'P&L':>12} {'Capital':>12}")
        print("-" * 65)
        for t in trades:
            reason = t.get('reason', '?')
            if 'trail' in reason.lower(): r = 'TRAIL'
            elif 'target' in reason.lower() or 'tgt' in reason.lower(): r = 'TGT'
            elif 'sl' in reason.lower() or 'stop' in reason.lower(): r = 'SL'
            elif 'time' in reason.lower() or 'eod' in reason.lower(): r = 'TIME'
            else: r = reason[:8]
            print(f"  {str(t['entry_time']):<20} {t['strategy']:<15} {r:>8} Rs {t['pnl']:>+9,.0f} Rs {t.get('capital_after',0):>9,.0f}")
    
    # Daily equity for short periods
    if td <= 30:
        ec = result['equity_curve']
        print(f"\n  DAILY P&L")
        print("-" * 65)
        for e in ec:
            if e['daily_pnl'] != 0 or True:
                bar_len = max(1, int(abs(e['daily_pnl']) / max(1, max(abs(x['daily_pnl']) for x in ec)) * 20))
                bar = "█" * bar_len
                sign = "+" if e['daily_pnl'] >= 0 else "-"
                print(f"  {e['date']}  Rs {e['daily_pnl']:>+10,.0f}  Cap Rs {e['capital']:>10,.0f}  {sign}{bar}")
    
    return result

# Load all data
df_all = load_real_data(days=100)
all_days = sorted(set(df_all.index.date))

# ═══════════════════════════════════════════════════════════════
# 1. LAST WEEK (last 4 trading days: Jun 9-12)
# ═══════════════════════════════════════════════════════════════
last4 = set(all_days[-4:])
df_week = df_all[df_all.index.map(lambda t: t.date() in last4)]
r1 = run_compound_backtest(df_week, starting_capital=10000, use_adaptive=False, use_risk_gates=False)
report(f"LAST WEEK ({min(last4)} → {max(last4)})", r1)

# ═══════════════════════════════════════════════════════════════
# 2. THIS MONTH (last ~22 trading days)
# ═══════════════════════════════════════════════════════════════
# Find all days in June 2026 (or last ~22 trading days)
import datetime
month_days = [d for d in all_days if d.month == 6 and d.year == 2026]
if len(month_days) < 5:
    # fallback to last 22 days
    month_days = all_days[-22:]
month_set = set(month_days)
df_month = df_all[df_all.index.map(lambda t: t.date() in month_set)]
r2 = run_compound_backtest(df_month, starting_capital=10000, use_adaptive=False, use_risk_gates=False)
report(f"THIS MONTH ({min(month_days)} → {max(month_days)})", r2)

# ═══════════════════════════════════════════════════════════════
# 3. LAST 100 DAYS
# ═══════════════════════════════════════════════════════════════
r3 = run_compound_backtest(df_all, starting_capital=10000, use_adaptive=False, use_risk_gates=False)
report("LAST 100 DAYS", r3)

# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
print()
print()
print("=" * 75)
print("     SUMMARY -- ALL THREE PERIODS (Rs 10,000 start each)")
print("=" * 75)
td1 = r1['trading_days']
td2 = r2['trading_days']
td3 = r3['trading_days']
cdgr1 = ((r1['final_capital'] / 10000) ** (1/max(td1,1)) - 1) * 100
cdgr2 = ((r2['final_capital'] / 10000) ** (1/max(td2,1)) - 1) * 100
cdgr3 = ((r3['final_capital'] / 10000) ** (1/max(td3,1)) - 1) * 100
wlr1 = abs(r1['avg_win'] / r1['avg_loss']) if r1['avg_loss'] != 0 else 0
wlr2 = abs(r2['avg_win'] / r2['avg_loss']) if r2['avg_loss'] != 0 else 0
wlr3 = abs(r3['avg_win'] / r3['avg_loss']) if r3['avg_loss'] != 0 else 0

fmt = "  {:<22} {:>12} {:>12} {:>12}"
print(fmt.format("Metric", "Last Week", "This Month", "100 Days"))
print("-" * 75)
print(fmt.format("Trades", str(r1['total_trades']), str(r2['total_trades']), str(r3['total_trades'])))
print(fmt.format("Win Rate", f"{r1['win_rate']:.1f}%", f"{r2['win_rate']:.1f}%", f"{r3['win_rate']:.1f}%"))
print(fmt.format("P&L", f"Rs {r1['total_pnl']:+,.0f}", f"Rs {r2['total_pnl']:+,.0f}", f"Rs {r3['total_pnl']:+,.0f}"))
print(fmt.format("Final Capital", f"Rs {r1['final_capital']:,.0f}", f"Rs {r2['final_capital']:,.0f}", f"Rs {r3['final_capital']:,.0f}"))
print(fmt.format("Profit Factor", f"{r1['profit_factor']:.2f}", f"{r2['profit_factor']:.2f}", f"{r3['profit_factor']:.2f}"))
print(fmt.format("Avg Win", f"Rs {r1['avg_win']:+,.0f}", f"Rs {r2['avg_win']:+,.0f}", f"Rs {r3['avg_win']:+,.0f}"))
print(fmt.format("Avg Loss", f"Rs {r1['avg_loss']:+,.0f}", f"Rs {r2['avg_loss']:+,.0f}", f"Rs {r3['avg_loss']:+,.0f}"))
print(fmt.format("Win/Loss Ratio", f"{wlr1:.2f}x", f"{wlr2:.2f}x", f"{wlr3:.2f}x"))
print(fmt.format("Winning Days", f"{r1['profitable_days']}/{td1}", f"{r2['profitable_days']}/{td2}", f"{r3['profitable_days']}/{td3}"))
print(fmt.format("Avg Trades/Day", f"{r1['total_trades']/max(td1,1):.1f}", f"{r2['total_trades']/max(td2,1):.1f}", f"{r3['total_trades']/max(td3,1):.1f}"))
print(fmt.format("CDGR", f"{cdgr1:.1f}%", f"{cdgr2:.1f}%", f"{cdgr3:.1f}%"))
print(fmt.format("Max Drawdown", f"{r1['max_drawdown_pct']:.1f}%", f"{r2['max_drawdown_pct']:.1f}%", f"{r3['max_drawdown_pct']:.1f}%"))
print("=" * 75)
