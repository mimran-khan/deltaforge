"""Comprehensive trade analysis: July 7 – September 10, 2026."""

import sqlite3
import math
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta

DB_PATH = "/Users/predatorroxy.macpro/TradingAgent/data/trades.db"
START_DATE = "2026-07-07"
END_DATE = "2026-09-10"
LOT_CAP = 6
MAX_LOSS = -5000.0

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute(
    "SELECT * FROM trades WHERE date >= ? AND date <= ? ORDER BY date, time",
    (START_DATE, END_DATE),
)
rows = [dict(r) for r in cur.fetchall()]
conn.close()

print(f"{'='*80}")
print(f"  TRADING SYSTEM ANALYSIS — {START_DATE} to {END_DATE}")
print(f"  Total trades in period: {len(rows)}")
print(f"{'='*80}\n")


def safe_div(a, b, default=0.0):
    return a / b if b else default


def capped_pnl(row):
    """PnL capped at LOT_CAP lots and MAX_LOSS per trade."""
    lots = row["lots"]
    pnl = row["pnl"]
    if lots > LOT_CAP:
        pnl = pnl * (LOT_CAP / lots)
    return max(pnl, MAX_LOSS)


def profit_factor(trades):
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    return safe_div(gross_profit, gross_loss, float("inf"))


def win_rate(trades):
    if not trades:
        return 0.0
    return 100.0 * sum(1 for t in trades if t["pnl"] > 0) / len(trades)


# ─────────────────────────────────────────────────────────────────────────────
# 1. WEEKLY PNL CURVE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  1. WEEKLY PnL CURVE")
print("=" * 80)

weeks = OrderedDict()
for t in rows:
    d = datetime.strptime(t["date"], "%Y-%m-%d")
    week_start = d - timedelta(days=d.weekday())
    wk = week_start.strftime("%Y-%m-%d")
    weeks.setdefault(wk, []).append(t)

cum = 0.0
print(f"{'Week Start':<14} {'Trades':>6} {'Win%':>7} {'Week PnL':>12} {'Cum PnL':>12}")
print("-" * 55)
for wk, trades in weeks.items():
    wpnl = sum(t["pnl"] for t in trades)
    cum += wpnl
    wr = win_rate(trades)
    print(f"{wk:<14} {len(trades):>6} {wr:>6.1f}% {wpnl:>+12.2f} {cum:>+12.2f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. DAILY PNL WITH MARKET REGIME
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  2. DAILY PnL WITH MARKET REGIME (Nifty close proxy = last entry_price)")
print("=" * 80)

days = OrderedDict()
for t in rows:
    days.setdefault(t["date"], []).append(t)

print(f"{'Date':<14} {'Trades':>6} {'PnL':>12} {'NiftyProxy':>12} {'Cum PnL':>12}")
print("-" * 60)
cum = 0.0
for day, trades in days.items():
    dpnl = sum(t["pnl"] for t in trades)
    cum += dpnl
    last_entry = trades[-1]["entry_price"]
    print(f"{day:<14} {len(trades):>6} {dpnl:>+12.2f} {last_entry:>12.2f} {cum:>+12.2f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. STRATEGY × DIRECTION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  3. STRATEGY × DIRECTION MATRIX (raw + capped at 6 lots / -5k max loss)")
print("=" * 80)

combos = defaultdict(list)
for t in rows:
    combos[(t["strategy"], t["direction"])].append(t)

print(f"\n{'Strat + Dir':<28} {'N':>5} {'Win%':>7} {'TotalPnL':>12} {'AvgPnL':>10} {'PF':>7}")
print("-" * 72)
for (strat, dirn), trades in sorted(combos.items()):
    n = len(trades)
    tot = sum(t["pnl"] for t in trades)
    avg = safe_div(tot, n)
    wr = win_rate(trades)
    pf = profit_factor(trades)
    label = f"{strat} {dirn}"
    print(f"{label:<28} {n:>5} {wr:>6.1f}% {tot:>+12.2f} {avg:>+10.2f} {pf:>7.2f}")

print(f"\n--- With 6-lot cap + 5k max-loss ---")
print(f"{'Strat + Dir':<28} {'N':>5} {'Win%':>7} {'TotalPnL':>12} {'AvgPnL':>10}")
print("-" * 65)
for (strat, dirn), trades in sorted(combos.items()):
    n = len(trades)
    capped = [capped_pnl(t) for t in trades]
    tot = sum(capped)
    avg = safe_div(tot, n)
    wr_c = 100.0 * sum(1 for p in capped if p > 0) / n if n else 0
    label = f"{strat} {dirn}"
    print(f"{label:<28} {n:>5} {wr_c:>6.1f}% {tot:>+12.2f} {avg:>+10.2f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. EXIT REASON ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  4. EXIT REASON ANALYSIS")
print("=" * 80)

exit_groups = defaultdict(list)
for t in rows:
    exit_groups[t["exit_reason"] or "UNKNOWN"].append(t)

print(f"{'Exit Reason':<16} {'Count':>6} {'TotalPnL':>12} {'AvgPnL':>10} {'AvgHoldBars':>12}")
print("-" * 60)
for reason, trades in sorted(exit_groups.items()):
    n = len(trades)
    tot = sum(t["pnl"] for t in trades)
    avg = safe_div(tot, n)
    avg_bars = safe_div(sum(t["hold_bars"] or 0 for t in trades), n)
    print(f"{reason:<16} {n:>6} {tot:>+12.2f} {avg:>+10.2f} {avg_bars:>12.1f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. TIME-OF-DAY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  5. TIME-OF-DAY ANALYSIS (by entry hour)")
print("=" * 80)

hours = defaultdict(list)
for t in rows:
    h = t["time"][:2] if t["time"] else "??"
    hours[h].append(t)

print(f"{'Hour':<6} {'Count':>6} {'Win%':>7} {'TotalPnL':>12} {'AvgPnL':>10}")
print("-" * 45)
for h in sorted(hours.keys()):
    trades = hours[h]
    n = len(trades)
    tot = sum(t["pnl"] for t in trades)
    avg = safe_div(tot, n)
    wr = win_rate(trades)
    print(f"{h}:xx  {n:>6} {wr:>6.1f}% {tot:>+12.2f} {avg:>+10.2f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 6. LOT SIZING IMPACT
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  6. LOT SIZING IMPACT (actual vs 6-lot cap + 5k max loss)")
print("=" * 80)

actual_total = sum(t["pnl"] for t in rows)
capped_total = sum(capped_pnl(t) for t in rows)
over_cap = [t for t in rows if t["lots"] > LOT_CAP]
under_maxloss = [t for t in rows if t["pnl"] < MAX_LOSS]

print(f"  Actual total PnL:          {actual_total:>+12.2f}")
print(f"  Capped total PnL:          {capped_total:>+12.2f}")
print(f"  Difference:                {capped_total - actual_total:>+12.2f}")
print(f"  Trades exceeding 6 lots:   {len(over_cap)}")
print(f"  Trades with loss > 5k:     {len(under_maxloss)}")
if over_cap:
    print(f"\n  Over-cap trades detail:")
    print(f"  {'Date':<12} {'Strategy':<14} {'Dir':<6} {'Lots':>5} {'ActualPnL':>12} {'CappedPnL':>12}")
    print(f"  {'-'*65}")
    for t in sorted(over_cap, key=lambda x: x["pnl"]):
        cp = capped_pnl(t)
        print(f"  {t['date']:<12} {t['strategy']:<14} {t['direction']:<6} {t['lots']:>5} {t['pnl']:>+12.2f} {cp:>+12.2f}")
if under_maxloss:
    print(f"\n  Large-loss trades (< -5000):")
    for t in sorted(under_maxloss, key=lambda x: x["pnl"]):
        cp = capped_pnl(t)
        print(f"  {t['date']} {t['strategy']:<14} {t['direction']:<6} lots={t['lots']} pnl={t['pnl']:+.2f} → capped={cp:+.2f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 7. CONSECUTIVE LOSS STREAKS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  7. CONSECUTIVE LOSS STREAKS")
print("=" * 80)

streaks = []
current_streak = []
for t in rows:
    if t["pnl"] < 0:
        current_streak.append(t)
    else:
        if current_streak:
            streaks.append(current_streak)
        current_streak = []
if current_streak:
    streaks.append(current_streak)

streaks.sort(key=lambda s: len(s), reverse=True)
print(f"\n  Top 10 losing streaks:")
print(f"  {'#':<4} {'Length':>6} {'Total PnL':>12} {'Start Date':<12} {'End Date':<12}")
print(f"  {'-'*50}")
for i, s in enumerate(streaks[:10], 1):
    stot = sum(t["pnl"] for t in s)
    print(f"  {i:<4} {len(s):>6} {stot:>+12.2f} {s[0]['date']:<12} {s[-1]['date']:<12}")

if streaks:
    longest = streaks[0]
    print(f"\n  Longest streak detail ({len(longest)} trades, {sum(t['pnl'] for t in longest):+.2f}):")
    for t in longest:
        print(f"    {t['date']} {t['time']} {t['strategy']:<14} {t['direction']:<6} pnl={t['pnl']:+.2f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 8. BEST AND WORST 10 TRADES
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  8. BEST AND WORST 10 TRADES")
print("=" * 80)

sorted_by_pnl = sorted(rows, key=lambda t: t["pnl"])

def print_trade_detail(trades, label):
    print(f"\n  {label}:")
    print(f"  {'#':<3} {'Date':<12} {'Time':<10} {'Strategy':<14} {'Dir':<6} {'Lots':>4} {'Entry':>10} {'Exit':>10} {'PnL':>12} {'Exit':>6} {'HoldB':>5}")
    print(f"  {'-'*97}")
    for i, t in enumerate(trades, 1):
        print(f"  {i:<3} {t['date']:<12} {t['time']:<10} {t['strategy']:<14} {t['direction']:<6} {t['lots']:>4} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} {t['pnl']:>+12.2f} {(t['exit_reason'] or '?'):<6} {(t['hold_bars'] or 0):>5}")

print_trade_detail(sorted_by_pnl[:10], "WORST 10 TRADES")
print_trade_detail(sorted_by_pnl[-10:][::-1], "BEST 10 TRADES")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 9. WIN RATE BY ADX BUCKET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  9. WIN RATE BY ADX BUCKET")
print("=" * 80)

adx_buckets = {"0-15": [], "15-25": [], "25-35": [], "35-50": [], "50+": [], "N/A": []}
for t in rows:
    adx = t["adx"]
    if adx is None:
        adx_buckets["N/A"].append(t)
    elif adx < 15:
        adx_buckets["0-15"].append(t)
    elif adx < 25:
        adx_buckets["15-25"].append(t)
    elif adx < 35:
        adx_buckets["25-35"].append(t)
    elif adx < 50:
        adx_buckets["35-50"].append(t)
    else:
        adx_buckets["50+"].append(t)

print(f"\n{'ADX Range':<12} {'Count':>6} {'Win%':>7} {'TotalPnL':>12} {'AvgPnL':>10} {'AvgADX':>8}")
print("-" * 58)
for bucket in ["0-15", "15-25", "25-35", "35-50", "50+", "N/A"]:
    trades = adx_buckets[bucket]
    if not trades:
        continue
    n = len(trades)
    tot = sum(t["pnl"] for t in trades)
    avg = safe_div(tot, n)
    wr = win_rate(trades)
    avg_adx = safe_div(sum(t["adx"] or 0 for t in trades), sum(1 for t in trades if t["adx"] is not None))
    print(f"{bucket:<12} {n:>6} {wr:>6.1f}% {tot:>+12.2f} {avg:>+10.2f} {avg_adx:>8.1f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 10. DIRECTION BIAS BY MONTH
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  10. DIRECTION BIAS BY MONTH")
print("=" * 80)

months = defaultdict(lambda: defaultdict(list))
for t in rows:
    m = t["date"][:7]
    months[m][t["direction"]].append(t)

print(f"\n{'Month':<10} {'Dir':<8} {'Count':>6} {'Win%':>7} {'TotalPnL':>12} {'AvgPnL':>10} {'PF':>7}")
print("-" * 64)
for m in sorted(months.keys()):
    for dirn in ["LONG", "SHORT"]:
        trades = months[m].get(dirn, [])
        if not trades:
            continue
        n = len(trades)
        tot = sum(t["pnl"] for t in trades)
        avg = safe_div(tot, n)
        wr = win_rate(trades)
        pf = profit_factor(trades)
        print(f"{m:<10} {dirn:<8} {n:>6} {wr:>6.1f}% {tot:>+12.2f} {avg:>+10.2f} {pf:>7.2f}")
    all_trades = [t for d in months[m].values() for t in d]
    tot_all = sum(t["pnl"] for t in all_trades)
    print(f"{m:<10} {'ALL':<8} {len(all_trades):>6} {win_rate(all_trades):>6.1f}% {tot_all:>+12.2f} {safe_div(tot_all, len(all_trades)):>+10.2f} {profit_factor(all_trades):>7.2f}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  SUMMARY STATISTICS")
print("=" * 80)
total_pnl = sum(t["pnl"] for t in rows)
total_wins = sum(1 for t in rows if t["pnl"] > 0)
total_losses = sum(1 for t in rows if t["pnl"] <= 0)
avg_win = safe_div(sum(t["pnl"] for t in rows if t["pnl"] > 0), total_wins)
avg_loss = safe_div(sum(t["pnl"] for t in rows if t["pnl"] <= 0), total_losses)
max_dd_trade = min(rows, key=lambda t: t["pnl"])
best_trade = max(rows, key=lambda t: t["pnl"])

# Calculate max drawdown on cumulative PnL
cum_pnls = []
c = 0
for t in rows:
    c += t["pnl"]
    cum_pnls.append(c)
peak = cum_pnls[0]
max_dd = 0
for cp in cum_pnls:
    if cp > peak:
        peak = cp
    dd = peak - cp
    if dd > max_dd:
        max_dd = dd

print(f"  Total trades:       {len(rows)}")
print(f"  Wins / Losses:      {total_wins} / {total_losses}")
print(f"  Win rate:           {win_rate(rows):.1f}%")
print(f"  Total PnL:          {total_pnl:+.2f}")
print(f"  Avg win:            {avg_win:+.2f}")
print(f"  Avg loss:           {avg_loss:+.2f}")
print(f"  Profit factor:      {profit_factor(rows):.2f}")
print(f"  Max drawdown (cum): {max_dd:+.2f}")
print(f"  Best trade:         {best_trade['pnl']:+.2f} ({best_trade['date']} {best_trade['strategy']} {best_trade['direction']})")
print(f"  Worst trade:        {max_dd_trade['pnl']:+.2f} ({max_dd_trade['date']} {max_dd_trade['strategy']} {max_dd_trade['direction']})")
print(f"  Capped PnL (6L/5k): {capped_total:+.2f}")
print(f"{'='*80}")
