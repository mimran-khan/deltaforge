"""One-shot capital correction. Run AFTER engine shutdown.
Reads trades.db to compute correct capital from sum(pnl) and writes capital.json.
Safe to run multiple times (idempotent).
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
DATA = Path(__file__).resolve().parent.parent / "data"
DB = DATA / "trades.db"
CAP_FILE = DATA / "capital.json"

INITIAL_CAPITAL = 10_000.0
INJECTIONS = [{"date": "2026-06-22", "amount": 35000.0, "note": "Manual injection, total deployed now 45k"}]
TOTAL_DEPLOYED = INITIAL_CAPITAL + sum(i["amount"] for i in INJECTIONS)

conn = sqlite3.connect(str(DB))
total_pnl = conn.execute(
    "SELECT COALESCE(sum(pnl), 0) FROM trades WHERE instrument = 'NIFTY'"
).fetchone()[0]
conn.close()

capital = TOTAL_DEPLOYED + total_pnl

data = json.load(open(CAP_FILE)) if CAP_FILE.exists() else {}
data.update({
    "current_capital": round(capital, 2),
    "day_start_capital": round(capital, 2),
    "total_pnl": round(capital - TOTAL_DEPLOYED, 2),
    "peak_capital": round(max(capital, data.get("peak_capital", capital)), 2),
    "initial_capital": INITIAL_CAPITAL,
    "capital_injections": INJECTIONS,
    "weekly_pnl": round(capital - data.get("week_start_capital", capital), 2),
    "daily_pnl": 0.0,
    "trades_today": 0,
    "wins_today": 0,
    "losses_today": 0,
    "consecutive_losses": 0,
    "last_updated": datetime.now(IST).isoformat(),
})

with open(CAP_FILE, "w") as f:
    json.dump(data, f, indent=2)

print(f"Capital corrected: Rs {capital:,.2f}")
print(f"Total deployed: Rs {TOTAL_DEPLOYED:,.0f} | Total PnL: Rs {capital - TOTAL_DEPLOYED:,.2f}")
print(f"Saved to {CAP_FILE}")
