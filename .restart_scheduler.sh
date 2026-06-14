#!/bin/bash
cd "$(dirname "$0")"
pkill -9 -f "daily_scheduler" 2>/dev/null || true
pkill -9 -f "run_monitored" 2>/dev/null || true
pkill -f "dashboard.server" 2>/dev/null || true
sleep 2
rm -f data/session.lock data/HALT
mkdir -p logs
nohup venv/bin/python -m dashboard.server >> logs/dashboard.log 2>&1 &
echo "Dashboard started on http://localhost:${DASHBOARD_PORT:-8900} (PID: $!)"
screen -dmS deltaforge bash -c "cd $(pwd) && venv/bin/python -m automation.daily_scheduler 2>&1 | tee logs/scheduler.log"
sleep 12
venv/bin/python -c "
import sys, json; sys.path.insert(0, '.')
with open('data/capital.json') as f:
    c = json.load(f)
print(f'Capital: Rs {c[\"current_capital\"]:,.2f}')
print(f'PnL:     Rs {c[\"daily_pnl\"]:,.2f}')
print(f'Trades:  {c[\"trades_today\"]}')
" 2>/dev/null
pgrep -f daily_scheduler > /dev/null && echo "RUNNING" || echo "FAILED"
if [ -f data/HALT ]; then echo "HALT: YES"; cat data/HALT; else echo "HALT: No"; fi
