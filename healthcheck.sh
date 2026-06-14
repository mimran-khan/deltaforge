#!/bin/bash
# DeltaForge Health Check & Fallback Starter
# Runs every 15 minutes via crontab. If the main trading process
# is not running during market hours, it restarts it.

export TZ=Asia/Kolkata
cd "$(dirname "$0")"

LOG="logs/healthcheck.log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $1" >> "$LOG"
}

# Only check Mon-Fri
DOW=$(date +%u)
if [ "$DOW" -gt 5 ]; then
    exit 0
fi

# Only check during market-relevant hours (08:45 - 15:30 IST)
HOUR=$(date +%H)
MIN=$(date +%M)
HHMM=$((10#$HOUR * 100 + 10#$MIN))

if [ "$HHMM" -lt 845 ] || [ "$HHMM" -gt 1530 ]; then
    exit 0
fi

# Check if holidays
TODAY=$(date +%Y-%m-%d)
if grep -q "\"$TODAY\"" config/holidays.json 2>/dev/null; then
    log "Holiday ($TODAY) -- skip"
    exit 0
fi

# Ensure dashboard API is up during market hours
if ! lsof -i :8900 > /dev/null 2>&1; then
    log "Dashboard not running on :8900 -- starting..."
    source venv/bin/activate
    nohup python -m dashboard.server >> logs/dashboard.log 2>&1 &
    sleep 2
    if lsof -i :8900 > /dev/null 2>&1; then
        log "Dashboard started (PID: $(lsof -t -i :8900 2>/dev/null | head -1))"
    else
        log "WARNING -- Dashboard failed to start"
    fi
fi

# Check if trading process is running
if pgrep -f "automation.daily_scheduler|df trade|run_monitored" > /dev/null 2>&1; then
    log "OK -- trading process running (PID: $(pgrep -f 'automation.daily_scheduler|df trade|run_monitored' | head -1))"
    exit 0
fi

# Not running! Attempt restart
log "ALERT -- No trading process found. Attempting fallback start..."

# Check if HALT flag exists (don't restart if halted intentionally)
DATA_DIR="data"
# Auto-clear stale HALT from previous days
if [ -f "$DATA_DIR/HALT" ]; then
    HALT_DATE=$(python3 -c "import json; print(json.load(open('$DATA_DIR/HALT'))['halted_at'][:10])" 2>/dev/null)
    TODAY=$(date +%Y-%m-%d)
    if [ "$HALT_DATE" != "$TODAY" ]; then
        log "Clearing stale HALT from $HALT_DATE (today is $TODAY)"
        rm "$DATA_DIR/HALT"
        source venv/bin/activate 2>/dev/null
        python3 -c "
import sys
sys.path.insert(0, '.')
from config import settings
_method = getattr(settings, 'ALERT_METHOD', 'slack')
if _method == 'imessage':
    from alerts.imessage_bot import send_system_alert
elif _method == 'slack':
    from alerts.slack_bot import send_system_alert
else:
    from alerts.telegram_bot import send_system_alert
send_system_alert('HALT Auto-Cleared', 'Stale HALT from ${HALT_DATE} cleared on ${TODAY}')
" 2>/dev/null || true
    fi
fi
if [ -f "$DATA_DIR/HALT" ]; then
    log "HALT flag exists -- not restarting (manual intervention needed)"
    exit 1
fi

# Check if session lock exists (another instance might be starting)
if [ -f "data/session.lock" ]; then
    LOCK_PID=$(python3 -c "import json; print(json.load(open('data/session.lock')).get('pid', 0))" 2>/dev/null)
    if [ -n "$LOCK_PID" ] && [ "$LOCK_PID" != "0" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        log "Session already running (PID: $LOCK_PID) -- waiting"
        exit 0
    fi
    log "Stale session lock (PID $LOCK_PID dead) -- removing"
    rm -f data/session.lock
fi

# Start the trading process in background
log "Starting run_monitored.sh as fallback..."
nohup ./run_monitored.sh >> logs/fallback_stdout.log 2>> logs/fallback_stderr.log &
FALLBACK_PID=$!
log "Fallback started (PID: $FALLBACK_PID)"

# Verify it actually started
sleep 5
if kill -0 "$FALLBACK_PID" 2>/dev/null; then
    log "Fallback process confirmed running (PID: $FALLBACK_PID)"
else
    log "CRITICAL -- Fallback process died immediately. Manual intervention required."
    exit 2
fi
