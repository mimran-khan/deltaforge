#!/bin/bash
# Wrapper: starts dashboard + keeps restarting `make trade` until a full trading day completes.
# On non-trading days (expiry, weekends, holidays) the process exits quickly;
# this wrapper sleeps and retries so it auto-starts on the next trading day.

cd "$(dirname "$0")"
export TZ=Asia/Kolkata

# ── Sync latest code from development repo ──
DEV_REPO="${DELTAFORGE_DEV_REPO:-}"
if [ -n "$DEV_REPO" ] && [ -d "$DEV_REPO" ] && ls "$DEV_REPO/main.py" > /dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Syncing code from $DEV_REPO ..."
    rsync -a --exclude='venv/' --exclude='data/' --exclude='logs/' --exclude='.git/' \
          --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
          "$DEV_REPO/" "$(pwd)/" 2>&1
    RSYNC_EXIT=$?
    if [ "$RSYNC_EXIT" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Code sync complete"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] WARNING: rsync failed (code $RSYNC_EXIT) -- TCC? Grant Full Disk Access to /bin/bash"
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Continuing with existing code in $(pwd)"
    fi
    # Also sync data CSV files (but not capital.json, trades.db, events, etc.)
    if [ -d "$DEV_REPO/data" ]; then
        rsync -a --include='*.csv' --include='*.json' --exclude='capital.json' \
              --exclude='trades.db' --exclude='events*.jsonl' --exclude='engine_state.json' \
              --exclude='paper_positions.json' --exclude='session.lock' \
              --exclude='*.tmp' --exclude='*.bak' --exclude='*.lock' \
              --include='*/' --exclude='*' \
              "$DEV_REPO/data/" "$(pwd)/data/" 2>&1 || true
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Data sync complete"
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] DELTAFORGE_DEV_REPO not set or not accessible -- using existing code in $(pwd)"
fi

# ── Data integrity check ──
if [ -f "data/nifty_5m_all_merged.csv" ]; then
    LINE_COUNT=$(wc -l < data/nifty_5m_all_merged.csv)
    if [ "$LINE_COUNT" -lt 1000 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] WARNING: nifty_5m_all_merged.csv has only $LINE_COUNT lines"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Data OK: nifty_5m_all_merged.csv has $LINE_COUNT lines"
    fi
fi

# ── Start dashboard in background ──
DASH_PID=""
start_dashboard() {
    # Dashboard is managed by LaunchAgent (always-on). Skip if already running.
    if lsof -i :8900 > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Dashboard already running on :8900 (managed by LaunchAgent)"
        DASH_PID=""
        return
    fi
    # Fallback: start if LaunchAgent didn't
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Dashboard not running -- starting as fallback..."
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Starting dashboard on http://localhost:8900 ..."
    source venv/bin/activate
    python -m dashboard.server >> logs/dashboard.log 2>&1 &
    DASH_PID=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Dashboard started (PID: $DASH_PID)"
}

stop_dashboard() {
    # Leave LaunchAgent-managed dashboard running (always-on)
    if lsof -i :8900 > /dev/null 2>&1 && [ -z "$DASH_PID" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Dashboard left running (managed by LaunchAgent)"
        return
    fi
    if [ -n "$DASH_PID" ] && kill -0 "$DASH_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Stopping dashboard (PID: $DASH_PID)"
        kill "$DASH_PID" 2>/dev/null
        wait "$DASH_PID" 2>/dev/null
    fi
}

MONITOR_PID=""
start_monitor() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Starting trade monitor..."
    source venv/bin/activate
    python -m automation.trade_monitor >> logs/monitor.log 2>&1 &
    MONITOR_PID=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Trade monitor started (PID: $MONITOR_PID)"
}

stop_monitor() {
    if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Stopping trade monitor (PID: $MONITOR_PID)"
        kill "$MONITOR_PID" 2>/dev/null
        wait "$MONITOR_PID" 2>/dev/null
    fi
    pkill -f "trade_monitor" 2>/dev/null
}

# Ensure dashboard and monitor are stopped on script exit
trap 'stop_dashboard; stop_monitor' EXIT

start_dashboard
start_monitor

# ── Check for existing session ──
check_session_lock() {
    LOCK_FILE="data/session.lock"
    if [ -f "$LOCK_FILE" ]; then
        LOCK_PID=$(python3 -c "import json; print(json.load(open('$LOCK_FILE')).get('pid', 0))" 2>/dev/null)
        if [ -n "$LOCK_PID" ] && [ "$LOCK_PID" != "0" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Trading session already running (PID: $LOCK_PID). Exiting."
            exit 0
        fi
    fi
}

check_session_lock

# ── Trading loop ──
while true; do
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Starting make trade..."
    echo "=========================================="
    make trade
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] make trade exited with code $EXIT_CODE"

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Clean exit -- done for today."
        break
    fi

    HOUR=$(date +%H)
    MIN=$(date +%M)
    if [ "$HOUR" -gt 15 ] || { [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 15 ]; }; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Post-market (after 15:15 IST) -- done for today."
        break
    fi

    # Restart dashboard if it died
    if [ -n "$DASH_PID" ] && ! kill -0 "$DASH_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Dashboard died -- restarting..."
        start_dashboard
    fi

    # Restart trade monitor if it died
    if [ -n "$MONITOR_PID" ] && ! kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Trade monitor died -- restarting..."
        start_monitor
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Sleeping 30 minutes before retry..."
    sleep 1800
done
