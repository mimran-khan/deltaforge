#!/bin/bash
# Wrapper: starts dashboard + keeps restarting `make trade` until a full trading day completes.
# On non-trading days (expiry, weekends, holidays) the process exits quickly;
# this wrapper sleeps and retries so it auto-starts on the next trading day.

cd "$(dirname "$0")"
export TZ=Asia/Kolkata

# TLS for curl/pip/tools invoked from bash (Python uses certifi via config/settings.py).
if [ -x "./venv/bin/python" ]; then
    _CA="$(./venv/bin/python -c "import certifi; print(certifi.where())" 2>/dev/null || true)"
    if [ -n "$_CA" ]; then
        export SSL_CERT_FILE="$_CA"
        export REQUESTS_CA_BUNDLE="$_CA"
    fi
fi

# ── Sync latest code from development repo ──
DEV_REPO="${DELTAFORGE_DEV_REPO:-$HOME/Documents/TradingAgent}"
if [ -d "$DEV_REPO" ] && ls "$DEV_REPO/main.py" > /dev/null 2>&1; then
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
    # IMPORTANT: excludes must come BEFORE includes — rsync uses first-match-wins
    if [ -d "$DEV_REPO/data" ]; then
        rsync -a \
              --exclude='capital.json' --exclude='capital.json.bak' \
              --exclude='trades.db' --exclude='events*.jsonl' --exclude='engine_state.json' \
              --exclude='paper_positions.json' --exclude='session.lock' \
              --exclude='multi_asset_capital.json' --exclude='HALT' \
              --exclude='*.tmp' --exclude='*.bak' --exclude='*.lock' \
              --include='*.csv' --include='*.json' \
              --include='*/' --exclude='*' \
              "$DEV_REPO/data/" "$(pwd)/data/" 2>&1 || true
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Data sync complete"
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] DELTAFORGE_DEV_REPO not set or not accessible -- using existing code in $(pwd)"
fi

# ── Import smoke test (catches broken imports before market open) ──
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Running import smoke test..."
source venv/bin/activate
if python -c "from automation.daily_scheduler import DailyScheduler; from alerts.command_listener import CommandPoller; print('OK')" 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Import smoke test PASSED"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] CRITICAL: Import smoke test FAILED -- code is broken!"
    python -c "from alerts import send_alert; send_alert('CRITICAL: DeltaForge import smoke test FAILED after code sync. Engine cannot start. Check logs immediately.')" 2>/dev/null || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Alert sent. Waiting 5 minutes before retrying with existing code..."
    sleep 300
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
        LOCK_DATE=$(python3 -c "import json; print(json.load(open('$LOCK_FILE')).get('started_at','')[:10])" 2>/dev/null)
        TODAY=$(date +%Y-%m-%d)
        if [ -n "$LOCK_PID" ] && [ "$LOCK_PID" != "0" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
            if [ "$LOCK_DATE" != "$TODAY" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Stale engine from $LOCK_DATE still alive (PID $LOCK_PID) -- killing it"
                kill "$LOCK_PID" 2>/dev/null
                sleep 5
                kill -0 "$LOCK_PID" 2>/dev/null && kill -9 "$LOCK_PID" 2>/dev/null
                sleep 2
                rm -f "$LOCK_FILE"
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Trading session already running (PID: $LOCK_PID). Exiting."
                exit 0
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Removing stale session lock (PID $LOCK_PID dead)"
            rm -f "$LOCK_FILE"
        fi
    fi
    rm -f "data/poller.lock"
}

check_session_lock

# ── Passive network check (20s max) — never block hours; never roam/power-cycle Wi-Fi ──
if ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Network OK"
elif [ -x "./scripts/ensure_network.sh" ]; then
  export DELTAFORGE_NETWORK_WAIT_SEC="${DELTAFORGE_NETWORK_WAIT_SEC:-20}"
  export DELTAFORGE_WIFI_SSID="${DELTAFORGE_WIFI_SSID:-predatorroxy}"
  ./scripts/ensure_network.sh || echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] WARNING: no internet — starting trade anyway"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] WARNING: no internet — starting trade anyway"
fi

# ── Trading loop ──
DF_BIN="$(pwd)/venv/bin/df"
if [ ! -x "$DF_BIN" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ERROR: $DF_BIN not found -- run: make install"
    exit 1
fi

CONSECUTIVE_FAST_CRASHES=0

while true; do
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Starting df trade..."
    echo "=========================================="
    source venv/bin/activate
    START_TS=$(date +%s)
    "$DF_BIN" trade
    EXIT_CODE=$?
    END_TS=$(date +%s)
    RUNTIME=$((END_TS - START_TS))
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] df trade exited with code $EXIT_CODE (ran ${RUNTIME}s)"

    HOUR=$(date +%H)
    MIN=$(date +%M)

    # MCX runs until 23:30, so "done for today" is after 23:35 or before 09:00.
    if [ "$EXIT_CODE" -eq 0 ]; then
        CONSECUTIVE_FAST_CRASHES=0
        if { [ "$HOUR" -ge 23 ] && [ "$MIN" -ge 35 ]; } || [ "$HOUR" -lt 9 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Clean exit (all markets closed) -- done for today."
            break
        elif [ "$HOUR" -eq 8 ] && [ "$MIN" -le 30 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Clean exit (early morning, likely holiday/weekend) -- done for today."
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] WARNING: exit 0 during market hours ($HOUR:$MIN) -- will retry."
        fi
    fi

    # Hard cutoff: don't retry after midnight
    if [ "$HOUR" -eq 0 ] || { [ "$HOUR" -ge 23 ] && [ "$MIN" -ge 45 ]; }; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] All markets closed (after 23:45 IST) -- done for today."
        break
    fi

    # Fast crash detection: if df trade dies within 15 seconds, it's a startup failure
    if [ "$RUNTIME" -lt 15 ] && [ "$EXIT_CODE" -ne 0 ]; then
        CONSECUTIVE_FAST_CRASHES=$((CONSECUTIVE_FAST_CRASHES + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] FAST CRASH #${CONSECUTIVE_FAST_CRASHES} (died in ${RUNTIME}s)"

        if [ "$CONSECUTIVE_FAST_CRASHES" -eq 1 ]; then
            python -c "from alerts import send_alert; send_alert('WARNING: DeltaForge crashed on startup (exit code $EXIT_CODE in ${RUNTIME}s). Retrying in 2 minutes.')" 2>/dev/null || true
        fi

        if [ "$CONSECUTIVE_FAST_CRASHES" -ge 3 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] CRITICAL: 3 consecutive startup crashes -- sending alert"
            python -c "from alerts import send_alert; send_alert('CRITICAL: DeltaForge has crashed $CONSECUTIVE_FAST_CRASHES times in a row on startup. Code is likely broken. Needs manual intervention.')" 2>/dev/null || true
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Fast retry in 2 minutes..."
        sleep 120
        continue
    else
        CONSECUTIVE_FAST_CRASHES=0
    fi

    # Normal crash during runtime -- send alert
    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Engine crashed after ${RUNTIME}s (code $EXIT_CODE)"
        python -c "from alerts import send_alert; send_alert('DeltaForge crashed after ${RUNTIME}s (exit code $EXIT_CODE). Restarting in 30 minutes.')" 2>/dev/null || true
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

# Post-session: run one-shot capital correction if script exists
if [ -f "scripts/fix_capital_once.py" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Running post-session capital correction..."
    python scripts/fix_capital_once.py 2>&1
fi
