#!/bin/bash
# Install the weekday 8:15 AM wake LaunchAgent (keeps Mac awake for trading).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Production runtime used by LaunchAgent; prefer it when present.
if [ -d "$HOME/TradingAgent/main.py" ]; then
    ROOT="$HOME/TradingAgent"
fi
PLIST_SRC="$ROOT/automation/com.deltaforge.wake.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.deltaforge.wake.plist"

sed "s|__PROJECT_ROOT__|$ROOT|g" "$PLIST_SRC" > "$PLIST_DST"
launchctl bootout "gui/$(id -u)/com.deltaforge.wake" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/com.deltaforge.wake" 2>/dev/null || true

echo "Installed com.deltaforge.wake — Mac stays awake 8:15–15:15 IST on weekdays."
echo "Logs: $ROOT/logs/wake_stdout.log"
