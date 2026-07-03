#!/bin/bash
# Passive network check — does NOT power-cycle Wi-Fi or roam across saved networks.
# Aggressive Wi-Fi management was disconnecting predatorroxy and blocking trading for hours.
#
# Behaviour:
#   1. Internet OK          → exit 0 immediately (touch nothing)
#   2. On home SSID       → wait briefly for DNS, never switch away
#   3. Truly offline      → one gentle join attempt to primary SSID (interactive/TCC only)
#   4. Cron/launchd       → check only, never call networksetup (AuthorizationCreate -60008)
#
# Env:
#   DELTAFORGE_WIFI_SSID          — home SSID (default: predatorroxy)
#   DELTAFORGE_NETWORK_WAIT_SEC   — max wait when on home SSID (default: 20)
#   DELTAFORGE_NETWORK_LOG        — log file (default: logs/network.log)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export TZ="${TZ:-Asia/Kolkata}"

LOG="${DELTAFORGE_NETWORK_LOG:-logs/network.log}"
LOCK_DIR="${DELTAFORGE_NETWORK_LOCK:-/tmp/deltaforge-network.lock}"
HOME_SSID="${DELTAFORGE_WIFI_SSID:-predatorroxy}"
MAX_WAIT="${DELTAFORGE_NETWORK_WAIT_SEC:-20}"

mkdir -p "$(dirname "$LOG")"

log() {
    local line="[$(date '+%Y-%m-%d %H:%M:%S %Z')] [network] $*"
    echo "$line"
    echo "$line" >> "$LOG"
}

detect_wifi_device() {
    networksetup -listallhardwareports 2>/dev/null | awk '/Hardware Port: Wi-Fi/{getline; print $2; exit}'
}

current_ssid() {
    local dev="$1"
    local info
    info="$(networksetup -getairportnetwork "$dev" 2>/dev/null || true)"
    if echo "$info" | grep -qi 'not associated'; then
        return 1
    fi
    echo "$info" | sed -n 's/^Current Wi-Fi Network: //p'
}

has_internet() {
    ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1
}

can_manage_wifi() {
    # networksetup needs GUI/TCC auth; cron/launchd often get AuthorizationCreate -60008
    if [ -n "${DELTAFORGE_SKIP_WIFI_MANAGE:-}" ]; then
        return 1
    fi
    if [ -z "${DISPLAY:-}" ] && [ -z "${SSH_TTY:-}" ]; then
        # Background agent — don't touch Wi-Fi radio
        return 1
    fi
    networksetup -getairportpower "$(detect_wifi_device)" >/dev/null 2>&1
}

try_join_home_once() {
    local dev="$1"
    local pw

    if ! can_manage_wifi; then
        log "Skipping Wi-Fi join (no TCC auth in this context)"
        return 1
    fi

    log "One join attempt: $HOME_SSID"
    pw="$(security find-generic-password -D "AirPort network password" -a "$HOME_SSID" -w 2>/dev/null \
        || security find-generic-password -l "$HOME_SSID" -w 2>/dev/null || true)"
    if [ -n "$pw" ]; then
        networksetup -setairportnetwork "$dev" "$HOME_SSID" "$pw" >/dev/null 2>&1 || true
    else
        networksetup -setairportnetwork "$dev" "$HOME_SSID" >/dev/null 2>&1 || true
    fi
    sleep 5
    has_internet
}

ensure_network() {
    local dev ssid elapsed=0

    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        log "Another network check running — skipping"
        return 0
    fi
    trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

    dev="$(detect_wifi_device)"
    if [ -z "$dev" ]; then
        log "No Wi-Fi interface — skip"
        return 0
    fi

    ssid="$(current_ssid "$dev" 2>/dev/null || true)"

    if has_internet; then
        log "OK — internet up${ssid:+ on $ssid}"
        return 0
    fi

    log "No internet${ssid:+ (SSID: $ssid)} — waiting up to ${MAX_WAIT}s (passive)"

    # On home network: just wait for router/DNS — do NOT roam or power-cycle
    while [ "$elapsed" -lt "$MAX_WAIT" ]; do
        if has_internet; then
            log "OK — internet returned after ${elapsed}s"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done

    if has_internet; then
        return 0
    fi

    # Only try join if not on home SSID
    ssid="$(current_ssid "$dev" 2>/dev/null || true)"
    if [ "$ssid" = "$HOME_SSID" ]; then
        log "Still on $HOME_SSID but no internet — not switching networks"
        return 1
    fi

    if try_join_home_once "$dev"; then
        log "OK — joined $HOME_SSID"
        return 0
    fi

    log "WARNING — no internet (left Wi-Fi unchanged)"
    return 1
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    ensure_network
fi
