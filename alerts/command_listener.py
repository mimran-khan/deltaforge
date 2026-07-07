"""Command listener -- always-on daemon that polls for /commands.

Supports multiple backends: Slack channel polling or iMessage chat.db polling.
Backend is selected automatically based on ALERT_METHOD in settings.

Commands:
  /status  -- capital, P&L, trades, drawdown, risk state
  /stop    -- halt trading, square off positions
  /start   -- launch a trading session (or resume if halted)
  /market  -- Nifty/BankNifty LTP, market state
  /paper N -- run N-day backtest and send results
  /help    -- list all commands with descriptions

Architecture:
  - CommandPoller: factory that returns the right poller backend
  - _SlackPoller / _IMessagePoller: backend-specific polling
  - CommandRouter: dispatches to handler functions
  - SessionManager: manages trading session lifecycle from /start and /stop
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pytz
from loguru import logger

from config import settings
from risk.kill_switch import (
    acquire_poller_lock,
    clear_halt,
    force_release_session_lock,
    is_halted,
    is_poller_running,
    is_session_running,
    release_poller_lock,
    set_halt,
)

_alert_method = getattr(settings, "ALERT_METHOD", "slack")
if _alert_method == "imessage":
    from alerts.imessage_bot import send_alert
elif _alert_method == "slack":
    from alerts.slack_bot import send_alert
else:
    from alerts.telegram_bot import send_alert

if TYPE_CHECKING:
    from engine.trading_engine import TradingEngine

IST = pytz.timezone("Asia/Kolkata")
CHAT_DB = Path.home() / "Library" / "Messages" / "chat.db"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CMD_PREFIX = "!" if getattr(settings, "ALERT_METHOD", "slack") == "slack" else "/"

HELP_TEXT = (
    "DELTAFORGE COMMANDS\n"
    "---\n"
    f"{CMD_PREFIX}status - P&L, capital, drawdown, risk state\n"
    f"{CMD_PREFIX}stop - halt trading, square off positions\n"
    f"{CMD_PREFIX}start - launch or resume trading session\n"
    f"{CMD_PREFIX}market - Nifty & BankNifty live prices\n"
    f"{CMD_PREFIX}paper N - backtest N days (e.g. {CMD_PREFIX}paper 60)\n"
    f"{CMD_PREFIX}help - show this list"
)


class SessionManager:
    """Manages the trading session subprocess lifecycle."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        with self._lock:
            if self._proc is None:
                return False
            poll = self._proc.poll()
            if poll is not None:
                self._proc = None
                return False
            return True

    def start_session(self) -> str:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return "DeltaForge trading session is already running."

            running, existing_pid = is_session_running()
            if running:
                return (f"DeltaForge session already running (PID {existing_pid}).\n"
                        f"Send {CMD_PREFIX}stop first, then {CMD_PREFIX}start.")

            venv_python = PROJECT_ROOT / "venv" / "bin" / "python3"
            if not venv_python.exists():
                venv_python = Path(sys.executable)

            try:
                self._proc = subprocess.Popen(
                    [str(venv_python), "-m", "cli", "trade", "--no-commands"],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("Trading session started (PID {})", self._proc.pid)
                return (
                    "TRADING SESSION STARTED\n\n"
                    f"PID: {self._proc.pid}\n"
                    "Mode: PAPER\n"
                    "The session will handle market hours,\n"
                    "login, and trading automatically.\n"
                    f"Send {CMD_PREFIX}stop to halt."
                )
            except Exception as e:
                logger.error("Failed to start session: {}", e)
                return f"Failed to start trading session: {e}"

    def stop_session(self) -> bool:
        """Terminate the session subprocess. Returns True if a process was killed."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = None
                return False
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            logger.info("Trading session terminated")
            self._proc = None
            return True


class CommandRouter:
    """Maps /commands to handler functions."""

    def __init__(self, engine: Optional[TradingEngine] = None,
                 session_mgr: Optional[SessionManager] = None):
        self.engine = engine
        self.session_mgr = session_mgr or SessionManager()

    def dispatch(self, command: str) -> str:
        parts = command.strip().split()
        raw_cmd = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []

        # Normalise prefix: accept both !cmd and /cmd
        cmd = raw_cmd
        if cmd.startswith("!"):
            cmd = "/" + cmd[1:]

        handlers = {
            "/status": self._handle_status,
            "/stop": self._handle_stop,
            "/start": self._handle_start,
            "/market": self._handle_market,
            "/paper": self._handle_paper,
            "/help": self._handle_help,
        }

        handler = handlers.get(cmd)
        if not handler:
            return f"Unknown command: {raw_cmd}\n\nSend {CMD_PREFIX}help for available commands."

        try:
            if cmd == "/paper":
                return handler(args)
            return handler()
        except Exception as e:
            logger.error("Command handler error for {}: {}", cmd, e)
            return f"Error processing {cmd}: {e}"

    def _handle_status(self) -> str:
        if self.engine:
            return self._engine_status()
        return self._file_status()

    def _engine_status(self) -> str:
        """Live status from running engine instance."""
        s = self.engine.get_status_dict()
        halted = is_halted()
        pnl = s["daily_pnl"]
        sign = "+" if pnl >= 0 else ""
        wr = (s["wins"] / s["trades"] * 100) if s["trades"] > 0 else 0

        lines = [
            "DELTAFORGE STATUS",
            "---",
            f"Capital: Rs {s['capital']:,.0f}",
            f"Day P&L: Rs {sign}{pnl:,.0f} ({s['daily_pnl_pct']:+.1f}%)",
            f"Trades: {s['trades']} (W:{s['wins']} L:{s['losses']})",
            f"Win Rate: {wr:.0f}%",
            f"Drawdown: {s['drawdown']:.1f}%",
            f"Consec Losses: {s['consecutive_losses']}",
            f"Positions: {s['open_positions']}",
            f"Mode: {settings.TRADING_MODE.upper()}",
            f"Kill Switch: {'HALTED' if halted else 'OK'}",
        ]
        return "\n".join(lines)

    def _file_status(self) -> str:
        """Offline status from capital file on disk."""
        cap_data = {}
        if settings.CAPITAL_FILE.exists():
            try:
                with open(settings.CAPITAL_FILE) as f:
                    cap_data = json.load(f)
            except Exception:
                pass

        if not cap_data:
            return f"DeltaForge is offline.\nNo capital data found.\nSend {CMD_PREFIX}start to begin."

        capital = cap_data.get("current_capital", settings.STARTING_CAPITAL)
        peak = cap_data.get("peak_capital", capital)
        dd = (peak - capital) / peak * 100 if peak > 0 else 0
        pnl = cap_data.get("daily_pnl", 0)
        trades = cap_data.get("trades_today", 0)
        wins = cap_data.get("wins_today", 0)
        losses = cap_data.get("losses_today", 0)
        consec = cap_data.get("consecutive_losses", 0)
        raw_updated = cap_data.get("last_updated", "unknown")
        try:
            updated = datetime.fromisoformat(raw_updated).astimezone(IST).strftime("%d %b %Y, %I:%M %p IST")
        except (ValueError, TypeError):
            updated = raw_updated
        sign = "+" if pnl >= 0 else ""
        wr = (wins / trades * 100) if trades > 0 else 0
        session_running = self.session_mgr.is_running

        lines = [
            "DELTAFORGE STATUS (OFFLINE)",
            "---",
            f"Capital: Rs {capital:,.0f}",
            f"Day P&L: Rs {sign}{pnl:,.0f}",
            f"Trades: {trades} (W:{wins} L:{losses})",
            f"Win Rate: {wr:.0f}%",
            f"Drawdown: {dd:.1f}%",
            f"Consec Losses: {consec}",
            f"Session: {'RUNNING' if session_running else 'STOPPED'}",
            f"Kill Switch: {'HALTED' if is_halted() else 'OK'}",
            f"Last Updated: {updated}",
        ]
        return "\n".join(lines)

    def _handle_stop(self) -> str:
        set_halt("Manual stop via command")

        if self.engine:
            try:
                if self.engine._paper_positions:
                    self.engine._square_off_all("MANUAL_STOP")
                if self.engine.position_mgr and self.engine.position_mgr.has_open_positions:
                    self.engine.position_mgr.square_off_all("MANUAL_STOP")
            except Exception as e:
                logger.error("Error during manual stop square-off: {}", e)

        killed = self.session_mgr.stop_session()
        force_release_session_lock()

        msg = "TRADING STOPPED\n\nKill switch activated."
        if killed:
            msg += "\nSession process terminated."
        if self.engine:
            msg += "\nAll positions squared off."
        msg += f"\nSend {CMD_PREFIX}start to resume."
        return msg

    def _handle_start(self) -> str:
        if is_halted():
            clear_halt()
            if self.engine and self.engine.risk:
                self.engine.risk.resume()

        if self.engine and self.engine._running:
            return "DeltaForge is already running."

        if self.session_mgr.is_running:
            return "DeltaForge trading session is already running."

        running, pid = is_session_running()
        if running:
            return (f"DeltaForge session already active (PID {pid}).\n"
                    f"Send {CMD_PREFIX}stop first if you want to restart.")

        return self.session_mgr.start_session()

    def _handle_market(self) -> str:
        if self.engine and self.engine.broker.is_active:
            snapshot = self.engine.get_market_snapshot()
        else:
            snapshot = {"nifty": None, "banknifty": None}

        now = datetime.now(IST)
        t = now.strftime("%H:%M")
        if t < settings.MARKET_OPEN:
            market_state = "PRE-MARKET"
        elif t > settings.MARKET_CLOSE:
            market_state = "CLOSED"
        else:
            market_state = "OPEN"

        lines = [
            f"MARKET ({market_state})",
            "---",
        ]

        if snapshot["nifty"]:
            lines.append(f"Nifty 50: Rs {snapshot['nifty']:,.2f}")
        else:
            lines.append("Nifty 50: N/A (broker offline)")

        if snapshot["banknifty"]:
            lines.append(f"BankNifty: Rs {snapshot['banknifty']:,.2f}")
        else:
            lines.append("BankNifty: N/A (broker offline)")

        lines.append(f"Time: {now.strftime('%H:%M:%S IST')}")
        return "\n".join(lines)

    def _handle_paper(self, args: list) -> str:
        days = 60
        if args:
            try:
                days = int(args[0])
            except ValueError:
                return f"Invalid days: {args[0]}\nUsage: /paper 60"

        if days < 5 or days > 500:
            return f"Days must be between 5 and 500 (got {days})"

        send_alert(f"Running {days}-day backtest...\nThis takes a few seconds.")

        thread = threading.Thread(
            target=self._run_backtest_async, args=(days,), daemon=True
        )
        thread.start()
        return None  # response sent async

    def _run_backtest_async(self, days: int):
        try:
            from backtest.run_backtest import load_real_data, run_compound_backtest

            df = load_real_data(days=days)
            results = run_compound_backtest(
                df,
                starting_capital=settings.STARTING_CAPITAL,
                lot_size=settings.NIFTY_LOT_SIZE,
            )

            pnl = results["total_pnl"]
            sign = "+" if pnl >= 0 else ""

            if results["profit_factor"] > 1.5 and results["win_rate"] > 45:
                verdict = "STRONG edge"
            elif results["profit_factor"] > 1.2:
                verdict = "Moderate edge"
            elif results["profit_factor"] > 1.0:
                verdict = "Marginal -- needs tuning"
            else:
                verdict = "Negative expectancy"

            strat_stats = {}
            for t in results["trades"]:
                s = t["strategy"]
                if s not in strat_stats:
                    strat_stats[s] = {"w": 0, "l": 0, "pnl": 0, "n": 0}
                strat_stats[s]["n"] += 1
                if t["pnl"] > 0:
                    strat_stats[s]["w"] += 1
                else:
                    strat_stats[s]["l"] += 1
                strat_stats[s]["pnl"] += t["pnl"]

            lines = [
                f"BACKTEST ({days} DAYS)",
                "---",
                f"Rs {results['starting_capital']:,.0f} -> Rs {results['final_capital']:,.0f}",
                f"P&L: Rs {sign}{pnl:,.0f} ({results['return_pct']}%)",
                f"Avg Daily: {results['avg_daily_return_pct']}%",
                f"Trades: {results['total_trades']}",
                f"W: {results['wins']} L: {results['losses']} ({results['win_rate']}%)",
                f"PF: {results['profit_factor']} | DD: {results['max_drawdown_pct']}%",
                f"Days: {results['active_trading_days']}/{results['trading_days']} active",
                f"Green: {results['profitable_days']} | Red: {results['loss_days']}",
            ]

            if strat_stats:
                lines.append("---")
                for name, st in sorted(strat_stats.items()):
                    swr = st["w"] / st["n"] * 100 if st["n"] > 0 else 0
                    lines.append(f"{name}: {st['n']}t {swr:.0f}%WR Rs{st['pnl']:+,.0f}")

            lines.extend(["---", f"Verdict: {verdict}"])

            send_alert("\n".join(lines))

        except Exception as e:
            logger.error("Backtest command error: {}", e)
            send_alert(f"Backtest failed: {e}")

    def _handle_help(self) -> str:
        return HELP_TEXT


# ─────────────────────────────────────────────────────────────────
#  Backend pollers
# ─────────────────────────────────────────────────────────────────

class _SlackPoller:
    """Polls a Slack channel for /commands using conversations.history."""

    def __init__(self, router: CommandRouter, poll_interval: int):
        self.router = router
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_ts: str = ""
        self._processed: set[str] = set()
        self._channel = getattr(settings, "SLACK_CHANNEL_ID", "")
        self._client = None
        self._bot_user_id: Optional[str] = None

    def _init_client(self) -> bool:
        token = getattr(settings, "SLACK_BOT_TOKEN", "")
        if not token or not self._channel:
            return False
        from slack_sdk import WebClient
        self._client = WebClient(token=token)
        try:
            auth = self._client.auth_test()
            self._bot_user_id = auth["user_id"]
        except Exception:
            self._bot_user_id = None
        return True

    def _init_watermark(self):
        """Set watermark to latest message so we skip old history."""
        try:
            resp = self._client.conversations_history(
                channel=self._channel, limit=1
            )
            msgs = resp.get("messages", [])
            if msgs:
                self._last_ts = msgs[0]["ts"]
            logger.debug("SlackPoller watermark set to ts {}", self._last_ts)
        except Exception as e:
            logger.warning("SlackPoller watermark init failed: {}", e)

    def start(self):
        if not self._init_client():
            logger.warning("SlackPoller: SLACK_BOT_TOKEN/SLACK_CHANNEL_ID not set -- disabled")
            return

        if not acquire_poller_lock():
            running, pid = is_poller_running()
            logger.warning("Another command listener is running (PID {}), skipping", pid)
            return

        self._init_watermark()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("SlackPoller started (polling every {}s)", self.poll_interval)

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        release_poller_lock()
        logger.info("SlackPoller stopped")

    def run_forever(self):
        if not self._init_client():
            logger.error("SLACK_BOT_TOKEN or SLACK_CHANNEL_ID not set.")
            return

        if not acquire_poller_lock():
            running, pid = is_poller_running()
            logger.error("Another command listener is already running (PID {}). "
                         "Stop it first or use the existing one.", pid)
            return

        self._init_watermark()
        self._running = True

        logger.info("DeltaForge Command Listener running on Slack (Ctrl+C to stop)")
        logger.info("Post !help in your Slack channel to see commands")

        send_alert("DeltaForge Command Listener started.\nSend !help for commands.")

        try:
            while self._running:
                self._check_new_commands()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Command listener interrupted")
        finally:
            self.router.session_mgr.stop_session()
            release_poller_lock()
            logger.info("Command listener stopped")

    def _poll_loop(self):
        while self._running:
            try:
                self._check_new_commands()
            except Exception as e:
                logger.debug("SlackPoller poll error: {}", e)
            time.sleep(self.poll_interval)

    def _check_new_commands(self):
        try:
            kwargs = {"channel": self._channel, "limit": 10}
            if self._last_ts:
                kwargs["oldest"] = self._last_ts
                kwargs["inclusive"] = False
            resp = self._client.conversations_history(**kwargs)
        except Exception as e:
            logger.debug("SlackPoller API error: {}", e)
            return

        msgs = resp.get("messages", [])
        for msg in reversed(msgs):
            ts = msg.get("ts", "")
            if ts <= self._last_ts or ts in self._processed:
                continue

            self._last_ts = ts
            self._processed.add(ts)
            if len(self._processed) > 100:
                self._processed = set(sorted(self._processed)[-50:])

            if self._bot_user_id and msg.get("user") == self._bot_user_id:
                continue
            if msg.get("bot_id"):
                continue

            text = msg.get("text", "").strip()
            if not text.startswith("!"):
                continue

            logger.info("Slack command received: {}", text)
            response = self.router.dispatch(text)
            if response:
                send_alert(response)


class _IMessagePoller:
    """Polls macOS chat.db for /commands from IMESSAGE_RECIPIENT."""

    def __init__(self, router: CommandRouter, poll_interval: int):
        self.router = router
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_rowid = 0
        self._recipient = getattr(settings, "IMESSAGE_RECIPIENT", "")

    def _init_watermark(self):
        try:
            conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
            cursor = conn.execute("SELECT MAX(ROWID) FROM message")
            row = cursor.fetchone()
            self._last_rowid = row[0] or 0
            conn.close()
            logger.debug("IMessagePoller watermark set to ROWID {}", self._last_rowid)
        except Exception as e:
            logger.warning("IMessagePoller watermark init failed: {}", e)
            self._last_rowid = 0

    def start(self):
        if not self._recipient:
            logger.warning("IMessagePoller: IMESSAGE_RECIPIENT not set -- disabled")
            return

        if not CHAT_DB.exists():
            logger.warning("IMessagePoller: chat.db not found at {} -- disabled", CHAT_DB)
            return

        if not acquire_poller_lock():
            running, pid = is_poller_running()
            logger.warning("Another command listener is running (PID {}), skipping", pid)
            return

        self._init_watermark()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("IMessagePoller started (polling every {}s)", self.poll_interval)

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        release_poller_lock()
        logger.info("IMessagePoller stopped")

    def run_forever(self):
        if not self._recipient:
            logger.error("IMESSAGE_RECIPIENT not set. Cannot start command listener.")
            return

        if not CHAT_DB.exists():
            logger.error("chat.db not found at {}. Grant Full Disk Access to your terminal.", CHAT_DB)
            return

        if not acquire_poller_lock():
            running, pid = is_poller_running()
            logger.error("Another command listener is already running (PID {}). "
                         "Stop it first or use the existing one.", pid)
            return

        self._init_watermark()
        self._running = True

        logger.info("DeltaForge Command Listener running (Ctrl+C to stop)")
        logger.info("Listening for commands from {}", self._recipient)
        logger.info("Send /help from iMessage to see available commands")

        send_alert(f"DeltaForge Command Listener started.\nSend {CMD_PREFIX}help for commands.")

        try:
            while self._running:
                self._check_new_commands()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Command listener interrupted")
        finally:
            self.router.session_mgr.stop_session()
            release_poller_lock()
            logger.info("Command listener stopped")

    def _poll_loop(self):
        while self._running:
            try:
                self._check_new_commands()
            except Exception as e:
                logger.debug("IMessagePoller poll error: {}", e)
            time.sleep(self.poll_interval)

    def _check_new_commands(self):
        try:
            conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
        except Exception:
            return

        try:
            query = """
                SELECT m.ROWID, m.text
                FROM message m
                JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                JOIN chat c ON cmj.chat_id = c.ROWID
                WHERE m.is_from_me = 0
                  AND m.ROWID > ?
                  AND c.chat_identifier = ?
                ORDER BY m.ROWID ASC
                LIMIT 10
            """
            cursor = conn.execute(query, (self._last_rowid, self._recipient))
            rows = cursor.fetchall()

            for rowid, text in rows:
                self._last_rowid = rowid
                if not text:
                    continue

                text = text.strip()
                if not text.startswith("/"):
                    continue

                logger.info("iMessage command received: {}", text)
                response = self.router.dispatch(text)
                if response:
                    send_alert(response)

        except Exception as e:
            logger.debug("IMessagePoller query error: {}", e)
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────
#  Public API (preserves backward-compatible CommandPoller interface)
# ─────────────────────────────────────────────────────────────────

class CommandPoller:
    """Facade that delegates to the right backend poller.

    Public API is identical to the old iMessage-only CommandPoller so
    all call-sites (cli.py, daily_scheduler.py, etc.) keep working.
    """

    def __init__(self, engine: Optional[TradingEngine] = None,
                 poll_interval: int = 5,
                 session_mgr: Optional[SessionManager] = None):
        self._session_mgr = session_mgr or SessionManager()
        self.router = CommandRouter(engine, self._session_mgr)

        method = getattr(settings, "ALERT_METHOD", "slack")
        if method == "slack":
            self._backend = _SlackPoller(self.router, poll_interval)
        else:
            self._backend = _IMessagePoller(self.router, poll_interval)

        self.poll_interval = poll_interval

    @property
    def engine(self):
        return self.router.engine

    @engine.setter
    def engine(self, value):
        self.router.engine = value

    def start(self):
        self._backend.start()

    def stop(self):
        self._backend.stop()

    def run_forever(self):
        self._backend.run_forever()
