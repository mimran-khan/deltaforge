#!/usr/bin/env python3
"""DeltaForge CLI -- single entry point for all operations.

Install:
    pip install -e .

Usage:
    df commands            # start always-on iMessage command listener
    df trade               # start daily trading session
    df trade --live        # live mode (real orders)
    df backtest            # run backtest
    df status              # show current capital, positions, risk state
    df logs                # tail today's log
    df logs --json         # tail JSON log (machine-readable)
    df test                # run full test suite
    df db summary          # today's trade summary from perf DB
    df db stats            # per-strategy stats
    df selftest            # run pre-flight self-test only
    df alert "message"     # send a test alert via iMessage/Telegram
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import click
import pytz
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

IST = pytz.timezone("Asia/Kolkata")
console = Console()


# ─── Main group ──────────────────────────────────────────────────

@click.group()
@click.version_option("2.0.0", prog_name="DeltaForge")
def cli():
    """DeltaForge -- Autonomous Nifty Options Trading System."""
    pass


# ─── trade ───────────────────────────────────────────────────────

@cli.command()
@click.option("--live", is_flag=True, help="Enable LIVE trading (real orders)")
@click.option("--paper", is_flag=True, default=True, help="Paper trading mode (default)")
@click.option("--no-commands", is_flag=True, help="Disable iMessage command listener")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level console logs")
@click.option("-q", "--quiet", is_flag=True, help="Warning-level console logs only")
def trade(live, paper, no_commands, verbose, quiet):
    """Start the daily trading session."""
    from config import settings
    from config.logging import setup_logging

    if live:
        settings.TRADING_MODE = "live"
        console.print("[bold red]LIVE TRADING MODE[/bold red] -- real orders will be placed")
        if not click.confirm("Are you sure?"):
            raise SystemExit(0)
    else:
        settings.TRADING_MODE = "paper"

    setup_logging(verbose=verbose, quiet=quiet)

    _print_banner(settings)

    from automation.daily_scheduler import DailyScheduler
    scheduler = DailyScheduler(enable_commands=not no_commands)
    scheduler.run()


# ─── backtest ────────────────────────────────────────────────────

@cli.command()
@click.option("--days", default=180, help="Number of trading days to simulate")
@click.option("--capital", default=10000.0, help="Starting capital (Rs)")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level console logs")
def backtest(days, capital, verbose):
    """Run walk-forward backtest on historical data."""
    from config.logging import setup_logging
    setup_logging(verbose=verbose)

    console.print(f"[cyan]Backtest:[/cyan] {days} days, Rs {capital:,.0f} starting capital")

    from backtest.run_backtest import main as bt_main
    sys.argv = ["backtest", "--days", str(days), "--capital", str(capital)]
    bt_main()


# ─── status ──────────────────────────────────────────────────────

@cli.command()
def status():
    """Show current system status: capital, risk, positions."""
    from config import settings
    from risk.kill_switch import is_halted

    table = Table(title="DeltaForge Status", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    # Capital state
    cap_data = {}
    if settings.CAPITAL_FILE.exists():
        try:
            with open(settings.CAPITAL_FILE) as f:
                cap_data = json.load(f)
        except Exception:
            pass

    capital = cap_data.get("current_capital", settings.STARTING_CAPITAL)
    peak = cap_data.get("peak_capital", capital)
    dd = (peak - capital) / peak * 100 if peak > 0 else 0
    daily_pnl = cap_data.get("daily_pnl", 0)
    total_pnl = cap_data.get("total_pnl", 0)
    trades = cap_data.get("trades_today", 0)
    wins = cap_data.get("wins_today", 0)
    losses = cap_data.get("losses_today", 0)
    consec = cap_data.get("consecutive_losses", 0)
    updated = cap_data.get("last_updated", "unknown")

    pnl_color = "green" if daily_pnl >= 0 else "red"
    dd_color = "green" if dd < 10 else ("yellow" if dd < 15 else "red")

    table.add_row("Mode", settings.TRADING_MODE.upper())
    table.add_row("Capital", f"Rs {capital:,.2f}")
    table.add_row("Peak Capital", f"Rs {peak:,.2f}")
    table.add_row("Daily P&L", f"[{pnl_color}]Rs {daily_pnl:+,.2f}[/{pnl_color}]")
    table.add_row("Total P&L", f"Rs {total_pnl:+,.2f}")
    table.add_row("Drawdown", f"[{dd_color}]{dd:.1f}%[/{dd_color}]")
    table.add_row("Trades Today", f"{trades} (W:{wins} L:{losses})")
    table.add_row("Consecutive Losses", str(consec))
    table.add_row("Kill Switch", "[red]HALTED[/red]" if is_halted() else "[green]OK[/green]")
    table.add_row("Last Updated", updated)

    console.print(table)

    # DB summary
    try:
        from persistence.performance_db import PerformanceDB
        db = PerformanceDB()
        summary = db.daily_summary()
        if summary["trades"] > 0:
            console.print(f"\n[dim]DB: {summary['trades']} trades today, "
                          f"WR {summary['wr']:.0f}%, "
                          f"PnL Rs {summary['total_pnl']:+,.0f}[/dim]")
        db.close()
    except Exception:
        pass


# ─── logs ────────────────────────────────────────────────────────

@cli.command()
@click.option("--json", "json_mode", is_flag=True, help="Show JSON logs instead of text")
@click.option("-n", "--lines", default=50, help="Number of lines to show")
@click.option("-f", "--follow", is_flag=True, help="Follow log output (like tail -f)")
def logs(json_mode, lines, follow):
    """View today's trading logs."""
    from config import settings

    if json_mode:
        log_dir = settings.LOG_DIR / "json"
        log_file = log_dir / f"trading_{datetime.now(IST).date().isoformat()}.jsonl"
    else:
        log_file = settings.LOG_DIR / f"trading_{datetime.now(IST).date().isoformat()}.log"

    if not log_file.exists():
        console.print(f"[yellow]No log file found:[/yellow] {log_file}")
        return

    if follow:
        os.execvp("tail", ["tail", "-f", "-n", str(lines), str(log_file)])
    else:
        with open(log_file) as f:
            all_lines = f.readlines()
        for line in all_lines[-lines:]:
            click.echo(line.rstrip())


# ─── test ────────────────────────────────────────────────────────

@cli.command("test")
@click.option("--component", is_flag=True, help="Run component tests only")
@click.option("--e2e", is_flag=True, help="Run end-to-end tests only")
@click.option("-v", "--verbose", is_flag=True, help="Verbose test output")
def run_tests(component, e2e, verbose):
    """Run the test suite."""
    venv_python = str(PROJECT_ROOT / "venv" / "bin" / "python")
    if not Path(venv_python).exists():
        venv_python = sys.executable
    args = [venv_python, "-m", "pytest"]

    if component:
        args.append("tests/test_components.py")
    elif e2e:
        args.append("tests/test_e2e.py")
    else:
        args.append("tests/")

    if verbose:
        args.append("-v")
    args.append("--tb=short")

    os.execvp(venv_python, args)


# ─── db ──────────────────────────────────────────────────────────

@cli.group()
def db():
    """Query the performance database."""
    pass


@db.command("summary")
@click.option("--date", "target_date", default=None, help="Date (YYYY-MM-DD), default=today")
def db_summary(target_date):
    """Show daily trade summary."""
    from persistence.performance_db import PerformanceDB
    pdb = PerformanceDB()
    s = pdb.daily_summary(target_date)
    pdb.close()

    if s["trades"] == 0:
        console.print(f"[dim]No trades on {s['date']}[/dim]")
        return

    table = Table(title=f"Trade Summary -- {s['date']}", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    wr_color = "green" if s["wr"] >= 60 else ("yellow" if s["wr"] >= 40 else "red")
    pnl_color = "green" if s["total_pnl"] >= 0 else "red"

    table.add_row("Trades", str(s["trades"]))
    table.add_row("Wins", str(s["wins"]))
    table.add_row("Losses", str(s["losses"]))
    table.add_row("Win Rate", f"[{wr_color}]{s['wr']:.1f}%[/{wr_color}]")
    table.add_row("Total P&L", f"[{pnl_color}]Rs {s['total_pnl']:+,.2f}[/{pnl_color}]")

    console.print(table)


@db.command("stats")
@click.option("--strategy", default=None, help="Filter by strategy name")
@click.option("--min-trades", default=1, help="Minimum trades to include")
def db_stats(strategy, min_trades):
    """Show per-strategy performance statistics."""
    from persistence.performance_db import PerformanceDB
    pdb = PerformanceDB()
    stats = pdb.strategy_stats(strategy=strategy, min_trades=min_trades)
    pdb.close()

    if not stats:
        console.print("[dim]No strategy data yet[/dim]")
        return

    table = Table(title="Strategy Performance", box=box.ROUNDED)
    table.add_column("Strategy", style="cyan")
    table.add_column("Trades", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("WR%", justify="right")
    table.add_column("Total P&L", justify="right")
    table.add_column("Avg P&L", justify="right")

    for s in stats:
        wr_color = "green" if s["wr"] >= 60 else ("yellow" if s["wr"] >= 40 else "red")
        pnl_color = "green" if s["total_pnl"] >= 0 else "red"
        table.add_row(
            s["strategy"],
            str(s["trades"]),
            str(s["wins"]),
            f"[{wr_color}]{s['wr']:.1f}%[/{wr_color}]",
            f"[{pnl_color}]Rs {s['total_pnl']:+,.0f}[/{pnl_color}]",
            f"Rs {s['avg_pnl']:+,.0f}",
        )

    console.print(table)


# ─── selftest ────────────────────────────────────────────────────

@cli.command()
def selftest():
    """Run the pre-flight self-test (signal generation check)."""
    from config import settings
    from config.logging import setup_logging
    setup_logging()

    console.print("[cyan]Running pre-flight self-test...[/cyan]")

    import pandas as pd

    from engine.multi_strategy_engine import MultiStrategyEngine

    selftest_path = Path(settings.DATA_DIR) / "selftest_candles.csv"
    if not selftest_path.exists():
        console.print("[red]Self-test data not found[/red]")
        raise SystemExit(1)

    df = pd.read_csv(selftest_path)
    dt_col = "Datetime" if "Datetime" in df.columns else "datetime"
    df[dt_col] = pd.to_datetime(df[dt_col])
    df.set_index(dt_col, inplace=True)

    engine = MultiStrategyEngine()
    engine.reset_day()
    indicators = engine.precompute(df)

    signals = 0
    for i in range(10, len(df)):
        ts = df.index[i]
        t = ts.strftime("%H:%M") if hasattr(ts, "strftime") else ""
        signals += len(engine.scan(indicators, i, t))

    console.print(f"[green]Self-test passed:[/green] {signals} signals on {len(df)} bars")


# ─── alert ───────────────────────────────────────────────────────

@cli.command()
@click.argument("message")
def alert(message):
    """Send a test alert via the configured method (Slack/iMessage/Telegram)."""
    from alerts import send_alert

    ok = send_alert(message)
    if ok:
        console.print("[green]Alert sent successfully[/green]")
    else:
        console.print("[red]Alert failed[/red]")
        raise SystemExit(1)


# ─── broker ──────────────────────────────────────────────────────

@cli.command()
def broker():
    """Test broker connection and show account info."""
    from config.logging import setup_logging
    setup_logging(quiet=True)

    from engine.broker import BrokerConnection

    console.print("[cyan]Connecting to Angel One...[/cyan]")
    b = BrokerConnection()

    if not b.login():
        console.print("[red]Login FAILED -- check .env credentials[/red]")
        raise SystemExit(1)

    table = Table(title="Broker Connection", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Client ID", b.client_code)
    table.add_row("Session", "[green]ACTIVE[/green]")
    table.add_row("Auth Token", b.auth_token[:30] + "...")
    table.add_row("Feed Token", (b.feed_token[:30] + "...") if b.feed_token else "N/A")

    from config import settings as _settings
    ltp = b.get_ltp("NSE", "Nifty 50", _settings.NIFTY_INDEX_TOKEN)
    table.add_row("Nifty 50 LTP", f"Rs {ltp:,.2f}" if ltp else "[yellow]Market closed[/yellow]")

    ltp_bn = b.get_ltp("NSE", "Nifty Bank", _settings.BANKNIFTY_INDEX_TOKEN)
    table.add_row("BankNifty LTP", f"Rs {ltp_bn:,.2f}" if ltp_bn else "[yellow]Market closed[/yellow]")

    try:
        profile = b.api.getProfile(b.refresh_token)
        if profile and profile.get("data"):
            table.add_row("Account Name", profile["data"].get("name", "N/A"))
    except Exception:
        pass

    positions = b.get_positions()
    table.add_row("Open Positions", str(len(positions)))

    console.print(table)
    b.logout()
    console.print("[dim]Session terminated[/dim]")


# ─── commands ─────────────────────────────────────────────────

@cli.command()
@click.option("-v", "--verbose", is_flag=True, help="Debug-level console logs")
def commands(verbose):
    """Start the always-on iMessage command listener.

    Runs as a foreground daemon that listens for /commands via iMessage.
    Can start/stop trading sessions, run backtests, and report status
    even when no trading session is active.

    Requires Full Disk Access for your terminal app.
    """
    from config.logging import setup_logging
    setup_logging(verbose=verbose)

    console.print(Panel(
        "[bold cyan]DeltaForge Command Listener[/bold cyan]\n"
        "Listening for iMessage commands...\n"
        "Send [bold]/help[/bold] from your phone to see commands\n"
        "Press Ctrl+C to stop",
        title="DeltaForge", border_style="cyan",
    ))

    from alerts.command_listener import CommandPoller
    poller = CommandPoller()
    poller.run_forever()


# ─── Banner ──────────────────────────────────────────────────────

def _print_banner(settings):
    banner = (
        f"[bold cyan]DeltaForge V11[/bold cyan] -- Multi-Strategy (76% WR, PF 2.62)\n"
        f"Mode: [bold]{'[red]LIVE' if settings.TRADING_MODE == 'live' else '[green]PAPER'}[/bold]\n"
        f"Capital: Rs {settings.STARTING_CAPITAL:,.0f}\n"
        f"Strategy: DeltaForge V11 -- Multi-Strategy (76% WR, PF 2.62)"
    )
    console.print(Panel(banner, title="DeltaForge", border_style="cyan"))


# ─── Entry point ─────────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()
