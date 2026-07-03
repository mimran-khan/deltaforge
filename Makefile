# ──────────────────────────────────────────────────────────────
#  DeltaForge -- Makefile
#
#  Usage:
#    make install     # create venv, install deps, register `df` CLI
#    make commands    # start always-on iMessage command listener
#    make trade       # start paper trading session
#    make trade-live  # start LIVE trading session (real orders)
#    make backtest    # run walk-forward backtest
#    make status      # show capital, risk, positions
#    make test        # run full test suite
#    make broker      # test broker connection
#    make logs        # tail today's log
#    make logs-json   # tail JSON log
#    make selftest    # run pre-flight engine check
#    make db-summary  # today's trade summary
#    make db-stats    # per-strategy performance
#    make clean       # remove caches, temp files
#    make clean-all   # remove venv + all caches
# ──────────────────────────────────────────────────────────────

SHELL := /bin/bash
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
DF := $(VENV)/bin/df

.DEFAULT_GOAL := help

# ─── Setup ───────────────────────────────────────────────────

.PHONY: install
install: $(VENV)/bin/df  ## Create venv and install everything

$(VENV)/bin/df: $(VENV)/bin/activate
	@$(PIP) install -e ".[test]" --quiet
	@echo "✓ DeltaForge installed. Run: make trade"

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip setuptools wheel --quiet

.PHONY: update
update:  ## Update all dependencies
	@$(PIP) install -e ".[test]" --upgrade --quiet
	@echo "✓ Dependencies updated"

# ─── Trading ─────────────────────────────────────────────────

.PHONY: trade
trade: $(VENV)/bin/df  ## Start paper trading session
	@$(DF) trade

.PHONY: trade-live
trade-live: $(VENV)/bin/df  ## Start LIVE trading (real orders -- confirm required)
	@$(DF) trade --live

.PHONY: trade-verbose
trade-verbose: $(VENV)/bin/df  ## Paper trading with debug-level logs
	@$(DF) trade --verbose

.PHONY: run
run: $(VENV)/bin/df  ## Start trading + dashboard together
	@echo "Starting dashboard at http://localhost:8900 ..."
	@$(PYTHON) -m dashboard.server &
	@DASHBOARD_PID=$$!; \
	trap "kill $$DASHBOARD_PID 2>/dev/null" EXIT; \
	echo "Starting trading engine ..."; \
	$(DF) trade; \
	kill $$DASHBOARD_PID 2>/dev/null

.PHONY: run-live
run-live: $(VENV)/bin/df  ## Start LIVE trading + dashboard together
	@echo "Starting dashboard at http://localhost:8900 ..."
	@$(PYTHON) -m dashboard.server &
	@DASHBOARD_PID=$$!; \
	trap "kill $$DASHBOARD_PID 2>/dev/null" EXIT; \
	echo "Starting LIVE trading engine ..."; \
	$(DF) trade --live; \
	kill $$DASHBOARD_PID 2>/dev/null

.PHONY: commands
commands: install  ## Start always-on iMessage command listener
	@$(DF) commands

# ─── Backtest ────────────────────────────────────────────────

.PHONY: backtest
backtest: install  ## Run walk-forward backtest (180 days, Rs 10K)
	@$(DF) backtest

.PHONY: backtest-custom
backtest-custom: install  ## Run backtest with custom params (DAYS=, CAPITAL=)
	@$(DF) backtest --days $(or $(DAYS),180) --capital $(or $(CAPITAL),10000)

# ─── Monitoring ──────────────────────────────────────────────

.PHONY: status
status: install  ## Show current system status
	@$(DF) status

.PHONY: broker
broker: install  ## Test broker connection + account info
	@$(DF) broker

.PHONY: selftest
selftest: install  ## Run pre-flight engine self-test
	@$(DF) selftest

.PHONY: logs
logs: install  ## Tail today's trading log
	@$(DF) logs -f

.PHONY: logs-json
logs-json: install  ## Tail today's JSON log
	@$(DF) logs --json -f

# ─── Database ────────────────────────────────────────────────

.PHONY: db-summary
db-summary: install  ## Show today's trade summary
	@$(DF) db summary

.PHONY: db-stats
db-stats: install  ## Show per-strategy statistics
	@$(DF) db stats

# ─── Dashboard ───────────────────────────────────────────────

.PHONY: dashboard
dashboard: install  ## Start dashboard (API + UI at http://localhost:8900)
	@$(PYTHON) -m dashboard.server

.PHONY: test-dashboard
test-dashboard: install  ## Run dashboard API tests
	@$(VENV)/bin/pytest tests/test_dashboard.py -v

# ─── Alerts ──────────────────────────────────────────────────

.PHONY: alert-test
alert-test: install  ## Send a test alert
	@$(DF) alert "DeltaForge test alert -- system is operational"

# ─── Testing ─────────────────────────────────────────────────

.PHONY: test
test: install  ## Run full test suite
	@$(VENV)/bin/pytest tests/ -v --tb=short

.PHONY: test-e2e
test-e2e: install  ## Run E2E tests only
	@$(DF) test --e2e -v

.PHONY: test-component
test-component: install  ## Run component tests only
	@$(DF) test --component -v

# ─── Auto-Start Setup ────────────────────────────────────────

.PHONY: install-autostart
install-autostart:  ## Install LaunchAgent + crontab for daily auto-start
	@mkdir -p $(HOME)/Library/LaunchAgents
	@sed 's|__PROJECT_ROOT__|$(CURDIR)|g' automation/com.tradingagent.daily.plist \
		> $(HOME)/Library/LaunchAgents/com.deltaforge.daily.plist
	@launchctl unload $(HOME)/Library/LaunchAgents/com.deltaforge.daily.plist 2>/dev/null || true
	@launchctl load $(HOME)/Library/LaunchAgents/com.deltaforge.daily.plist
	@echo "✓ LaunchAgent installed (triggers 08:25 IST daily)"
	@(crontab -l 2>/dev/null | grep -v healthcheck || true; \
	  echo "*/15 8-15 * * 1-5 cd $(CURDIR) && ./healthcheck.sh") | crontab -
	@echo "✓ Healthcheck crontab installed (every 15 min, Mon-Fri 08:00-15:59)"

# ─── Cleanup ─────────────────────────────────────────────────

.PHONY: clean
clean:  ## Remove caches and temp files
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .mypy_cache
	@echo "✓ Caches cleaned"

.PHONY: clean-all
clean-all: clean  ## Remove venv + all caches
	@rm -rf $(VENV) *.egg-info
	@echo "✓ Full clean complete"

# ─── Help ────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help
	@echo ""
	@echo "  DeltaForge -- Available Commands"
	@echo "  ──────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
