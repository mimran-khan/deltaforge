# Changelog

All notable changes to DeltaForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- MIT License
- Financial disclaimer (DISCLAIMER.md)
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- GitHub Actions CI workflow (pytest on Python 3.9-3.12)
- Dashboard security: localhost-only default, CORS restrictions, API token auth for halt endpoint
- `.gitignore` expanded for IDE files, caches, and credential patterns

### Changed
- README rewritten for open-source readiness
- LaunchAgent plist templatized (no hardcoded paths)
- Dashboard defaults to `127.0.0.1` instead of `0.0.0.0`
- Scripts use project-relative paths instead of hardcoded user directories

### Fixed
- Settings drift between `.env.example`, `settings.py`, and README

## [2.0.0] - 2026-06-01

### Added
- Multi-strategy engine (Stochastic Cross, Pullback-in-Trend, Supertrend Flip, VWAP Bias, RSI Mean Reversion)
- 9-gate pre-trade risk engine
- Real-time dashboard with FastAPI and WebSocket
- Slack, iMessage, and Telegram alert channels
- Walk-forward backtester
- CLI interface (`df` command)
- Comprehensive test suite
