<!-- Banner — replace with your own banner image -->
<!-- <p align="center"><img src="docs/assets/banner.png" alt="DeltaForge" width="800"></p> -->

<h1 align="center">DeltaForge</h1>

<p align="center"><strong>Stop watching charts. Start trading systems.</strong></p>

<p align="center">
  <a href="https://github.com/mimran-khan/deltaforge/actions/workflows/ci.yml"><img src="https://github.com/mimran-khan/deltaforge/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/mimran-khan/deltaforge/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python 3.9+"></a>
  <a href="https://hits.sh/github.com/mimran-khan/deltaforge"><img src="https://hits.sh/github.com/mimran-khan/deltaforge.svg?label=views&color=6e5494" alt="Views"></a>
</p>

<p align="center">
  <a href="https://github.com/mimran-khan/deltaforge/stargazers"><img src="https://img.shields.io/github/stars/mimran-khan/deltaforge?style=social" alt="GitHub Stars"></a>
  <a href="https://github.com/mimran-khan/deltaforge/network/members"><img src="https://img.shields.io/github/forks/mimran-khan/deltaforge?style=social" alt="GitHub Forks"></a>
  <a href="https://github.com/mimran-khan/deltaforge/watchers"><img src="https://img.shields.io/github/watchers/mimran-khan/deltaforge?style=social" alt="GitHub Watchers"></a>
  <a href="https://github.com/mimran-khan/deltaforge/releases"><img src="https://img.shields.io/github/v/release/mimran-khan/deltaforge?color=brightgreen&logo=github" alt="Release"></a>
  <a href="https://github.com/mimran-khan/deltaforge/issues"><img src="https://img.shields.io/github/issues/mimran-khan/deltaforge" alt="Issues"></a>
</p>

<p align="center">
  <a href="https://twitter.com/intent/tweet?text=Check%20out%20DeltaForge%20—%20autonomous%20Nifty%20options%20trading&url=https%3A//github.com/mimran-khan/deltaforge/"><img src="https://img.shields.io/badge/-Twitter-1DA1F2?style=flat-square&logo=twitter&logoColor=white" alt="Twitter"></a>
  <a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/mimran-khan/deltaforge/"><img src="https://img.shields.io/badge/-LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="https://www.reddit.com/submit?url=https%3A//github.com/mimran-khan/deltaforge/&title=DeltaForge%20—%20Autonomous%20Nifty%20Options%20Trading"><img src="https://img.shields.io/badge/-Reddit-FF4500?style=flat-square&logo=reddit&logoColor=white" alt="Reddit"></a>
  <a href="https://api.whatsapp.com/send?text=Check%20out%20DeltaForge%20—%20autonomous%20Nifty%20options%20trading%3A%20https%3A//github.com/mimran-khan/deltaforge/"><img src="https://img.shields.io/badge/-WhatsApp-25D366?style=flat-square&logo=whatsapp&logoColor=white" alt="WhatsApp"></a>
  <a href="https://t.me/share/url?url=https%3A//github.com/mimran-khan/deltaforge/&text=DeltaForge%20—%20Autonomous%20Nifty%20Options%20Trading"><img src="https://img.shields.io/badge/-Telegram-26A5E4?style=flat-square&logo=telegram&logoColor=white" alt="Telegram"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#why-deltaforge">Why DeltaForge</a> · <a href="#features">Features</a> · <a href="#documentation">Docs</a> · <a href="DISCLAIMER.md">Disclaimer</a>
</p>

---

### **Your strategy deserves better than a spreadsheet and a prayer.**

DeltaForge is an autonomous intraday options trading system for Nifty 50 on the NSE. You define the rules — it watches the market, finds entries, manages risk, sizes positions, and gets out. Paper or live, same engine, same signals, no surprises.

**Works with:** [Angel One SmartAPI](https://smartapi.angelone.in/) · [Slack](https://slack.com/) · [iMessage](https://support.apple.com/messages) · [Telegram](https://telegram.org/)

> **Disclaimer** — This is educational software, not financial advice. Options trading involves substantial risk of loss. See [DISCLAIMER.md](DISCLAIMER.md).

<!-- TODO: Add screenshots when ready -->
<!--
<p align="center">
  <table>
    <tr>
      <td align="center"><a href="docs/assets/dashboard.png"><img src="docs/assets/dashboard.png" width="400" alt="Dashboard"></a><br><strong>Real-time Dashboard</strong></td>
      <td align="center"><a href="docs/assets/terminal.png"><img src="docs/assets/terminal.png" width="400" alt="Terminal"></a><br><strong>CLI Trading Session</strong></td>
    </tr>
    <tr>
      <td align="center"><a href="docs/assets/slack.png"><img src="docs/assets/slack.png" width="400" alt="Slack Alerts"></a><br><strong>Slack Alerts</strong></td>
      <td align="center"><a href="docs/assets/backtest.png"><img src="docs/assets/backtest.png" width="400" alt="Backtest"></a><br><strong>Backtest Results</strong></td>
    </tr>
  </table>
</p>
-->

---

## Quick Start

**Prerequisites:** Python 3.9+, macOS or Linux, [Angel One](https://www.angelone.in/) trading account

```bash
git clone https://github.com/mimran-khan/deltaforge.git
cd deltaforge
make install              # venv + deps + registers `df` CLI
cp .env.example .env      # add your Angel One creds
df selftest               # verify everything works
```

Start paper trading:

```bash
$ df trade

[09:15:01] Engine started in PAPER mode
[09:15:02] WebSocket connected — streaming Nifty ticks
[09:22:15] SIGNAL  Stochastic Cross  BUY  conf=72
[09:22:15] RISK    9/9 gates passed
[09:22:16] ENTRY   NIFTY 24500 CE @ ₹185.40  qty=50
[10:14:33] EXIT    ₹213.20  P&L +₹1,390  (+15.0%)
[15:30:00] EOD     3 trades | Net P&L +₹3,820 | Capital ₹13,820
```

Go live when you're ready:

```bash
df trade --live           # requires confirmation
```

---

## Why DeltaForge

Most retail options tools are either overpriced terminals you can't customize, or fragile scripts taped together with cron jobs. DeltaForge is neither.

| | Manual / Spreadsheet | Paid Terminals | DeltaForge |
|:--|:--|:--|:--|
| **Execution** | You click buttons all day | Alerts, you still click | Fully autonomous — entry to exit |
| **Risk management** | "I'll stop at 5% loss" (you won't) | Basic SL/target | 9-gate filter + independent kill switch |
| **Position sizing** | Fixed lots, guesswork | Basic calculators | Auto-compounds with equity, scales down on drawdown |
| **Backtesting** | Copy-paste into Excel | Separate tool, different logic | Same engine — backtest = paper = live |
| **Alerts** | Check the app | Email (if lucky) | Instant Slack, iMessage, or Telegram |
| **Cost** | Free (your time isn't) | ₹2,000–₹15,000/mo | Free and open source |

---

## Features

| **Feature** | **Description** |
|---|---|
| 🧠 **Multi-Strategy Engine** | Trend Ride, Pullback-in-Trend, Supertrend Flip, Stochastic Cross — all run in parallel with regime-aware confidence scoring |
| 🌐 **Multi-Asset Trading** | Nifty 50 options (NSE) + MCX Crude Oil Mini + CDS USDINR futures — each with independent risk pools |
| 🛡️ **Adaptive Risk System** | Daily loss limits (10%), consecutive-loss brakes, drawdown tiers, adaptive mode that adjusts confidence/lots in real-time |
| 🔴 **Kill Switch** | Independent watchdog process. Halts trading on breach, manual override via CLI or dashboard. Works even if the main engine hangs. |
| 📈 **Compound Sizing** | Scales lots with capital growth. Position sizing adapts to equity curve with configurable per-lot capital allocation. |
| 📊 **Live Dashboard** | FastAPI + WebSocket — capital, trades, risk status, halt toggle. All at `localhost:8900` |
| 💬 **Instant Alerts** | Slack (recommended), iMessage (macOS), or Telegram — entries, exits, errors, end-of-day P&L reports |
| 🔄 **Paper ↔ Live Parity** | Identical code path across backtest, paper, and live. If it works in paper, it works in production. Zero divergence. |
| 📉 **Walk-Forward Backtest** | 100+ day replays with production engine. Realistic costs (brokerage + STT + slippage). Compound lot sizing. |
| ⏰ **Automated Scheduling** | macOS launchd agent for daily auto-start, code sync, health monitoring, and session management |

---

## What It Does

```mermaid
graph LR
    A["📡 Angel One WebSocket"] --> B["🕯 CandleBuilder"]
    B --> C["🧠 MultiStrategyEngine"]
    C --> D{"🛡️ RiskEngine + AdaptiveMode"}
    D -->|"✅ pass"| E["💰 PositionManager"]
    D -->|"❌ reject"| F(("Skip"))
    E --> G["📊 Dashboard"]
    E --> H["💬 Alerts"]
    I["🔴 Kill Switch"] -.->|"halt"| D
    J["📡 Futures Feed"] --> K["🌐 MultiAssetEngine"]
    K --> D
```

---

## Usage

```bash
# Trading
df trade                    # paper mode (default)
df trade --live             # live (confirmation required)
df trade -v                 # verbose / debug output

# Monitoring
df status                   # capital, drawdown, risk state
df logs -f                  # follow today's log
df db stats                 # per-strategy performance
df db summary               # today's trade summary

# Backtesting
df backtest --days 180      # default capital ₹10,000
df backtest --days 365 --capital 50000

# Testing & Verification
df selftest                 # pre-flight engine check
df broker                   # test broker connection
df alert "test"             # verify alert channel

# Dashboard
make dashboard              # web UI at localhost:8900
make run                    # trading + dashboard together
```

---

## Configuration

Copy [`.env.example`](.env.example) and fill in your credentials:

```ini
# Angel One SmartAPI (required)
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret

# Trading
TRADING_MODE=paper
STARTING_CAPITAL=10000

# Risk Management
DAILY_LOSS_LIMIT_PCT=10          # halt after 10% daily loss
CAPITAL_PER_LOT=3000             # lot sizing aggressiveness

# Alerts — choose: "slack", "imessage", or "telegram"
ALERT_METHOD=slack
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_CHANNEL_ID=C0XXXXXXXXX
```

All parameters live in [`config/settings.py`](config/settings.py) and are overridable via `.env`. See [`.env.example`](.env.example) for every configurable option.

---

## Paper vs Live

| | Paper | Live |
|---|---|---|
| Signal generation | Identical engine | Identical engine |
| Order execution | Simulated (premium model) | Angel One SmartAPI |
| Costs | Realistic (brokerage + STT + slippage) | Actual |
| Risk gates | All 9 enforced | All 9 enforced |

**Same code. Same signals. Same risk.** If it works in paper, it works in live.

---

## Documentation

| Topic | Description |
|---|---|
| [Strategies & Indicators](docs/strategy.md) | How each strategy generates signals, indicator reference, options model |
| [Risk Model & Kill Switch](docs/risk.md) | 9-gate system, drawdown tiers, real-time monitoring, kill switch |
| [Dashboard & API](docs/dashboard.md) | REST endpoints, WebSocket, security configuration, alerts |
| [Architecture & Design](docs/architecture.md) | Data flow, system components, project structure, design decisions |

---

## Roadmap

| Timeline | Milestone |
|---|---|
| **Shipped** | Multi-strategy engine, adaptive risk system, multi-asset (Nifty + MCX + CDS), compound sizing, automated daily scheduling, walk-forward backtesting, Slack/iMessage/Telegram alerts, FastAPI dashboard |
| **Next** | BankNifty support, strategy marketplace, ML-based regime detection |
| **Future** | Mobile companion alerts, cloud sync, community strategies, broker-agnostic execution |

---

## Built With

- [Angel One SmartAPI](https://smartapi.angelone.in/) — Broker connectivity & WebSocket market data
- [FastAPI](https://fastapi.tiangolo.com/) — Dashboard HTTP layer
- [Loguru](https://github.com/Delgan/loguru) — Structured logging
- [Pandas](https://pandas.pydata.org/) — Indicator computation & data analysis
- [SQLite](https://www.sqlite.org/) — Trade persistence
- [WebSocket](https://websockets.readthedocs.io/) — Real-time market feed

---

## Contributing

We're building the trading system we wish existed — help us make it better.

**Contribute:** New strategies, risk improvements, broker integrations, indicators, bug fixes, docs

- [Contributing Guide](CONTRIBUTING.md)
- [Report Bug](https://github.com/mimran-khan/deltaforge/issues)
- [Request Feature](https://github.com/mimran-khan/deltaforge/issues)
- [Good First Issues](https://github.com/mimran-khan/deltaforge/labels/good%20first%20issue)

---

## Testing

```bash
make test                   # full test suite
make test-e2e               # end-to-end only
make test-component         # component tests
```

---

## License

MIT — see [LICENSE](LICENSE). Full trading disclaimer in [DISCLAIMER.md](DISCLAIMER.md).

---

<p align="center"><strong>Your strategy deserves better than a spreadsheet and a prayer.</strong></p>

<!-- Uncomment when repo is public -->
<!--
<p align="center">
  <a href="https://star-history.com/#mimran-khan/deltaforge&Date">
    <img src="https://api.star-history.com/svg?repos=mimran-khan/deltaforge&type=Date" alt="Star History Chart" width="600">
  </a>
</p>

<p align="center">
  <a href="https://repobeats.axiom.co">
    <img src="https://repobeats.axiom.co/api/embed/YOUR_REPO_ID.svg" alt="Repobeats analytics" width="600">
  </a>
</p>
-->

<p align="center">⭐ <strong>Star</strong> · 🔄 <strong>Share</strong> · 🤝 <strong>Contribute</strong></p>
