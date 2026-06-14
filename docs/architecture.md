# Architecture

## Data Flow

```mermaid
graph LR
    A["Angel One WebSocket"] --> B["CandleBuilder<br/>(tick → OHLCV)"]
    B --> C["MultiStrategyEngine<br/>(5 strategies)"]
    C --> D{"RiskEngine<br/>(9 gates)"}
    D -->|"pass"| E["PositionManager"]
    D -->|"reject"| F(("Skip"))
    E --> G["Angel One API<br/>(live) / Simulated (paper)"]
    H["Kill Switch<br/>(independent process)"] -.->|"halt"| D
    I["CapitalTracker"] -.->|"sizing"| E
    J["PerformanceDB<br/>(SQLite)"] -.->|"record"| E
```

## Strategy Pipeline

```mermaid
graph TD
    A["Closed Candle"] --> B["Indicator Computation"]
    B --> C["Stochastic Cross"]
    B --> D["Pullback-in-Trend"]
    B --> E["Supertrend Flip"]
    B --> F["VWAP Bias"]
    B --> G["RSI Mean Reversion"]
    C --> H{"Best Signal<br/>(highest confidence)"}
    D --> H
    E --> H
    F --> H
    G --> H
    H -->|"conf ≥ 50"| I["RiskEngine"]
    H -->|"conf < 50"| J(("Discard"))
```

## System Components

```mermaid
graph TB
    subgraph "Market Data"
        MF["market_feed.py<br/>WebSocket (SmartWebSocketV2)"]
        CB["candle_builder.py<br/>Tick → OHLCV"]
        IE["indicators_extended.py<br/>HTF (15m) indicators"]
    end

    subgraph "Signal Generation"
        MSE["multi_strategy_engine.py<br/>5 strategies in parallel"]
        PM["premium_model.py<br/>Black-Scholes delta/theta"]
        SS["strike_selector.py<br/>ATM strike selection"]
    end

    subgraph "Risk & Execution"
        RE["risk_engine.py<br/>9-gate pre-trade filter"]
        CT["capital_tracker.py<br/>Equity + compound sizing"]
        KS["kill_switch.py<br/>Independent watchdog"]
        SD["shock_detector.py<br/>Extreme move breaker"]
        POS["position_manager.py<br/>Lifecycle + trailing SL"]
    end

    subgraph "Infrastructure"
        BR["broker.py<br/>Angel One SmartAPI"]
        DB["performance_db.py<br/>SQLite trades"]
        AL["alerts/<br/>Slack / iMessage / Telegram"]
        DS["dashboard/<br/>FastAPI + WebSocket"]
    end

    MF --> CB --> IE --> MSE
    MSE --> RE
    RE --> POS
    POS --> BR
    POS --> DB
    PM --> POS
    SS --> POS
    CT --> RE
    KS -.-> RE
    SD -.-> KS
    POS --> AL
    DB --> DS
```

## Trading Session Lifecycle

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant E as TradingEngine
    participant W as WebSocket
    participant M as MultiStrategyEngine
    participant R as RiskEngine
    participant B as Broker

    S->>E: start_session()
    E->>W: connect()
    W-->>E: tick stream

    loop Every closed candle
        E->>M: evaluate(candle)
        M-->>E: Signal(direction, confidence)
        E->>R: check_gates(signal)
        alt All gates pass
            R-->>E: APPROVED
            E->>B: place_order()
            B-->>E: order_id
        else Any gate fails
            R-->>E: REJECTED (gate_name)
        end
    end

    S->>E: end_session()
    E->>W: disconnect()
```

## Project Structure

```
deltaforge/
├── cli.py                        # CLI entry point (`df` command)
├── main.py                       # Python entry point
├── config/
│   ├── settings.py               # All parameters (env-overridable)
│   ├── logging.py                # Console + file + JSON logging
│   └── holidays.json             # NSE holiday calendar
├── engine/
│   ├── trading_engine.py         # Main trading loop
│   ├── multi_strategy_engine.py  # 5-strategy signal generator
│   ├── candle_builder.py         # Tick → OHLCV
│   ├── market_feed.py            # WebSocket feed
│   ├── premium_model.py          # Black-Scholes model
│   ├── indicators_extended.py    # HTF indicators
│   ├── broker.py                 # Angel One wrapper
│   └── strike_selector.py        # ATM strike selection
├── risk/
│   ├── risk_engine.py            # 9-gate filter
│   ├── capital_tracker.py        # Equity tracking + sizing
│   ├── kill_switch.py            # Watchdog process
│   └── shock_detector.py         # Circuit breaker
├── execution/
│   └── position_manager.py       # Position lifecycle
├── persistence/
│   └── performance_db.py         # SQLite trade DB
├── alerts/                       # Slack / iMessage / Telegram
├── automation/
│   └── daily_scheduler.py        # Session lifecycle
├── dashboard/                    # FastAPI + WebSocket UI
├── backtest/                     # 30+ analysis scripts
└── tests/                        # e2e, component, integration
```

## Key Design Decisions

**Single engine for all modes** — `MultiStrategyEngine` runs identically in backtest, paper, and live. No code divergence between test and production.

**File-based kill switch** — uses filesystem signals rather than IPC, so it works even when the main process is hung.

**Atomic capital state** — `capital.json` is written via tmp file + rename to prevent corruption on crash.

**Compound position sizing** — lot count scales with equity (1 lot per Rs 6,000 of deployable capital) but is throttled by drawdown tiers.

## Logging

| Sink | Format | Level | Location |
|------|--------|-------|----------|
| Console | Colored, human-readable | INFO | stderr |
| File | Timestamped, module/function/line | DEBUG | `logs/trading_YYYY-MM-DD.log` |
| JSON | Machine-parseable JSONL | DEBUG | `logs/json/trading_YYYY-MM-DD.jsonl` |

Daily rotation, 30-day retention, gzip compression. Thread-safe via `enqueue=True`.
