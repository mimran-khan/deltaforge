# Risk Management

DeltaForge implements a 3-layer risk model: pre-trade gates, real-time monitoring, and position sizing.

## Gate Flow

```mermaid
graph TD
    S["Incoming Signal"] --> G0{"Gate 0<br/>System halted?"}
    G0 -->|"no"| G1{"Gate 1<br/>Capital ≥ ₹3,000?"}
    G0 -->|"yes"| R0(("REJECT"))
    G1 -->|"yes"| G2{"Gate 2<br/>Daily loss < 25%?"}
    G1 -->|"no"| H1["HALT"]
    G2 -->|"yes"| G3{"Gate 3<br/>Weekly loss < 50%?"}
    G2 -->|"no"| H2["HALT"]
    G3 -->|"yes"| G4{"Gate 4<br/>Consec. losses < 5?"}
    G3 -->|"no"| H3["HALT"]
    G4 -->|"yes"| G5{"Gate 5<br/>Drawdown < 35%?"}
    G4 -->|"no"| H4["HALT"]
    G5 -->|"yes"| G55{"Gate 5.5<br/>VIX ≤ 18?"}
    G5 -->|"no"| H5["HALT"]
    G55 -->|"yes"| G6{"Gate 6<br/>Not expiry day?"}
    G55 -->|"no"| R55(("REJECT"))
    G6 -->|"yes"| G7{"Gate 7<br/>Within entry window?"}
    G6 -->|"no"| R6(("REJECT"))
    G7 -->|"yes"| G8{"Gate 8<br/>Confidence ≥ 50?"}
    G7 -->|"no"| R7(("REJECT"))
    G8 -->|"yes"| PASS["✅ APPROVED"]
    G8 -->|"no"| R8(("REJECT"))

    style PASS fill:#22c55e,color:#fff
    style H1 fill:#ef4444,color:#fff
    style H2 fill:#ef4444,color:#fff
    style H3 fill:#ef4444,color:#fff
    style H4 fill:#ef4444,color:#fff
    style H5 fill:#ef4444,color:#fff
```

## Gate Reference

| Gate | Check | On Fail |
|------|-------|---------|
| 0 | System not halted | Reject |
| 1 | Capital ≥ ₹3,000 | **HALT** |
| 2 | Daily loss < 25% of day-start capital | **HALT** |
| 3 | Weekly loss < 50% of starting capital | **HALT** |
| 4 | Consecutive losses < 5 | **HALT** |
| 5 | Drawdown < 35% from peak | **HALT** |
| 5.5 | India VIX ≤ 18 | Reject |
| 6 | Not Nifty expiry day (if configured) | Reject |
| 7 | Within entry window | Reject |
| 8 | Signal confidence ≥ 50 | Reject |

**Reject** = signal skipped, system stays active. **HALT** = trading stopped for session.

## Real-Time Monitoring

```mermaid
graph LR
    T["Every Tick"] --> DL{"Daily loss<br/>limit breached?"}
    DL -->|"no"| DD{"Drawdown<br/>limit breached?"}
    DL -->|"yes"| HALT["HALT trading"]
    DD -->|"no"| CM{"Capital<br/>below minimum?"}
    DD -->|"yes"| HALT
    CM -->|"no"| OK["Continue"]
    CM -->|"yes"| HALT
    KS["Kill Switch<br/>(independent)"] -.->|"file signal"| HALT
```

## Position Sizing

```mermaid
graph TD
    C["Current Capital"] --> D{"Drawdown<br/>level?"}
    D -->|"< 20%"| FULL["Full size<br/>1 lot / ₹6,000"]
    D -->|"20–35%"| HALF["Half size"]
    D -->|"≥ 35%"| STOP["HALT — no trades"]
    FULL --> CAP["Cap at 20 lots max"]
    HALF --> CAP

    style FULL fill:#22c55e,color:#fff
    style HALF fill:#f59e0b,color:#fff
    style STOP fill:#ef4444,color:#fff
```

- **Deploy ratio**: 60% of total capital is deployable
- **Lot sizing**: 1 lot per ₹6,000 of deployable capital
- **Max position**: 20 lots
- **Atomic state**: `capital.json` written via tmp + rename (crash-safe)

## Kill Switch

The kill switch is an independent process that monitors system health via file-based signaling.

```mermaid
graph LR
    subgraph "Main Process"
        TE["TradingEngine"] --> RE["RiskEngine"]
    end

    subgraph "Watchdog (independent)"
        KS["KillSwitch"] -->|"checks"| HF["halt file on disk"]
    end

    HF -->|"read"| RE
    KS -->|"writes"| HF
    CLI["df halt"] -->|"writes"| HF
    DASH["Dashboard /api/halt"] -->|"writes"| HF
```

Triggers:
- Automatic: drawdown or loss breaches
- Manual: `df halt`, dashboard toggle, or creating a halt file on disk
