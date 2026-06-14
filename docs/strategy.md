# Trading Strategy

## Multi-Strategy Engine

DeltaForge runs five strategies in parallel. Each generates a signal with a confidence score; the risk engine then decides whether to act.

### Stochastic Cross

- Stochastic %K crosses back from extreme zone (<25 or >75)
- EMA(9) vs EMA(21) confirms trend direction
- HTF RSI boosts confidence when aligned

### Pullback-in-Trend

- 15-minute RSI shows moderate+ trend (|RSI-50| >= 10)
- Dead-zone filter [0,10) skips choppy markets
- 4 LTF oscillators (RSI, Stochastic, CCI, Williams %R) confirm pullback exhaustion
- Confidence >= 50 required

### Supertrend Flip

- Supertrend(10,3) direction change with EMA + ADX confirmation
- Fast Supertrend(7,2) used as confidence boost when both agree
- ADX >= 15 required for directional conviction

### VWAP Bias (confidence booster)

- Intraday VWAP computed with daily reset
- Signals aligned with VWAP position get +5 confidence

### RSI Mean Reversion (overflow strategy)

- Buy oversold dips (RSI < 38) in uptrends, sell overbought (RSI > 62) in downtrends
- Requires HTF trend confirmation and price vs EMA20 alignment

## Indicators

| Category | Indicators |
|----------|-----------|
| Trend | EMA(9, 20, 21, 50), Supertrend(10,3), Supertrend(7,2) |
| Momentum | RSI(14), Stochastic(14,3), CCI(20), Williams %R(14) |
| Volume | VWAP (intraday), Volume SMA(20) |
| Volatility | ATR(14), Bollinger Bands %B |
| Trend Strength | ADX(14) |
| Multi-TF | 15-minute RSI via candle resampling |

## Options Model

- **Instrument**: Nifty 50 weekly slightly-ITM options (CE for LONG, PE for SHORT)
- **Premium Model**: Black-Scholes delta approximation (delta=0.70)
- **Dynamic Theta**: Scales proportionally to Nifty index level
- **Stop Loss**: 15% of entry premium
- **Targets**: Strategy-specific multipliers via premium model
- **Exit**: Time-based (24 candles = 2 hours max hold) if target not hit
- **Costs**: Realistic brokerage (Rs 20/order), STT, stamp duty, slippage (Rs 1.5/unit)
