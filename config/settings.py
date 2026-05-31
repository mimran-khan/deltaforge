"""Centralized configuration -- single source of truth.

Every parameter used by any module is defined here.
No hardcoded values anywhere else in the codebase.

Architecture: modular monolith with event-driven data flow.
Reference: FIA Automated Trading Risk Controls, NautilusTrader patterns.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# ═══════════════════════════════════════════════════════════════════
#  BROKER: Angel One SmartAPI
# ═══════════════════════════════════════════════════════════════════
ANGEL_API_KEY = os.getenv("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
ANGEL_PASSWORD = os.getenv("ANGEL_PASSWORD", "")
ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "")

# ═══════════════════════════════════════════════════════════════════
#  ALERTS: Telegram
# ═══════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ═══════════════════════════════════════════════════════════════════
#  TRADING MODE
# ═══════════════════════════════════════════════════════════════════
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"

# ═══════════════════════════════════════════════════════════════════
#  INSTRUMENT SPECS (verify against NSE on startup)
# ═══════════════════════════════════════════════════════════════════
NIFTY_LOT_SIZE = 75
BANKNIFTY_LOT_SIZE = 30
NIFTY_EXPIRY_DAY = 1       # Tuesday = 1 (Monday=0, Sunday=6)

# ═══════════════════════════════════════════════════════════════════
#  CAPITAL & POSITION SIZING
# ═══════════════════════════════════════════════════════════════════
STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "10000"))

LOT_TIERS = [
    # (min_capital, max_lots, instrument)
    (10_000,  1, "NIFTY"),
    (25_000,  2, "NIFTY"),
    (50_000,  3, "NIFTY"),
    (100_000, 5, "NIFTY"),
]

CAPITAL_DEPLOY_PCT = 60.0       # deploy 60% per trade (down from 80%)
COMPOUND_DAILY = True
MAX_LOTS = 1                    # fixed 1 lot until capital > Rs 25,000

# ═══════════════════════════════════════════════════════════════════
#  RISK MANAGEMENT (3-layer: pre-trade, real-time, post-trade)
# ═══════════════════════════════════════════════════════════════════

# -- Pre-trade gates --
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "1"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "2"))
MIN_CAPITAL_TO_TRADE = 3000     # halt if capital drops below this

# -- Daily/weekly loss limits --
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "5"))
WEEKLY_LOSS_LIMIT_PCT = 10.0

# -- Drawdown tiers (path-dependent) --
DRAWDOWN_HALFSIZE_PCT = 10.0    # reduce to 50% size if DD > 10%
DRAWDOWN_HALT_PCT = 15.0        # full halt if DD > 15% from peak

# -- Volatility circuit breaker --
MAX_VIX_THRESHOLD = 18.0        # skip if India VIX > 18
VIX_SPIKE_HALT_PCT = 15.0       # halt if VIX jumps > 15% intraday

# -- Expiry day --
SKIP_EXPIRY_DAY = True           # no trading on expiry day (Tuesday)

# ═══════════════════════════════════════════════════════════════════
#  CONFLUENCE ENGINE
# ═══════════════════════════════════════════════════════════════════
CONFLUENCE_THRESHOLD = float(os.getenv("CONFLUENCE_THRESHOLD", "55"))
MIN_STRENGTH = os.getenv("MIN_STRENGTH", "STRONG")
CONFLUENCE_HOLD_CANDLES = 12     # 60 minutes (where edge was proven)

CONFLUENCE_CATEGORY_WEIGHTS = {
    "trend": 1.0,
    "momentum": 1.2,
    "volatility": 0.8,
    "volume": 0.9,
    "trend_strength": 1.1,
    "structure": 0.7,
    "candlestick": 0.6,
    "statistical": 0.9,
    "divergence": 1.3,
    "htf": 1.4,
    "derivative": 0.8,
    "candle_struct": 0.5,
}

# ═══════════════════════════════════════════════════════════════════
#  PREMIUM / OPTIONS MODEL
# ═══════════════════════════════════════════════════════════════════

# OTM option model (validated for Rs 10,000 capital)
PREMIUM_BASE = 25.0             # OTM Nifty option premium
PREMIUM_DELTA = 0.20            # OTM delta
PREMIUM_THETA_PER_CANDLE = 0.04 # theta decay per 5-min candle
PREMIUM_SL_PCT = 30.0           # stop loss as % of premium
PREMIUM_TARGET_PCT = 40.0       # target gain as % of premium
PREMIUM_TARGET_POINTS = 10      # target gain Rs per unit (OTM)

# Strike selection
STRIKE_MIN_DTE = 3              # minimum days to expiry
STRIKE_MAX_DTE = 5              # maximum days to expiry
STRIKE_OFFSET = 1               # 1-OTM strike (cheaper premium)

# Execution costs (per lot, realistic for NFO)
BROKERAGE_PER_ORDER = 20.0      # Rs 20 flat per order
STT_PCT = 0.0625                # STT on premium (buy side)
STAMP_DUTY_PCT = 0.003          # stamp duty
SLIPPAGE_POINTS = 1.0           # Rs 1.0 per unit slippage assumption

# ═══════════════════════════════════════════════════════════════════
#  TIMING (IST)
# ═══════════════════════════════════════════════════════════════════
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
ENTRY_START = "09:45"           # no entry before 9:45 (let ORB settle)
ENTRY_END = "13:00"             # no new entry after 13:00
NO_NEW_ENTRY_AFTER = "13:00"    # alias for backward compat
SQUARE_OFF_TIME = "15:15"       # hard exit
SESSION_LOGIN_TIME = "08:30"
INSTRUMENT_DOWNLOAD_TIME = "08:45"
EOD_REPORT_TIME = "15:35"
SESSION_LOGOUT_TIME = "23:55"

# ═══════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "trades.db"
CAPITAL_FILE = DATA_DIR / "capital.json"
INSTRUMENTS_FILE = DATA_DIR / "instruments.json"

LOG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  LEGACY ALIASES (backward compat for strategies)
# ═══════════════════════════════════════════════════════════════════
ORB_WINDOW_MINUTES = 15
ORB_MAX_RANGE_POINTS = 250
ORB_VOLUME_MULTIPLIER = 1.5
ORB_ENTRY_START = "09:30"
ORB_ENTRY_END = "11:30"
ORB_RR_RATIO = 1.5
VWAP_EMA_FAST = 9
VWAP_EMA_SLOW = 20
VWAP_RSI_PERIOD = 14
VWAP_RSI_LONG_THRESHOLD = 50
VWAP_RSI_SHORT_THRESHOLD = 50
VWAP_VOLUME_MULTIPLIER = 1.5
VWAP_ENTRY_START = "09:30"
VWAP_ENTRY_END = "14:00"
