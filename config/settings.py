"""Centralized configuration -- single source of truth.

Every tunable parameter is defined here. Secrets and user-facing knobs
are loaded from environment variables (via .env); domain constants that
rarely change are defined as module-level values.

See .env.example for the full list of overridable settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Homebrew Python on macOS often lacks a default CA bundle; certifi fixes Slack/API TLS.
if not os.environ.get("SSL_CERT_FILE"):
    try:
        import certifi

        _ca_bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", _ca_bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca_bundle)
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════
#  BROKER: Angel One SmartAPI
# ═══════════════════════════════════════════════════════════════════
ANGEL_API_KEY = os.getenv("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
ANGEL_PASSWORD = os.getenv("ANGEL_PASSWORD", "")
ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "")

# ═══════════════════════════════════════════════════════════════════
#  ALERTS
# ═══════════════════════════════════════════════════════════════════
ALERT_METHOD = os.getenv("ALERT_METHOD", "slack")  # "slack", "imessage", or "telegram"
IMESSAGE_RECIPIENT = os.getenv("IMESSAGE_RECIPIENT", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "")

# ═══════════════════════════════════════════════════════════════════
#  TRADING MODE
# ═══════════════════════════════════════════════════════════════════
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"

# ═══════════════════════════════════════════════════════════════════
#  MULTI-ASSET (Gold, Crude, Currency)
#  When False, the existing Nifty-only engine runs unchanged.
#  When True, the MultiAssetEngine runs alongside (separate process).
# ═══════════════════════════════════════════════════════════════════
MULTI_ASSET_ENABLED = os.getenv("MULTI_ASSET_ENABLED", "true").lower() in ("1", "true", "yes")

# ═══════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8900"))
DASHBOARD_CORS_ORIGINS = os.getenv("DASHBOARD_CORS_ORIGINS", "")
DASHBOARD_API_TOKEN = os.getenv("DASHBOARD_API_TOKEN", "")

# ═══════════════════════════════════════════════════════════════════
#  INSTRUMENT SPECS (verify against NSE on startup)
# ═══════════════════════════════════════════════════════════════════
NIFTY_LOT_SIZE = 65            # NSE revised Jan 2026 (was 75)
BANKNIFTY_LOT_SIZE = 30
NIFTY_EXPIRY_DAY = 3       # Thursday = 3 (Monday=0, Sunday=6)
NIFTY_INDEX_TOKEN = "99926000"
BANKNIFTY_INDEX_TOKEN = "99926009"

# ═══════════════════════════════════════════════════════════════════
#  CAPITAL & POSITION SIZING
# ═══════════════════════════════════════════════════════════════════
STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "10000"))
FUTURES_STARTING_CAPITAL = float(os.getenv("FUTURES_STARTING_CAPITAL", "50000"))

CAPITAL_PER_LOT = 20_000        # 1 lot per Rs 20,000 — balanced: 7 lots gives good PnL with HARD_CAP protection
MAX_LOTS_CAP = 50               # raised from 20 -- optimizer: 50 lots = +7% geo daily
MAX_LOSS_PER_TRADE = int(os.getenv("MAX_LOSS_PER_TRADE", "8000"))

DTE_LOT_CAPS = {1: 4, 2: 6}    # gamma protection: DTE<=1 max 4 lots, DTE<=2 max 6 lots

MAX_TRADES_PER_DIRECTION = int(os.getenv("MAX_TRADES_PER_DIRECTION", "2"))
DIRECTION_LOSS_CAP = int(os.getenv("DIRECTION_LOSS_CAP", "12000"))

CAPITAL_DEPLOY_PCT = float(os.getenv("CAPITAL_DEPLOY_PCT", "100"))
COMPOUND_DAILY = True

MAX_SIMULTANEOUS_POSITIONS = int(os.getenv("MAX_SIMULTANEOUS_POSITIONS", "3"))
TRAIL_TRIGGER_PCT = float(os.getenv("TRAIL_TRIGGER_PCT", "5"))
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "2"))
SCAN_WARMUP_BARS = int(os.getenv("SCAN_WARMUP_BARS", "50"))
LATE_START_CATCHUP_ENABLED = os.getenv("LATE_START_CATCHUP_ENABLED", "true").lower() in ("1", "true", "yes")
LATE_START_CATCHUP_MINUTES = int(os.getenv("LATE_START_CATCHUP_MINUTES", "15"))  # after ENTRY_START
LATE_START_CATCHUP_DRY_RUN = os.getenv("LATE_START_CATCHUP_DRY_RUN", "false").lower() in ("1", "true", "yes")
SHOCK_THRESHOLD_PCT = float(os.getenv("SHOCK_THRESHOLD_PCT", "1.5"))
SHOCK_LOOKBACK_BARS = int(os.getenv("SHOCK_LOOKBACK_BARS", "3"))

# ═══════════════════════════════════════════════════════════════════
#  RISK MANAGEMENT (3-layer: pre-trade, real-time, post-trade)
# ═══════════════════════════════════════════════════════════════════

# -- Pre-trade gates --
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))
MIN_CAPITAL_TO_TRADE = 3000     # halt if capital drops below this

# -- Daily/weekly loss limits --
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "10"))
DAILY_PROFIT_TARGET_PCT = float(os.getenv("DAILY_PROFIT_TARGET_PCT", "35"))  # stop entries after 35% daily gain
WEEKLY_LOSS_LIMIT_PCT = float(os.getenv("WEEKLY_LOSS_LIMIT_PCT", "50"))

# -- Drawdown tiers (path-dependent) --
DRAWDOWN_HALFSIZE_PCT = 20.0    # reduce to 50% size if DD > 20% from peak
DRAWDOWN_HALT_PCT = 35.0        # full halt if DD > 35% from peak

# -- Volatility circuit breaker --
MAX_VIX_THRESHOLD = float(os.getenv("MAX_VIX_THRESHOLD", "18"))  # skip if India VIX > 18
VIX_SPIKE_HALT_PCT = 15.0       # halt if VIX jumps > 15% intraday

# -- Expiry day --
SKIP_EXPIRY_DAY = False          # allow trading on expiry day (Tuesday)

# ═══════════════════════════════════════════════════════════════════
#  RUNNER MODE (let winners run when trend is strong)
# ═══════════════════════════════════════════════════════════════════
TREND_RUNNER_ENABLED = True
TREND_RUNNER_ADX_MIN = 38.0          # min ADX to activate runner on TGT hit (lowered from 40 for live/backtest parity)
TREND_RUNNER_ADX_EXIT = 35.0         # exit runner when ADX drops below this
TREND_RUNNER_TRAIL_PCT = 8.0         # trail 8% below peak premium in runner mode
TREND_RUNNER_MAX_BARS = 12           # max extra bars in runner mode (1 hour)
TREND_RUNNER_STRATEGIES = ["TREND_RIDE", "PULLBACK", "SUPERTREND", "STOCH_CROSS"]
TREND_RUNNER_CUTOFF_TIME = "15:00"   # don't activate runner after this time (extended to capture late rallies)

# ═══════════════════════════════════════════════════════════════════
#  ADAPTIVE MODE (intra-day performance-based parameter modulation)
# ═══════════════════════════════════════════════════════════════════
ADAPTIVE_MODE_ENABLED = False   # optimizer: disabling adaptive = +2.3% geo daily (was throttling good signals)
ADAPTIVE_AGGRESSIVE_PNL_PCT = 5.0    # promote to AGGRESSIVE above +5% daily
ADAPTIVE_AGGRESSIVE_CONSEC_WINS = 2  # need 2+ consecutive wins
ADAPTIVE_AGGRESSIVE_MIN_WR = 60.0    # need 60%+ win rate today
ADAPTIVE_DEFENSIVE_PNL_PCT = -7.0    # drop to DEFENSIVE below -7% daily
ADAPTIVE_DEFENSIVE_CONSECUTIVE = 2   # or 2 consecutive losses
ADAPTIVE_DEFENSIVE_WR = 40.0         # or WR < 40% with enough trades
ADAPTIVE_DEFENSIVE_WR_MIN_TRADES = 3 # WR gate needs 3+ trades to activate
ADAPTIVE_HALT_CONSECUTIVE = 3        # HALT after 3 consecutive losses
ADAPTIVE_HALT_LOSS_PCT = -10.0       # HALT below -10% daily

# ═══════════════════════════════════════════════════════════════════
#  PULLBACK ENGINE (replaces confluence engine as primary alpha)
# ═══════════════════════════════════════════════════════════════════
PULLBACK_MIN_CONFIDENCE = 68     # raised from 50 -- 70-79 band has 45.6% WR per analysis
PULLBACK_HOLD_CANDLES = 20       # max hold ~100 min (strategy-specific overrides in backtest)
PARTIAL_PROFIT_PCT = float(os.getenv("PARTIAL_PROFIT_PCT", "25"))
PULLBACK_MAX_SIGNALS_PER_DAY = int(os.getenv("PULLBACK_MAX_SIGNALS_PER_DAY", "5"))

# Legacy confluence (kept for monitoring/logging)
CONFLUENCE_THRESHOLD = float(os.getenv("CONFLUENCE_THRESHOLD", "55"))
MIN_STRENGTH = os.getenv("MIN_STRENGTH", "STRONG")
CONFLUENCE_HOLD_CANDLES = 12     # 60 minutes

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

# ATM option model (higher delta = better premium tracking)
PREMIUM_BASE = 100.0            # ATM Nifty option premium (~Rs 100)
PREMIUM_DELTA = 0.80            # ITM delta -- optimizer: 0.80 > 0.70 (+2% geo daily)
PREMIUM_THETA_PER_CANDLE = 0.15 # reduced theta -- optimizer: 0.15 halves decay pressure
PREMIUM_SL_PCT = float(os.getenv("PREMIUM_SL_PCT", "3"))
PREMIUM_TARGET_PCT = float(os.getenv("PREMIUM_TARGET_PCT", "50"))  # 50% target -- optimizer validated
PREMIUM_TARGET_POINTS = 0       # point-based TP disabled (use pct)

# Dynamic theta model: scales theta proportionally to Nifty level
THETA_REFERENCE_LEVEL = 24_000  # Nifty level where theta = THETA_BASE
THETA_BASE = 0.15               # halved from 0.30 -- optimizer validated lower decay


def get_scaled_theta(nifty_price: float) -> float:
    """Scale option theta decay proportionally to index level.

    At Nifty 24,000: theta = 0.30/bar (base).
    At Nifty 20,000: theta = 0.25/bar (lower absolute moves).
    """
    if nifty_price <= 0:
        return THETA_BASE
    return THETA_BASE * (nifty_price / THETA_REFERENCE_LEVEL)

# Strike selection
STRIKE_MIN_DTE = 3              # minimum days to expiry
STRIKE_MAX_DTE = 5              # maximum days to expiry
STRIKE_OFFSET = 0               # ATM strike (offset=0)

# Execution costs (realistic for NFO -- industry standard values)
BROKERAGE_PER_ORDER = 20.0      # Rs 20 flat per order (discount brokers)
STT_SELL_PCT = 0.05             # STT 0.05% on sell-side premium turnover
EXCHANGE_TXN_PCT = 0.05         # exchange + SEBI + stamp + GST combined
BID_ASK_SPREAD = 0.30           # Rs 0.30/unit ATM bid-ask half-spread
MARKET_IMPACT_PCT = 0.10        # 0.1% premium impact for 5+ lots
SLIPPAGE_POINTS = 0.30          # half-spread as slippage (buy at ask, sell at bid)

# ═══════════════════════════════════════════════════════════════════
#  TIMING (IST)
# ═══════════════════════════════════════════════════════════════════
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
ENTRY_START = "09:30"           # entry from 09:30 (after first candle)
ENTRY_END = os.getenv("ENTRY_END", "14:30")
NO_NEW_ENTRY_AFTER = ENTRY_END  # alias for backward compat
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


def ensure_dirs():
    """Create runtime directories. Called by CLI/scheduler, not at import time."""
    LOG_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def _validate_settings():
    """Validate critical settings at import time."""
    errors = []
    if STARTING_CAPITAL <= 0:
        errors.append("STARTING_CAPITAL must be > 0")
    if not (0 < DAILY_LOSS_LIMIT_PCT <= 100):
        errors.append("DAILY_LOSS_LIMIT_PCT must be in (0, 100]")
    if not (0 < WEEKLY_LOSS_LIMIT_PCT <= 100):
        errors.append("WEEKLY_LOSS_LIMIT_PCT must be in (0, 100]")
    if PREMIUM_SL_PCT <= 0:
        errors.append("PREMIUM_SL_PCT must be > 0")
    if ENTRY_START >= ENTRY_END:
        errors.append("ENTRY_START must be before ENTRY_END")
    if NIFTY_LOT_SIZE <= 0:
        errors.append("NIFTY_LOT_SIZE must be > 0")
    if errors:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))


_validate_settings()

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
