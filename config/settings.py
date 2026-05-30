import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# ── Angel One SmartAPI ──────────────────────────────────────────────
ANGEL_API_KEY = os.getenv("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
ANGEL_PASSWORD = os.getenv("ANGEL_PASSWORD", "")
ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "")

# ── Telegram ────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Trading Mode ────────────────────────────────────────────────────
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"

# ── Capital & Position Sizing ───────────────────────────────────────
STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "10000"))

NIFTY_LOT_SIZE = 75   # verify against current NSE specs on startup
BANKNIFTY_LOT_SIZE = 30

LOT_TIERS = [
    # (min_capital, max_lots, instrument)
    (10_000,  1, "NIFTY"),
    (25_000,  2, "NIFTY"),
    (50_000,  3, "NIFTY"),
    (100_000, 5, "NIFTY"),
]

# ── Risk Management ────────────────────────────────────────────────
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "20"))
WEEKLY_LOSS_LIMIT_PCT = 40.0
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "3"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "2"))
PREMIUM_SL_PCT = 40.0        # stop loss as % of premium paid
PREMIUM_TARGET_POINTS = 25   # target gain in premium Rs per unit
MAX_VIX_THRESHOLD = 20.0     # skip trading if India VIX > this

# ── Strategy: ORB ───────────────────────────────────────────────────
ORB_WINDOW_MINUTES = 15       # 9:15–9:30
ORB_MAX_RANGE_POINTS = 150    # skip if opening range > 150 Nifty pts
ORB_VOLUME_MULTIPLIER = 2.0   # breakout candle volume >= 2x avg
ORB_ENTRY_START = "09:30"
ORB_ENTRY_END = "11:00"
ORB_RR_RATIO = 1.5

# ── Strategy: VWAP Momentum ────────────────────────────────────────
VWAP_EMA_FAST = 9
VWAP_EMA_SLOW = 20
VWAP_RSI_PERIOD = 14
VWAP_RSI_LONG_THRESHOLD = 50
VWAP_RSI_SHORT_THRESHOLD = 50
VWAP_VOLUME_MULTIPLIER = 1.5
VWAP_ENTRY_START = "09:30"
VWAP_ENTRY_END = "14:00"

# ── Timing (IST) ───────────────────────────────────────────────────
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
NO_NEW_ENTRY_AFTER = "14:30"
SQUARE_OFF_TIME = "15:15"
SESSION_LOGIN_TIME = "08:30"
INSTRUMENT_DOWNLOAD_TIME = "08:45"
EOD_REPORT_TIME = "15:35"
SESSION_LOGOUT_TIME = "23:55"

# ── Paths ───────────────────────────────────────────────────────────
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "trades.db"
CAPITAL_FILE = DATA_DIR / "capital.json"
INSTRUMENTS_FILE = DATA_DIR / "instruments.json"

LOG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
