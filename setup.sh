#!/bin/bash
# TradingAgent Setup Script
# Run this once after cloning to set up the environment

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Setting up TradingAgent at $PROJECT_DIR"

# 1. Create virtual environment if not exists
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_DIR/venv"
fi

# 2. Install dependencies
echo "Installing dependencies..."
source "$PROJECT_DIR/venv/bin/activate"
pip install --upgrade pip --quiet
pip install --no-compile -r "$PROJECT_DIR/requirements.txt" --quiet

# 3. Create .env from example if not exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "Creating .env from .env.example..."
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo ""
    echo "IMPORTANT: Edit .env with your credentials:"
    echo "  $PROJECT_DIR/.env"
    echo ""
fi

# 4. Create data and log directories
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/data"

# 5. Install launchd job (macOS automation)
PLIST_SRC="$PROJECT_DIR/automation/com.tradingagent.daily.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.tradingagent.daily.plist"

echo ""
echo "To enable daily auto-start at 8:25 AM IST:"
echo "  cp $PLIST_SRC $PLIST_DST"
echo "  launchctl load $PLIST_DST"
echo ""
echo "To disable:"
echo "  launchctl unload $PLIST_DST"
echo ""

# 6. Run backtest to validate
echo "Running backtest on 180 days of synthetic data..."
python "$PROJECT_DIR/main.py" --backtest --bt-days 180 --bt-capital 10000

echo ""
echo "Setup complete!"
echo ""
echo "Quick start:"
echo "  source venv/bin/activate"
echo "  python main.py --backtest        # Validate strategies"
echo "  python main.py --paper           # Paper trading"
echo "  python main.py --live            # Live trading (AFTER paper validation)"
