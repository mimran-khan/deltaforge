#!/bin/bash
# DeltaForge Legacy Setup Script
#
# The recommended install method is:
#   make install
#
# This script is kept for environments without make.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Setting up DeltaForge at $PROJECT_DIR"

# 1. Create virtual environment if not exists
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_DIR/venv"
fi

# 2. Install dependencies via pyproject.toml
echo "Installing dependencies..."
source "$PROJECT_DIR/venv/bin/activate"
pip install --upgrade pip setuptools wheel --quiet
pip install -e ".[test]" --quiet

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

echo ""
echo "Setup complete!"
echo ""
echo "Quick start:"
echo "  make trade       # Paper trading"
echo "  make backtest    # Run backtest"
echo "  make test        # Run tests"
echo "  make help        # See all commands"
echo ""
echo "Or use the CLI directly:"
echo "  source venv/bin/activate"
echo "  df trade         # Paper trading"
echo "  df --help        # See all CLI commands"
