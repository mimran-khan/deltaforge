#!/usr/bin/env python3
"""DeltaForge entry point -- delegates to the CLI.

Usage:
    python main.py             # show help
    python main.py trade       # start paper trading
    python main.py status      # show system status
    python main.py --help      # full command list

Preferred:
    df trade                   # after `pip install -e .` or `make install`
"""
from cli import main

if __name__ == "__main__":
    main()
