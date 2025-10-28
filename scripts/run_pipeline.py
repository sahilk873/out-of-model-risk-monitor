#!/usr/bin/env python3
"""End-to-end out-of-model risk monitor pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --portfolio data/sample_portfolio.csv --start 2024-01-01
"""

import argparse
import sys
from datetime import datetime

from dotenv import load_dotenv

from risk_monitor.pipeline import run_pipeline
from risk_monitor.config import MonitorConfig
from risk_monitor.utils import setup_logging

logger = setup_logging("pipeline")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Out-of-Model Risk Monitor Pipeline")
    parser.add_argument(
        "--portfolio", default="data/sample_portfolio.csv", help="Portfolio CSV path"
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default: today)")
    parser.add_argument("--stress-start", default="2020-01-01", help="Stress test start date")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", default=None, help="Path to config file (optional)")
    args = parser.parse_args()

    if args.verbose:
        setup_logging("pipeline", level="DEBUG")

    config = MonitorConfig.from_env()
    end_date = args.end or datetime.today().strftime("%Y-%m-%d")

    try:
        ctx = run_pipeline(
            portfolio_file=args.portfolio,
            start_date=args.start,
            end_date=end_date,
            stress_start=args.stress_start,
            config=config,
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)
