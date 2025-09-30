from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(
    name: str = "risk_monitor",
    level: str = "INFO",
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def percentile_ci(
    values: list[float],
    ci: float = 0.95,
) -> tuple[float, float, float]:
    import numpy as np

    arr = np.array(sorted(values))
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)
    lower = np.percentile(arr, (1 - ci) / 2 * 100)
    upper = np.percentile(arr, (1 + ci) / 2 * 100)
    return float(np.mean(arr)), float(lower), float(upper)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
