from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    ticker: str
    n_expected: int
    n_actual: int
    pct_missing: float
    n_outliers: int = 0
    stale_days: Optional[int] = None
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    has_survivalship_bias: bool = False
    flags: List[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return len(self.flags) == 0


def check_data_freshness(
    prices: pd.DataFrame,
    max_stale_days: int = 5,
) -> Dict[str, DataQualityReport]:
    reports: Dict[str, DataQualityReport] = {}
    today = prices.index.max()
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if series.empty:
            reports[ticker] = DataQualityReport(
                ticker=ticker,
                n_expected=len(prices),
                n_actual=0,
                pct_missing=100.0,
                flags=["No data available"],
            )
            continue
        last_date = series.index.max()
        stale_days = (today - last_date).days
        flags: List[str] = []
        if stale_days > max_stale_days:
            flags.append(f"Stale ({stale_days}d since last observation)")
        reports[ticker] = DataQualityReport(
            ticker=ticker,
            n_expected=len(prices),
            n_actual=len(series),
            pct_missing=round((1 - len(series) / len(prices)) * 100, 2),
            stale_days=stale_days,
            first_date=str(series.index.min().date()),
            last_date=str(last_date.date()),
            flags=flags,
        )
    return reports


def detect_outliers(
    returns: pd.DataFrame,
    method: str = "mad",
    threshold: float = 5.0,
) -> Dict[str, List[str]]:
    outliers: Dict[str, List[str]] = {}
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        if len(series) < 20:
            continue
        if method == "mad":
            median = series.median()
            mad = (series - median).abs().median()
            if mad == 0:
                continue
            mad_scores = (series - median).abs() / mad
            outlier_dates = series.index[mad_scores > threshold].tolist()
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            z = (series - mean).abs() / std
            outlier_dates = series.index[z > threshold].tolist()
        else:
            raise ValueError(f"Unknown outlier method: {method}")
        if outlier_dates:
            outliers[ticker] = [str(d.date()) for d in outlier_dates]
    return outliers


def check_survivorship_bias(
    tickers: List[str],
    current_prices: pd.Series,
    min_price: float = 0.01,
) -> List[str]:
    flags: List[str] = []
    for ticker in tickers:
        if ticker not in current_prices.index:
            flags.append(f"{ticker}: missing from current prices (delisted/acquired?)")
        elif current_prices[ticker] < min_price:
            flags.append(f"{ticker}: price ${current_prices[ticker]:.2f} near zero")
    return flags


def detect_price_jumps(
    prices: pd.DataFrame,
    max_daily_pct: float = 50.0,
) -> Dict[str, List[str]]:
    jumps: Dict[str, List[str]] = {}
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < 2:
            continue
        daily_pct = series.pct_change().abs() * 100
        bad = daily_pct[daily_pct > max_daily_pct]
        if not bad.empty:
            jumps[ticker] = [f"{str(d.date())}: {v:.1f}%" for d, v in bad.items()]
    return jumps


def validate_returns(returns: pd.DataFrame) -> List[str]:
    issues: List[str] = []
    for ticker in returns.columns:
        series = returns[ticker]
        clean = series.dropna()
        if len(clean) < 20:
            issues.append(f"{ticker}: only {len(clean)} non-NaN observations")
            continue
        if series.isna().mean() > 0.5:
            issues.append(f"{ticker}: {series.isna().mean():.0%} missing values")
        inf_count = np.isinf(clean).sum()
        if inf_count > 0:
            issues.append(f"{ticker}: {inf_count} infinite values")
    return issues
