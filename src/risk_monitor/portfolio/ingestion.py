from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Portfolio:
    name: str
    weights: pd.DataFrame  # index=date, columns=ticker, values=weight
    side: Optional[str] = None  # "long", "long_short", "active"

    def __post_init__(self):
        if self.weights.empty:
            raise ValueError(f"Portfolio '{self.name}' has empty weights.")

    @property
    def tickers(self) -> List[str]:
        return list(self.weights.columns)

    @property
    def dates(self):
        return self.weights.index

    @property
    def is_long_only(self) -> bool:
        return (self.weights.min().min() >= -1e-8) and (self.side != "long_short")

    @property
    def gross_exposure(self) -> pd.Series:
        return self.weights.abs().sum(axis=1)

    @property
    def net_exposure(self) -> pd.Series:
        return self.weights.sum(axis=1)


@dataclass
class PortfolioSet:
    portfolios: Dict[str, Portfolio] = field(default_factory=dict)

    def get(self, name: str) -> Portfolio:
        if name not in self.portfolios:
            raise KeyError(f"Portfolio '{name}' not found. Available: {list(self.portfolios)}")
        return self.portfolios[name]

    @property
    def names(self) -> List[str]:
        return list(self.portfolios)


def validate_weights(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "weight", "side", "portfolio_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    dupes = df.duplicated(subset=["date", "ticker", "portfolio_name"], keep=False)
    if dupes.any():
        raise ValueError(
            f"Duplicate (date, ticker, portfolio_name) entries found:\n{df[dupes].head()}"
        )

    bad_weights = df["weight"].isna() | (df["weight"] == 0)
    if bad_weights.any():
        raise ValueError("Zero or NaN weights found.")

    return df


def load_portfolio_csv(filepath: str) -> PortfolioSet:
    raw = pd.read_csv(filepath)
    validated = validate_weights(raw)
    # groupby drops NaN keys; fill them so benchmark portfolio is captured
    validated["portfolio_name"] = validated["portfolio_name"].fillna("")

    ps = PortfolioSet()
    for pname, group in validated.groupby("portfolio_name"):
        name = pname if isinstance(pname, str) and pname.strip() else "__benchmark__"
        pivot = group.pivot_table(index="date", columns="ticker", values="weight", aggfunc="first")
        pivot = pivot.sort_index().fillna(0.0)
        side = group["side"].iloc[0] if "side" in group.columns else None
        if side and isinstance(side, str) and side.strip():
            ps.portfolios[name] = Portfolio(name=name, weights=pivot, side=side)
        else:
            ps.portfolios[name] = Portfolio(name=name, weights=pivot)

    return ps


def compute_active_weights(portfolio: Portfolio, benchmark: Optional[Portfolio]) -> pd.DataFrame:
    if benchmark is None:
        return portfolio.weights.copy()

    common_tickers = list(set(portfolio.tickers) | set(benchmark.tickers))
    pw = portfolio.weights.reindex(columns=common_tickers, fill_value=0.0)
    bw = benchmark.weights.reindex(columns=common_tickers, fill_value=0.0)

    aligned_bw = bw.reindex(index=pw.index, method="ffill").fillna(0.0)

    active = pw - aligned_bw
    return active


def validate_portfolio_set(ps: PortfolioSet) -> List[str]:
    issues = []
    for name, pf in ps.portfolios.items():
        if pf.weights.isna().any().any():
            issues.append(f"{name}: contains NaN weights")
        gross = pf.gross_exposure
        if gross.max() > 1.5:
            issues.append(f"{name}: gross exposure {gross.max():.2f} exceeds 150%")
        net = pf.net_exposure.abs()
        if net.max() > 0.3 and name != "__benchmark__":
            issues.append(f"{name}: net exposure {net.max():.4f} may indicate leverage")
    return issues
