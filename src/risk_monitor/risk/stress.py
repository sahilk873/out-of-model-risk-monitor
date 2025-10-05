from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

HISTORICAL_SCENARIOS: Dict[str, Dict] = {
    "2008_Financial_Crisis": {
        "start": "2008-09-01",
        "end": "2009-03-09",
        "description": "Global financial crisis peak-to-trough",
    },
    "2020_COVID_Shock": {
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "COVID-19 pandemic selloff",
    },
    "2022_Rate_Hike": {
        "start": "2022-01-03",
        "end": "2022-10-12",
        "description": "Federal Reserve aggressive rate hiking cycle",
    },
}

FACTOR_SHOCKS: Dict[str, Dict[str, float]] = {
    "Severe_Recession": {
        "Mkt-RF": -3.0,
        "SMB": 0.5,
        "HML": 1.0,
        "RMW": 0.5,
        "CMA": 0.5,
        "MOM": -1.0,
    },
    "Growth_Scare": {"Mkt-RF": -2.0, "SMB": -0.5, "HML": 1.5, "RMW": 0.0, "CMA": 1.0, "MOM": -0.5},
    "Momentum_Crash": {"Mkt-RF": -1.0, "SMB": 0.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0, "MOM": -4.0},
    "Inflation_Shock": {
        "Mkt-RF": -1.5,
        "SMB": -0.5,
        "HML": 0.5,
        "RMW": -0.5,
        "CMA": -0.5,
        "MOM": 0.0,
    },
    "Tech_Wreck": {"Mkt-RF": -2.5, "SMB": 1.0, "HML": -0.5, "RMW": -1.0, "CMA": -0.5, "MOM": -1.5},
}


def run_historical_replay(
    returns: pd.DataFrame,
    scenario_name: str,
    weights: Optional[pd.Series] = None,
) -> pd.Series:
    if scenario_name not in HISTORICAL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    scenario = HISTORICAL_SCENARIOS[scenario_name]
    period_returns = returns.loc[scenario["start"] : scenario["end"]]
    if period_returns.empty:
        raise ValueError(
            f"No return data for scenario '{scenario_name}' "
            f"({scenario['start']} to {scenario['end']})"
        )

    if weights is not None:
        w = weights.reindex(period_returns.columns).fillna(0.0)
        w = w / w.abs().sum() if w.abs().sum() > 0 else w
        weighted = period_returns.multiply(w, axis=1)
        cumulative = (1 + weighted.sum(axis=1)).prod() - 1
    else:
        cumulative = (1 + period_returns).prod() - 1
        cumulative = cumulative.mean() if isinstance(cumulative, pd.Series) else cumulative

    return cumulative * 100


def run_factor_shock(
    weights: pd.Series,
    factor_exposures: pd.DataFrame,
    shock_name: str,
) -> Tuple[float, pd.Series, pd.Series]:
    if shock_name not in FACTOR_SHOCKS:
        raise ValueError(f"Unknown shock: {shock_name}")

    shock = FACTOR_SHOCKS[shock_name]
    fcols = list(shock.keys())
    available = [c for c in fcols if c in factor_exposures.columns]

    betas = factor_exposures[available].values
    shock_vals = np.array([shock[c] for c in available])

    # Per-security shock P&L (%) — shock_vals are already in percent
    security_pnl = betas @ shock_vals

    # Weighted portfolio impact
    w = weights.reindex(factor_exposures.index).fillna(0.0).values
    portfolio_pnl = w @ security_pnl

    # Decompose by factor
    factor_contrib = pd.Series(
        (w @ betas) * shock_vals,
        index=available,
    )

    return (
        float(portfolio_pnl),
        pd.Series(security_pnl, index=factor_exposures.index),
        factor_contrib,
    )


def decompose_stress_losses(
    weights: pd.Series,
    returns: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    scenario_period_returns: pd.Series,
    sectors: Optional[pd.Series] = None,
) -> Dict:
    w = weights.reindex(returns.columns).fillna(0.0)
    total_loss = w @ scenario_period_returns

    by_security = scenario_period_returns * w
    by_security = by_security.sort_values()

    by_factor = pd.Series(dtype=float)
    for ticker in returns.columns:
        if ticker in factor_exposures.index:
            by_factor = pd.concat([by_factor, factor_exposures.loc[ticker] * w[ticker]])
    by_factor = by_factor.groupby(by_factor.index).sum().sort_values()

    by_sector = pd.Series(dtype=float)
    if sectors is not None:
        sector_map = sectors.reindex(returns.columns).fillna("Unknown")
        by_sector = by_security.groupby(sector_map).sum().sort_values()

    return {
        "total_loss_pct": float(total_loss),
        "by_security": by_security.to_dict(),
        "by_factor": by_factor.to_dict(),
        "by_sector": by_sector.to_dict(),
    }
