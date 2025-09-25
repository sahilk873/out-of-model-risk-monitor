from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

CORE_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def align_factors_and_returns(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    risk_free_col: str = "RF",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_dates = returns.index.intersection(factors.index)
    if len(common_dates) < 20:
        raise ValueError(
            f"Insufficient overlapping dates between returns ({len(returns)}) "
            f"and factors ({len(factors)})"
        )
    aligned_returns = returns.loc[common_dates]
    aligned_factors = factors.loc[common_dates]
    return aligned_returns, aligned_factors


def estimate_factor_exposures(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    window: int = 252,
    min_window: int = 60,
) -> pd.DataFrame:
    if len(returns) < min_window:
        raise ValueError(f"Need at least {min_window} observations, got {len(returns)}")

    factor_cols = [c for c in CORE_FACTORS if c in factors.columns]
    has_rf = "RF" in factors.columns
    rf = factors["RF"] if has_rf else pd.Series(0.0, index=returns.index)

    excess_ret = returns.sub(rf, axis=0)

    exposures = {}
    for ticker in returns.columns:
        ticker_ret = excess_ret[ticker].dropna()
        if len(ticker_ret) < min_window:
            continue

        joined = pd.concat(
            [ticker_ret.rename("ret")] + [factors[c].rename(c) for c in factor_cols],
            axis=1,
        ).dropna()

        if len(joined) < min_window:
            continue

        y = joined["ret"].values
        X = joined[factor_cols].values
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
            exposures[ticker] = {
                "Alpha": model.params[0],
                **{factor_cols[i]: model.params[i + 1] for i in range(len(factor_cols))},
                "R_squared": model.rsquared,
                "N_obs": len(joined),
            }
        except Exception:
            continue

    if not exposures:
        raise ValueError("No factor models could be estimated.")

    return pd.DataFrame(exposures).T


def rolling_factor_exposures(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    window: int = 252,
    step: int = 21,
    min_window: int = 60,
) -> Dict[str, pd.DataFrame]:
    exposures_over_time: Dict[str, list] = {}
    dates = returns.index.sort_values()

    for i in range(window, len(dates) + 1, step):
        end_date = dates[i - 1]
        start_date = dates[i - window]
        window_ret = returns.loc[start_date:end_date]
        window_fac = factors.loc[start_date:end_date]

        if len(window_ret) < min_window:
            continue

        try:
            ex = estimate_factor_exposures(window_ret, window_fac, window, min_window)
            ex["date"] = end_date
            for ticker in ex.index:
                if ticker not in exposures_over_time:
                    exposures_over_time[ticker] = []
                exposures_over_time[ticker].append(ex.loc[ticker].to_dict())
        except (ValueError, Exception):
            continue

    result = {}
    for ticker, records in exposures_over_time.items():
        result[ticker] = pd.DataFrame(records).set_index("date")
    return result


def compute_residual_returns(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    exposures: pd.DataFrame,
) -> pd.DataFrame:
    factor_cols = [c for c in CORE_FACTORS if c in factors.columns]
    has_rf = "RF" in factors.columns
    rf = factors["RF"] if has_rf else pd.Series(0.0, index=returns.index)

    excess_ret = returns.sub(rf, axis=0)
    common_dates = excess_ret.index.intersection(factors.index)
    excess_ret = excess_ret.loc[common_dates]
    factor_vals = factors.loc[common_dates]

    residuals = pd.DataFrame(index=excess_ret.index)
    for ticker in returns.columns:
        if ticker not in exposures.index:
            continue
        beta = exposures.loc[ticker]
        alpha = beta.get("Alpha", 0.0)
        loadings = np.array([beta.get(c, 0.0) for c in factor_cols])
        fvals = factor_vals[factor_cols].values
        explained = fvals @ loadings + alpha
        res = excess_ret[ticker].values[: len(explained)] - explained
        residuals[ticker] = res

    return residuals.dropna(how="all")
