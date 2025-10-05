from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from risk_monitor.factor_models.standard_factors import CORE_FACTORS


def compute_covariance(
    returns: pd.DataFrame,
    method: str = "ewma",
    halflife: int = 60,
) -> pd.DataFrame:
    if method == "ewma":
        ewm = returns.ewm(span=2 * halflife - 1)
        cov = returns.cov()
        try:
            cov = ewm.cov().iloc[-len(returns.columns) :]
            cov = cov.groupby(level=1).mean()
            cov = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
        except Exception:
            cov = returns.cov()
    elif method == "sample":
        cov = returns.cov()
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    return cov.fillna(0.0)


def factor_model_covariance(
    factor_returns: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    residual_variance: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    factor_cov = compute_covariance(factor_returns)

    securities = factor_exposures.index
    fcols = [c for c in CORE_FACTORS if c in factor_exposures.columns]

    if not fcols:
        raise ValueError("No factor columns found in exposures.")

    betas = factor_exposures[fcols].values
    systematic = betas @ factor_cov.loc[fcols, fcols].values @ betas.T
    total = systematic + np.diag(residual_variance.reindex(securities).fillna(0.0).values)

    sys_df = pd.DataFrame(systematic, index=securities, columns=securities)
    total_df = pd.DataFrame(total, index=securities, columns=securities)
    return sys_df, total_df


def compute_active_risk(
    active_weights: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    w = active_weights.reindex(covariance.index).fillna(0.0).values
    var = w @ covariance.values @ w
    return float(max(np.sqrt(var) * np.sqrt(252) * 100, 0.0))  # annualized vol %


def risk_contribution(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:
    w = weights.reindex(covariance.index).fillna(0.0).values
    port_var = w @ covariance.values @ w
    if port_var <= 0:
        return pd.Series(0.0, index=covariance.index)
    marginal = covariance.values @ w
    rc = w * marginal / port_var * 100  # percent contribution
    return pd.Series(rc, index=covariance.index)


def factor_risk_attribution(
    active_weights: pd.Series,
    factor_exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    residual_var: pd.Series,
) -> Dict:
    fcols = [c for c in CORE_FACTORS if c in factor_exposures.columns]
    securities = factor_exposures.index

    w = active_weights.reindex(securities).fillna(0.0)
    betas = factor_exposures[fcols].values
    factor_cov_m = factor_cov.loc[fcols, fcols].values

    port_beta = w.values @ betas
    factor_risk = port_beta @ factor_cov_m @ port_beta
    factor_vol = np.sqrt(max(factor_risk, 0.0))

    specific_risk = (w.values**2 * residual_var.reindex(securities).fillna(0.0).values).sum()
    specific_vol = np.sqrt(max(specific_risk, 0.0))

    total_risk = np.sqrt(max(factor_risk + specific_risk, 0.0))

    factor_contrib = pd.Series(
        (port_beta @ factor_cov_m) * port_beta / max(factor_risk, 1e-12) * 100,
        index=fcols,
    )

    return {
        "total_risk_annual_pct": float(total_risk * np.sqrt(252) * 100),
        "factor_risk_annual_pct": float(factor_vol * np.sqrt(252) * 100),
        "specific_risk_annual_pct": float(specific_vol * np.sqrt(252) * 100),
        "factor_risk_pct_of_total": float(factor_risk / max(total_risk**2, 1e-12) * 100),
        "specific_risk_pct_of_total": float(specific_risk / max(total_risk**2, 1e-12) * 100),
        "factor_contrib_pct": factor_contrib.to_dict(),
    }


def security_risk_contribution(
    active_weights: pd.Series,
    factor_exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    residual_var: pd.Series,
) -> pd.DataFrame:
    fcols = [c for c in CORE_FACTORS if c in factor_exposures.columns]
    securities = factor_exposures.index
    w = active_weights.reindex(securities).fillna(0.0)

    betas = factor_exposures[fcols].values
    fcm = factor_cov.loc[fcols, fcols].values

    sys_cov = betas @ fcm @ betas.T
    res_cov = np.diag(residual_var.reindex(securities).fillna(0.0).values)
    total_cov = sys_cov + res_cov

    total_var = w.values @ total_cov @ w.values
    if total_var <= 0:
        return pd.DataFrame(0.0, index=securities, columns=["marginal_ctr_pct", "total_ctr_pct"])

    marginal_covar = total_cov @ w.values
    ctr = w.values * marginal_covar / total_var * 100

    return pd.DataFrame(
        {
            "weight_pct": w.values * 100,
            "marginal_ctr_pct": (marginal_covar / (2 * np.sqrt(total_var)) * 100),
            "total_ctr_pct": ctr,
        },
        index=securities,
    )
