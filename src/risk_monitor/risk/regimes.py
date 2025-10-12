from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from hmmlearn import hmm


@dataclass
class RegimeSummary:
    n_regimes: int
    current_regime: int
    regime_probs: pd.DataFrame
    regime_stats: pd.DataFrame
    smoothed_probs: pd.DataFrame
    transition_matrix: np.ndarray
    half_life_days: pd.Series


def detect_volatility_regimes(
    returns: pd.DataFrame,
    n_regimes: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 1000,
    random_state: int = 42,
) -> RegimeSummary:
    returns = returns.dropna(how="all")
    if len(returns) < 100:
        raise ValueError(f"Need at least 100 observations for regime detection, got {len(returns)}")

    vol_estimates = returns.apply(lambda x: x.ewm(span=21).std().bfill().values, axis=0)
    train_data = np.column_stack(
        [vol_estimates[:, i] for i in range(min(3, vol_estimates.shape[1]))]
    )

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-4,
    )
    model.fit(train_data)

    hidden_states = model.predict(train_data)
    state_probs = model.predict_proba(train_data)

    state_probs_df = pd.DataFrame(
        state_probs,
        index=returns.index[-len(state_probs) :],
        columns=[f"Regime_{i}" for i in range(n_regimes)],
    )
    smoothed_probs_df = state_probs_df.ewm(span=10).mean()

    sorted_means = np.argsort(
        [np.mean(returns.values[-252:, :].std() * np.sqrt(252) * 100) for _ in range(n_regimes)]
    )
    if len(sorted_means) == n_regimes:
        state_map = {old: new for new, old in enumerate(sorted_means)}
        hidden_states = np.array([state_map[s] for s in hidden_states])

    regime_stats = []
    for i in range(n_regimes):
        mask = hidden_states == i
        regime_returns = returns.values[-len(hidden_states) :][mask]
        if len(regime_returns) == 0:
            continue
        annual_vol = float(np.std(regime_returns, ddof=1) * np.sqrt(252) * 100)
        mean_ret = float(np.mean(regime_returns) * 252 * 100)
        regime_stats.append(
            {
                "regime": f"Regime_{i}",
                "n_obs": int(mask.sum()),
                "freq_pct": float(mask.mean() * 100),
                "annualized_vol_pct": round(annual_vol, 2),
                "annualized_return_pct": round(mean_ret, 2),
                "description": _describe_regime(i, n_regimes, annual_vol),
            }
        )

    T = model.transmat_
    eigenvalues = np.linalg.eigvals(T.T)
    steady_state = np.abs(eigenvalues[0].real)
    if steady_state < 1:
        steady_state = 1.0
    half_life_days = pd.Series(
        {
            f"Regime_{i}": float(-np.log(2) / np.log(max(T[i, i], 0.01)))
            if T[i, i] < 1 and T[i, i] > 0
            else 9999.0
            for i in range(n_regimes)
        }
    )

    return RegimeSummary(
        n_regimes=n_regimes,
        current_regime=int(hidden_states[-1]),
        regime_probs=state_probs_df,
        regime_stats=pd.DataFrame(regime_stats),
        smoothed_probs=smoothed_probs_df,
        transition_matrix=T,
        half_life_days=half_life_days,
    )


def _describe_regime(i: int, n_regimes: int, annual_vol: float) -> str:
    if n_regimes == 2:
        return "Low Volatility" if i == 0 else "High Volatility"
    if n_regimes == 3:
        descriptions = ["Low Volatility", "Medium Volatility", "High Volatility"]
        return descriptions[i] if i < 3 else f"Regime_{i}"
    if n_regimes == 4:
        descriptions = ["Low Vol", "Low-Med Vol", "Med-High Vol", "High Vol"]
        return descriptions[i] if i < 4 else f"Regime_{i}"
    return f"Regime_{i}"


def regime_conditional_value_at_risk(
    returns: pd.DataFrame,
    regime_probs: pd.DataFrame,
    confidence_level: float = 0.05,
) -> pd.DataFrame:
    results = []
    for col in regime_probs.columns:
        prob = regime_probs[col].values
        weighted_returns = returns.values[-len(prob) :] * prob[:, np.newaxis]
        regime_returns = weighted_returns[prob > 0.01]
        if len(regime_returns) < 20:
            continue
        var = np.percentile(regime_returns, confidence_level * 100, axis=0)
        cvar = (
            regime_returns[regime_returns <= var].mean(axis=0)
            if regime_returns[regime_returns <= var].size > 0
            else var
        )
        results.append(
            {
                "regime": col,
                "VaR_95pct": float(np.mean(var) * np.sqrt(252) * 100),
                "CVaR_95pct": float(np.mean(cvar) * np.sqrt(252) * 100),
            }
        )
    return pd.DataFrame(results) if results else pd.DataFrame()


def compute_regime_covariances(
    returns: pd.DataFrame,
    regime_probs: pd.DataFrame,
    method: str = "ewma",
    halflife: int = 60,
) -> Dict[str, pd.DataFrame]:
    from risk_monitor.risk.attribution import compute_covariance

    regime_covs: Dict[str, pd.DataFrame] = {}
    for col in regime_probs.columns:
        prob = regime_probs[col].values.flatten()
        if prob.sum() < 10:
            continue
        weighted_returns = returns.values[-len(prob) :] * prob[:, np.newaxis]
        weighted_df = pd.DataFrame(
            weighted_returns, index=returns.index[-len(prob) :], columns=returns.columns
        )
        try:
            cov = compute_covariance(weighted_df, method=method, halflife=halflife)
            regime_covs[col] = cov
        except Exception:
            continue
    return regime_covs


def entropy_regime_concentration(regime_probs: pd.DataFrame) -> pd.Series:
    p = regime_probs.values.clip(1e-12, 1 - 1e-12)
    entropy = -(p * np.log(p)).sum(axis=1)
    max_entropy = np.log(regime_probs.shape[1])
    return pd.Series(
        entropy / max_entropy,
        index=regime_probs.index,
        name="regime_entropy",
    )


def regime_stability_index(hidden_states: np.ndarray, window: int = 21) -> float:
    if len(hidden_states) < window:
        return 0.0
    recent = hidden_states[-window:]
    switches = (recent[1:] != recent[:-1]).sum()
    return float(1.0 - switches / (window - 1))
