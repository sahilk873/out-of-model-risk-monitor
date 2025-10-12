from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class CrowdingResult:
    crowding_scores: pd.DataFrame
    aggregate_score: float
    top_crowded_tickers: List[str]
    factor_crowding: pd.Series


def compute_herfindahl_index(weights: pd.Series) -> float:
    hhi = (weights.values**2).sum()
    n = len(weights)
    hhi_normalized = (hhi - 1.0 / n) / (1.0 - 1.0 / n) if n > 1 else 0.0
    return float(max(hhi_normalized, 0.0))


def compute_factor_crowding(
    factor_exposures: pd.DataFrame,
    active_weights: pd.Series,
) -> pd.Series:
    factor_cols = [c for c in factor_exposures.columns if c not in ("Alpha", "R_squared", "N_obs")]
    w = active_weights.reindex(factor_exposures.index).fillna(0.0)
    w_abs = w.abs()
    w_abs = w_abs / w_abs.sum() if w_abs.sum() > 0 else w_abs

    crowding = {}
    for col in factor_cols:
        beta = factor_exposures[col].abs()
        weighted_cross = (
            w_abs.values[:, None]
            * beta.values[None, :]
            * (factor_exposures[col].values[:, None] != 0)
        ).sum()
        n_nonzero = (factor_exposures[col].abs() > 0.01).sum()
        n_overlap = ((factor_exposures[col].abs() > 0.01) & (w_abs > 0.001)).sum()
        crowding[col] = weighted_cross / max(n_nonzero, 1) * (n_overlap / max(n_nonzero, 1))

    return pd.Series(crowding, name="factor_crowding")


def detect_crowded_trades(
    active_weights: pd.Series,
    residual_correlations: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    theme_scores: Optional[pd.DataFrame] = None,
    top_n: int = 5,
) -> CrowdingResult:
    w = active_weights.abs()
    w = w / w.sum() if w.sum() > 0 else w

    weight_concentration = compute_herfindahl_index(w)
    factor_crowd = compute_factor_crowding(factor_exposures, active_weights)

    residual_crowding = pd.Series(0.0, index=residual_correlations.index)
    if not residual_correlations.empty:
        for ticker in residual_correlations.columns:
            others = [c for c in residual_correlations.columns if c != ticker]
            if others:
                residual_crowding[ticker] = residual_correlations.loc[ticker, others].abs().mean()

    theme_crowding = pd.Series(0.0, index=active_weights.index)
    if theme_scores is not None and not theme_scores.empty:
        for ticker in active_weights.index:
            theme_crowding[ticker] = theme_scores["holdings_exposure_pct"].sum() / max(
                len(theme_scores), 1
            )

    scores = []
    for ticker in active_weights.index:
        wgt = w.get(ticker, 0.0)
        res_crowd = residual_crowding.get(ticker.upper(), residual_crowding.get(ticker, 0.0))
        th_crowd = theme_crowding.get(ticker, 0.0) / 100.0

        crowding_score = (
            0.30 * wgt + 0.35 * res_crowd + 0.15 * weight_concentration + 0.20 * th_crowd
        )
        scores.append(
            {
                "ticker": ticker,
                "weight_pct": wgt * 100,
                "residual_crowding": round(float(res_crowd), 4),
                "theme_crowding": round(float(th_crowd), 4),
                "crowding_score": round(float(crowding_score), 4),
            }
        )

    crowding_df = pd.DataFrame(scores).sort_values("crowding_score", ascending=False)
    aggregate = float(
        0.30 * weight_concentration
        + 0.35 * (crowding_df["residual_crowding"].mean() if not crowding_df.empty else 0.0)
        + 0.20 * (crowding_df["theme_crowding"].mean() if not crowding_df.empty else 0.0)
    )

    top_tickers = crowding_df.head(top_n)["ticker"].tolist() if not crowding_df.empty else []

    return CrowdingResult(
        crowding_scores=crowding_df,
        aggregate_score=aggregate,
        top_crowded_tickers=top_tickers,
        factor_crowding=factor_crowd,
    )


def compute_effective_n(weights: pd.Series) -> float:
    hhi = (weights.values**2).sum()
    return 1.0 / hhi if hhi > 0 else 1.0


def compute_gini_coefficient(weights: pd.Series) -> float:
    w = weights.abs().values
    w = np.sort(w)
    n = len(w)
    if n == 0 or w.sum() == 0:
        return 0.0
    cumulative = np.cumsum(w)
    b = cumulative.sum() / (n * w.sum())
    return float(1.0 - 2.0 * b + 1.0 / n)


def compute_dispersion_ratio(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    cross_sectional_vol = returns.rolling(window).std().mean(axis=1)
    market_vol = returns.mean(axis=1).rolling(window).std()
    ratio = cross_sectional_vol / (market_vol + 1e-10)
    return ratio.rename("dispersion_ratio")
