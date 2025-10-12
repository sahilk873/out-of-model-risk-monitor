from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BrinsonAttribution:
    """
    Brinson-style performance attribution decomposing active return into
    allocation, selection, and interaction effects.
    """

    def decompose(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        sector_map: pd.Series,
    ) -> Dict:
        common = sector_map.index.intersection(portfolio_weights.index)
        pw = portfolio_weights.reindex(common).fillna(0.0)
        bw = benchmark_weights.reindex(common).fillna(0.0)
        pr = portfolio_returns.reindex(common).fillna(0.0)
        br = benchmark_returns.reindex(common).fillna(0.0)
        sec = sector_map.reindex(common)

        sectors = sec.unique()
        total = {
            "allocation": 0.0,
            "selection": 0.0,
            "interaction": 0.0,
            "total": 0.0,
        }
        by_sector: Dict[str, Dict] = {}

        for sector in sectors:
            mask = sec == sector
            w_p = pw[mask].sum()
            w_b = bw[mask].sum()

            r_p = (pr[mask] * pw[mask] / w_p).sum() if w_p > 0 else 0.0
            r_b = (br[mask] * bw[mask] / w_b).sum() if w_b > 0 else 0.0
            r_b_total = (br[mask] * bw[mask]).sum() / bw.sum() if bw.sum() > 0 else 0.0

            alloc = (w_p - w_b) * r_b_total * 100
            select = w_b * (r_p - r_b) * 100
            interact = (w_p - w_b) * (r_p - r_b) * 100

            by_sector[sector] = {
                "allocation_bps": round(alloc * 100, 2),
                "selection_bps": round(select * 100, 2),
                "interaction_bps": round(interact * 100, 2),
                "total_bps": round((alloc + select + interact) * 100, 2),
                "active_weight_pct": round((w_p - w_b) * 100, 2),
            }
            total["allocation"] += alloc
            total["selection"] += select
            total["interaction"] += interact

        total["total"] = total["allocation"] + total["selection"] + total["interaction"]
        return {"total": total, "by_sector": by_sector}

    @staticmethod
    def geometric_attribution(
        portfolio_return: float,
        benchmark_return: float,
    ) -> Dict:
        active = portfolio_return - benchmark_return
        ratio = (1 + portfolio_return) / (1 + benchmark_return) - 1
        return {
            "active_return_bps": round(active * 10000, 2),
            "geometric_excess_bps": round(ratio * 10000, 2),
            "portfolio_return_pct": round(portfolio_return * 100, 4),
            "benchmark_return_pct": round(benchmark_return * 100, 4),
        }


def compute_turnover(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: Optional[pd.DataFrame] = None,
) -> pd.Series:
    if benchmark_weights is not None:
        common = portfolio_weights.columns.intersection(benchmark_weights.columns)
        pw = portfolio_weights[common]
        bw = benchmark_weights[common].reindex(pw.index).fillna(0.0)
        active = pw.sub(bw, axis=1)
    else:
        active = portfolio_weights.copy()

    diffs = active.diff().abs().sum(axis=1)
    return diffs / 2


def compute_exposure_drift(
    exposures: pd.DataFrame,
    reference: Optional[pd.Series] = None,
) -> pd.Series:
    if reference is not None:
        drifts = exposures.sub(reference, axis=1).abs().max(axis=1)
    else:
        drifts = exposures.diff().abs().sum(axis=1)
    return drifts


def factor_performance_attribution(
    portfolio_returns: pd.Series,
    factor_exposures: pd.Series,
    factor_returns: pd.DataFrame,
    window: int = 252,
) -> Dict:
    factor_cols = [c for c in factor_returns.columns if c in factor_exposures.index]
    if not factor_cols:
        return {"error": "No overlapping factor columns"}

    betas = np.array([factor_exposures.get(c, 0.0) for c in factor_cols])
    fwd_returns = (
        factor_returns[factor_cols].iloc[-window:]
        if len(factor_returns) > window
        else factor_returns[factor_cols]
    )

    factor_contrib = (fwd_returns * betas).sum(axis=1)
    alpha_contrib = portfolio_returns.iloc[-len(fwd_returns) :].values - factor_contrib.values

    cumulative_factor = (1 + factor_contrib).prod() - 1
    cumulative_alpha = (1 + pd.Series(alpha_contrib, index=fwd_returns.index)).prod() - 1
    factor_ic = pd.Series(dtype=float)
    if len(factor_contrib) > 60:
        lookback = 60
        ic_values = []
        for i in range(lookback, len(factor_contrib)):
            est_ret = factor_contrib.iloc[i - lookback : i]
            act_ret = (
                portfolio_returns.iloc[
                    -(len(factor_contrib) - i + lookback) : -(len(factor_contrib) - i)
                ]
                if i < len(factor_contrib)
                else portfolio_returns.iloc[-lookback:]
            )
            if len(est_ret) == len(act_ret) and len(est_ret) > 0:
                ic = est_ret.corr(act_ret)
                if not np.isnan(ic):
                    ic_values.append(ic)
        factor_ic = pd.Series(ic_values, name="factor_ic") if ic_values else pd.Series(dtype=float)

    return {
        "cumulative_factor_return_pct": round(float(cumulative_factor * 100), 2),
        "cumulative_alpha_return_pct": round(float(cumulative_alpha * 100), 2),
        "factor_contribution_series": factor_contrib,
        "alpha_contribution_series": pd.Series(alpha_contrib, index=fwd_returns.index),
        "factor_ic_mean": round(float(factor_ic.mean()), 4) if not factor_ic.empty else None,
        "factor_ic_std": round(float(factor_ic.std()), 4) if not factor_ic.empty else None,
        "factor_ic_ratio": round(float(factor_ic.mean() / max(factor_ic.std(), 1e-8)), 2)
        if not factor_ic.empty
        else None,
    }


def factor_timing_analysis(
    portfolio_returns: pd.Series,
    factor_exposures_over_time: Dict[str, pd.DataFrame],
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    results = []
    for ticker, exposure_df in factor_exposures_over_time.items():
        if exposure_df.empty:
            continue
        timing_scores = []
        for date in exposure_df.index:
            if date not in factor_returns.index:
                continue
            betas = exposure_df.loc[date]
            fwd_ret = factor_returns.loc[date]
            common = [
                c
                for c in betas.index
                if c in fwd_ret.index and c not in ("Alpha", "R_squared", "N_obs")
            ]
            if not common:
                continue
            cross = betas[common].rank().corr(fwd_ret[common].rank())
            if not np.isnan(cross):
                timing_scores.append(cross)

        if timing_scores:
            results.append(
                {
                    "ticker": ticker,
                    "timing_ic_mean": float(np.mean(timing_scores)),
                    "timing_ic_std": float(np.std(timing_scores)),
                    "timing_ic_ratio": float(
                        np.mean(timing_scores) / max(np.std(timing_scores), 1e-8)
                    ),
                }
            )

    return (
        pd.DataFrame(results).sort_values("timing_ic_ratio", ascending=False)
        if results
        else pd.DataFrame()
    )
