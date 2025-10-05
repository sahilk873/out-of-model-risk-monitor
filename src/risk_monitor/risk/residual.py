from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def compute_residual_correlations(residuals: pd.DataFrame) -> pd.DataFrame:
    residuals = residuals.dropna(how="all").fillna(0.0)
    if residuals.shape[1] < 2:
        return pd.DataFrame()
    corr = residuals.corr()
    return corr.fillna(0.0)


def cluster_residuals(
    residuals: pd.DataFrame,
    threshold: float = 0.5,
    method: str = "average",
) -> Dict[str, List[str]]:
    corr = compute_residual_correlations(residuals)
    if corr.empty:
        return {}

    dist_mat = np.clip(1.0 - corr.values, 0, 2)
    np.fill_diagonal(dist_mat, 0.0)

    condensed = pdist(dist_mat)
    if np.isnan(condensed).any():
        condensed = np.nan_to_num(condensed, nan=1.0)

    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=threshold, criterion="distance")
    tickers = list(corr.columns)

    clusters: Dict[str, List[str]] = {}
    for ticker, label in zip(tickers, labels):
        key = f"RC{label}"
        if key not in clusters:
            clusters[key] = []
        clusters[key].append(ticker)

    return clusters


def summarize_clusters(
    clusters: Dict[str, List[str]],
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cl_name, members in clusters.items():
        if len(members) < 2:
            continue
        cl_res = residuals[members].fillna(0.0)
        avg_corr = cl_res.corr().values[np.triu_indices_from(cl_res.corr().values, k=1)].mean()
        rows.append(
            {
                "cluster": cl_name,
                "n_securities": len(members),
                "members": ", ".join(sorted(members)),
                "avg_residual_corr": float(avg_corr),
                "total_var_explained": float(cl_res.sum(axis=1).var()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "cluster",
                "n_securities",
                "members",
                "avg_residual_corr",
                "total_var_explained",
            ]
        )
    return pd.DataFrame(rows).sort_values("avg_residual_corr", ascending=False)
