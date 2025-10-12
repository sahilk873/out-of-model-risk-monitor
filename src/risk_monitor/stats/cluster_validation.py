from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score


def compute_silhouette_scores(
    residuals: pd.DataFrame,
    max_clusters: int = 10,
) -> pd.DataFrame:
    residuals = residuals.dropna(how="all").fillna(0.0)
    if residuals.shape[1] < 2:
        return pd.DataFrame()

    dist_mat = np.clip(1.0 - residuals.corr().values, 0, 2)
    np.fill_diagonal(dist_mat, 0.0)
    condensed = pdist(dist_mat)
    condensed = np.nan_to_num(condensed, nan=1.0)

    results = []
    for n_clusters in range(2, min(max_clusters + 1, residuals.shape[1])):
        Z = linkage(condensed, method="average")
        from scipy.cluster.hierarchy import fcluster

        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(dist_mat, labels, metric="precomputed")
            results.append({"n_clusters": n_clusters, "silhouette_score": float(score)})
        except Exception:
            continue

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("silhouette_score", ascending=False)


def gap_statistic(
    residuals: pd.DataFrame,
    max_clusters: int = 10,
    n_reference: int = 100,
) -> pd.DataFrame:
    residuals = residuals.dropna(how="all").fillna(0.0)
    if residuals.shape[1] < 2:
        return pd.DataFrame()

    corr = residuals.corr().values
    dist_mat = np.clip(1.0 - corr, 0, 2)
    np.fill_diagonal(dist_mat, 0.0)

    tickers = list(residuals.columns)
    n = len(tickers)

    def _within_cluster_disp(labels: np.ndarray, n_clusters: int) -> float:
        disp = 0.0
        for k in range(1, n_clusters + 1):
            members = np.where(labels == k)[0]
            if len(members) < 2:
                continue
            sub_idx = np.ix_(members, members)
            D_k = dist_mat[sub_idx].sum() / (2 * len(members))
            disp += D_k
        return disp

    results = []
    for n_clusters in range(2, min(max_clusters + 1, n)):
        Z = linkage(pdist(dist_mat), method="average")
        from scipy.cluster.hierarchy import fcluster

        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        log_w = np.log(_within_cluster_disp(labels, n_clusters))

        ref_disp = []
        for _ in range(n_reference):
            ref_corr = np.random.uniform(-0.5, 0.5, (n, n))
            ref_corr = (ref_corr + ref_corr.T) / 2
            np.fill_diagonal(ref_corr, 1.0)
            ref_dist = np.clip(1.0 - ref_corr, 0, 2)
            np.fill_diagonal(ref_dist, 0.0)
            Z_ref = linkage(pdist(ref_dist), method="average")
            ref_labels = fcluster(Z_ref, t=n_clusters, criterion="maxclust")
            ref_disp.append(np.log(_within_cluster_disp(ref_labels, n_clusters)))

        mean_ref = np.mean(ref_disp)
        gap = mean_ref - log_w
        std_ref = np.std(ref_disp)
        gap_std = std_ref * np.sqrt(1 + 1 / n_reference)

        results.append(
            {
                "n_clusters": n_clusters,
                "gap": float(gap),
                "gap_std": float(gap_std),
            }
        )

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("gap", ascending=False)
