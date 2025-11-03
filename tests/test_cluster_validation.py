import numpy as np
import pandas as pd

from risk_monitor.stats.cluster_validation import compute_silhouette_scores, gap_statistic


def test_compute_silhouette_scores(sample_residuals):
    scores = compute_silhouette_scores(sample_residuals, max_clusters=5)
    assert isinstance(scores, pd.DataFrame)
    if not scores.empty:
        assert "silhouette_score" in scores.columns
        assert scores["silhouette_score"].iloc[0] > -1
        assert scores["silhouette_score"].iloc[0] < 1


def test_compute_silhouette_scores_insufficient_data():
    residuals = pd.DataFrame({"A": [0.1, 0.2, 0.3]})
    scores = compute_silhouette_scores(residuals)
    assert scores.empty


def test_gap_statistic(sample_residuals):
    gap = gap_statistic(sample_residuals, max_clusters=5, n_reference=10)
    assert isinstance(gap, pd.DataFrame)
    if not gap.empty:
        assert "gap" in gap.columns


def test_silhouette_optimal_clusters():
    """With clearly separated clusters, silhouette should identify structure."""
    np.random.seed(42)
    n = 30
    # Two clear clusters
    residuals = pd.DataFrame(
        {
            "A1": np.random.normal(0, 0.5, n),
            "A2": np.random.normal(0, 0.5, n),
            "A3": np.random.normal(0, 0.5, n),
            "B1": np.random.normal(2, 0.5, n),
            "B2": np.random.normal(2, 0.5, n),
            "B3": np.random.normal(2, 0.5, n),
        }
    )
    scores = compute_silhouette_scores(residuals, max_clusters=5)
    if not scores.empty:
        assert scores["silhouette_score"].iloc[0] > 0
