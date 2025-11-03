import numpy as np
import pandas as pd
import pytest

from risk_monitor.portfolio.optimization import (
    PortfolioOptimizer,
    compute_efficient_frontier,
)


@pytest.fixture
def cov_matrix():
    np.random.seed(42)
    n = 5
    tickers = [f"A{i}" for i in range(n)]
    vol = np.random.uniform(0.15, 0.35, n)
    # Generate well-conditioned correlation matrix via random factor model
    f = np.random.normal(0, 1, (n, 2))
    corr = f @ f.T + np.eye(n) * 0.3
    corr = corr / np.sqrt(np.outer(np.diag(corr), np.diag(corr)))
    corr = corr.clip(-1, 1)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vol, vol) * corr
    return pd.DataFrame(cov, index=tickers, columns=tickers)


def test_minimum_variance(cov_matrix):
    opt = PortfolioOptimizer()
    w = opt.minimum_variance(cov_matrix, max_weight=0.30)
    assert abs(w.sum() - 1.0) < 1e-6
    assert all(w >= -1e-6)
    assert all(w <= 0.30 + 1e-6)


def test_risk_parity(cov_matrix):
    opt = PortfolioOptimizer()
    w = opt.risk_parity(cov_matrix)
    assert abs(w.sum() - 1.0) < 1e-6


def test_max_diversification(cov_matrix):
    opt = PortfolioOptimizer()
    w = opt.max_diversification(cov_matrix)
    assert abs(w.sum() - 1.0) < 1e-6


def test_hrp(cov_matrix):
    opt = PortfolioOptimizer()
    w = opt.hrp(cov_matrix)
    assert abs(w.sum() - 1.0) < 1e-6


def test_min_variance_lower_vol_than_equal_weight(cov_matrix):
    opt = PortfolioOptimizer()
    w_mv = opt.minimum_variance(cov_matrix)
    n = len(cov_matrix)
    w_ew = pd.Series(np.ones(n) / n, index=cov_matrix.index)

    vol_mv = np.sqrt(w_mv.values @ cov_matrix.values @ w_mv.values)
    vol_ew = np.sqrt(w_ew.values @ cov_matrix.values @ w_ew.values)
    assert vol_mv <= vol_ew + 1e-8


def test_compute_efficient_frontier():
    np.random.seed(42)
    n = 5
    tickers = [f"A{i}" for i in range(n)]
    cov = pd.DataFrame(np.random.uniform(0.01, 0.04, (n, n)), index=tickers, columns=tickers)
    cov = cov @ cov.T / n
    returns = pd.Series(np.random.uniform(0.0005, 0.002, n), index=tickers)
    ef = compute_efficient_frontier(cov, returns, n_points=10)
    assert isinstance(ef, pd.DataFrame)
    if not ef.empty:
        assert "vol" in ef.columns
        assert "ret" in ef.columns
        assert "sharpe" in ef.columns


def test_hrp_single_asset():
    opt = PortfolioOptimizer()
    cov = pd.DataFrame({"A": [0.04]}, index=["A"])
    w = opt.hrp(cov)
    assert abs(w["A"] - 1.0) < 1e-6
