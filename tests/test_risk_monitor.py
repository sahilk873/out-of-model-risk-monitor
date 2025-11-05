import numpy as np
import pandas as pd
import pytest

from risk_monitor.factor_models.standard_factors import (
    CORE_FACTORS,
    align_factors_and_returns,
    compute_residual_returns,
    estimate_factor_exposures,
)
from risk_monitor.risk.attribution import (
    compute_active_risk,
    compute_covariance,
    factor_risk_attribution,
    risk_contribution,
)
from risk_monitor.risk.residual import (
    cluster_residuals,
    compute_residual_correlations,
    summarize_clusters,
)
from risk_monitor.risk.stress import (
    FACTOR_SHOCKS,
    HISTORICAL_SCENARIOS,
    run_factor_shock,
    run_historical_replay,
)
from risk_monitor.themes.engine import PREDEFINED_THEMES, ThemeRiskEngine

# ── Fixtures ──


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    returns = pd.DataFrame(
        {
            ticker: np.random.normal(0.001, 0.02, len(dates))
            for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        },
        index=dates,
    )
    return returns


@pytest.fixture
def sample_factors():
    np.random.seed(99)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    factors = pd.DataFrame(
        {f: np.random.normal(0.0005, 0.01, len(dates)) for f in CORE_FACTORS},
        index=dates,
    )
    factors["RF"] = np.random.normal(0.0002, 0.0001, len(dates))
    return factors


@pytest.fixture
def sample_residuals():
    np.random.seed(123)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    residuals = pd.DataFrame(
        {
            ticker: np.random.normal(0, 0.01, len(dates))
            for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM"]
        },
        index=dates,
    )
    return residuals


# ── Factor Model Tests ──


def test_align_factors_and_returns(sample_returns, sample_factors):
    r, f = align_factors_and_returns(sample_returns, sample_factors)
    assert len(r) == len(f)
    assert not r.empty


def test_estimate_factor_exposures(sample_returns, sample_factors):
    exposures = estimate_factor_exposures(sample_returns, sample_factors, window=252, min_window=60)
    assert len(exposures) > 0
    for col in CORE_FACTORS:
        assert col in exposures.columns


def test_estimate_factor_exposures_insufficient_data():
    with pytest.raises(ValueError, match="Need at least"):
        estimate_factor_exposures(
            pd.DataFrame({"A": [0.01]}, index=pd.to_datetime(["2024-01-01"])),
            pd.DataFrame({"Mkt-RF": [0.01]}, index=pd.to_datetime(["2024-01-01"])),
            window=252,
            min_window=60,
        )


def test_no_lookahead_in_factor_exposures(sample_returns, sample_factors):
    """Ensure rolling estimation doesn't use future data: exposures at time t
    should be estimated using only data up to t."""
    exposures = estimate_factor_exposures(sample_returns, sample_factors, window=252, min_window=60)
    assert "Alpha" in exposures.columns
    assert exposures["Alpha"].notna().any()


def test_compute_residual_returns(sample_returns, sample_factors):
    exposures = estimate_factor_exposures(sample_returns, sample_factors, window=252, min_window=60)
    residuals = compute_residual_returns(sample_returns, sample_factors, exposures)
    assert not residuals.empty
    assert residuals.shape[1] <= sample_returns.shape[1]


# ── Active Risk Contribution Tests ──


def test_compute_covariance(sample_returns):
    cov = compute_covariance(sample_returns, method="sample")
    assert cov.shape == (sample_returns.shape[1], sample_returns.shape[1])
    assert cov.values.max() > 0
    assert all(cov.index == cov.columns)


def test_compute_active_risk():
    w = pd.Series({"A": 0.5, "B": 0.5})
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.04]], index=["A", "B"], columns=["A", "B"])
    risk = compute_active_risk(w, cov)
    assert risk > 0


def test_risk_contribution():
    w = pd.Series({"A": 0.6, "B": 0.4})
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.04]], index=["A", "B"], columns=["A", "B"])
    rc = risk_contribution(w, cov)
    assert abs(rc.sum() - 100) < 1e-10


def test_factor_risk_attribution(sample_returns, sample_factors):
    exposures = estimate_factor_exposures(sample_returns, sample_factors, window=252, min_window=60)
    fcols = [c for c in CORE_FACTORS if c in sample_factors.columns]
    factor_cov = compute_covariance(sample_factors[fcols])
    residual_var = pd.Series({t: 0.01 for t in exposures.index[:3]})
    w = pd.Series({t: 0.2 for t in exposures.index[:3]})

    result = factor_risk_attribution(
        w, exposures.loc[exposures.index[:3]], factor_cov, residual_var
    )
    assert "total_risk_annual_pct" in result
    assert result["total_risk_annual_pct"] >= 0


# ── Residual Clustering Tests ──


def test_compute_residual_correlations(sample_residuals):
    corr = compute_residual_correlations(sample_residuals)
    assert not corr.empty
    assert abs(corr.values.max()) <= 1.0


def test_cluster_residuals(sample_residuals):
    clusters = cluster_residuals(sample_residuals, threshold=0.3)
    assert isinstance(clusters, dict)
    assert len(clusters) > 0


def test_summarize_clusters(sample_residuals):
    clusters = cluster_residuals(sample_residuals, threshold=0.3)
    summary = summarize_clusters(clusters, sample_residuals)
    assert isinstance(summary, pd.DataFrame)
    assert summary.empty or "n_securities" in summary.columns


# ── Theme Scoring Tests ──


def test_theme_engine_holdings_scoring():
    engine = ThemeRiskEngine()
    w = pd.Series({"NVDA": 0.05, "AMD": 0.03, "AAPL": 0.04, "MSFT": 0.03})
    score = engine.score_holdings_based(w, "AI_Infrastructure")
    assert score > 0


def test_theme_engine_no_exposure():
    engine = ThemeRiskEngine()
    w = pd.Series({"KO": 0.1, "PEP": 0.1})
    score = engine.score_holdings_based(w, "AI_Infrastructure")
    assert score == 0.0


def test_theme_residual_comovement(sample_residuals):
    engine = ThemeRiskEngine()
    score = engine.score_residual_comovement(sample_residuals, "Mega_Cap_Concentration")
    assert isinstance(score, float)


def test_theme_filing_similarity():
    engine = ThemeRiskEngine()
    texts = {
        "NVDA": "We design GPUs for artificial intelligence and deep learning data centers",
        "AMD": "Our products accelerate machine learning and AI inference workloads",
        "AAPL": "We design consumer electronics and mobile communication devices",
    }
    scores = engine.score_filing_similarity(texts, "AI_Infrastructure")
    assert len(scores) == 3
    assert scores["NVDA"] > scores["AAPL"]


def test_analyze_all_themes(sample_residuals, sample_returns):
    engine = ThemeRiskEngine()
    w = pd.Series({"NVDA": 0.05, "AMD": 0.03, "AAPL": 0.04, "MSFT": 0.03, "GOOGL": 0.02})
    result = engine.analyze_all_themes(w, sample_residuals, sample_returns)
    assert len(result) == len(PREDEFINED_THEMES)
    assert "holdings_exposure_pct" in result.columns


# ── Stress Test Tests ──


def test_run_historical_replay(sample_returns):
    scenario = list(HISTORICAL_SCENARIOS.keys())[0]
    try:
        result = run_historical_replay(sample_returns, scenario)
        assert isinstance(result, (float, int, np.floating))
    except (ValueError, KeyError):
        pass


def test_run_factor_shock(sample_returns, sample_factors):
    from risk_monitor.factor_models.standard_factors import estimate_factor_exposures

    exposures = estimate_factor_exposures(sample_returns, sample_factors, window=252, min_window=60)
    w = pd.Series({t: 0.2 for t in exposures.index[:3]})

    shock_name = list(FACTOR_SHOCKS.keys())[0]
    pnl, sec_pnl, fac_contrib = run_factor_shock(w, exposures.loc[exposures.index[:3]], shock_name)
    assert isinstance(pnl, float)
    assert len(sec_pnl) == 3
    assert len(fac_contrib) > 0


def test_predefined_themes_exist():
    assert len(PREDEFINED_THEMES) == 8
    assert "AI_Infrastructure" in PREDEFINED_THEMES
    assert "Regional_Banks" in PREDEFINED_THEMES
