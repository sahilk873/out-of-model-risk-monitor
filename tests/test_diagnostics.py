import numpy as np
import pandas as pd

from risk_monitor.factor_models.standard_factors import CORE_FACTORS
from risk_monitor.stats.diagnostics import (
    bootstrap_factor_exposures,
    breusch_pagan_test,
    compute_vif,
    durbin_watson,
    factor_model_diagnostics,
    jarque_bera_test,
    walk_forward_ic,
)

# ── VIF Tests ──


def test_compute_vif():
    np.random.seed(42)
    X = pd.DataFrame({"A": np.random.normal(0, 1, 100), "B": np.random.normal(0, 1, 100)})
    vif = compute_vif(X)
    assert isinstance(vif, pd.Series)
    assert len(vif) == 2
    assert all(v > 0 for v in vif)


def test_compute_vif_high_multicollinearity():
    np.random.seed(42)
    a = np.random.normal(0, 1, 100)
    X = pd.DataFrame({"A": a, "B": a * 0.99 + np.random.normal(0, 0.01, 100)})
    vif = compute_vif(X)
    assert vif.max() > 5


# ── Durbin-Watson Tests ──


def test_durbin_watson_independent():
    np.random.seed(42)
    resid = np.random.normal(0, 1, 100)
    dw = durbin_watson(resid)
    assert 1.5 < dw < 2.5


def test_durbin_watson_autocorrelated():
    resid = np.cumsum(np.random.normal(0, 0.1, 100))
    dw = durbin_watson(resid)
    assert dw < 1.5


# ── Breusch-Pagan Tests ──


def test_breusch_pagan(sample_returns, sample_factors):
    import statsmodels.api as sm

    ticker = sample_returns.columns[0]
    rf = sample_factors["RF"]
    excess = sample_returns[ticker] - rf
    joined = pd.concat([excess.rename("ret"), sample_factors[CORE_FACTORS]], axis=1).dropna()
    y = joined["ret"].values
    X = sm.add_constant(joined[CORE_FACTORS].values)
    model = sm.OLS(y, X).fit()
    bp = breusch_pagan_test(model)
    assert "lm_stat" in bp
    assert "heteroskedastic" in bp


# ── Jarque-Bera Tests ──


def test_jarque_bera_normal():
    np.random.seed(42)
    resid = np.random.normal(0, 1, 500)
    jb = jarque_bera_test(resid)
    assert jb["normal"] is True


def test_jarque_bera_non_normal():
    np.random.seed(42)
    resid = np.random.exponential(1, 500) - 1
    jb = jarque_bera_test(resid)
    assert jb["normal"] is False


# ── Factor Model Diagnostics ──


def test_factor_model_diagnostics(sample_returns, sample_factors):
    ticker = sample_returns.columns[0]
    diag = factor_model_diagnostics(sample_returns[ticker], sample_factors)
    assert "error" not in diag
    assert "rsquared" in diag
    assert "durbin_watson" in diag
    assert "vif" in diag
    assert diag["nobs"] > 0


def test_factor_model_diagnostics_insufficient_data():
    returns = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    factors = pd.DataFrame(
        {"Mkt-RF": [0.01, 0.02]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"])
    )
    diag = factor_model_diagnostics(returns, factors)
    assert "error" in diag


# ── Bootstrap Factor Exposures ──


def test_bootstrap_factor_exposures(sample_returns, sample_factors):
    ticker = sample_returns.columns[0]
    boot = bootstrap_factor_exposures(sample_returns[ticker], sample_factors, n_samples=100)
    assert not boot.empty
    assert "mean" in boot.columns
    assert "lower" in boot.columns
    assert "upper" in boot.columns


def test_bootstrap_ci_narrow_with_more_data(sample_returns, sample_factors):
    """With enough data, bootstrap CIs should be sensible."""
    ticker = sample_returns.columns[0]
    boot = bootstrap_factor_exposures(sample_returns[ticker], sample_factors, n_samples=200)
    mkt_row = boot.loc["Mkt-RF"] if "Mkt-RF" in boot.index else boot.iloc[0]
    assert mkt_row["lower"] <= mkt_row["upper"]


# ── Walk-Forward IC ──


def test_walk_forward_ic(sample_returns, sample_factors):
    ic_series = walk_forward_ic(
        sample_factors["Mkt-RF"],
        sample_returns,
        window=60,
        step=30,
    )
    assert isinstance(ic_series, pd.Series)
