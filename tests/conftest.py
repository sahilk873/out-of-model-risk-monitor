import numpy as np
import pandas as pd
import pytest

from risk_monitor.factor_models.standard_factors import CORE_FACTORS


@pytest.fixture
def sample_dates():
    return pd.date_range("2024-01-01", periods=252, freq="B")


@pytest.fixture
def sample_tickers():
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


@pytest.fixture
def sample_returns(sample_dates, sample_tickers):
    np.random.seed(42)
    return pd.DataFrame(
        {ticker: np.random.normal(0.001, 0.02, len(sample_dates)) for ticker in sample_tickers},
        index=sample_dates,
    )


@pytest.fixture
def sample_factors(sample_dates):
    np.random.seed(99)
    factors = pd.DataFrame(
        {f: np.random.normal(0.0005, 0.01, len(sample_dates)) for f in CORE_FACTORS},
        index=sample_dates,
    )
    factors["RF"] = np.random.normal(0.0002, 0.0001, len(sample_dates))
    return factors


@pytest.fixture
def sample_residuals():
    np.random.seed(123)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    return pd.DataFrame(
        {
            ticker: np.random.normal(0, 0.01, len(dates))
            for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM"]
        },
        index=dates,
    )


@pytest.fixture
def sample_weights():
    return pd.Series({"AAPL": 0.05, "MSFT": 0.04, "GOOGL": 0.03, "AMZN": 0.03, "NVDA": 0.02})


@pytest.fixture
def sample_cov():
    tickers = ["A", "B"]
    return pd.DataFrame([[0.04, 0.01], [0.01, 0.04]], index=tickers, columns=tickers)
