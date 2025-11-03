import numpy as np
import pandas as pd

from risk_monitor.data.quality import (
    check_data_freshness,
    detect_outliers,
    detect_price_jumps,
    validate_returns,
)


def test_check_data_freshness():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = pd.DataFrame(
        {"AAPL": np.random.uniform(150, 200, 100), "MSFT": np.random.uniform(300, 400, 100)},
        index=dates,
    )
    reports = check_data_freshness(prices)
    assert "AAPL" in reports
    assert "MSFT" in reports
    assert reports["AAPL"].passes


def test_detect_outliers_mad():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    returns = pd.DataFrame({"AAPL": np.random.normal(0, 0.02, 100)}, index=dates)
    returns.loc[dates[50], "AAPL"] = 0.5
    outliers = detect_outliers(returns, method="mad", threshold=5.0)
    assert len(outliers["AAPL"]) >= 1


def test_detect_outliers_zscore():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    returns = pd.DataFrame({"AAPL": np.random.normal(0, 0.02, 100)}, index=dates)
    returns.loc[dates[50], "AAPL"] = 0.5
    outliers = detect_outliers(returns, method="zscore", threshold=4.0)
    assert len(outliers.get("AAPL", [])) >= 1


def test_detect_price_jumps():
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(100, 200, 50)}, index=dates)
    prices.loc[dates[25], "AAPL"] = 500
    jumps = detect_price_jumps(prices)
    assert len(jumps.get("AAPL", [])) >= 1


def test_validate_returns_ok():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    returns = pd.DataFrame(
        {"AAPL": np.random.normal(0, 0.02, 100)},
        index=dates,
    )
    issues = validate_returns(returns)
    assert len(issues) == 0


def test_validate_returns_insufficient():
    returns = pd.DataFrame({"AAPL": [0.01]})
    issues = validate_returns(returns)
    assert len(issues) == 1


def test_validate_returns_missing():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    returns = pd.DataFrame({"AAPL": np.random.normal(0, 0.02, 100)}, index=dates)
    returns.iloc[:60] = np.nan
    issues = validate_returns(returns)
    assert len(issues) == 1
