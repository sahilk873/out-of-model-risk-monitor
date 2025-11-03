import pandas as pd

from risk_monitor.portfolio.attribution import (
    BrinsonAttribution,
    compute_exposure_drift,
    compute_turnover,
)


def test_brinson_decomposition():
    attr = BrinsonAttribution()
    pw = pd.Series({"AAPL": 0.05, "MSFT": 0.04, "GOOGL": 0.03, "JPM": 0.02})
    bw = pd.Series({"AAPL": 0.03, "MSFT": 0.03, "GOOGL": 0.02, "JPM": 0.02})
    pr = pd.Series({"AAPL": 0.01, "MSFT": 0.02, "GOOGL": 0.015, "JPM": 0.005})
    br = pd.Series({"AAPL": 0.008, "MSFT": 0.015, "GOOGL": 0.012, "JPM": 0.006})
    sector_map = pd.Series(
        {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "JPM": "Financials",
        }
    )
    result = attr.decompose(pw, bw, pr, br, sector_map)
    assert "total" in result
    assert "by_sector" in result
    assert "allocation" in result["total"]
    assert "Technology" in result["by_sector"]


def test_geometric_attribution():
    attr = BrinsonAttribution()
    result = attr.geometric_attribution(0.15, 0.10)
    assert result["active_return_bps"] == 500.0
    assert result["portfolio_return_pct"] == 15.0


def test_compute_turnover():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    pf = pd.DataFrame(
        {"AAPL": [0.05, 0.04, 0.06], "MSFT": [0.03, 0.04, 0.02]},
        index=dates,
    )
    turnover = compute_turnover(pf)
    assert len(turnover) == len(dates)
    assert turnover.iloc[0] == 0.0
    assert turnover.iloc[1] > 0


def test_compute_exposure_drift():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    exposures = pd.DataFrame(
        {"Mkt-RF": [1.0, 0.9, 1.1], "SMB": [0.2, 0.3, 0.1]},
        index=dates,
    )
    drift = compute_exposure_drift(exposures)
    assert len(drift) == len(dates)
