import pandas as pd
import pytest

from risk_monitor.portfolio.ingestion import (
    Portfolio,
    PortfolioSet,
    compute_active_weights,
    load_portfolio_csv,
    validate_portfolio_set,
    validate_weights,
)


def test_validate_weights_ok():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "ticker": ["AAPL", "MSFT"],
            "weight": [0.05, 0.03],
            "side": ["long", "long"],
            "portfolio_name": ["active", "active"],
        }
    )
    result = validate_weights(df)
    assert not result.empty
    assert "date" in result.columns


def test_validate_weights_missing_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        validate_weights(pd.DataFrame({"ticker": ["AAPL"]}))


def test_validate_weights_duplicates():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "ticker": ["AAPL", "AAPL"],
            "weight": [0.05, 0.05],
            "side": ["long", "long"],
            "portfolio_name": ["active", "active"],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_weights(df)


def test_validate_weights_zero_weight():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "ticker": ["AAPL"],
            "weight": [0.0],
            "side": ["long"],
            "portfolio_name": ["active"],
        }
    )
    with pytest.raises(ValueError, match="Zero or NaN"):
        validate_weights(df)


def test_load_portfolio_csv(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text(
        "date,ticker,weight,side,portfolio_name\n"
        "2024-01-01,AAPL,0.05,long,active\n"
        "2024-01-01,MSFT,0.03,long,active\n"
        "2024-01-01,SPY,0.10,long,\n"
        "2024-01-01,QQQ,0.05,long,\n"
    )
    ps = load_portfolio_csv(str(csv))
    assert "active" in ps.names
    bench_names = [n for n in ps.names if n != "active"]
    assert len(bench_names) == 1
    assert "benchmark" in bench_names[0].lower()


def test_compute_active_weights():
    pf = Portfolio(
        name="active",
        weights=pd.DataFrame(
            {"AAPL": [0.05], "MSFT": [0.03]}, index=pd.to_datetime(["2024-01-01"])
        ),
    )
    bm = Portfolio(
        name="benchmark",
        weights=pd.DataFrame({"AAPL": [0.03]}, index=pd.to_datetime(["2024-01-01"])),
    )
    active = compute_active_weights(pf, bm)
    assert "AAPL" in active.columns
    assert "MSFT" in active.columns


def test_compute_active_weights_no_benchmark():
    pf = Portfolio(
        name="active", weights=pd.DataFrame({"AAPL": [0.05]}, index=pd.to_datetime(["2024-01-01"]))
    )
    active = compute_active_weights(pf, None)
    assert active["AAPL"].iloc[0] == 0.05


def test_portfolio_set():
    ps = PortfolioSet()
    pf = Portfolio(
        name="test", weights=pd.DataFrame({"A": [0.1]}, index=pd.to_datetime(["2024-01-01"]))
    )
    ps.portfolios["test"] = pf
    assert ps.get("test").name == "test"
    with pytest.raises(KeyError):
        ps.get("nonexistent")


def test_portfolio_properties():
    w = pd.DataFrame(
        {"AAPL": [0.05, 0.06], "MSFT": [0.03, 0.02]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    pf = Portfolio(name="test", weights=w)
    assert pf.is_long_only
    assert pf.tickers == ["AAPL", "MSFT"]
    assert len(pf.dates) == 2
    assert pf.gross_exposure.iloc[0] == 0.08


def test_validate_portfolio_set():
    w = pd.DataFrame({"AAPL": [0.05]}, index=pd.to_datetime(["2024-01-01"]))
    ps = PortfolioSet(portfolios={"ok": Portfolio(name="ok", weights=w)})
    assert len(validate_portfolio_set(ps)) == 0
