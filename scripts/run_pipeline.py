#!/usr/bin/env python3
"""End-to-end out-of-model risk monitor pipeline."""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from risk_monitor.portfolio.ingestion import (
    load_portfolio_csv,
    compute_active_weights,
    validate_portfolio_set,
)
from risk_monitor.data.prices import YFinanceProvider, PriceCache
from risk_monitor.data.factors import get_ff_factors
from risk_monitor.factor_models.standard_factors import (
    estimate_factor_exposures,
    compute_residual_returns,
)
from risk_monitor.factor_models.sector_macro_proxies import SECTOR_ETFS
from risk_monitor.risk.attribution import (
    compute_covariance,
    factor_risk_attribution,
    security_risk_contribution,
)
from risk_monitor.risk.residual import cluster_residuals, summarize_clusters
from risk_monitor.themes.engine import ThemeRiskEngine
from risk_monitor.risk.stress import (
    run_historical_replay,
    run_factor_shock,
    HISTORICAL_SCENARIOS,
    FACTOR_SHOCKS,
)
from risk_monitor.alerts.generator import AlertGenerator
from risk_monitor.reports.report import generate_report, save_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")

STRESS_START = "2020-01-01"


def setup_output_dirs():
    for d in ["reports/tables", "reports/figures", "data"]:
        os.makedirs(d, exist_ok=True)


def run_pipeline(
    portfolio_file: str = "data/sample_portfolio.csv",
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
):
    logger.info("=" * 60)
    logger.info("OUT-OF-MODEL RISK MONITOR — Full Pipeline")
    logger.info("=" * 60)

    setup_output_dirs()
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    # ── Phase 2: Portfolio Ingestion ──
    logger.info("[Phase 2] Loading portfolio...")
    ps = load_portfolio_csv(portfolio_file)
    issues = validate_portfolio_set(ps)
    for iss in issues:
        logger.warning(f"Portfolio issue: {iss}")

    active_portfolio = ps.get("active")
    try:
        benchmark = ps.get("__benchmark__")
        logger.info(f"Benchmark found with {len(benchmark.tickers)} tickers")
    except KeyError:
        benchmark = None
        logger.info("No benchmark found; using absolute weights.")

    active_weights = compute_active_weights(active_portfolio, benchmark)
    latest_weights = active_weights.iloc[-1]
    latest_weights = latest_weights[latest_weights.abs() > 0]
    logger.info(
        f"Active portfolio: {len(active_portfolio.tickers)} positions, "
        f"{len(latest_weights)} non-zero active weights"
    )

    # ── Phase 3: Data Pipeline ──
    logger.info("[Phase 3] Fetching price data...")
    all_tickers = list(
        set(active_portfolio.tickers)
        | (set(benchmark.tickers) if benchmark else set())
        | set(SECTOR_ETFS.keys())
    )
    price_provider = PriceCache(YFinanceProvider())

    # Fetch extended history for stress tests
    try:
        returns_wide = price_provider.get_returns(all_tickers, STRESS_START, end_date)
        logger.info(
            f"  Got returns for {len(returns_wide.columns)} tickers, {len(returns_wide)} days"
        )
    except ValueError as e:
        logger.error(f"Price data error: {e}")
        sys.exit(1)

    returns = price_provider.get_returns(all_tickers, start_date, end_date)
    logger.info(f"  Analysis period: {len(returns)} trading days")

    missing_holdings = [t for t in active_portfolio.tickers if t not in returns.columns]
    if missing_holdings:
        logger.warning(f"  Missing price data for: {missing_holdings}")

    # Fetch Fama-French factors
    logger.info("Fetching Fama-French factors...")
    ff_factors = pd.DataFrame()
    try:
        ff_raw = get_ff_factors(STRESS_START, end_date)
        ff_factors = get_ff_factors(start_date, end_date)
        logger.info(f"  Got {len(ff_factors.columns)} factors, {len(ff_factors)} days")
    except Exception as e:
        logger.error(f"FF factor fetch failed: {e}")

    # ── Phase 4: Factor Model ──
    logger.info("[Phase 4] Estimating factor exposures...")
    factor_exposures = pd.DataFrame()
    residuals = pd.DataFrame()
    if ff_factors.empty:
        logger.warning("  Skipping factor model (no factor data).")
    else:
        portfolio_returns = returns[[t for t in active_portfolio.tickers if t in returns.columns]]
        factor_exposures = estimate_factor_exposures(
            portfolio_returns, ff_factors, window=126, min_window=60
        )
        logger.info(f"  Estimated exposures for {len(factor_exposures)} tickers")

        residuals = compute_residual_returns(portfolio_returns, ff_factors, factor_exposures)
        logger.info(f"  Residual returns: {residuals.shape}")

    # ── Phase 5: Active Risk Attribution ──
    logger.info("[Phase 5] Computing active risk attribution...")
    risk_attribution_result: Dict = {}
    if ff_factors.empty or factor_exposures.empty:
        logger.warning("  Skipping risk attribution (no factor model).")
    else:
        fcols_available = [
            c
            for c in factor_exposures.columns
            if c in ff_factors.columns and c not in ("Alpha", "R_squared", "N_obs")
        ]
        factor_ret = ff_factors[fcols_available]
        factor_cov = compute_covariance(factor_ret)
        residual_var_series = residuals.var().fillna(0.0)

        risk_attribution_result = factor_risk_attribution(
            latest_weights, factor_exposures, factor_cov, residual_var_series
        )
        logger.info(
            f"  Active risk: {risk_attribution_result.get('total_risk_annual_pct', 0):.2f}% "
            f"(factor: {risk_attribution_result.get('factor_risk_pct_of_total', 0):.1f}%, "
            f"specific: {risk_attribution_result.get('specific_risk_pct_of_total', 0):.1f}%)"
        )

        sec_contrib = security_risk_contribution(
            latest_weights, factor_exposures, factor_cov, residual_var_series
        )
        save_table(sec_contrib, "security_risk_contribution")

    # ── Phase 6: Residual Risk Model ──
    logger.info("[Phase 6] Clustering residual returns...")
    cluster_summary = pd.DataFrame()
    if not residuals.empty:
        clusters = cluster_residuals(residuals, threshold=0.3)
        cluster_summary = summarize_clusters(clusters, residuals)
        logger.info(f"  Found {len(cluster_summary)} residual clusters")
        for _, row in cluster_summary.iterrows():
            logger.info(
                f"    {row['cluster']}: {row['n_securities']} members, "
                f"corr={row['avg_residual_corr']:.3f}"
            )

    # ── Phase 7: Theme Risk Engine ──
    logger.info("[Phase 7] Analyzing theme risks...")
    theme_engine = ThemeRiskEngine()
    portfolio_returns = returns[[t for t in active_portfolio.tickers if t in returns.columns]]
    theme_scores = theme_engine.analyze_all_themes(
        active_weights=latest_weights,
        residuals=residuals if not residuals.empty else pd.DataFrame(),
        returns=portfolio_returns,
        ticker_texts=None,
    )
    logger.info(f"  Analyzed {len(theme_scores)} themes")
    for _, row in theme_scores.iterrows():
        logger.info(
            f"    {row['theme']}: holdings={row['holdings_exposure_pct']:.1f}%, "
            f"resid_corr={row['residual_avg_corr']:.3f}"
        )

    # ── Phase 9: Stress Tests ──
    logger.info("[Phase 9] Running stress tests...")
    historical_results: Dict[str, float] = {}
    for scenario_name in HISTORICAL_SCENARIOS:
        try:
            loss = run_historical_replay(
                returns_wide,
                scenario_name,
                weights=latest_weights if scenario_name != "2008_Financial_Crisis" else None,
            )
            loss_val = float(np.asarray(loss).flat[0])
            historical_results[scenario_name] = loss_val
            logger.info(f"  {scenario_name}: {loss_val:.2f}%")
        except (ValueError, KeyError) as e:
            logger.warning(f"  {scenario_name}: {e}")

    factor_shock_results: Dict[str, float] = {}
    for shock_name in FACTOR_SHOCKS:
        if not factor_exposures.empty:
            try:
                pnl, _, _ = run_factor_shock(latest_weights, factor_exposures, shock_name)
                factor_shock_results[shock_name] = pnl
                logger.info(f"  {shock_name}: {pnl:.2f}%")
            except ValueError as e:
                logger.warning(f"  {shock_name}: {e}")
        else:
            logger.info(f"  {shock_name}: skipped (no factor model)")

    # ── Phase 8: Risk Alerts ──
    logger.info("[Phase 8] Generating risk alerts...")
    alert_gen = AlertGenerator()
    if risk_attribution_result:
        alert_gen.check_factor_concentration(risk_attribution_result.get("factor_contrib_pct", {}))
    if not theme_scores.empty:
        alert_gen.check_theme_exposure(theme_scores)
    if not cluster_summary.empty:
        alert_gen.check_residual_clusters(cluster_summary)
    alert_gen.check_stress_loss(historical_results, factor_shock_results)

    alerts = alert_gen.generate_report()
    logger.info(f"  Generated {len(alerts)} risk alerts:")
    for a in alerts:
        logger.info(f"    [{a['severity'].upper()}] {a['title']}")

    # ── Phase 10: Reports ──
    logger.info("[Phase 10] Generating reports...")
    portfolio_summary = {
        "name": "active",
        "n_positions": len(active_portfolio.tickers),
        "dates": f"{active_portfolio.dates[0].date()} to {active_portfolio.dates[-1].date()}",
        "gross_exposure": float(active_portfolio.gross_exposure.iloc[-1]),
        "net_exposure": float(active_portfolio.net_exposure.iloc[-1]),
    }

    stress_results = {
        "historical": {"by_security": {}, "by_factor": {}, "by_sector": {}},
        "factor_shock": factor_shock_results,
    }

    generate_report(
        portfolio_summary=portfolio_summary,
        factor_exposures=factor_exposures,
        risk_attribution=risk_attribution_result,
        cluster_summary=cluster_summary,
        theme_scores=theme_scores,
        stress_results=stress_results,
        alerts=alerts,
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — Reports in reports/")
    logger.info("=" * 60)

    return {
        "portfolio": ps,
        "factor_exposures": factor_exposures,
        "risk_attribution": risk_attribution_result,
        "residual_clusters": cluster_summary,
        "theme_scores": theme_scores,
        "stress_results": stress_results,
        "alerts": alerts,
    }


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Out-of-Model Risk Monitor Pipeline")
    parser.add_argument(
        "--portfolio", default="data/sample_portfolio.csv", help="Portfolio CSV path"
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    args = parser.parse_args()
    run_pipeline(args.portfolio, args.start, args.end)
