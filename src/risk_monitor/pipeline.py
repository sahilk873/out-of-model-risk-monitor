from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from risk_monitor.alerts.generator import AlertGenerator
from risk_monitor.config import MonitorConfig
from risk_monitor.data.cache import CachedPriceProvider, DiskCache
from risk_monitor.data.factors import get_ff_factors
from risk_monitor.data.prices import YFinanceProvider
from risk_monitor.data.quality import (
    check_data_freshness,
    detect_outliers,
    detect_price_jumps,
    validate_returns,
)
from risk_monitor.factor_models.sector_macro_proxies import SECTOR_ETFS
from risk_monitor.factor_models.standard_factors import (
    compute_residual_returns,
    estimate_factor_exposures,
)
from risk_monitor.portfolio.attribution import (
    compute_turnover,
    factor_performance_attribution,
)
from risk_monitor.portfolio.ingestion import (
    Portfolio,
    PortfolioSet,
    compute_active_weights,
    load_portfolio_csv,
    validate_portfolio_set,
)
from risk_monitor.reports.report import generate_report, save_table
from risk_monitor.risk.attribution import (
    compute_covariance,
    factor_risk_attribution,
    security_risk_contribution,
)
from risk_monitor.risk.crowding import (
    compute_dispersion_ratio,
    compute_herfindahl_index,
    detect_crowded_trades,
)
from risk_monitor.risk.regimes import (
    detect_volatility_regimes,
    entropy_regime_concentration,
    regime_stability_index,
)
from risk_monitor.risk.residual import cluster_residuals, summarize_clusters
from risk_monitor.risk.stress import (
    FACTOR_SHOCKS,
    HISTORICAL_SCENARIOS,
    run_factor_shock,
    run_historical_replay,
)
from risk_monitor.risk.transaction_costs import TransactionCostModel
from risk_monitor.stats.diagnostics import (
    bootstrap_factor_exposures,
    factor_model_diagnostics,
)
from risk_monitor.themes.engine import ThemeRiskEngine
from risk_monitor.utils import setup_logging

logger = setup_logging("pipeline")


@dataclass
class PipelineContext:
    portfolio_file: str
    start_date: str
    end_date: str
    stress_start: str
    config: MonitorConfig = field(default_factory=MonitorConfig.from_env)

    portfolio_set: Optional[PortfolioSet] = None
    active_portfolio: Optional[Portfolio] = None
    benchmark: Optional[Portfolio] = None
    active_weights: Optional[pd.DataFrame] = None
    latest_weights: Optional[pd.Series] = None
    returns_wide: Optional[pd.DataFrame] = None
    returns: Optional[pd.DataFrame] = None
    ff_factors: Optional[pd.DataFrame] = None
    factor_exposures: Optional[pd.DataFrame] = None
    residuals: Optional[pd.DataFrame] = None
    risk_attribution_result: Dict = field(default_factory=dict)
    cluster_summary: Optional[pd.DataFrame] = None
    theme_scores: Optional[pd.DataFrame] = None
    alerts: List[Dict] = field(default_factory=list)
    stress_results: Dict = field(default_factory=dict)
    diagnostics: Dict = field(default_factory=dict)
    regime_result: Optional[Dict] = None
    crowding_result: Optional[Dict] = None
    transaction_costs: Optional[pd.Series] = None
    factor_perf_attr: Dict = field(default_factory=dict)
    cv_results: Optional[pd.DataFrame] = None
    dispersion: Optional[pd.Series] = None
    turnover: Optional[pd.Series] = None

    def save_checkpoint(self, phase: str) -> None:
        os.makedirs("data/checkpoints", exist_ok=True)
        path = f"data/checkpoints/{phase}.json"
        with open(path, "w") as f:
            json.dump({"phase": phase, "timestamp": datetime.now().isoformat()}, f)


def run_pipeline(
    portfolio_file: str = "data/sample_portfolio.csv",
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    stress_start: str = "2020-01-01",
    config: Optional[MonitorConfig] = None,
) -> PipelineContext:
    cfg = config or MonitorConfig.from_env()
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    ctx = PipelineContext(
        portfolio_file=portfolio_file,
        start_date=start_date,
        end_date=end_date,
        stress_start=stress_start,
        config=cfg,
    )

    logger.info("=" * 60)
    logger.info(
        "OUT-OF-MODEL RISK MONITOR — Full Pipeline v%s", __import__("risk_monitor").__version__
    )
    logger.info("=" * 60)

    # ── Phase 1: Portfolio Ingestion ──
    logger.info("[Phase 1] Loading portfolio from %s ...", portfolio_file)
    ctx.portfolio_set = load_portfolio_csv(portfolio_file)
    issues = validate_portfolio_set(ctx.portfolio_set)
    for iss in issues:
        logger.warning("Portfolio issue: %s", iss)

    ctx.active_portfolio = ctx.portfolio_set.get("active")
    try:
        ctx.benchmark = ctx.portfolio_set.get("__benchmark__")
        logger.info("Benchmark found with %d tickers", len(ctx.benchmark.tickers))
    except KeyError:
        ctx.benchmark = None
        logger.info("No benchmark found; using absolute weights.")

    ctx.active_weights = compute_active_weights(ctx.active_portfolio, ctx.benchmark)
    ctx.latest_weights = ctx.active_weights.iloc[-1]
    ctx.latest_weights = ctx.latest_weights[ctx.latest_weights.abs() > 0]
    logger.info(
        "Active portfolio: %d positions, %d non-zero active weights",
        len(ctx.active_portfolio.tickers),
        len(ctx.latest_weights),
    )

    ctx.turnover = compute_turnover(ctx.active_weights)
    ctx.save_checkpoint("01_portfolio")

    # ── Phase 2: Data Pipeline ──
    logger.info("[Phase 2] Fetching data...")
    all_tickers = list(
        set(ctx.active_portfolio.tickers)
        | (set(ctx.benchmark.tickers) if ctx.benchmark else set())
        | set(SECTOR_ETFS.keys())
    )
    price_provider = CachedPriceProvider(YFinanceProvider(), DiskCache(cfg.data))
    ctx.returns_wide = price_provider.get_returns(all_tickers, stress_start, end_date)
    ctx.returns = price_provider.get_returns(all_tickers, start_date, end_date)
    logger.info(
        "Returns: %d tickers, %d days (stress), %d days (analysis)",
        len(ctx.returns_wide.columns),
        len(ctx.returns_wide),
        len(ctx.returns),
    )

    missing_holdings = [t for t in ctx.active_portfolio.tickers if t not in ctx.returns.columns]
    if missing_holdings:
        logger.warning("Missing price data for: %s", missing_holdings)

    freshness = check_data_freshness(price_provider.get_prices(all_tickers, start_date, end_date))
    stale_tickers = [t for t, r in freshness.items() if r.flags]
    if stale_tickers:
        logger.warning("Stale data: %s", stale_tickers)

    outliers = detect_outliers(ctx.returns)
    for t, dates in outliers.items():
        logger.warning("Outliers detected for %s on %s", t, dates[:3])

    jump_issues = detect_price_jumps(price_provider.get_prices(all_tickers, start_date, end_date))
    for t, jumps in jump_issues.items():
        logger.warning("Large price jumps for %s: %s", t, jumps[:2])

    ret_issues = validate_returns(ctx.returns)
    for iss in ret_issues:
        logger.warning("Return validation: %s", iss)

    logger.info("Fetching Fama-French factors...")
    try:
        ctx.ff_factors = get_ff_factors(start_date, end_date)
        logger.info("Got %d factors, %d days", len(ctx.ff_factors.columns), len(ctx.ff_factors))
    except Exception as e:
        logger.error("FF factor fetch failed: %s", e)
        ctx.ff_factors = None

    ctx.dispersion = compute_dispersion_ratio(ctx.returns, window=60)
    ctx.save_checkpoint("02_data")

    # ── Phase 3: Factor Model ──
    logger.info("[Phase 3] Estimating factor exposures...")
    if ctx.ff_factors is None or ctx.ff_factors.empty:
        logger.warning("Skipping factor model (no factor data).")
    else:
        portfolio_returns = ctx.returns[
            [t for t in ctx.active_portfolio.tickers if t in ctx.returns.columns]
        ]
        ctx.factor_exposures = estimate_factor_exposures(
            portfolio_returns,
            ctx.ff_factors,
            window=cfg.factor_model.window,
            min_window=cfg.factor_model.min_window,
        )
        logger.info("Estimated exposures for %d tickers", len(ctx.factor_exposures))

        for ticker in ctx.factor_exposures.index[:5]:
            sr = portfolio_returns[ticker]
            diag = factor_model_diagnostics(sr, ctx.ff_factors)
            if "error" not in diag:
                ctx.diagnostics[f"{ticker}_factor_model"] = diag
                logger.debug(
                    "Factor model diag %s: R²=%.3f, DW=%.2f",
                    ticker,
                    diag["rsquared"],
                    diag["durbin_watson"],
                )

        for ticker in list(ctx.factor_exposures.index)[:3]:
            sr = portfolio_returns[ticker]
            boot = bootstrap_factor_exposures(
                sr,
                ctx.ff_factors,
                n_samples=cfg.factor_model.bootstrap_samples,
                ci=cfg.factor_model.bootstrap_ci,
            )
            if not boot.empty:
                ctx.diagnostics[f"{ticker}_bootstrap"] = boot.to_dict()
                logger.debug("Bootstrapped exposures for %s", ticker)

        ctx.residuals = compute_residual_returns(
            portfolio_returns, ctx.ff_factors, ctx.factor_exposures
        )
        logger.info("Residual returns: %s", ctx.residuals.shape)

        # Factor performance attribution
        for ticker in list(ctx.factor_exposures.index)[:5]:
            sr = portfolio_returns[ticker]
            fe = ctx.factor_exposures.loc[ticker]
            perf = factor_performance_attribution(sr, fe, ctx.ff_factors)
            if "error" not in perf:
                ctx.factor_perf_attr[ticker] = perf
                logger.debug(
                    "Factor perf attr %s: factor=%.2f%%, alpha=%.2f%%, IC=%.4f",
                    ticker,
                    perf.get("cumulative_factor_return_pct", 0),
                    perf.get("cumulative_alpha_return_pct", 0),
                    perf.get("factor_ic_mean", 0),
                )

    ctx.save_checkpoint("03_factors")

    # ── Phase 4: Active Risk Attribution (Denoised) ──
    logger.info("[Phase 4] Computing active risk attribution (denoised)...")
    if ctx.factor_exposures is not None and not ctx.factor_exposures.empty:
        assert ctx.ff_factors is not None
        assert ctx.residuals is not None
        fcols_available = [
            c
            for c in ctx.factor_exposures.columns
            if c in ctx.ff_factors.columns and c not in ("Alpha", "R_squared", "N_obs")
        ]
        factor_ret = ctx.ff_factors[fcols_available]
        factor_cov = compute_covariance(
            factor_ret,
            method=cfg.risk.covariance_method,
            denoise=cfg.risk.denoise_covariance,
            denoise_method=cfg.risk.denoise_method,
        )
        residual_var_series = ctx.residuals.var().fillna(0.0)

        ctx.risk_attribution_result = factor_risk_attribution(
            ctx.latest_weights, ctx.factor_exposures, factor_cov, residual_var_series
        )
        logger.info(
            "Active risk: %.2f%% (factor: %.1f%%, specific: %.1f%%)",
            ctx.risk_attribution_result.get("total_risk_annual_pct", 0),
            ctx.risk_attribution_result.get("factor_risk_pct_of_total", 0),
            ctx.risk_attribution_result.get("specific_risk_pct_of_total", 0),
        )
        sec_contrib = security_risk_contribution(
            ctx.latest_weights, ctx.factor_exposures, factor_cov, residual_var_series
        )
        save_table(sec_contrib, "security_risk_contribution")
    else:
        logger.warning("Skipping risk attribution (no factor model).")
    ctx.save_checkpoint("04_attribution")

    # ── Phase 5: Residual Risk Model ──
    logger.info("[Phase 5] Clustering residual returns...")
    if ctx.residuals is not None and not ctx.residuals.empty:
        ctx.cluster_summary = summarize_clusters(
            cluster_residuals(ctx.residuals, threshold=cfg.theme.residual_corr_threshold),
            ctx.residuals,
        )
        logger.info(
            "Found %d residual clusters",
            len(ctx.cluster_summary) if ctx.cluster_summary is not None else 0,
        )
    ctx.save_checkpoint("05_residual")

    # ── Phase 6: Theme Risk Engine ──
    logger.info("[Phase 6] Analyzing theme risks...")
    portfolio_returns = ctx.returns[
        [t for t in ctx.active_portfolio.tickers if t in ctx.returns.columns]
    ]
    theme_engine = ThemeRiskEngine()
    ctx.theme_scores = theme_engine.analyze_all_themes(
        active_weights=ctx.latest_weights,
        residuals=ctx.residuals if ctx.residuals is not None else pd.DataFrame(),
        returns=portfolio_returns,
        ticker_texts=None,
    )
    logger.info("Analyzed %d themes", len(ctx.theme_scores))
    ctx.save_checkpoint("06_themes")

    # ── Phase 7a: Regime Detection ──
    logger.info("[Phase 7a] Detecting volatility regimes...")
    try:
        regime_result = detect_volatility_regimes(
            ctx.returns_wide if ctx.returns_wide is not None else ctx.returns,
            n_regimes=cfg.regime.n_regimes,
        )
        regime_entropy = entropy_regime_concentration(regime_result.regime_probs)
        stability = regime_stability_index(
            regime_result.regime_probs.idxmax(axis=1)
            .map({f"Regime_{i}": i for i in range(regime_result.n_regimes)})
            .values
        )
        ctx.regime_result = {
            "n_regimes": regime_result.n_regimes,
            "current_regime": regime_result.current_regime,
            "regime_stats": regime_result.regime_stats,
            "transition_matrix": regime_result.transition_matrix.tolist(),
            "half_life_days": regime_result.half_life_days.to_dict(),
            "stability_index": round(float(stability), 4),
            "regime_entropy_mean": round(float(regime_entropy.mean()), 4),
            "current_description": regime_result.regime_stats.loc[
                regime_result.regime_stats["regime"] == f"Regime_{regime_result.current_regime}",
                "description",
            ].values[0]
            if f"Regime_{regime_result.current_regime}"
            in regime_result.regime_stats["regime"].values
            else "Unknown",
        }
        logger.info(
            "Current regime: %s (stability=%.2f)",
            ctx.regime_result["current_description"],
            stability,
        )
    except Exception as e:
        logger.warning("Regime detection failed: %s", e)
        ctx.regime_result = None
    ctx.save_checkpoint("07a_regimes")

    # ── Phase 7b: Crowded Trade Detection ──
    logger.info("[Phase 7b] Detecting crowded trades...")
    if ctx.residuals is not None and not ctx.residuals.empty and ctx.factor_exposures is not None:
        residual_corr = ctx.residuals.corr().fillna(0.0)
        try:
            crowd = detect_crowded_trades(
                active_weights=ctx.latest_weights,
                residual_correlations=residual_corr,
                factor_exposures=ctx.factor_exposures,
                theme_scores=ctx.theme_scores,
                top_n=cfg.crowding.top_n,
            )
            hhi = compute_herfindahl_index(ctx.latest_weights)
            ctx.crowding_result = {
                "aggregate_score": round(float(crowd.aggregate_score), 4),
                "herfindahl_index": round(float(hhi), 4),
                "top_crowded": crowd.top_crowded_tickers,
                "factor_crowding": crowd.factor_crowding.to_dict(),
            }
            logger.info(
                "Crowding aggregate score: %.4f (HHI: %.4f)",
                crowd.aggregate_score,
                hhi,
            )
        except Exception as e:
            logger.warning("Crowded trade detection failed: %s", e)
            ctx.crowding_result = None
    ctx.save_checkpoint("07b_crowding")

    # ── Phase 7c: Transaction Cost Analysis ──
    logger.info("[Phase 7c] Estimating transaction costs...")
    try:
        tc_model = TransactionCostModel(
            participation_rate=cfg.transaction_costs.participation_rate,
            impact_alpha=cfg.transaction_costs.impact_alpha,
            impact_beta=cfg.transaction_costs.impact_beta,
        )
        all_prices = price_provider.get_prices(all_tickers, start_date, end_date)
        if isinstance(all_prices, pd.DataFrame):
            avg_dv = pd.Series(1e6, index=all_prices.columns)
        else:
            avg_dv = pd.Series(1e6, index=ctx.returns.columns)

        ctx.transaction_costs = tc_model.compute_portfolio_impact(
            weights=ctx.latest_weights,
            portfolio_value=cfg.transaction_costs.portfolio_value,
            avg_daily_volume=avg_dv,
            prices=all_prices if isinstance(all_prices, pd.DataFrame) else pd.DataFrame(),
            returns=ctx.returns,
        )
        logger.info(
            "Avg transaction cost: %.4f bps",
            ctx.transaction_costs.mean() * 100 if ctx.transaction_costs is not None else 0,
        )
    except Exception as e:
        logger.warning("Transaction cost analysis failed: %s", e)
        ctx.transaction_costs = None
    ctx.save_checkpoint("07c_tcosts")

    # ── Phase 7d: Stress Tests ──
    logger.info("[Phase 7d] Running stress tests...")
    historical_results: Dict[str, float] = {}
    for scenario_name in HISTORICAL_SCENARIOS:
        try:
            loss = run_historical_replay(
                ctx.returns_wide,
                scenario_name,
                weights=ctx.latest_weights if scenario_name != "2008_Financial_Crisis" else None,
            )
            loss_val = float(np.asarray(loss).flat[0])
            historical_results[scenario_name] = loss_val
            logger.info("  %s: %.2f%%", scenario_name, loss_val)
        except (ValueError, KeyError) as e:
            logger.warning("  %s: %s", scenario_name, e)

    factor_shock_results: Dict[str, float] = {}
    if ctx.factor_exposures is not None:
        for shock_name in FACTOR_SHOCKS:
            try:
                pnl, _, _ = run_factor_shock(ctx.latest_weights, ctx.factor_exposures, shock_name)
                factor_shock_results[shock_name] = pnl
                logger.info("  %s: %.2f%%", shock_name, pnl)
            except ValueError as e:
                logger.warning("  %s: %s", shock_name, e)

    ctx.stress_results = {
        "historical": historical_results,
        "factor_shock": factor_shock_results,
    }
    ctx.save_checkpoint("07d_stress")

    # ── Phase 8: Risk Alerts ──
    logger.info("[Phase 8] Generating risk alerts...")
    alert_gen = AlertGenerator()
    if ctx.risk_attribution_result:
        alert_gen.check_factor_concentration(
            ctx.risk_attribution_result.get("factor_contrib_pct", {}),
            threshold=cfg.alerts.factor_contrib_threshold,
        )
    if ctx.theme_scores is not None and not ctx.theme_scores.empty:
        alert_gen.check_theme_exposure(
            ctx.theme_scores,
            holdings_threshold=cfg.alerts.theme_holdings_threshold,
            residual_corr_threshold=cfg.alerts.theme_residual_corr_threshold,
        )
    if ctx.cluster_summary is not None and not ctx.cluster_summary.empty:
        alert_gen.check_residual_clusters(
            ctx.cluster_summary,
            min_size=cfg.alerts.cluster_min_size,
            min_corr=cfg.alerts.cluster_min_corr,
        )
    alert_gen.check_stress_loss(
        historical_results,
        factor_shock_results,
        loss_threshold=cfg.alerts.stress_loss_threshold,
    )
    if ctx.regime_result is not None:
        alert_gen.check_regime_stability(
            ctx.regime_result.get("stability_index", 1.0),
            threshold=cfg.alerts.regime_stability_threshold,
        )
    if ctx.crowding_result is not None:
        alert_gen.check_crowding(
            ctx.crowding_result.get("aggregate_score", 0.0),
            threshold=cfg.alerts.crowding_threshold,
        )
    if ctx.transaction_costs is not None and not ctx.transaction_costs.empty:
        alert_gen.check_liquidity(
            ctx.transaction_costs,
            threshold=cfg.alerts.liquidity_threshold,
        )
    ctx.alerts = alert_gen.generate_report()
    logger.info("Generated %d risk alerts:", len(ctx.alerts))
    for a in ctx.alerts:
        logger.info("  [%s] %s", a["severity"].upper(), a["title"])
    ctx.save_checkpoint("08_alerts")

    # ── Phase 9: Reports ──
    logger.info("[Phase 9] Generating reports...")
    portfolio_summary = {
        "name": "active",
        "n_positions": len(ctx.active_portfolio.tickers),
        "dates": f"{ctx.active_portfolio.dates[0].date()} to "
        f"{ctx.active_portfolio.dates[-1].date()}",
        "gross_exposure": float(ctx.active_portfolio.gross_exposure.iloc[-1]),
        "net_exposure": float(ctx.active_portfolio.net_exposure.iloc[-1]),
    }

    stress_for_report = {
        "historical": historical_results,
        "factor_shock": factor_shock_results,
    }

    generate_report(
        portfolio_summary=portfolio_summary,
        factor_exposures=ctx.factor_exposures
        if ctx.factor_exposures is not None
        else pd.DataFrame(),
        risk_attribution=ctx.risk_attribution_result,
        cluster_summary=ctx.cluster_summary if ctx.cluster_summary is not None else pd.DataFrame(),
        theme_scores=ctx.theme_scores if ctx.theme_scores is not None else pd.DataFrame(),
        stress_results=stress_for_report,
        alerts=ctx.alerts,
        regime_result=ctx.regime_result,
        crowding_result=ctx.crowding_result,
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — Reports in reports/")
    logger.info("=" * 60)
    ctx.save_checkpoint("09_complete")

    return ctx
