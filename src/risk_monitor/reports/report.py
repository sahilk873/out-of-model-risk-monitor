from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def save_table(df: pd.DataFrame, name: str, output_dir: str = "reports/tables"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(path, index=True)
    return path


def plot_factor_exposures(
    exposures: pd.DataFrame,
    save_path: str = "reports/figures/factor_exposures.png",
):
    fcols = [c for c in exposures.columns if c != "Alpha" and c != "R_squared" and c != "N_obs"]
    if not fcols:
        return
    data = exposures[fcols].copy()
    data.columns = [c.replace("-", "\n") for c in data.columns]

    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.3)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Factor Beta"},
    )
    ax.set_title("Factor Exposures (Rolling OLS Betas)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Ticker")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_risk_attribution(
    attribution_result: Dict,
    save_path: str = "reports/figures/risk_attribution.png",
):
    if "factor_contrib_pct" not in attribution_result:
        return
    fac_contrib = attribution_result["factor_contrib_pct"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    factors = list(fac_contrib.keys())
    values = list(fac_contrib.values())
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]
    bars = ax1.barh(factors, values, color=colors)
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.set_title("Factor Risk Contribution (% of Active Risk)")
    ax1.set_xlabel("Risk Contribution (%)")
    for bar, v in zip(bars, values):
        ax1.text(
            bar.get_width() + 0.5 if v >= 0 else bar.get_width() - 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
        )

    labels = ["Factor Risk", "Specific Risk"]
    sizes = [
        attribution_result.get("factor_risk_pct_of_total", 0),
        attribution_result.get("specific_risk_pct_of_total", 0),
    ]
    ax2.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#3498db", "#95a5a6"], startangle=90)
    ax2.set_title(
        f"Total Active Risk: {attribution_result.get('total_risk_annual_pct', 0):.1f}% annualized"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residual_clusters(
    cluster_summary: pd.DataFrame,
    save_path: str = "reports/figures/residual_clusters.png",
):
    if cluster_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        cluster_summary["cluster"],
        cluster_summary["avg_residual_corr"],
        color=sns.color_palette("viridis", len(cluster_summary)),
    )
    ax.set_xlabel("Average Residual Correlation")
    ax.set_title("Residual Return Clusters")
    for bar, row in zip(bars, cluster_summary.itertuples()):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"n={row.n_securities}",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_theme_radar(
    theme_scores: pd.DataFrame,
    save_path: str = "reports/figures/theme_radar.png",
):
    if theme_scores.empty:
        return
    metrics = ["holdings_exposure_pct", "residual_avg_corr", "covariance_concentration_ratio"]
    available = [m for m in metrics if m in theme_scores.columns]

    if not available:
        return

    labels = theme_scores["theme"].tolist()
    n = len(labels)
    if n == 0:
        return

    data = theme_scores[available].values
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    data_norm = np.concatenate([data_norm, data_norm[:1]], axis=0)
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    for i, metric in enumerate(available):
        values = data_norm[:, i]
        ax.plot(angles, values, "o-", label=metric.replace("_", " ").title(), linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Theme Risk Profile", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_stress_decomposition(
    stress_results: Dict,
    save_path: str = "reports/figures/stress_decomposition.png",
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    if "by_security" in stress_results.get("historical", {}):
        sec = pd.Series(stress_results["historical"]["by_security"])
        top_sec = sec.abs().sort_values(ascending=False).head(10)
        axes[0].barh(
            range(len(top_sec)),
            top_sec.values,
            color=["#e74c3c" if v < 0 else "#2ecc71" for v in top_sec.values],
        )
        axes[0].set_yticks(range(len(top_sec)))
        axes[0].set_yticklabels(top_sec.index)
        axes[0].set_title("Top Security Contributions")
        axes[0].axvline(0, color="black", linewidth=0.5)

    if "by_factor" in stress_results.get("historical", {}):
        fac = pd.Series(stress_results["historical"]["by_factor"])
        axes[1].barh(
            range(len(fac)),
            fac.values,
            color=["#3498db" if v < 0 else "#2ecc71" for v in fac.values],
        )
        axes[1].set_yticks(range(len(fac)))
        axes[1].set_yticklabels(fac.index)
        axes[1].set_title("Factor Contributions")
        axes[1].axvline(0, color="black", linewidth=0.5)

    if "by_sector" in stress_results.get("historical", {}):
        sec2 = pd.Series(stress_results["historical"]["by_sector"])
        axes[2].barh(
            range(len(sec2)),
            sec2.values,
            color=["#9b59b6" if v < 0 else "#2ecc71" for v in sec2.values],
        )
        axes[2].set_yticks(range(len(sec2)))
        axes[2].set_yticklabels(sec2.index)
        axes[2].set_title("Sector Contributions")
        axes[2].axvline(0, color="black", linewidth=0.5)

    plt.suptitle("Stress Test Loss Decomposition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_regime_summary(
    regime_result: Optional[Dict] = None,
    save_path: str = "reports/figures/regime_summary.png",
):
    if regime_result is None:
        return
    regime_stats = regime_result.get("regime_stats")
    if regime_stats is None or regime_stats.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    regimes = regime_stats["regime"].tolist()
    vols = regime_stats["annualized_vol_pct"].tolist()
    freqs = regime_stats["freq_pct"].tolist()

    colors_vol = sns.color_palette("YlOrRd", len(regimes))
    axes[0].bar(regimes, vols, color=colors_vol)
    axes[0].set_title("Regime Annualized Volatility")
    axes[0].set_ylabel("Volatility (%)")
    for i, v in enumerate(vols):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    axes[1].bar(regimes, freqs, color=sns.color_palette("Blues", len(regimes)))
    axes[1].set_title("Regime Frequency")
    axes[1].set_ylabel("Occurrence (%)")
    for i, f in enumerate(freqs):
        axes[1].text(i, f + 0.5, f"{f:.1f}%", ha="center", fontsize=9)

    if "transition_matrix" in regime_result:
        tm = np.array(regime_result["transition_matrix"])
        sns.heatmap(
            tm,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=regimes,
            yticklabels=regimes,
            ax=axes[2],
            cbar_kws={"label": "Transition Prob"},
        )
        axes[2].set_title("Regime Transition Matrix")
        axes[2].set_ylabel("From")
        axes[2].set_xlabel("To")

    if "half_life_days" in regime_result:
        extra_text = (
            f"Stability: {regime_result.get('stability_index', 'N/A')} | "
            f"Entropy: {regime_result.get('regime_entropy_mean', 'N/A')}"
        )
        fig.suptitle(
            f"Volatility Regime Analysis — {extra_text}",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_crowding_summary(
    crowding_result: Optional[Dict] = None,
    save_path: str = "reports/figures/crowding_summary.png",
):
    if crowding_result is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    top = crowding_result.get("top_crowded", [])
    if top:
        axes[0].barh(
            range(len(top)), list(range(len(top), 0, -1)), color=sns.color_palette("Reds", len(top))
        )
        axes[0].set_yticks(range(len(top)))
        axes[0].set_yticklabels(top)
        axes[0].set_title("Top Crowded Tickers")
        axes[0].invert_xaxis()

    agg = crowding_result.get("aggregate_score", 0)
    hhi = crowding_result.get("herfindahl_index", 0)
    metrics = ["Aggregate\nCrowding", "Herfindahl\nIndex"]
    values = [agg, hhi]
    colors = ["#e74c3c" if v > 0.5 else "#f39c12" for v in values]
    bars = axes[1].bar(metrics, values, color=colors)
    axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Warning Threshold")
    axes[1].set_title("Crowding Metrics")
    axes[1].legend()
    for bar, v in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.3f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_report(
    portfolio_summary: Dict,
    factor_exposures: pd.DataFrame,
    risk_attribution: Dict,
    cluster_summary: pd.DataFrame,
    theme_scores: pd.DataFrame,
    stress_results: Dict,
    alerts: List[Dict],
    regime_result: Optional[Dict] = None,
    crowding_result: Optional[Dict] = None,
    output_dir: str = "reports",
):
    save_table(factor_exposures, "factor_exposures")
    save_table(cluster_summary, "residual_clusters")
    save_table(theme_scores, "theme_scores")
    save_table(pd.DataFrame(alerts), "risk_alerts")

    plot_factor_exposures(factor_exposures)
    plot_risk_attribution(risk_attribution)
    plot_residual_clusters(cluster_summary)
    plot_theme_radar(theme_scores)
    plot_stress_decomposition(stress_results)
    plot_regime_summary(regime_result)
    plot_crowding_summary(crowding_result)
