from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class RiskAlert:
    title: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    evidence: List[str] = field(default_factory=list)
    module: str = ""
    metric_value: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence,
            "module": self.module,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
        }


class AlertGenerator:
    def __init__(self):
        self.alerts: List[RiskAlert] = []

    def check_factor_concentration(
        self,
        factor_contrib: Dict[str, float],
        threshold: float = 40.0,
    ) -> List[RiskAlert]:
        for factor, contrib in factor_contrib.items():
            if abs(contrib) > threshold:
                self.alerts.append(
                    RiskAlert(
                        title=f"High {factor} factor concentration",
                        severity="high" if abs(contrib) > 60 else "medium",
                        description=(
                            f"Active risk contribution from {factor} is {contrib:.1f}%, "
                            f"exceeding {threshold}% threshold."
                        ),
                        evidence=[
                            f"Factor risk contribution: {contrib:.2f}%",
                            f"Threshold: {threshold}%",
                        ],
                        module="attribution",
                        metric_value=contrib,
                        threshold=threshold,
                    )
                )
        return self.alerts

    def check_theme_exposure(
        self,
        theme_scores: pd.DataFrame,
        holdings_threshold: float = 10.0,
        residual_corr_threshold: float = 0.3,
    ) -> List[RiskAlert]:
        for _, row in theme_scores.iterrows():
            holdings = (
                row.get("holdings_exposure_pct", 0)
                if not isinstance(row, dict)
                else row["holdings_exposure_pct"]
            )
            res_corr = (
                row.get("residual_avg_corr", 0)
                if not isinstance(row, dict)
                else row["residual_avg_corr"]
            )
            theme_name = row.get("theme", "Unknown") if not isinstance(row, dict) else row["theme"]
            desc = row.get("description", "") if not isinstance(row, dict) else row["description"]

            if holdings is None or res_corr is None:
                continue

            if isinstance(row, pd.Series):
                holdings = row.get("holdings_exposure_pct", 0)
                res_corr = row.get("residual_avg_corr", 0)
                theme_name = row.get("theme", "Unknown")
                desc = row.get("description", "")

            evidence_parts = []
            severity = "low"

            if holdings >= holdings_threshold:
                evidence_parts.append(
                    f"Holdings-based exposure: {holdings:.1f}% of active risk budget"
                )
                severity = "medium"

            if res_corr >= residual_corr_threshold:
                evidence_parts.append(
                    f"Residual return correlation: {res_corr:.2f} "
                    f"(exceeds {residual_corr_threshold:.0%} threshold)"
                )
                severity = (
                    "high" if res_corr >= 0.5 and holdings >= holdings_threshold else severity
                )

            if holdings >= holdings_threshold * 2:
                severity = "high"

            if not evidence_parts:
                continue

            self.alerts.append(
                RiskAlert(
                    title=f"Theme concentration: {theme_name}",
                    severity=severity,
                    description=(
                        f"Portfolio has meaningful exposure to {theme_name} "
                        f"({desc}). Evidence suggests hidden theme risk."
                    ),
                    evidence=evidence_parts,
                    module="theme_risk",
                    metric_value=holdings,
                    threshold=holdings_threshold,
                )
            )
        return self.alerts

    def check_residual_clusters(
        self,
        cluster_summary: pd.DataFrame,
        min_size: int = 3,
        min_corr: float = 0.4,
    ) -> List[RiskAlert]:
        for _, row in cluster_summary.iterrows():
            n = row["n_securities"]
            corr = row["avg_residual_corr"]
            if n >= min_size and corr >= min_corr:
                self.alerts.append(
                    RiskAlert(
                        title=f"Residual return cluster detected: {row['cluster']}",
                        severity="medium" if n >= 5 else "low",
                        description=(
                            f"Cluster with {n} securities showing average residual "
                            f"correlation of {corr:.2f}"
                        ),
                        evidence=[
                            f"Average residual correlation: {corr:.3f}",
                            f"Members ({n}): {row['members']}",
                        ],
                        module="residual",
                        metric_value=corr,
                        threshold=min_corr,
                    )
                )
        return self.alerts

    def check_stress_loss(
        self,
        scenario_results: Dict[str, float],
        factor_shock_results: Dict[str, float],
        loss_threshold: float = 5.0,
    ) -> List[RiskAlert]:
        for scenario, loss in scenario_results.items():
            abs_loss = abs(loss)
            if abs_loss >= loss_threshold:
                self.alerts.append(
                    RiskAlert(
                        title=f"Stress test: {scenario}",
                        severity="critical"
                        if abs_loss >= 15
                        else "high"
                        if abs_loss >= 10
                        else "medium",
                        description=(
                            f"Historical scenario '{scenario}' results in "
                            f"{loss:.1f}% portfolio loss."
                        ),
                        evidence=[
                            f"Scenario loss: {loss:.2f}%",
                            f"Threshold: {loss_threshold}%",
                        ],
                        module="stress_test",
                        metric_value=abs_loss,
                        threshold=loss_threshold,
                    )
                )

        for shock, loss in factor_shock_results.items():
            abs_loss = abs(loss)
            if abs_loss >= loss_threshold:
                self.alerts.append(
                    RiskAlert(
                        title=f"Factor shock: {shock}",
                        severity="critical"
                        if abs_loss >= 15
                        else "high"
                        if abs_loss >= 10
                        else "medium",
                        description=(
                            f"Factor shock scenario '{shock}' results in "
                            f"{loss:.1f}% portfolio loss."
                        ),
                        evidence=[
                            f"Shock loss: {loss:.2f}%",
                            f"Threshold: {loss_threshold}%",
                        ],
                        module="stress_test",
                        metric_value=abs_loss,
                        threshold=loss_threshold,
                    )
                )
        return self.alerts

    def check_data_quality(
        self,
        missing_data: Dict[str, List[str]],
        data_start: Dict[str, str],
    ) -> List[RiskAlert]:
        for ticker, missing in missing_data.items():
            if missing:
                self.alerts.append(
                    RiskAlert(
                        title=f"Missing data for {ticker}",
                        severity="medium",
                        description=f"No {', '.join(missing)} data for {ticker}.",
                        evidence=[f"Missing data types: {missing}"],
                        module="data_quality",
                    )
                )
        return self.alerts

    def check_regime_stability(
        self,
        stability_index: float,
        threshold: float = 0.6,
    ) -> List[RiskAlert]:
        if stability_index < threshold:
            self.alerts.append(
                RiskAlert(
                    title="Unstable volatility regime",
                    severity="high" if stability_index < 0.4 else "medium",
                    description=(
                        f"Volatility regime stability index is {stability_index:.2f}, "
                        f"below {threshold:.2f} threshold. Portfolio risk estimates "
                        f"may be unreliable during regime transitions."
                    ),
                    evidence=[
                        f"Regime stability index: {stability_index:.2f}",
                        f"Threshold: {threshold:.2f}",
                        "Low stability indicates frequent regime switching",
                    ],
                    module="regime_detection",
                    metric_value=stability_index,
                    threshold=threshold,
                )
            )
        return self.alerts

    def check_crowding(
        self,
        aggregate_score: float,
        threshold: float = 0.5,
    ) -> List[RiskAlert]:
        if aggregate_score > threshold:
            self.alerts.append(
                RiskAlert(
                    title="Elevated crowding risk detected",
                    severity="high" if aggregate_score > 0.7 else "medium",
                    description=(
                        f"Crowding aggregate score is {aggregate_score:.2f}, "
                        f"exceeding {threshold:.2f} threshold. Portfolio may "
                        f"have concentrated positions in crowded trades."
                    ),
                    evidence=[
                        f"Crowding score: {aggregate_score:.2f}",
                        f"Threshold: {threshold:.2f}",
                        "High crowding indicates unwind risk",
                    ],
                    module="crowding",
                    metric_value=aggregate_score,
                    threshold=threshold,
                )
            )
        return self.alerts

    def check_liquidity(
        self,
        transaction_costs: pd.Series,
        threshold: float = 0.3,
    ) -> List[RiskAlert]:
        high_cost = transaction_costs[transaction_costs > threshold]
        for ticker, cost in high_cost.items():
            self.alerts.append(
                RiskAlert(
                    title=f"Liquidity risk: {ticker}",
                    severity="medium" if cost > threshold * 1.5 else "low",
                    description=(
                        f"Estimated transaction cost for {ticker} is {cost:.2%}, "
                        f"indicating elevated liquidity risk."
                    ),
                    evidence=[
                        f"Transaction cost: {cost:.2%}",
                        f"Threshold: {threshold:.2%}",
                        "High costs suggest limited liquidity",
                    ],
                    module="transaction_costs",
                    metric_value=float(cost),
                    threshold=threshold,
                )
            )
        return self.alerts

    def generate_report(self) -> List[Dict]:
        sorted_alerts = sorted(
            self.alerts,
            key=lambda a: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(a.severity, 4),
        )
        return [a.to_dict() for a in sorted_alerts]
