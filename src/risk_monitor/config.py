from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import ClassVar, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DataConfig:
    start_date: str = "2024-01-01"
    end_date: Optional[str] = None
    stress_start: str = "2020-01-01"
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    cache_ttl_days: int = 7


@dataclass(frozen=True)
class FactorModelConfig:
    window: int = 126
    min_window: int = 60
    step: int = 21
    bootstrap_samples: int = 1000
    bootstrap_ci: float = 0.95


@dataclass(frozen=True)
class RiskConfig:
    covariance_method: str = "ewma"
    halflife: int = 60
    risk_free_rate: float = 0.05
    denoise_covariance: bool = True
    denoise_method: str = "marchenko_pastur"


@dataclass(frozen=True)
class ThemeConfig:
    residual_corr_threshold: float = 0.3
    holdings_threshold_pct: float = 10.0


@dataclass(frozen=True)
class RegimeConfig:
    n_regimes: int = 3
    covariance_type: str = "diag"
    regime_cov_method: str = "ewma"


@dataclass(frozen=True)
class TransactionCostConfig:
    participation_rate: float = 0.10
    portfolio_value: float = 100_000_000
    impact_alpha: float = 0.142
    impact_beta: float = 0.500


@dataclass(frozen=True)
class CrowdingConfig:
    top_n: int = 5
    crowding_threshold: float = 0.5


@dataclass(frozen=True)
class AlertConfig:
    factor_contrib_threshold: float = 40.0
    theme_holdings_threshold: float = 10.0
    theme_residual_corr_threshold: float = 0.3
    cluster_min_size: int = 3
    cluster_min_corr: float = 0.4
    stress_loss_threshold: float = 5.0
    regime_stability_threshold: float = 0.6
    crowding_threshold: float = 0.5
    liquidity_threshold: float = 0.3


@dataclass(frozen=True)
class ReportConfig:
    output_dir: str = "reports"
    tables_dir: str = "reports/tables"
    figures_dir: str = "reports/figures"
    dpi: int = 150


@dataclass(frozen=True)
class MonitorConfig:
    data: DataConfig = field(default_factory=DataConfig)
    factor_model: FactorModelConfig = field(default_factory=FactorModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    transaction_costs: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    crowding: CrowdingConfig = field(default_factory=CrowdingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)

    _instances: ClassVar[dict] = {}

    @classmethod
    def from_env(cls) -> MonitorConfig:
        return cls(
            data=DataConfig(
                start_date=os.getenv("START_DATE", "2024-01-01"),
                end_date=os.getenv("END_DATE") or None,
                stress_start=os.getenv("STRESS_START", "2020-01-01"),
                data_dir=os.getenv("DATA_DIR", "data"),
                cache_dir=os.getenv("CACHE_DIR", "data/cache"),
                cache_ttl_days=int(os.getenv("CACHE_TTL_DAYS", "7")),
            ),
            factor_model=FactorModelConfig(
                window=int(os.getenv("FACTOR_WINDOW", "126")),
                min_window=int(os.getenv("FACTOR_MIN_WINDOW", "60")),
                step=int(os.getenv("FACTOR_STEP", "21")),
                bootstrap_samples=int(os.getenv("BOOTSTRAP_SAMPLES", "1000")),
                bootstrap_ci=float(os.getenv("BOOTSTRAP_CI", "0.95")),
            ),
            risk=RiskConfig(
                denoise_covariance=os.getenv("DENOISE_COVARIANCE", "True").lower() == "true",
                denoise_method=os.getenv("DENOISE_METHOD", "marchenko_pastur"),
            ),
            regime=RegimeConfig(
                n_regimes=int(os.getenv("N_REGIMES", "3")),
            ),
        )

    @classmethod
    def get_instance(cls) -> MonitorConfig:
        if cls._instances:
            instance = cls._instances.get(cls)
            if instance is not None:
                return instance
        instance = cls.from_env()
        cls._instances[cls] = instance
        return instance
