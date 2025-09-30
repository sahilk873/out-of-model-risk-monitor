from __future__ import annotations


class RiskMonitorError(Exception):
    """Base exception for all risk-monitor errors."""


class DataQualityError(RiskMonitorError):
    """Raised when input data fails quality checks."""


class InsufficientDataError(RiskMonitorError):
    """Raised when there isn't enough data for a computation."""


class FactorModelError(RiskMonitorError):
    """Raised when factor model estimation fails."""


class PortfolioError(RiskMonitorError):
    """Raised for portfolio construction or validation failures."""


class ThemeError(RiskMonitorError):
    """Raised when theme analysis encounters an issue."""


class ConfigurationError(RiskMonitorError):
    """Raised when the system is misconfigured."""


class CacheError(RiskMonitorError):
    """Raised when caching operations fail."""


class DataFreshnessError(RiskMonitorError):
    """Raised when data is stale or insufficiently recent."""
