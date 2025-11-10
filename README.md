# Out-of-Model Risk Monitor

**Detect hidden risks in equity portfolios beyond standard factor models.**

A production-quality research system for identifying thematic concentrations, residual return clusters, and model-blind risk factors that standard factor models (Barra, Axioma, etc.) miss. Built entirely on public data.

---

## Features

### Core Risk Analysis
- **Portfolio Ingestion**: CSV-based portfolio loading with validation, active weight computation vs. benchmark
- **Factor Model**: Rolling OLS estimation of Fama-French 5-factor + Momentum exposures
- **Sector/Macro Proxies**: ETF-based sector exposure decomposition + macro sensitivity analysis
- **Factor Model Diagnostics**: VIF multicollinearity testing, Durbin-Watson autocorrelation, Breusch-Pagan heteroskedasticity, Jarque-Bera normality tests
- **Bootstrap Confidence Intervals**: Resampled uncertainty bounds on all factor exposures
- **Walk-Forward Information Coefficient**: Cross-sectional IC tracking for factor predictive power

### Risk Attribution
- **Factor Risk Decomposition**: Systematic vs. specific risk breakdown with EWMA covariance
- **Security-Level Contribution**: Marginal and total contribution to active risk
- **Risk Contribution (RC)**: Percentage-based risk budgeting decomposition

### Residual Risk Model
- **Hierarchical Clustering**: HAC-based residual return correlation clustering
- **Cluster Validation**: Silhouette scores and Gap statistic for optimal cluster selection
- **Cluster Summarization**: Average intra-cluster correlation, variance explained

### Theme Risk Engine
Eight predefined investable themes with multi-signal detection:
| Theme | Detection Signals |
|-------|------------------|
| AI Infrastructure | Holdings overlap, residual co-movement, TF-IDF text similarity |
| Rate-Sensitive Growth | Holdings, residual correlation |
| Regional Banks | Holdings, residual co-movement |
| China Supply Chain | Holdings, text similarity |
| GLP-1 Obesity Drugs | Holdings, text keywords |
| Unprofitable Tech | Holdings, fundamental screening |
| Oil Sensitivity | Holdings, residual co-movement |
| Mega-Cap Concentration | Holdings weight, covariance risk |

### Stress Testing
- **Historical Replay**: 2008 Financial Crisis, 2020 COVID Shock, 2022 Rate Hike cycle
- **Factor Shocks**: Severe Recession, Growth Scare, Momentum Crash, Inflation Shock, Tech Wreck
- **Loss Decomposition**: By security, factor exposure, and sector

### Risk Alerting
Severity-graded alerts (low/medium/high/critical) with evidence traces:
- Factor concentration breaches
- Theme exposure warnings
- Residual cluster detection
- Stress loss thresholds
- Data quality issues

### Portfolio Construction
- **Optimization**: Minimum variance, risk parity, max diversification, hierarchical risk parity (HRP)
- **Performance Attribution**: Brinson-style allocation/selection/interaction decomposition
- **Turnover Analysis**: Portfolio churn tracking
- **Exposure Drift**: Time-varying factor exposure monitoring
- **Efficient Frontier**: Mean-variance frontier computation

---

## Architecture

```
src/risk_monitor/
├── __init__.py          # Package root, version
├── config.py            # Configuration management (dataclass-based)
├── exceptions.py        # Custom exception hierarchy
├── pipeline.py          # Production pipeline orchestrator
├── utils.py             # Logging, statistics utilities
├── py.typed             # PEP 561 compliance marker
│
├── portfolio/           # Portfolio ingestion, validation, optimization
│   ├── ingestion.py     # Portfolio dataclasses, CSV loading, active weights
│   ├── optimization.py  # Min variance, risk parity, HRP, efficient frontier
│   └── attribution.py   # Brinson decomposition, turnover, drift
│
├── data/                # Data providers and quality
│   ├── prices.py        # YFinance price provider with in-memory cache
│   ├── cache.py         # Disk-based cache with TTL expiration
│   ├── factors.py       # Fama-French factor downloader and parser
│   ├── macro.py         # FRED API client for macro series
│   ├── sec_edgar.py     # SEC EDGAR API client (CIK lookup, filings)
│   └── quality.py       # Data quality checks, outlier detection, freshness
│
├── factor_models/       # Factor model estimation
│   ├── standard_factors.py   # FF5 + Momentum OLS estimation, residuals
│   └── sector_macro_proxies.py # ETF sector & macro proxy regressions
│
├── stats/               # Statistical methods
│   ├── diagnostics.py        # VIF, DW, BP, JB tests, bootstrap CIs, walk-forward IC
│   └── cluster_validation.py # Silhouette scores, Gap statistic
│
├── risk/                # Risk analytics
│   ├── attribution.py   # Covariance estimation, factor attribution, risk contribution
│   ├── residual.py      # Residual correlation, HAC clustering, summarization
│   └── stress.py        # Historical replay, factor shocks, loss decomposition
│
├── themes/              # Thematic risk detection
│   └── engine.py        # Theme definitions, holdings/text/residual scoring
│
├── alerts/              # Risk alert generation
│   └── generator.py     # Severity-graded alerting with evidence traces
│
└── reports/             # Reporting and visualization
    └── report.py        # Table saving, factor heatmap, risk pie, cluster bars, radar
```

---

## Data Flow

```
Portfolio CSV → Validate → Active Weights → Factor Model → Residuals
                                 │               │              │
                                 ▼               ▼              ▼
                          Risk Attribution   Diagnostics   Clustering
                                 │               │              │
                                 ▼               ▼              ▼
                          Theme Engine ←─── Residual Corr ────┤
                                 │                             │
                                 ▼                             ▼
                          Stress Tests ──────────────→ Alerts → Reports
```

---

## Installation

```bash
# Clone and install
git clone <repo>
cd out-of-model-risk-monitor
make install-dev

# Create .env (optional — core features work without API keys)
cp .env.example .env
# Edit .env to add FRED_API_KEY and SEC_USER_AGENT for extended features

# Run full pipeline
make run-pipeline

# Run tests
make test
make coverage

# Lint and type check
make lint
make typecheck
```

---

## Configuration

All parameters are configurable via environment variables or the `MonitorConfig` dataclass:

| Variable | Default | Description |
|----------|---------|-------------|
| `START_DATE` | 2024-01-01 | Analysis start date |
| `FACTOR_WINDOW` | 126 | Rolling OLS estimation window (days) |
| `FACTOR_MIN_WINDOW` | 60 | Minimum observations for factor model |
| `BOOTSTRAP_SAMPLES` | 1000 | Bootstrap iterations for CI |
| `CACHE_TTL_DAYS` | 7 | Disk cache expiration |
| `DATA_DIR` | data/ | Data storage location |
| `FRED_API_KEY` | — | FRED API key (macro data) |
| `SEC_USER_AGENT` | — | SEC EDGAR user agent (filing access) |

---

## Development

```bash
# Setup
make install-dev
pre-commit install

# Code Quality
make lint       # ruff
make typecheck  # mypy
make test       # pytest

# Full CI
make ci
```

---

## Constraints & Design Principles

- **No fake AI oracle** — every result is a reproducible computation
- **Every alert includes evidence** — holdings %, residual correlation, text similarity
- **Missing data detected and reported** — not silently accepted
- **No lookahead bias** — all factor regressions are backward-looking
- **Theme labels auditable** — seed tickers, residual correlations, and text evidence visible
- **Crowding metrics labeled** as public-data proxies (not true fund positioning)
- **Statistical significance reported** — bootstrap CIs, p-values, diagnostic tests
- **Cluster validation** — silhouette scores and gap statistics, not arbitrary thresholds

---

## Requirements

- Python >= 3.10
- See `pyproject.toml` for full dependency list

---

## Disclaimer

This is a research prototype using only public data. It is **not** a replacement for commercial risk models (Barra, Axioma, Northfield, etc.). Crowding metrics are public-data proxies, not true fund positioning data.

---

## License

MIT
