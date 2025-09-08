# Out-of-Model Risk Monitor

**Detect hidden risks in systematic equity portfolios beyond standard factor models.**

A public-data research prototype for identifying thematic concentrations, residual return clusters, and model-blind risk factors that standard factor models miss.

> **Disclaimer:** This is a research prototype using only public data. It is **not** a replacement for commercial risk models (Barra, Axioma, Northfield, etc.). Crowding metrics are public-data proxies, not true hedge-fund positioning data.

---

## Architecture

```
src/risk_monitor/
├── portfolio/        # Portfolio ingestion, validation, active weight computation
├── data/             # Data providers (yfinance, Fama-French, SEC EDGAR, FRED)
├── factor_models/    # Standard factor model + sector/macro proxy regressions
├── risk/             # Active risk attribution, residual clustering, stress tests
├── themes/           # Theme-risk engine (holdings, text, residual co-movement)
├── alerts/           # Evidence-based risk alert generation
└── reports/          # Table/Figure generation
```

## Pipeline

| Phase | Component | Description |
|-------|-----------|-------------|
| 2 | Portfolio Ingestion | CSV parsing, validation, active weight computation vs benchmark |
| 3 | Data Pipeline | Price data (yfinance), Fama-French factors, SEC EDGAR, FRED macro |
| 4 | Factor Model | Rolling OLS on market, size, value, profitability, investment, momentum |
| 5 | Risk Attribution | Factor-model covariance, active risk decomposition by security/factor |
| 6 | Residual Model | Residual return correlation clustering (HAC) |
| 7 | Theme Engine | 8 predefined themes scored via holdings, text similarity, co-movement |
| 8 | Risk Alerts | Evidence-based severity-graded alerts (low/med/high/critical) |
| 9 | Stress Tests | Historical replay (COVID, Rate Hike) + factor shock scenarios |
| 10 | Reports | CSV tables + PNG figures |

## Quick Start

```bash
# Clone and install
git clone <repo>
cd out-of-model-risk-monitor
make install

# Create .env (optional — most features work without API keys)
cp .env.example .env
# Edit .env to add FRED_API_KEY and SEC_USER_AGENT

# Run full pipeline
make run-pipeline

# Run tests
make test

# View reports
open reports/figures/   # PNG visualizations
cat reports/tables/*.csv # Summary tables
```

## Predefined Themes

| Theme | Seed Tickers | Detection Method |
|-------|-------------|-----------------|
| AI Infrastructure | NVDA, AMD, TSM, AVGO | Holdings, filing text, residual corr |
| Rate-Sensitive Growth | PLTR, DDOG, SNOW | Holdings, residual co-movement |
| Regional Banks | KEY, HBAN, RF, FITB | Holdings, text keywords |
| China Supply Chain | TSM, BABA, JD, PDD | Holdings, filing text |
| GLP-1 Obesity Drugs | LLY, NVO, AMGN | Holdings, text keywords |
| Unprofitable Tech | U, SNAP, RBLX, DOCU | Holdings, fundamentals |
| Oil Sensitivity | XOM, CVX, COP, OXY | Holdings, residual corr |
| Mega-Cap Concentration | AAPL, MSFT, GOOGL | Holdings weight, covariance |

## Stress Test Scenarios

**Historical:**
- 2008 Financial Crisis (peak-to-trough)
- 2020 COVID-19 Shock
- 2022 Rate Hike Cycle

**Factor Shocks:**
- Severe Recession
- Growth Scare
- Momentum Crash
- Inflation Shock
- Tech Wreck

## Constraints Met

- No fake AI risk oracle — every result is a reproducible computation
- Every alert includes evidence (holdings %, residual correlation, text similarity)
- Missing data is detected and reported (not silently accepted)
- Factor models use no lookahead — all regressions are backward-looking
- Theme labels are auditable via seed tickers, residual correlations, and text evidence
- Crowding metrics are labeled as public-data proxies
- Core logic lives in `src/`, not notebooks

## Requirements

- Python >= 3.10
- See `pyproject.toml` for dependencies

## Environment Variables

See `.env.example`:
- `FRED_API_KEY` — for macro data (get from https://fred.stlouisfed.org)
- `SEC_USER_AGENT` — required for SEC EDGAR API access
- `DATA_DIR` — data storage location (default: `data/`)
