from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm

SECTOR_ETFS: Dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
}

MACRO_PROXY_ETFS: Dict[str, str] = {
    "TLT": "Long-term Treasury Rates",
    "SHY": "Short-term Rates",
    "GLD": "Gold / Inflation Hedge",
    "DXY": "US Dollar (via UUP)",
    "HYG": "High Yield Credit Spread",
}


def get_sector_exposures(
    returns: pd.DataFrame,
    sector_etf_returns: pd.DataFrame,
    window: int = 252,
    min_window: int = 60,
) -> pd.DataFrame:
    common_dates = returns.index.intersection(sector_etf_returns.index)
    if len(common_dates) < min_window:
        raise ValueError(f"Insufficient overlap: {len(common_dates)} days (need >= {min_window})")

    r = returns.loc[common_dates]
    s = sector_etf_returns.loc[common_dates]

    exposures = {}
    for ticker in r.columns:
        tr = r[ticker].dropna()
        if len(tr) < min_window:
            continue

        joined = pd.concat([tr.rename("ret"), s], axis=1).dropna()
        if len(joined) < min_window:
            continue

        y = joined["ret"].values
        X = joined[s.columns].values
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
            exposures[ticker] = {
                "Alpha": model.params[0],
                **{s.columns[i]: model.params[i + 1] for i in range(len(s.columns))},
                "R_squared": model.rsquared,
                "N_obs": len(joined),
            }
        except Exception:
            continue

    if not exposures:
        raise ValueError("No sector exposure models could be estimated.")

    return pd.DataFrame(exposures).T


def get_macro_exposures(
    returns: pd.DataFrame,
    macro_series: pd.DataFrame,
    window: int = 252,
    min_window: int = 60,
) -> pd.DataFrame:
    macro_returns = macro_series.pct_change().dropna(how="all").replace([np.inf, -np.inf], np.nan)
    return get_sector_exposures(returns, macro_returns, window, min_window)
