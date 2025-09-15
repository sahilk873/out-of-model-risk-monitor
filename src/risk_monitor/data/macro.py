from __future__ import annotations

from typing import Dict, List

import pandas as pd
import requests

MACRO_SERIES: Dict[str, str] = {
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "T10YIE": "10-Year Breakeven Inflation Rate",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "DTWEXBGS": "Trade-Weighted US Dollar Index",
    "VIXCLS": "CBOE Volatility Index (VIX)",
    "BAA10Y": "Moody's BAA Corporate Bond Yield Relative to 10Y Treasury",
    "T5YIE": "5-Year Breakeven Inflation Rate",
}


class FREDClient:
    BASE = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str):
        if not api_key or api_key == "your_fred_api_key_here":
            raise ValueError(
                "Valid FRED API key required. Set FRED_API_KEY in .env "
                "(get one at https://fred.stlouisfed.org/docs/api/api_key.html)"
            )
        self.api_key = api_key

    def get_series(self, series_id: str, start: str, end: str) -> pd.Series:
        url = f"{self.BASE}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "sort_order": "asc",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "observations" not in data:
            raise ValueError(f"No observations for series {series_id}")

        records = []
        for obs in data["observations"]:
            if obs["value"] != ".":
                records.append({"date": obs["date"], "value": float(obs["value"])})

        if not records:
            raise ValueError(f"No valid observations for series {series_id}")

        srs = pd.DataFrame(records).set_index("date")
        srs.index = pd.to_datetime(srs.index)
        srs = srs.sort_index()
        return srs["value"]

    def get_multiple_series(self, series_ids: List[str], start: str, end: str) -> pd.DataFrame:
        frames = {}
        for sid in series_ids:
            try:
                frames[sid] = self.get_series(sid, start, end)
            except Exception:
                frames[sid] = pd.Series(dtype=float, name=sid)
        return pd.DataFrame(frames).ffill().dropna(how="all")
