from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import requests

FACTOR_URLS: Dict[str, str] = {
    "F-F_Research_Data_5_Factors_2x3_daily": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    ),
    "F-F_Momentum_Factor_daily": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Momentum_Factor_daily_CSV.zip"
    ),
}


def _download_zip_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            raw = f.read().decode("utf-8")

    lines = raw.splitlines()
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped[0].isdigit():
            start = i
            break
    header_candidates = [i for i, ln in enumerate(lines[:start]) if "," in ln]
    header_line = header_candidates[-1] if header_candidates else start - 1

    header = [h.strip() for h in lines[header_line].split(",")]
    data_lines = lines[start:]

    import io as io_module

    csv_text = ",".join(header) + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io_module.StringIO(csv_text))
    return df


def parse_ff_daily(df: pd.DataFrame) -> pd.DataFrame:
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace(-99.99, float("nan")).replace(-999, float("nan"))
    return df / 100


def get_ff_factors(start: str, end: str) -> pd.DataFrame:
    ff5_raw, mom_raw = fetch_ff_factors()
    ff5 = parse_ff_daily(ff5_raw)
    mom = parse_ff_daily(mom_raw)

    ff5["MOM"] = mom.iloc[:, 0] if not mom.empty else 0.0
    ff5 = ff5.dropna(how="all")
    ff5 = ff5.ffill().dropna()

    return ff5.loc[start:end]


def fetch_ff_factors() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ff5 = _download_zip_csv(FACTOR_URLS["F-F_Research_Data_5_Factors_2x3_daily"])
    mom = _download_zip_csv(FACTOR_URLS["F-F_Momentum_Factor_daily"])
    return ff5, mom


FACTOR_DESCRIPTIONS = {
    "Mkt-RF": "Excess return of broad market over risk-free rate",
    "SMB": "Small-cap minus large-cap (size factor)",
    "HML": "High book-to-market minus low (value factor)",
    "RMW": "Robust profitability minus weak (profitability factor)",
    "CMA": "Conservative investment minus aggressive (investment factor)",
    "MOM": "Momentum: winners minus losers (12-1 month)",
}
