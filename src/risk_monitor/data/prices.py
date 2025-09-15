from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import yfinance as yf


class PriceProvider(ABC):
    @abstractmethod
    def get_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame: ...

    @abstractmethod
    def get_returns(self, tickers: List[str], start: str, end: str) -> pd.DataFrame: ...


class YFinanceProvider(PriceProvider):
    def get_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        missing = [t for t in tickers if not isinstance(t, str) or not t.strip()]
        if missing:
            raise ValueError(f"Invalid tickers: {missing}")

        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,
        )

        if data.empty:
            raise ValueError(f"No price data returned for: {tickers}")

        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].copy()
        else:
            if len(tickers) == 1:
                close = data.to_frame("Close") if "Close" in data else data
                if "Close" in data:
                    close = data[["Close"]].rename(columns={"Close": tickers[0].upper()})
                else:
                    close = data
                    close.columns = [t.upper() for t in tickers]
            else:
                close = data

        close = close.dropna(how="all")
        close.columns = [c.upper() for c in close.columns]
        close = close.ffill()

        actual = [c for c in close.columns if c.upper() in {t.upper() for t in tickers}]
        if not actual:
            raise ValueError("No matching tickers found in price data.")

        return close[actual]

    def get_returns(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        prices = self.get_prices(tickers, start, end)
        returns = prices.pct_change().dropna(how="all")
        if returns.empty:
            raise ValueError("No return data could be computed.")
        return returns


class PriceCache:
    def __init__(self, provider: PriceProvider):
        self._provider = provider
        self._cache: dict = {}

    def get_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        key = (tuple(sorted(tickers)), start, end)
        if key not in self._cache:
            self._cache[key] = self._provider.get_prices(tickers, start, end)
        return self._cache[key].copy()

    def get_returns(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        key_r = ("ret", tuple(sorted(tickers)), start, end)
        if key_r not in self._cache:
            prices = self.get_prices(tickers, start, end)
            self._cache[key_r] = prices.pct_change().dropna(how="all")
        return self._cache[key_r].copy()
