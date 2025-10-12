from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TransactionCostModel:
    trading_volume_window: int = 21
    bid_ask_spread_default: float = 0.0010
    impact_alpha: float = 0.142
    impact_beta: float = 0.500
    impact_gamma: float = 0.200
    participation_rate: float = 0.10
    annual_trading_days: int = 252

    def estimate_bid_ask_spread(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty or "High" not in prices.columns or "Low" not in prices.columns:
            return pd.Series(self.bid_ask_spread_default, index=prices.columns)
        high = prices["High"]
        low = prices["Low"]
        mid = (high + low) / 2
        roll_estimate = (high - low).rolling(21).mean() / mid.rolling(21).mean()
        if roll_estimate.empty:
            return pd.Series(self.bid_ask_spread_default, index=prices.columns)
        spread = roll_estimate.iloc[-1]
        return spread.clip(0.0001, 0.05).fillna(self.bid_ask_spread_default)

    def estimate_amihud_illiquidity(
        self,
        returns: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.Series:
        common = returns.index.intersection(volume.index)
        if len(common) < 20:
            return pd.Series(0.0, index=returns.columns)
        r = returns.loc[common].abs()
        v = volume.loc[common].replace(0, np.nan)
        amihud = (r / v).mean()
        return amihud.fillna(0.0) * 1e6

    def estimate_market_impact(
        self,
        ticker: str,
        trade_size_usd: float,
        avg_daily_volume_usd: float,
        volatility: float,
        spread: float,
    ) -> float:
        if avg_daily_volume_usd <= 0:
            return spread
        participation = trade_size_usd / avg_daily_volume_usd
        if participation <= 0:
            return 0.0

        permanent = self.impact_alpha * (participation**self.impact_beta) * volatility
        temporary = (
            self.impact_gamma
            * np.sign(trade_size_usd)
            * volatility
            * (participation**self.impact_beta)
        )
        return float(abs(permanent) + abs(temporary))

    def compute_portfolio_impact(
        self,
        weights: pd.Series,
        portfolio_value: float,
        avg_daily_volume: pd.Series,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.Series:
        spreads = self.estimate_bid_ask_spread(prices)
        annual_vol = returns.std() * np.sqrt(self.annual_trading_days)

        impacts = []
        for ticker in weights.index:
            if ticker not in avg_daily_volume.index or ticker not in spreads.index:
                continue
            trade_size = abs(weights.get(ticker, 0.0)) * portfolio_value
            adv = avg_daily_volume.get(ticker, 0.0)
            vol = annual_vol.get(ticker, 0.0)
            sp = spreads.get(ticker, self.bid_ask_spread_default)
            impact = self.estimate_market_impact(ticker, trade_size, adv, vol, sp)
            impacts.append(impact)

        return pd.Series(
            impacts,
            index=[t for t in weights.index if t in avg_daily_volume.index and t in spreads.index],
        )

    def cost_adjusted_returns(
        self,
        returns: pd.DataFrame,
        turnover: pd.Series,
        weights: pd.DataFrame,
        portfolio_value: float,
        avg_daily_volume: pd.Series,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        impacts = self.compute_portfolio_impact(
            weights.iloc[-1] if isinstance(weights, pd.DataFrame) else weights,
            portfolio_value,
            avg_daily_volume,
            prices,
            returns,
        )
        avg_impact = impacts.mean() if not impacts.empty else 0.001
        roundtrip_cost = 2 * avg_impact
        cost_penalty = turnover * roundtrip_cost / 100
        return returns.subtract(cost_penalty, axis=0)

    def compute_liquidity_score(
        self,
        volume: pd.DataFrame,
        prices: pd.DataFrame,
        market_cap: Optional[pd.Series] = None,
    ) -> pd.Series:
        spreads = self.estimate_bid_ask_spread(prices)
        adv = (
            volume.rolling(self.trading_volume_window).mean().iloc[-1]
            if len(volume) >= self.trading_volume_window
            else volume.mean()
        )
        dollar_volume = adv * prices["Close"].iloc[-1] if "Close" in prices.columns else adv

        score = pd.Series(0.0, index=spreads.index)
        for t in score.index:
            s = spreads.get(t, 0.01)
            dv = dollar_volume.get(t, 0.0)
            inv_spread = 1.0 / max(s, 0.0001)
            log_dv = np.log10(max(dv, 1.0))
            score[t] = inv_spread * log_dv

        return (score - score.min()) / (score.max() - score.min() + 1e-10)
