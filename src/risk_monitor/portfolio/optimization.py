from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.optimize as opt


def _regularize(cov: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    values = cov.values.copy()
    values[np.diag_indices_from(values)] += epsilon
    return pd.DataFrame(values, index=cov.index, columns=cov.columns)


@dataclass
class PortfolioOptimizer:
    """
    Portfolio optimization with configurable objective and constraints.
    Uses SLSQP with increased iteration budget and regularized covariance.
    Falls back to direct quadratic programming for minimum variance.
    """

    maxiter: int = 2000
    regularization: float = 1e-6

    def _solve(self, objective, n, bounds, constraints, w0=None):
        w0 = w0 if w0 is not None else np.ones(n) / n
        options = {"maxiter": self.maxiter, "ftol": 1e-15}
        result = opt.minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
        return result

    def _min_variance_closed_form(self, cov: pd.DataFrame) -> Optional[pd.Series]:
        """Direct analytical solution for unconstrained min variance portfolio."""
        n = len(cov)
        try:
            inv_cov = np.linalg.inv(cov.values)
            ones = np.ones(n)
            w = inv_cov @ ones / (ones @ inv_cov @ ones)
            return pd.Series(w, index=cov.index)
        except np.linalg.LinAlgError:
            return None

    def minimum_variance(
        self,
        cov: pd.DataFrame,
        bounds: Optional[List[Tuple[float, float]]] = None,
        max_weight: float = 0.15,
    ) -> pd.Series:
        n = len(cov)
        tickers = list(cov.index)
        bounds = bounds or [(0.0, max_weight)] * n
        reg_cov = _regularize(cov, self.regularization)

        def objective(w):
            return w @ reg_cov.values @ w

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        closed = self._min_variance_closed_form(reg_cov)
        w0 = None
        if closed is not None:
            w0 = closed.clip(0.0, max_weight).values
            w0 = w0 / w0.sum()
        result = self._solve(objective, n, bounds, constraints, w0=w0)
        if result.success:
            return pd.Series(result.x, index=tickers)
        if closed is not None:
            clipped = closed.clip(0.0, max_weight)
            return clipped / clipped.sum()
        raise ValueError(f"Minimum variance optimization failed: {result.message}")

    def risk_parity(
        self,
        cov: pd.DataFrame,
        bounds: Optional[List[Tuple[float, float]]] = None,
        max_weight: float = 0.20,
    ) -> pd.Series:
        n = len(cov)
        tickers = list(cov.index)
        bounds = bounds or [(0.0, max_weight)] * n
        reg_cov = _regularize(cov, self.regularization)

        def _risk_contributions(w):
            port_var = w @ reg_cov.values @ w
            if port_var <= 0:
                return np.ones(n) * (1 / n)
            marginal = reg_cov.values @ w
            rc = w * marginal
            return rc / rc.sum()

        def objective(w):
            rc = _risk_contributions(w)
            target = 1.0 / n
            return np.sum((rc - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        result = self._solve(objective, n, bounds, constraints)
        if result.success:
            return pd.Series(result.x, index=tickers)
        return pd.Series(np.ones(n) / n, index=tickers)

    def max_diversification(
        self,
        cov: pd.DataFrame,
        bounds: Optional[List[Tuple[float, float]]] = None,
        max_weight: float = 0.15,
    ) -> pd.Series:
        n = len(cov)
        tickers = list(cov.index)
        bounds = bounds or [(0.0, max_weight)] * n
        reg_cov = _regularize(cov, self.regularization)
        std = np.sqrt(np.diag(reg_cov.values))

        def objective(w):
            port_std = np.sqrt(w @ reg_cov.values @ w)
            weighted_std = w @ std
            if port_std <= 0:
                return 0.0
            return -weighted_std / port_std

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        result = self._solve(objective, n, bounds, constraints)
        if result.success:
            return pd.Series(result.x, index=tickers)
        return pd.Series(np.ones(n) / n, index=tickers)

    def hrp(
        self,
        cov: pd.DataFrame,
    ) -> pd.Series:
        tickers = list(cov.index)
        n = len(tickers)
        if n < 2:
            return pd.Series([1.0], index=tickers)

        corr = cov.corr().values
        dist = np.sqrt(2 * (1 - corr.clip(-1, 1)))
        np.fill_diagonal(dist, 0.0)

        linkage = sch.linkage(squareform_to_vector(dist), method="single")
        order = sch.leaves_list(linkage)

        sorted_tickers = [tickers[i] for i in order]
        sorted_cov = cov.values[order][:, order]

        def _get_weights(cov_sub: np.ndarray) -> np.ndarray:
            n_sub = cov_sub.shape[0]
            if n_sub == 1:
                return np.array([1.0])
            inv_diag = 1.0 / np.diag(cov_sub)
            w = inv_diag / inv_diag.sum()
            return w

        def _cluster_weights(cov_sub: np.ndarray, node: int) -> np.ndarray:
            if node < 0:
                return _get_weights(cov_sub)
            n = cov_sub.shape[0]
            if n == 1:
                return np.array([1.0])
            mid = n // 2
            cov1 = cov_sub[:mid, :mid]
            cov2 = cov_sub[mid:, mid:]
            alpha = 1.0 - _get_weights(cov_sub)[0]
            w1 = alpha * _cluster_weights(cov1, -1)
            w2 = (1 - alpha) * _cluster_weights(cov2, -1)
            return np.concatenate([w1, w2])

        try:
            weights = _cluster_weights(sorted_cov, 1)
            result = pd.Series(weights, index=sorted_tickers)
            result = result.reindex(tickers).fillna(0.0)
            return result / result.sum()
        except Exception:
            return pd.Series(np.ones(n) / n, index=tickers)


def squareform_to_vector(square: np.ndarray) -> np.ndarray:
    n = square.shape[0]
    return square[np.triu_indices(n, k=1)]


def compute_efficient_frontier(
    cov: pd.DataFrame,
    returns: pd.Series,
    n_points: int = 50,
    max_weight: float = 0.15,
) -> pd.DataFrame:
    n = len(cov)
    bounds = [(0.0, max_weight)] * n
    reg_cov = _regularize(cov, 1e-8)

    points = []
    for target in np.linspace(returns.min(), returns.max(), n_points):

        def objective(w):
            return w @ reg_cov.values @ w

        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, r=target: w @ returns.values - r},
        ]
        w0 = np.ones(n) / n
        result = opt.minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        if result.success:
            w = result.x
            vol = np.sqrt(w @ reg_cov.values @ w) * np.sqrt(252)
            ret = w @ returns.values * 252
            sharpe = ret / vol if vol > 0 else 0.0
            points.append({"vol": vol, "ret": ret, "sharpe": sharpe})

    if not points:
        return pd.DataFrame()
    return pd.DataFrame(points)
