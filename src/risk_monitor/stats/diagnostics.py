from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def compute_vif(exog: pd.DataFrame) -> pd.Series:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = sm.add_constant(exog, has_constant="skip")
    if X.shape[1] <= 1:
        return pd.Series(dtype=float)
    vif_vals = []
    cols = list(X.columns)
    for i in range(1, len(cols)):
        vif_vals.append(variance_inflation_factor(X.values, i))
    return pd.Series(vif_vals, index=cols[1:])


def durbin_watson(residuals: np.ndarray) -> float:
    from statsmodels.stats.stattools import durbin_watson as dw

    return float(dw(residuals))


def breusch_pagan_test(model: sm.OLS) -> Dict:
    from statsmodels.stats.diagnostic import het_breuschpagan

    bp_test = het_breuschpagan(model.resid, model.model.exog)
    return {
        "lm_stat": float(bp_test[0]),
        "lm_pvalue": float(bp_test[1]),
        "f_stat": float(bp_test[2]),
        "f_pvalue": float(bp_test[3]),
        "heteroskedastic": bool(bp_test[1] < 0.05),
    }


def jarque_bera_test(residuals: np.ndarray) -> Dict:
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    return {
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pvalue),
        "normal": bool(jb_pvalue >= 0.05),
    }


def factor_model_diagnostics(
    returns: pd.Series,
    factors: pd.DataFrame,
) -> Dict:
    factor_cols = [c for c in factors.columns if c != "RF" and c in factors.columns]
    excess_ret = returns.copy()
    if "RF" in factors.columns:
        excess_ret = returns - factors["RF"]

    joined = pd.concat([excess_ret.rename("ret"), factors[factor_cols]], axis=1).dropna()
    if len(joined) < 30:
        return {"error": "Insufficient data for diagnostics"}

    y = joined["ret"].values
    X = sm.add_constant(joined[factor_cols].values)

    try:
        model = sm.OLS(y, X).fit()

        vif = compute_vif(joined[factor_cols])

        dw_stat = durbin_watson(model.resid)

        bp = breusch_pagan_test(model)

        jb = jarque_bera_test(model.resid)

        serial_corr = pd.Series(model.resid).autocorr() if len(model.resid) > 1 else 0.0

        return {
            "rsquared": float(model.rsquared),
            "rsquared_adj": float(model.rsquared_adj),
            "f_stat": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "aic": float(model.aic),
            "bic": float(model.bic),
            "nobs": int(model.nobs),
            "durbin_watson": float(dw_stat),
            "vif": vif.to_dict(),
            "breusch_pagan": bp,
            "jarque_bera": jb,
            "residual_autocorr": float(serial_corr),
        }
    except Exception as e:
        return {"error": str(e)}


def bootstrap_factor_exposures(
    returns: pd.Series,
    factors: pd.DataFrame,
    n_samples: int = 1000,
    ci: float = 0.95,
) -> pd.DataFrame:
    factor_cols = [c for c in factors.columns if c != "RF" and c in factors.columns]
    excess_ret = returns.copy()
    if "RF" in factors.columns:
        excess_ret = returns - factors["RF"]

    joined = pd.concat([excess_ret.rename("ret"), factors[factor_cols]], axis=1).dropna()
    if len(joined) < 30:
        return pd.DataFrame()

    n = len(joined)
    results: dict[str, list[float]] = {c: [] for c in ["Alpha"] + factor_cols}

    for _ in range(n_samples):
        idx = np.random.randint(0, n, n)
        sample = joined.iloc[idx]
        y = sample["ret"].values
        X = sm.add_constant(sample[factor_cols].values)
        try:
            model = sm.OLS(y, X).fit()
            results["Alpha"].append(model.params[0])
            for i, c in enumerate(factor_cols):
                results[c].append(model.params[i + 1])
        except Exception:
            continue

    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100

    summary = {}
    for col_name, values in results.items():
        if not values:
            continue
        arr = np.array(values)
        summary[col_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "lower": float(np.percentile(arr, lower_pct)),
            "upper": float(np.percentile(arr, upper_pct)),
            "p_value": float(2 * (1 - stats.norm.cdf(abs(np.mean(arr)) / max(np.std(arr), 1e-12)))),
        }

    return pd.DataFrame(summary).T


def walk_forward_ic(
    factor_returns: pd.Series,
    security_returns: pd.DataFrame,
    window: int = 252,
    step: int = 21,
) -> pd.Series:
    dates = security_returns.index.sort_values()
    ic_values = []

    for i in range(window, len(dates) + 1, step):
        est_end = dates[i - 1]
        est_start = dates[i - window]
        fwd_start = est_end
        fwd_end = dates[min(i - 1 + step, len(dates) - 1)]

        est_factor = factor_returns.loc[est_start:est_end]
        est_returns = security_returns.loc[est_start:est_end]

        if len(est_factor) < 60 or est_returns.empty:
            continue

        betas = {}
        for ticker in est_returns.columns:
            sr = est_returns[ticker].dropna()
            j = pd.concat([sr.rename("ret"), est_factor.rename("f")], axis=1).dropna()
            if len(j) < 30:
                continue
            try:
                beta = sm.OLS(j["ret"].values, sm.add_constant(j["f"].values)).fit().params[1]
                betas[ticker] = beta
            except Exception:
                continue

        if not betas:
            continue

        fwd_returns = security_returns.loc[fwd_start:fwd_end]
        if fwd_returns.empty:
            continue

        betas_s = pd.Series(betas)
        fwd_ret = fwd_returns.mean()
        common = betas_s.index.intersection(fwd_ret.index)
        if len(common) < 5:
            continue

        ic = betas_s[common].corr(fwd_ret[common])
        if not np.isnan(ic):
            ic_values.append(ic)

    if not ic_values:
        return pd.Series(dtype=float, name="ic")
    return pd.Series(ic_values, name="ic")


@dataclass
class PurgedCV:
    n_test: int = 252
    n_train: int = 756
    embargo: int = 21
    n_splits: int = 4
    purge: int = 21

    def split(
        self,
        index: pd.DatetimeIndex,
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        n = len(index)
        step = (n - self.n_train - self.n_test) // max(self.n_splits - 1, 1)
        step = max(step, self.n_test + self.embargo)

        splits = []
        for i in range(self.n_splits):
            test_end = n - i * step
            test_start = max(0, test_end - self.n_test)
            if test_start < self.purge:
                break
            train_end = max(0, test_start - self.embargo)
            train_start = max(0, train_end - self.n_train)

            train_idx = index[train_start:train_end]
            test_idx = index[test_start:test_end]

            if len(train_idx) < 60 or len(test_idx) < 20:
                break
            splits.append((train_idx, test_idx))

        return list(reversed(splits))


def walk_forward_purged_cv(
    returns: pd.Series,
    factors: pd.DataFrame,
    model_fn: Callable,
    cv: Optional[PurgedCV] = None,
    metric: str = "rsquared",
) -> pd.DataFrame:
    cv = cv or PurgedCV()
    results = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(returns.index)):
        train_ret = returns.loc[train_idx]
        train_fac = factors.loc[train_idx]
        test_ret = returns.loc[test_idx]
        test_fac = factors.loc[test_idx]

        try:
            model = model_fn(train_ret, train_fac)
            if "error" in model:
                raise ValueError(model["error"])

            y_test = test_ret.values
            X_test = sm.add_constant(test_fac[[c for c in factors.columns if c != "RF"]].values)
            predictions = X_test @ np.array(
                [model.get(c, 0.0) for c in ["const"] + [c for c in factors.columns if c != "RF"]]
            )
            actual = y_test

            residuals = actual - predictions[: len(actual)]
            ss_res = (residuals**2).sum()
            ss_tot = ((actual - actual.mean()) ** 2).sum()
            oos_r2 = 1 - ss_res / max(ss_tot, 1e-10)

            results.append(
                {
                    "fold": fold,
                    "train_start": train_idx[0],
                    "train_end": train_idx[-1],
                    "test_start": test_idx[0],
                    "test_end": test_idx[-1],
                    "oos_rsquared": float(round(oos_r2, 4)),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )
        except Exception as e:
            results.append(
                {
                    "fold": fold,
                    "train_start": train_idx[0],
                    "train_end": train_idx[-1],
                    "test_start": test_idx[0],
                    "test_end": test_idx[-1],
                    "error": str(e),
                }
            )

    return pd.DataFrame(results) if results else pd.DataFrame()
