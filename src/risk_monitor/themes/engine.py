from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

PREDEFINED_THEMES: Dict[str, Dict] = {
    "AI_Infrastructure": {
        "seed_tickers": ["NVDA", "AMD", "TSM", "AVGO", "MRVL", "ANET"],
        "keywords": [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "GPU",
            "data center",
            "accelerated computing",
            "large language model",
            "AI inference",
            "AI training",
            "neural network",
        ],
        "description": "Companies exposed to AI hardware and infrastructure spending",
    },
    "Rate_Sensitive_Growth": {
        "seed_tickers": ["PLTR", "DDOG", "ZS", "SNOW", "CRWD", "MDB", "HUBS"],
        "keywords": [
            "high growth",
            "unprofitable",
            "negative free cash flow",
            "long duration",
            "present value of growth",
        ],
        "description": "High-duration growth equities sensitive to interest rate changes",
    },
    "Regional_Banks": {
        "seed_tickers": ["SIVBQ", "FRC", "KEY", "HBAN", "RF", "FITB", "CMA", "ZION"],
        "keywords": [
            "regional bank",
            "commercial real estate",
            "net interest margin",
            "deposit franchise",
            "uninsured deposit",
        ],
        "description": "Regional banks exposed to CRE and deposit franchise risk",
    },
    "China_Supply_Chain": {
        "seed_tickers": ["TSM", "BABA", "JD", "PDD", "NIO", "LI", "XPEV"],
        "keywords": [
            "China",
            "supply chain",
            "tariff",
            "export control",
            "semiconductor",
            "manufacturing",
        ],
        "description": "Companies with significant China exposure in supply chain or revenue",
    },
    "GLP1_Obesity_Drugs": {
        "seed_tickers": ["LLY", "NVO", "AMGN", "VKTX", "ALT"],
        "keywords": [
            "GLP-1",
            "obesity",
            "weight management",
            "diabetes",
            "semaglutide",
            "tirzepatide",
            "incretin",
        ],
        "description": "Companies involved in GLP-1 receptor agonist drug development",
    },
    "Unprofitable_Tech": {
        "seed_tickers": ["U", "SNAP", "RBLX", "DOCU", "ZM", "PTON", "CAR"],
        "keywords": [
            "negative earnings",
            "negative free cash flow",
            "burn rate",
            "cash runway",
            "operating loss",
        ],
        "description": "Technology companies with negative earnings and high cash burn",
    },
    "Oil_Sensitivity": {
        "seed_tickers": ["XOM", "CVX", "COP", "EOG", "OXY", "SLB", "HAL"],
        "keywords": [
            "crude oil",
            "upstream",
            "exploration and production",
            "oil price",
            "hydrocarbon",
        ],
        "description": "Companies with direct exposure to crude oil prices",
    },
    "Mega_Cap_Concentration": {
        "seed_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
        "keywords": [
            "mega cap",
            "large market capitalization",
            "index weight",
            "systemically important",
        ],
        "description": "Largest market capitalization stocks with high index weighting",
    },
}


class ThemeRiskEngine:
    def __init__(self):
        self.themes = PREDEFINED_THEMES.copy()
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)

    def score_holdings_based(
        self,
        active_weights: pd.Series,
        theme_name: str,
    ) -> float:
        theme = self.themes.get(theme_name)
        if theme is None:
            return 0.0
        seed_set = set(t.upper() for t in theme["seed_tickers"])
        total_active = active_weights.abs().sum()
        if total_active == 0:
            return 0.0
        theme_active = (
            active_weights.loc[[t for t in active_weights.index if t.upper() in seed_set]]
            .abs()
            .sum()
        )
        return float(theme_active / total_active * 100)

    def score_filing_similarity(
        self,
        ticker_texts: Dict[str, str],
        theme_name: str,
    ) -> Dict[str, float]:
        theme = self.themes.get(theme_name)
        if theme is None:
            return {}

        tickers = list(ticker_texts.keys())
        texts = list(ticker_texts.values())

        if len(tickers) < 2:
            return {t: 0.0 for t in tickers}

        all_texts = texts + [" ".join(theme["keywords"])]
        try:
            tfidf = self._vectorizer.fit_transform(all_texts)
            keyword_vec = tfidf[-1:]
            doc_vecs = tfidf[:-1]
            scores = cosine_similarity(doc_vecs, keyword_vec).flatten()
            return dict(zip(tickers, scores))
        except ValueError:
            return {t: 0.0 for t in tickers}

    def score_residual_comovement(
        self,
        residuals: pd.DataFrame,
        theme_name: str,
    ) -> float:
        theme = self.themes.get(theme_name)
        if theme is None:
            return 0.0
        seed_set = set(t.upper() for t in theme["seed_tickers"])

        theme_tickers = [t for t in residuals.columns if t.upper() in seed_set]
        if len(theme_tickers) < 2:
            return 0.0

        corr = residuals[theme_tickers].corr().fillna(0.0).values
        n = corr.shape[0]
        if n < 2:
            return 0.0
        upper_tri = corr[np.triu_indices_from(corr, k=1)]
        mean_corr = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
        return mean_corr

    def compute_theme_covariance_concentration(
        self,
        active_weights: pd.Series,
        returns: pd.DataFrame,
        theme_name: str,
    ) -> float:
        theme = self.themes.get(theme_name)
        if theme is None:
            return 0.0
        seed_set = set(t.upper() for t in theme["seed_tickers"])
        theme_tickers = [t for t in returns.columns if t.upper() in seed_set]
        if len(theme_tickers) < 2:
            return 0.0

        theme_returns = returns[theme_tickers].dropna()
        theme_cov = theme_returns.cov().fillna(0.0)
        weights = active_weights.reindex(theme_tickers).fillna(0.0).values
        var = weights @ theme_cov.values @ weights
        port_weights = active_weights.reindex(returns.columns).fillna(0.0).values
        total_returns = returns.dropna()
        if total_returns.empty:
            return 0.0
        port_var = port_weights @ total_returns.cov().fillna(0.0).values @ port_weights
        if port_var <= 0:
            return 0.0
        return float(np.sqrt(var) / np.sqrt(port_var))

    def analyze_all_themes(
        self,
        active_weights: pd.Series,
        residuals: pd.DataFrame,
        returns: pd.DataFrame,
        ticker_texts: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        rows = []
        for tname in self.themes:
            holdings_score = self.score_holdings_based(active_weights, tname)
            residual_corr = self.score_residual_comovement(residuals, tname)
            cov_conc = self.compute_theme_covariance_concentration(active_weights, returns, tname)

            filing_scores: Dict[str, float] = {}
            if ticker_texts:
                filing_scores = self.score_filing_similarity(ticker_texts, tname)

            rows.append(
                {
                    "theme": tname,
                    "description": self.themes[tname]["description"],
                    "holdings_exposure_pct": round(holdings_score, 2),
                    "residual_avg_corr": round(residual_corr, 4),
                    "covariance_concentration_ratio": round(cov_conc, 2),
                    "avg_filing_similarity": round(
                        np.mean(list(filing_scores.values())) if filing_scores else 0.0, 4
                    ),
                    "n_seed_tickers": len(self.themes[tname]["seed_tickers"]),
                    "n_active_holdings": sum(
                        1
                        for t in self.themes[tname]["seed_tickers"]
                        if t.upper() in active_weights.index
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values("holdings_exposure_pct", ascending=False)
