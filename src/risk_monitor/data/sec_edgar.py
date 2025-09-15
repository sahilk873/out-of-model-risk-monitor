from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests


class SECEdgarClient:
    BASE = "https://data.sec.gov"
    SUBMISSIONS = "https://data.sec.gov/submissions"
    XBRL = "https://data.sec.gov/api/xbrl/companyfacts"

    def __init__(self, user_agent: str):
        if not user_agent or "email" not in user_agent.lower():
            raise ValueError(
                "SEC requires a User-Agent with contact email. "
                "Set SEC_USER_AGENT in .env (e.g. 'YourName your@email.com')"
            )
        self.headers = {"User-Agent": user_agent, "Accept": "application/json"}

    def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        time.sleep(0.1)
        resp = requests.get(url, headers=self.headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def search_cik(self, ticker: str) -> Optional[str]:
        url = f"{self.BASE}/files/company_tickers.json"
        data = self._get(url)
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
        return None

    def get_company_concept(self, cik: str, taxonomy: str, concept: str) -> Dict:
        url = f"{self.BASE}/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json"
        return self._get(url)

    def get_company_facts(self, cik: str) -> Dict:
        url = f"{self.XBRL}/CIK{cik}.json"
        return self._get(url)

    def get_submissions(self, cik: str) -> Dict:
        url = f"{self.SUBMISSIONS}/CIK{cik}.json"
        return self._get(url)

    def get_filing_text_urls(self, cik: str, form: str = "10-K", count: int = 5) -> List[Dict]:
        subs = self.get_submissions(cik)
        filings = subs.get("filings", {}).get("recent", {})
        urls = []
        for i, frm in enumerate(filings.get("form", [])):
            if frm == form and len(urls) < count:
                acc = filings["accessionNumber"][i]
                primary = filings.get("primaryDocument", [""])[i]
                urls.append(
                    {
                        "accession": acc.replace("-", ""),
                        "form": form,
                        "filing_date": filings.get("filingDate", [""])[i],
                        "primary_doc": primary,
                        "url": (
                            f"https://www.sec.gov/Archives/edgar/data/"
                            f"{int(cik)}/{acc.replace('-', '')}/{primary}"
                        ),
                    }
                )
        return urls

    def get_filing_text(self, url: str) -> str:
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.text

    def get_ticker_metadata(self, ticker: str) -> Optional[Dict]:
        cik = self.search_cik(ticker)
        if cik is None:
            return None
        facts = self.get_company_facts(cik)
        subs = self.get_submissions(cik)
        return {
            "cik": cik,
            "ticker": ticker.upper(),
            "name": subs.get("name", ""),
            "sic": subs.get("sicDescription", ""),
            "sector": self._extract_sector(facts),
        }

    @staticmethod
    def _extract_sector(facts: Dict) -> str:
        return "Unknown"
