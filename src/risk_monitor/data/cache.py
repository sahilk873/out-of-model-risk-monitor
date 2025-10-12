from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from risk_monitor.config import DataConfig
from risk_monitor.exceptions import CacheError


def _stored_prefix(meta_path: str) -> Optional[str]:
    """Read the raw key from metadata and return its first segment."""
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("prefix_hint")
    except Exception:
        return None


class DiskCache:
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.cache_dir = self.config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, prefix: str, *args: str) -> str:
        raw = ":".join([prefix] + list(args))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def _meta_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.meta.json")

    def get(self, prefix: str, *args: str) -> Optional[Any]:
        key = self._key(prefix, *args)
        path = self._path(key)
        meta_path = self._meta_path(key)
        if not os.path.exists(path) or not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            ts = datetime.fromisoformat(meta["cached_at"])
            if (datetime.now() - ts).days > self.config.cache_ttl_days:
                return None
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise CacheError(f"Cache read failed for {key}: {e}")

    def set(self, obj: Any, prefix: str, *args: str) -> None:
        key = self._key(prefix, *args)
        path = self._path(key)
        meta_path = self._meta_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "cached_at": datetime.now().isoformat(),
                        "key": key,
                        "prefix_hint": prefix,
                    },
                    f,
                )
        except Exception as e:
            raise CacheError(f"Cache write failed for {key}: {e}")

    def clear(self, prefix: Optional[str] = None) -> int:
        count = 0
        for fname in list(os.listdir(self.cache_dir)):
            if not fname.endswith(".meta.json"):
                continue
            fpath = os.path.join(self.cache_dir, fname)
            remove = not prefix
            if prefix:
                hint = _stored_prefix(fpath)
                if hint == prefix:
                    remove = True
            if remove:
                pkl_path = fpath.replace(".meta.json", ".pkl")
                if os.path.exists(pkl_path):
                    os.remove(pkl_path)
                os.remove(fpath)
                count += 1
        return count


class CachedPriceProvider:
    def __init__(self, provider, cache: Optional[DiskCache] = None):
        self._provider = provider
        self._cache = cache or DiskCache()
        self._mem_cache: Dict[Tuple, pd.DataFrame] = {}

    def get_prices(self, tickers, start, end):
        key_tuple = (tuple(sorted(tickers)), start, end)
        if key_tuple in self._mem_cache:
            return self._mem_cache[key_tuple].copy()
        cached = self._cache.get("prices", *key_tuple)
        if cached is not None:
            self._mem_cache[key_tuple] = cached
            return cached.copy()
        result = self._provider.get_prices(tickers, start, end)
        self._cache.set(result, "prices", *key_tuple)
        self._mem_cache[key_tuple] = result
        return result.copy()

    def get_returns(self, tickers, start, end):
        key_tuple = ("ret", tuple(sorted(tickers)), start, end)
        if key_tuple in self._mem_cache:
            return self._mem_cache[key_tuple].copy()
        cached = self._cache.get("returns", *key_tuple)
        if cached is not None:
            self._mem_cache[key_tuple] = cached
            return cached.copy()
        prices = self.get_prices(tickers, start, end)
        result = prices.pct_change().dropna(how="all")
        self._cache.set(result, "returns", *key_tuple)
        self._mem_cache[key_tuple] = result
        return result.copy()
