
from risk_monitor.config import DataConfig
from risk_monitor.data.cache import DiskCache


def test_disk_cache_set_get(tmp_path):
    config = DataConfig(cache_dir=str(tmp_path / "cache"), cache_ttl_days=7)
    cache = DiskCache(config)
    obj = {"test": [1, 2, 3], "nested": {"a": 1}}
    cache.set(obj, "test", "key1")
    retrieved = cache.get("test", "key1")
    assert retrieved == obj


def test_disk_cache_miss(tmp_path):
    config = DataConfig(cache_dir=str(tmp_path / "cache"))
    cache = DiskCache(config)
    result = cache.get("nonexistent", "key")
    assert result is None


def test_disk_cache_clear(tmp_path):
    config = DataConfig(cache_dir=str(tmp_path / "cache"))
    cache = DiskCache(config)
    cache.set("data1", "pfx", "key1")
    cache.set("data2", "pfx", "key2")
    cache.set("data3", "other", "key3")
    count = cache.clear(prefix="pfx")
    assert count == 2
    assert cache.get("pfx", "key1") is None
    assert cache.get("other", "key3") is not None


def test_disk_cache_clear_all(tmp_path):
    config = DataConfig(cache_dir=str(tmp_path / "cache"))
    cache = DiskCache(config)
    cache.set("data", "pfx", "key")
    count = cache.clear()
    assert count > 0
