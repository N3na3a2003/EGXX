from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any

from config import CACHE_DIR, CACHE_TTL_SECONDS


class FileCache:
    """Tiny disk cache for API responses and computed frames."""

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(f"{namespace}:{key}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{namespace}_{digest}.pkl"

    def get(self, namespace: str, key: str) -> Any | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                record = pickle.load(fh)
        except Exception:
            self._remove_quietly(path)
            return None
        if not isinstance(record, dict) or "created_at" not in record or "value" not in record:
            self._remove_quietly(path)
            return None
        if time.time() - float(record["created_at"]) > self.ttl_seconds:
            return None
        return record["value"]

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        try:
            with path.open("wb") as fh:
                pickle.dump({"created_at": time.time(), "value": value}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return

    def clear_expired(self) -> int:
        removed = 0
        now = time.time()
        for path in self.cache_dir.glob("*.pkl"):
            try:
                with path.open("rb") as fh:
                    record = pickle.load(fh)
                if not isinstance(record, dict) or now - float(record.get("created_at", 0)) > self.ttl_seconds:
                    path.unlink(missing_ok=True)
                    removed += 1
            except Exception:
                if self._remove_quietly(path):
                    removed += 1
        return removed

    def clear_all(self) -> int:
        removed = 0
        for path in self.cache_dir.glob("*.pkl"):
            if self._remove_quietly(path):
                removed += 1
        return removed

    @staticmethod
    def _remove_quietly(path: Path) -> bool:
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError:
            return False
