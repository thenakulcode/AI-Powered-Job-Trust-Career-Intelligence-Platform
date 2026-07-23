"""Tiny in-memory TTL cache for extraction results.

Job posting pages don't change second-to-second, and the same URL is often
re-submitted (user re-checks a job, frontend retries, etc.). This avoids
re-launching Playwright or re-hitting an ATS API for a URL we already
resolved recently.

Deliberately simple (dict + monotonic timestamps) rather than a dependency
like `cachetools` — this is a single-process, low-cardinality cache and
doesn't need eviction policies beyond a max size guard.
"""

from __future__ import annotations

import threading
import time
from typing import Any

_DEFAULT_TTL_SECONDS = 60 * 30  # 30 minutes
_MAX_ENTRIES = 500

_lock = threading.Lock()
_store: dict[str, tuple[float, Any]] = {}


def get(key: str) -> Any | None:
    with _lock:
        entry = _store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            _store.pop(key, None)
            return None
        return value


def set(key: str, value: Any, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
    with _lock:
        if len(_store) >= _MAX_ENTRIES:
            # Evict the entry that expires soonest rather than tracking
            # insertion order — cheap enough at this scale.
            oldest_key = min(_store, key=lambda k: _store[k][0])
            _store.pop(oldest_key, None)
        _store[key] = (time.monotonic() + ttl_seconds, value)


def clear() -> None:
    with _lock:
        _store.clear()
