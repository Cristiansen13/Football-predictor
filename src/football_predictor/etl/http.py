"""Polite HTTP client: per-host rate limiting, retries and an on-disk cache."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..config import USER_AGENT

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpClient:
    def __init__(
        self,
        cache_dir: Path | None = None,
        delays: dict[str, float] | None = None,
        *,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.delays = delays or {}
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request: dict[str, float] = {}

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(url.encode()).hexdigest()
        return self.cache_dir / urlparse(url).netloc / f"{digest}.txt"

    def _throttle(self, host: str) -> None:
        delay = self.delays.get(host, 1.0)
        elapsed = time.monotonic() - self._last_request.get(host, float("-inf"))
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request[host] = time.monotonic()

    def get_text(
        self,
        url: str,
        *,
        use_cache: bool = True,
        encoding: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        cache_path = self._cache_path(url)
        if use_cache and cache_path is not None and cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        host = urlparse(url).netloc
        for attempt in range(self.max_retries + 1):
            self._throttle(host)
            response = self.session.get(url, timeout=self.timeout, headers=headers)
            if response.status_code not in RETRY_STATUS or attempt == self.max_retries:
                break
            wait = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
            log.warning("GET %s -> %s, retrying in %.0fs", url, response.status_code, wait)
            time.sleep(wait)
        response.raise_for_status()
        if encoding:
            response.encoding = encoding
        text = response.text

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        return text
