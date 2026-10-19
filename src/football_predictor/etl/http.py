"""Polite HTTP client: per-host rate limiting, retries, an on-disk cache and a
browser fallback for pages behind anti-bot protection."""

from __future__ import annotations

import hashlib
import logging
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import requests

from ..config import USER_AGENT
from .browser import is_challenge_page

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
BLOCKED_STATUS = {401, 403, 429, 503}


class BlockedError(RuntimeError):
    """The site refused a plain HTTP client (anti-bot page or 403)."""


def _retry_after(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return default


class HttpClient:
    def __init__(
        self,
        cache_dir: Path | None = None,
        delays: dict[str, float] | None = None,
        *,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
        fallback: Callable[[str], str] | None = None,
    ) -> None:
        """
        ``fallback`` renders a URL some other way (e.g. a :class:`BrowserFetcher`)
        and is used when the site answers with a block status or a JavaScript
        challenge instead of content.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.delays = delays or {}
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.timeout = timeout
        self.max_retries = max_retries
        self.fallback = fallback
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

    def _request(self, url: str, headers: dict[str, str] | None) -> requests.Response:
        host = urlparse(url).netloc
        for attempt in range(self.max_retries + 1):
            self._throttle(host)
            try:
                response = self.session.get(url, timeout=self.timeout, headers=headers)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.max_retries:
                    raise
                log.warning("GET %s failed (%s), retrying", url, exc)
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code not in RETRY_STATUS or attempt == self.max_retries:
                return response
            wait = _retry_after(response.headers.get("Retry-After"), 2 ** (attempt + 1))
            log.warning("GET %s -> %s, retrying in %.0fs", url, response.status_code, wait)
            time.sleep(min(wait, 120))
        raise AssertionError("unreachable")

    def get_text(
        self,
        url: str,
        *,
        use_cache: bool = True,
        encoding: str | None = None,
        headers: dict[str, str] | None = None,
        allow_fallback: bool = True,
    ) -> str:
        cache_path = self._cache_path(url)
        if use_cache and cache_path is not None and cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        response = self._request(url, headers)
        if encoding:
            response.encoding = encoding
        blocked = response.status_code in BLOCKED_STATUS or (response.ok and is_challenge_page(response.text))
        if blocked:
            if self.fallback is None or not allow_fallback:
                raise BlockedError(
                    f"{url} refused a plain HTTP client (HTTP {response.status_code}); "
                    "enable the browser fallback (needs Playwright + Chromium)"
                )
            text = self.fallback(url)
        else:
            response.raise_for_status()
            text = response.text

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        return text
