"""Headless Chromium (Playwright) for sites that block plain HTTP clients."""

from __future__ import annotations

import logging
import time
from types import TracebackType

log = logging.getLogger(__name__)

CHALLENGE_MARKERS = (
    "<title>Just a moment...</title>",
    "challenge-platform",
    "cf-browser-verification",
    "Attention Required! | Cloudflare",
    "Enable JavaScript and cookies to continue",
)


def is_challenge_page(html: str) -> bool:
    """True for anti-bot interstitials (Cloudflare and similar) instead of content."""
    head = html[:20_000]
    return any(marker in head for marker in CHALLENGE_MARKERS)


# One Playwright driver per process. Playwright's sync API refuses to start a
# second driver in a thread that already runs one, so the HTTP fallback browser
# and the SOFIFA browser (often open at the same time) share this one and
# release it when the last of them closes.
_driver = {"playwright": None, "users": 0}


def _acquire_playwright():
    if _driver["playwright"] is None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'scraping' extra: pip install -e '.[scraping]'") from exc
        _driver["playwright"] = sync_playwright().start()
    _driver["users"] += 1
    return _driver["playwright"]


def _release_playwright() -> None:
    _driver["users"] = max(0, _driver["users"] - 1)
    if _driver["users"] == 0 and _driver["playwright"] is not None:
        _driver["playwright"].stop()
        _driver["playwright"] = None


class BrowserFetcher:
    """Lazily started browser session; usable as a context manager or via close()."""

    def __init__(
        self,
        *,
        delay: float = 3.0,
        executable_path: str | None = None,
        headless: bool = True,
        timeout_ms: int = 45_000,
    ) -> None:
        self.delay = delay
        self.executable_path = executable_path
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._last = float("-inf")
        self._playwright = None
        self._browser = None
        self._page = None

    def start(self) -> "BrowserFetcher":
        if self._page is not None:
            return self
        self._playwright = _acquire_playwright()
        try:
            self._browser = self._playwright.chromium.launch(
                headless=self.headless, executable_path=self.executable_path
            )
            self._page = self._browser.new_page()
        except BaseException:
            _release_playwright()
            self._playwright = None
            raise
        return self

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            _release_playwright()
        self._playwright = self._browser = self._page = None

    def __enter__(self) -> "BrowserFetcher":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def get_html(self, url: str, wait_for: str | None = None) -> str:
        """Render ``url``; waits out a JavaScript challenge, then for ``wait_for``."""
        self.start()
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        deadline = time.monotonic() + self.timeout_ms / 1000
        while is_challenge_page(self._page.content()) and time.monotonic() < deadline:
            self._page.wait_for_timeout(1000)
        if wait_for:
            self._page.wait_for_selector(wait_for, timeout=self.timeout_ms)
        html = self._page.content()
        if is_challenge_page(html):
            hint = ("try headless=False (--headful)" if self.headless
                    else "the site demands an interactive challenge, so it cannot be scraped right now")
            raise RuntimeError(f"Still blocked by an anti-bot challenge at {url}; {hint}")
        return html

    # HttpClient fallback interface
    def __call__(self, url: str) -> str:
        log.info("browser fallback: %s", url)
        return self.get_html(url)
