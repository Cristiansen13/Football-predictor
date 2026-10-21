import pytest
import requests

from football_predictor.etl import http
from football_predictor.etl.browser import is_challenge_page
from football_predictor.etl.http import BlockedError, HttpClient, _retry_after

CHALLENGE = "<html><head><title>Just a moment...</title></head><body>challenge-platform</body></html>"


def _response(status: int, text: str, headers: dict | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = text.encode()
    response.encoding = "utf-8"
    response.headers.update(headers or {})
    response.url = "https://example.com/page"
    return response


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = 0

    def get(self, url, timeout=None, headers=None):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda seconds: None)


def client(responses, **kwargs):
    return HttpClient(session=FakeSession(responses), delays={"example.com": 0.0}, **kwargs)


def test_challenge_detection():
    assert is_challenge_page(CHALLENGE)
    assert not is_challenge_page("<html><title>Premier League Scores</title><table></table></html>")


def test_blocked_page_uses_browser_fallback_and_is_cached(tmp_path):
    rendered = []

    def browser(url):
        rendered.append(url)
        return "<table id='sched_x'></table>"

    c = client([_response(403, CHALLENGE)], cache_dir=tmp_path, fallback=browser)
    assert c.get_text("https://example.com/page") == "<table id='sched_x'></table>"
    assert rendered == ["https://example.com/page"]
    # Served from the cache the second time: no request, no browser.
    assert c.get_text("https://example.com/page") == "<table id='sched_x'></table>"
    assert c.session.calls == 1 and len(rendered) == 1


def test_challenge_page_with_200_is_not_mistaken_for_content():
    c = client([_response(200, CHALLENGE)])
    with pytest.raises(BlockedError, match="browser fallback"):
        c.get_text("https://example.com/page")


def test_fallback_can_be_disabled_per_request():
    c = client([_response(403, "denied")], fallback=lambda url: "should not be used")
    with pytest.raises(BlockedError):
        c.get_text("https://example.com/api", allow_fallback=False)


def test_transient_errors_are_retried():
    c = client(
        [requests.ConnectionError("reset"), _response(502, "bad gateway"), _response(200, "ok")]
    )
    assert c.get_text("https://example.com/page") == "ok"
    assert c.session.calls == 3


def test_not_found_raises_without_retry():
    c = client([_response(404, "missing")])
    with pytest.raises(requests.HTTPError):
        c.get_text("https://example.com/page")
    assert c.session.calls == 1


def test_retry_after_parsing():
    assert _retry_after("7", 1.0) == 7.0
    assert _retry_after(None, 3.0) == 3.0
    assert _retry_after("not a date", 2.0) == 2.0
    assert _retry_after("Wed, 21 Oct 2015 07:28:00 GMT", 5.0) == 0.0  # in the past


def test_browser_sessions_share_one_playwright_driver(monkeypatch):
    """Two open browser sessions (HTTP fallback + SOFIFA) must not start a second
    Playwright driver: its sync API refuses that in the same thread."""
    import sys
    import types

    from football_predictor.etl import browser as browser_module

    started, stopped = [], []

    class FakePage:
        pass

    class FakeBrowser:
        def new_page(self):
            return FakePage()

        def close(self):
            pass

    class FakeChromium:
        def launch(self, **kwargs):
            return FakeBrowser()

    class FakeDriver:
        chromium = FakeChromium()

        def stop(self):
            stopped.append(self)

    class FakeContext:
        def start(self):
            driver = FakeDriver()
            started.append(driver)
            return driver

    fake_module = types.ModuleType("playwright.sync_api")
    fake_module.sync_playwright = lambda: FakeContext()
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)
    monkeypatch.setattr(browser_module, "_driver", {"playwright": None, "users": 0})

    first = browser_module.BrowserFetcher().start()
    second = browser_module.BrowserFetcher().start()
    assert len(started) == 1 and first._playwright is second._playwright
    first.close()
    assert stopped == []  # still in use by the second session
    second.close()
    assert stopped == started
    browser_module.BrowserFetcher().start().close()
    assert len(started) == 2  # a fresh driver once every session has closed
