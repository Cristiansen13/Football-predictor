"""SOFIFA (EA FC ratings): point-in-time player overall ratings.

SOFIFA sits behind a JavaScript challenge, so pages are rendered with a real
browser through Playwright. Every roster update ("r=" parameter) is a dated
snapshot of all ratings, which is what makes the ratings point-in-time.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from types import TracebackType

import pandas as pd
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

BASE_URL = "https://sofifa.com"
# lg=13 is the Premier League; 60 players per page.
PLAYERS_URL = BASE_URL + "/players?type=all&lg%5B0%5D=13&r={roster}&set=true&offset={offset}"
PAGE_SIZE = 60

_PLAYER_ID = re.compile(r"/player/(\d+)")
_ROSTER_ID = re.compile(r"[?&]r=(\d{6})")
_DATE_FORMATS = ("%b %d, %Y", "%d %b %Y", "%Y-%m-%d", "%B %d, %Y")


def _parse_date(text: str) -> datetime | None:
    text = " ".join(text.split())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_roster_versions(html: str) -> pd.DataFrame:
    """Roster update ids and their release dates from the version selector."""
    soup = BeautifulSoup(html, "lxml")
    versions: dict[str, datetime] = {}
    for option in soup.find_all("option"):
        value, date = str(option.get("value", "")), _parse_date(option.get_text())
        roster = value if value.isdigit() and len(value) == 6 else None
        if roster is None:
            match = _ROSTER_ID.search(value)
            roster = match.group(1) if match else None
        if roster and date:
            versions[roster] = date
    for link in soup.find_all("a", href=True):
        match, date = _ROSTER_ID.search(link["href"]), _parse_date(link.get_text())
        if match and date:
            versions.setdefault(match.group(1), date)
    frame = pd.DataFrame(
        {"roster_id": list(versions), "snapshot_date": pd.to_datetime(list(versions.values()))}
    )
    return frame.sort_values("snapshot_date").reset_index(drop=True)


def parse_players_page(html: str) -> pd.DataFrame:
    """Player rows (id, name, team, overall, positions) from a players list page."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.select("table tbody tr"):
        player_link = tr.find("a", href=_PLAYER_ID)
        overall_cell = tr.find("td", attrs={"data-col": "oa"})
        if player_link is None or overall_cell is None:
            continue
        overall = re.search(r"\d+", overall_cell.get_text())
        team_link = tr.find("a", href=re.compile(r"/team/\d+"))
        name = (
            player_link.get("data-tippy-content")
            or player_link.get("title")
            or player_link.get_text(" ", strip=True)
        )
        rows.append(
            {
                "sofifa_id": _PLAYER_ID.search(player_link["href"]).group(1),
                "player_name": " ".join(name.split()),
                "team": team_link.get_text(" ", strip=True) if team_link else None,
                "overall": int(overall.group()) if overall else None,
                "positions": ",".join(span.get_text(strip=True) for span in tr.select("span.pos")),
            }
        )
    return pd.DataFrame(rows, columns=["sofifa_id", "player_name", "team", "overall", "positions"])


class SofifaBrowser:
    """Headless Chromium session that renders SOFIFA pages."""

    def __init__(self, *, delay: float = 3.0, executable_path: str | None = None, headless: bool = True):
        self.delay = delay
        self.executable_path = executable_path
        self.headless = headless
        self._last = float("-inf")

    def __enter__(self) -> "SofifaBrowser":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'scraping' extra: pip install football-predictor[scraping]") from exc
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(
                headless=self.headless, executable_path=self.executable_path
            )
            self._page = self._browser.new_page()
        except BaseException:
            self._playwright.stop()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._browser.close()
        self._playwright.stop()

    def get_html(self, url: str, wait_for: str = "table tbody tr") -> str:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        self._page.goto(url, wait_until="domcontentloaded")
        self._page.wait_for_selector(wait_for, timeout=30_000)
        return self._page.content()

    def roster_versions(self) -> pd.DataFrame:
        return parse_roster_versions(self.get_html(BASE_URL + "/", wait_for="body"))

    def league_ratings(self, roster_id: str, max_pages: int = 20) -> pd.DataFrame:
        """All Premier League player ratings in one roster snapshot."""
        pages = []
        for page in range(max_pages):
            frame = parse_players_page(
                self.get_html(PLAYERS_URL.format(roster=roster_id, offset=page * PAGE_SIZE))
            )
            pages.append(frame)
            if len(frame) < PAGE_SIZE:
                break
        ratings = pd.concat(pages, ignore_index=True).drop_duplicates("sofifa_id")
        return ratings.assign(roster_id=roster_id)


def select_snapshots(versions: pd.DataFrame, start: datetime, end: datetime, every_days: int = 28) -> pd.DataFrame:
    """Thin the roster versions to roughly one per ``every_days`` within [start, end].

    The latest snapshot before ``start`` is kept so the first matches of the
    window also have ratings.
    """
    versions = versions.sort_values("snapshot_date")
    before = versions[versions["snapshot_date"] < start].tail(1)
    window = versions[(versions["snapshot_date"] >= start) & (versions["snapshot_date"] <= end)]
    keep, last = [], None
    for idx, row in window.iterrows():
        if last is None or (row["snapshot_date"] - last).days >= every_days:
            keep.append(idx)
            last = row["snapshot_date"]
    return pd.concat([before, window.loc[keep]]).reset_index(drop=True)
