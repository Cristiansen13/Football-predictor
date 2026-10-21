"""SOFIFA (EA FC ratings): point-in-time player overall ratings.

SOFIFA sits behind a JavaScript challenge, so pages are rendered with a real
browser through Playwright. Every roster update ("r=" parameter) is a dated
snapshot of all ratings, which is what makes the ratings point-in-time.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup

from .browser import BrowserFetcher

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


class SofifaBrowser(BrowserFetcher):
    """Browser session that renders SOFIFA pages (behind a JavaScript challenge)."""

    def get_html(self, url: str, wait_for: str | None = "table tbody tr") -> str:
        return super().get_html(url, wait_for=wait_for)

    def roster_versions(self, editions: Iterable[int] = ()) -> pd.DataFrame:
        """Roster update ids and dates, from the home page and each game edition.

        Roster ids are ``EEUUUU``: game edition (``24`` = EA FC 24, covering the
        2023/24 season) and update number, so ``EE0001`` opens an edition and
        its version selector lists that edition's updates.
        """
        urls = [BASE_URL + "/"] + [f"{BASE_URL}/?r={edition:02d}0001&set=true" for edition in editions]
        frames = []
        for url in urls:
            try:
                frames.append(parse_roster_versions(self.get_html(url, wait_for="body")))
            except Exception as exc:  # noqa: BLE001 - one edition failing should not stop the rest
                log.warning("sofifa: could not read versions from %s: %s", url, exc)
        versions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if versions.empty:
            raise RuntimeError(
                "Could not read any SOFIFA roster versions; the version selector markup may have "
                "changed (see parse_roster_versions) or the site blocked the browser"
            )
        return versions.drop_duplicates("roster_id").sort_values("snapshot_date").reset_index(drop=True)

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
        if ratings.empty:
            raise RuntimeError(f"No players parsed for roster {roster_id}; the players table markup may have changed")
        return ratings.assign(roster_id=roster_id)


def editions_for_seasons(first_season: int, last_season: int) -> list[int]:
    """Game editions whose ratings cover the given seasons (2023/24 -> 24)."""
    return [(season + 1) % 100 for season in range(first_season, last_season + 1)]


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
