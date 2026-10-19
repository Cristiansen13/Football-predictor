"""FBref: fixtures, results, possession and shot counts.

FBref tables are parsed through their ``data-stat`` attributes rather than the
visible headers, which change with the stats provider. Some tables are shipped
inside HTML comments and revealed by JavaScript, so comments are searched too.
"""

from __future__ import annotations

import io
import re
from typing import Iterator

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment, Tag

from ..config import season_label
from .http import HttpClient

BASE_URL = "https://fbref.com"
SCHEDULE_URL = BASE_URL + "/en/comps/9/{label}/schedule/{label}-Premier-League-Scores-and-Fixtures"
TEAM_SCHEDULE_URL = (
    BASE_URL
    + "/en/squads/{squad_id}/{label}/matchlogs/all_comps/schedule/{slug}-Scores-and-Fixtures-All-Competitions"
)
TEAM_SHOOTING_URL = (
    BASE_URL + "/en/squads/{squad_id}/{label}/matchlogs/c9/shooting/{slug}-Match-Logs-Premier-League"
)

_SCORE = re.compile(r"(\d+)\s*[–\-]\s*(\d+)")
_TIME = re.compile(r"(\d{1,2}:\d{2})")
_SQUAD = re.compile(r"/squads/(?P<id>[0-9a-f]{8})/(?:[^/]+/)?(?P<slug>[^/]+?)-Stats")


def iter_tables(html: str) -> Iterator[Tag]:
    soup = BeautifulSoup(html, "lxml")
    yield from soup.find_all("table")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "<table" in comment:
            yield from BeautifulSoup(str(comment), "lxml").find_all("table")


def find_table(html: str, id_prefix: str) -> Tag:
    for table in iter_tables(html):
        if str(table.get("id", "")).startswith(id_prefix):
            return table
    raise ValueError(f"No table with id starting {id_prefix!r}")


def table_to_frame(table: Tag) -> pd.DataFrame:
    """Rows of a stats table keyed by ``data-stat``; links are kept as ``<stat>_href``."""
    body = table.find("tbody") or table
    rows = []
    for tr in body.find_all("tr"):
        classes = tr.get("class") or []
        if any(c in classes for c in ("thead", "over_header", "spacer")):
            continue
        row: dict[str, str] = {}
        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if not stat:
                continue
            row[stat] = cell.get_text(" ", strip=True)
            link = cell.find("a", href=True)
            if link is not None:
                row[f"{stat}_href"] = link["href"]
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def read_html_table(table: Tag) -> pd.DataFrame:
    """Header-based view of a table via ``pandas.read_html`` (for ad-hoc exploration)."""
    return pd.read_html(io.StringIO(str(table)))[0]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    cleaned = frame[column].astype(str).str.replace(",", "", regex=False).str.rstrip("%")
    return pd.to_numeric(cleaned, errors="coerce")


def _kickoff(frame: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(frame["date"], errors="coerce")
    if "start_time" in frame:
        time = frame["start_time"].astype(str).str.extract(_TIME, expand=False).fillna("00:00")
        date = date + pd.to_timedelta(time + ":00", errors="coerce").fillna(pd.Timedelta(0))
    return date


def _squad_info(hrefs: pd.Series) -> pd.DataFrame:
    return hrefs.fillna("").str.extract(_SQUAD)


def parse_schedule(html: str, season: int) -> pd.DataFrame:
    raw = table_to_frame(find_table(html, "sched_"))
    raw = raw[raw.get("home_team", pd.Series(dtype=str)).fillna("") != ""].reset_index(drop=True)
    score = raw.get("score", pd.Series("", index=raw.index)).fillna("").str.extract(_SCORE)
    home_squad = _squad_info(raw.get("home_team_href", pd.Series(index=raw.index, dtype=str)))
    away_squad = _squad_info(raw.get("away_team_href", pd.Series(index=raw.index, dtype=str)))
    frame = pd.DataFrame(
        {
            "season": season,
            "kickoff": _kickoff(raw),
            "home_team": raw["home_team"],
            "away_team": raw["away_team"],
            "home_goals": pd.to_numeric(score[0], errors="coerce").astype("Int64"),
            "away_goals": pd.to_numeric(score[1], errors="coerce").astype("Int64"),
            "home_xg": _numeric(raw, "home_xg"),
            "away_xg": _numeric(raw, "away_xg"),
            "home_squad_id": home_squad["id"],
            "home_squad_slug": home_squad["slug"],
            "away_squad_id": away_squad["id"],
            "away_squad_slug": away_squad["slug"],
        }
    )
    return frame.dropna(subset=["kickoff"]).reset_index(drop=True)


def squads_from_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Unique (team, squad_id, slug) triples needed to request team match logs."""
    home = schedule[["home_team", "home_squad_id", "home_squad_slug"]]
    away = schedule[["away_team", "away_squad_id", "away_squad_slug"]]
    home.columns = away.columns = ["team", "squad_id", "slug"]
    return pd.concat([home, away]).dropna().drop_duplicates("squad_id").reset_index(drop=True)


def parse_team_matchlogs(html: str, team: str) -> pd.DataFrame:
    """Team match log (any competition) -> one row per match from the team's view."""
    raw = table_to_frame(find_table(html, "matchlogs_for"))
    raw = raw[raw.get("date", pd.Series(dtype=str)).fillna("") != ""].reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "team": team,
            "kickoff": _kickoff(raw),
            "competition": raw.get("comp", pd.Series("", index=raw.index)),
            "venue": raw.get("venue", pd.Series("", index=raw.index)).str.lower(),
            "opponent": raw.get("opponent", pd.Series("", index=raw.index)),
            "goals_for": _numeric(raw, "goals_for"),
            "goals_against": _numeric(raw, "goals_against"),
            "xg_for": _numeric(raw, "xg_for"),
            "xg_against": _numeric(raw, "xg_against"),
            "possession": _numeric(raw, "possession"),
            "shots": _numeric(raw, "shots"),
            "shots_on_target": _numeric(raw, "shots_on_target"),
        }
    )
    return frame.dropna(subset=["kickoff"]).reset_index(drop=True)


def fetch_schedule(client: HttpClient, season: int, **kwargs) -> pd.DataFrame:
    label = season_label(season)
    return parse_schedule(client.get_text(SCHEDULE_URL.format(label=label), **kwargs), season)


def fetch_team_logs(
    client: HttpClient, squad_id: str, slug: str, team: str, season: int, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all-competition schedule incl. possession, Premier League shooting log)."""
    label = season_label(season)
    schedule_html = client.get_text(
        TEAM_SCHEDULE_URL.format(squad_id=squad_id, label=label, slug=slug), **kwargs
    )
    shooting_html = client.get_text(
        TEAM_SHOOTING_URL.format(squad_id=squad_id, label=label, slug=slug), **kwargs
    )
    return parse_team_matchlogs(schedule_html, team), parse_team_matchlogs(shooting_html, team)


def combine_team_logs(schedule_log: pd.DataFrame, shooting_log: pd.DataFrame) -> pd.DataFrame:
    """Premier League rows of the schedule log, enriched with shot counts."""
    league = schedule_log[schedule_log["competition"].str.contains("Premier League", na=False)]
    league = league.drop(columns=["shots", "shots_on_target"])
    shots = shooting_log[["team", "kickoff", "shots", "shots_on_target"]]
    return league.merge(shots, on=["team", "kickoff"], how="left").reset_index(drop=True)
