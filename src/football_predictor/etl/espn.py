"""ESPN's public site API: all-competition team schedules and per-match possession.

FBref answers scripted clients with a Cloudflare challenge, so the two things
the model took from it come from here instead: every fixture of a club in cup
and European competitions (for the fatigue index) and possession and shot
counts per league match. The endpoints are the ones ESPN's own site calls;
they are public and need no key, but they are not a documented API.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .http import HttpClient

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
SCOREBOARD_URL = BASE_URL + "/{league}/scoreboard?dates={date:%Y%m%d}"
TEAM_SCHEDULE_URL = BASE_URL + "/{league}/teams/{team_id}/schedule?season={season}"

LEAGUES = {"E0": "eng.1", "E1": "eng.2"}
LEAGUE_NAMES = {"eng.1": "Premier League", "eng.2": "Championship"}
# Competitions a Premier League club can play besides the league.
CUPS = {
    "eng.fa": "FA Cup",
    "eng.league_cup": "EFL Cup",
    "uefa.champions": "Champions League",
    "uefa.europa": "Europa League",
    "uefa.europa.conf": "Conference League",
}
STATISTICS = {"possessionPct": "possession", "totalShots": "shots", "shotsOnTarget": "shots_on_target"}
LOCAL_TZ = "Europe/London"

SCOREBOARD_COLUMNS = ["espn_event_id", "kickoff", "home_team", "away_team", "team", "espn_team_id", "venue",
                      "goals", "possession", "shots", "shots_on_target", "is_final"]
SCHEDULE_COLUMNS = ["team", "kickoff", "competition", "venue", "opponent"]


def _kickoff(text: str) -> pd.Timestamp:
    """ESPN publishes UTC; the rest of the warehouse keeps UK local time without a zone."""
    return pd.Timestamp(text).tz_convert(LOCAL_TZ).tz_localize(None)


def _score(competitor: dict[str, Any]) -> float:
    score = competitor.get("score")
    if isinstance(score, dict):
        score = score.get("value", score.get("displayValue"))
    try:
        return float(score)
    except (TypeError, ValueError):
        return np.nan


def _sides(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    competition = (event.get("competitions") or [{}])[0]
    competitors = competition.get("competitors") or []
    return competition, {c.get("homeAway"): c for c in competitors}


def parse_scoreboard(payload: str | dict[str, Any]) -> pd.DataFrame:
    """One row per (match, team): kickoff, venue, ESPN team id, goals and match statistics."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    rows = []
    for event in data.get("events", []):
        competition, by_side = _sides(event)
        if not {"home", "away"} <= set(by_side):
            continue
        final = str(competition.get("status", {}).get("type", {}).get("name", "")).startswith("STATUS_F")
        names = {side: by_side[side]["team"]["displayName"] for side in ("home", "away")}
        for side, competitor in by_side.items():
            stats = {
                STATISTICS[s["name"]]: pd.to_numeric(s.get("displayValue"), errors="coerce")
                for s in competitor.get("statistics", [])
                if s.get("name") in STATISTICS
            }
            rows.append(
                {
                    "espn_event_id": str(event.get("id")),
                    "kickoff": _kickoff(event["date"]),
                    "home_team": names["home"],
                    "away_team": names["away"],
                    "team": names[side],
                    "espn_team_id": str(competitor["team"]["id"]),
                    "venue": side,
                    "goals": _score(competitor) if final else np.nan,
                    "possession": stats.get("possession", np.nan),
                    "shots": stats.get("shots", np.nan),
                    "shots_on_target": stats.get("shots_on_target", np.nan),
                    "is_final": final,
                }
            )
    return pd.DataFrame(rows, columns=SCOREBOARD_COLUMNS)


def parse_team_schedule(payload: str | dict[str, Any], team: str, competition: str) -> pd.DataFrame:
    """A club's fixtures in one competition: kickoff, venue, opponent."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    team_id = str(data.get("team", {}).get("id", ""))
    rows = []
    for event in data.get("events", []):
        _, by_side = _sides(event)
        if not {"home", "away"} <= set(by_side):
            continue
        mine = next((side for side, c in by_side.items() if str(c["team"]["id"]) == team_id), None)
        opponent_side = "away" if mine == "home" else "home"
        rows.append(
            {
                "team": team,
                "kickoff": _kickoff(event["date"]),
                "competition": competition,
                "venue": mine or "neutral",
                "opponent": by_side[opponent_side]["team"]["displayName"],
            }
        )
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def fetch_scoreboard(client: HttpClient, league: str, date: pd.Timestamp, **kwargs) -> pd.DataFrame:
    url = SCOREBOARD_URL.format(league=league, date=pd.Timestamp(date))
    return parse_scoreboard(client.get_text(url, **kwargs))


def fetch_team_schedule(
    client: HttpClient, league: str, team_id: str, team: str, season: int, competition: str, **kwargs
) -> pd.DataFrame:
    url = TEAM_SCHEDULE_URL.format(league=league, team_id=team_id, season=season)
    return parse_team_schedule(client.get_text(url, **kwargs), team, competition)
