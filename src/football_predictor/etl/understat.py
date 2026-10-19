"""Understat: xG, xA and PPDA extracted from JSON embedded in <script> tags.

League pages embed ``var datesData = JSON.parse('\\x5B...')``; the payload is a
JavaScript string literal of hex escapes that has to be unescaped before it is
valid JSON. If Understat stops embedding the data, the same payload is
requested from its AJAX endpoints instead.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from .http import HttpClient

LEAGUE_URL = "https://understat.com/league/EPL/{season}"
MATCH_URL = "https://understat.com/match/{match_id}"
LEAGUE_API_URL = "https://understat.com/getLeagueData/EPL/{season}"
MATCH_API_URL = "https://understat.com/getMatchData/{match_id}"
AJAX_HEADERS = {"X-Requested-With": "XMLHttpRequest"}

log = logging.getLogger(__name__)

_JS_ESCAPE = re.compile(r"\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)", re.S)
_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


def _unescape_js_string(raw: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token[0] in "xu" and len(token) > 1:
            return chr(int(token[1:], 16))
        return _SIMPLE_ESCAPES.get(token, token)

    return _JS_ESCAPE.sub(replace, raw)


def extract_json_var(html: str, var_name: str) -> Any:
    """Return the decoded value of ``var <var_name> = JSON.parse('...')``."""
    pattern = re.compile(
        rf"(?:var|let|const)\s+{re.escape(var_name)}\s*=\s*JSON\.parse\(\s*'(?P<data>(?:[^'\\]|\\.)*)'\s*\)",
        re.S,
    )
    match = pattern.search(html)
    if match is None:
        raise ValueError(f"Variable {var_name!r} not found in page")
    return json.loads(_unescape_js_string(match.group("data")))


def _ajax(client: HttpClient, url: str, referer: str, **kwargs) -> Any:
    """Understat's own data endpoint; it expects an XHR from the page it serves."""
    headers = {**AJAX_HEADERS, "Referer": referer}
    return json.loads(client.get_text(url, headers=headers, allow_fallback=False, **kwargs))


def _league_payload(client: HttpClient, season: int, **kwargs) -> dict[str, Any]:
    page = LEAGUE_URL.format(season=season)
    html = client.get_text(page, **kwargs)
    try:
        return {
            "dates": extract_json_var(html, "datesData"),
            "teams": extract_json_var(html, "teamsData"),
        }
    except ValueError:
        payload = _ajax(client, LEAGUE_API_URL.format(season=season), page, **kwargs)
        if not {"dates", "teams"} <= set(payload):
            raise ValueError(f"Unexpected Understat league payload keys: {sorted(payload)}")
        return payload


def fetch_league(client: HttpClient, season: int, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download one EPL season and return (matches, team_match_stats)."""
    payload = _league_payload(client, season, **kwargs)
    matches = parse_league_matches(payload["dates"], season)
    history = parse_team_history(payload["teams"])
    linked = link_team_history(matches, history)
    if len(linked) < len(history):
        log.warning("understat %s: %d of %d team-match rows matched no fixture",
                    season, len(history) - len(linked), len(history))
    return matches, linked


def fetch_match_rosters(client: HttpClient, understat_id: str, **kwargs) -> pd.DataFrame:
    page = MATCH_URL.format(match_id=understat_id)
    html = client.get_text(page, **kwargs)
    try:
        rosters = extract_json_var(html, "rostersData")
    except ValueError:
        rosters = _ajax(client, MATCH_API_URL.format(match_id=understat_id), page, **kwargs)["rosters"]
    return parse_match_rosters(rosters, understat_id)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def parse_league_matches(dates_data: list[dict[str, Any]], season: int) -> pd.DataFrame:
    rows = []
    for item in dates_data:
        played = bool(item.get("isResult"))
        goals, xg = item.get("goals") or {}, item.get("xG") or {}
        rows.append(
            {
                "understat_id": str(item["id"]),
                "season": season,
                "kickoff": pd.Timestamp(item["datetime"]),
                "home_team": item["h"]["title"],
                "away_team": item["a"]["title"],
                "home_goals": int(goals["h"]) if played and goals.get("h") is not None else pd.NA,
                "away_goals": int(goals["a"]) if played and goals.get("a") is not None else pd.NA,
                "home_xg": _to_float(xg.get("h")) if played else np.nan,
                "away_xg": _to_float(xg.get("a")) if played else np.nan,
                "is_result": played,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame[["home_goals", "away_goals"]] = frame[["home_goals", "away_goals"]].astype("Int64")
    return frame


def _ratio(stat: dict[str, Any] | None) -> float:
    if not stat:
        return np.nan
    attempts, actions = _to_float(stat.get("att")), _to_float(stat.get("def"))
    return attempts / actions if actions else np.nan


def parse_team_history(teams_data: dict[str, Any]) -> pd.DataFrame:
    """Per-team, per-match rows from ``teamsData`` (xG, xGA, PPDA, deep completions)."""
    rows = []
    for team in teams_data.values():
        for entry in team.get("history", []):
            rows.append(
                {
                    "team": team["title"],
                    "kickoff": pd.Timestamp(entry["date"]),
                    "venue": "home" if entry.get("h_a") == "h" else "away",
                    "xg": _to_float(entry.get("xG")),
                    "xga": _to_float(entry.get("xGA")),
                    "npxg": _to_float(entry.get("npxG")),
                    "npxga": _to_float(entry.get("npxGA")),
                    # Passes allowed per defensive action: lower = more intense press.
                    "ppda": _ratio(entry.get("ppda")),
                    "ppda_allowed": _ratio(entry.get("ppda_allowed")),
                    "deep": _to_float(entry.get("deep")),
                    "deep_allowed": _to_float(entry.get("deep_allowed")),
                    "xpts": _to_float(entry.get("xpts")),
                }
            )
    return pd.DataFrame(rows)


def link_team_history(matches: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Attach the fixture (home/away team names) to each team history row.

    ``teamsData`` rows carry no opponent, so they are matched to fixtures on
    (team, kickoff, venue).
    """
    if history.empty:
        return history.assign(home_team=pd.Series(dtype=str), away_team=pd.Series(dtype=str))
    home = matches[["kickoff", "home_team", "away_team"]].assign(team=matches["home_team"], venue="home")
    away = matches[["kickoff", "home_team", "away_team"]].assign(team=matches["away_team"], venue="away")
    keys = pd.concat([home, away], ignore_index=True)
    return history.merge(keys, on=["team", "kickoff", "venue"], how="inner")


def parse_match_rosters(rosters: dict[str, Any], understat_id: str) -> pd.DataFrame:
    """Players who featured in a match; ``position == 'Sub'`` marks substitutes."""
    rows = []
    for side in ("h", "a"):
        for player in (rosters.get(side) or {}).values():
            position = player.get("position", "")
            minutes = _to_float(player.get("time"))
            rows.append(
                {
                    "understat_id": str(understat_id),
                    "venue": "home" if side == "h" else "away",
                    "player_name": player.get("player"),
                    "position": position,
                    "is_starter": position != "Sub",
                    "minutes": 0 if np.isnan(minutes) else int(minutes),
                    "xg": _to_float(player.get("xG")),
                    "xa": _to_float(player.get("xA")),
                }
            )
    return pd.DataFrame(rows)
