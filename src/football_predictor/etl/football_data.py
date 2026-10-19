"""Football-Data.co.uk: results, basic match stats and bookmaker odds (direct CSV)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from ..config import season_code
from .http import HttpClient

CSV_URL = "https://www.football-data.co.uk/mmz4281/{code}/{division}.csv"

# bookmaker -> market time -> (home, draw, away) column names. Closing odds were
# added in 2012/13 for Pinnacle and in 2019/20 for the market average; older
# files publish the average as BbAv*.
ODDS_COLUMNS = {
    "pinnacle": {"closing": ("PSCH", "PSCD", "PSCA"), "opening": ("PSH", "PSD", "PSA")},
    "average": {"closing": ("AvgCH", "AvgCD", "AvgCA"), "opening": ("AvgH", "AvgD", "AvgA")},
    "average_legacy": {"opening": ("BbAvH", "BbAvD", "BbAvA")},
    "bet365": {"closing": ("B365CH", "B365CD", "B365CA"), "opening": ("B365H", "B365D", "B365A")},
    "max": {"closing": ("MaxCH", "MaxCD", "MaxCA"), "opening": ("MaxH", "MaxD", "MaxA")},
}

STAT_COLUMNS = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
}


def fetch_season_csv(client: HttpClient, season: int, division: str = "E0", **kwargs) -> str:
    url = CSV_URL.format(code=season_code(season), division=division)
    return client.get_text(url, encoding="latin-1", **kwargs)


def _read_csv(csv_text: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(csv_text), on_bad_lines="skip")
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    return frame.dropna(subset=["HomeTeam", "AwayTeam"])


def _kickoff(frame: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(frame["Date"], dayfirst=True, format="mixed")
    if "Time" in frame:
        time = pd.to_timedelta(frame["Time"].fillna("00:00").astype(str) + ":00", errors="coerce")
        date = date + time.fillna(pd.Timedelta(0))
    return date


def parse_season(csv_text: str, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one season file into (matches, odds).

    ``matches`` has one row per fixture with goals and shot counts; ``odds`` is
    long format with one row per (fixture, bookmaker, market time).
    """
    raw = _read_csv(csv_text)
    matches = pd.DataFrame(
        {
            "season": season,
            "kickoff": _kickoff(raw),
            "home_team": raw["HomeTeam"].str.strip(),
            "away_team": raw["AwayTeam"].str.strip(),
            "home_goals": pd.to_numeric(raw["FTHG"], errors="coerce").astype("Int64"),
            "away_goals": pd.to_numeric(raw["FTAG"], errors="coerce").astype("Int64"),
        }
    )
    for column, name in STAT_COLUMNS.items():
        matches[name] = pd.to_numeric(raw[column], errors="coerce") if column in raw else np.nan

    odds_frames = []
    for bookmaker, markets in ODDS_COLUMNS.items():
        for market_time, columns in markets.items():
            if not all(c in raw for c in columns):
                continue
            block = raw[list(columns)].apply(pd.to_numeric, errors="coerce")
            block.columns = ["home", "draw", "away"]
            block.insert(0, "market_time", market_time)
            block.insert(0, "bookmaker", bookmaker.removesuffix("_legacy"))
            block.insert(0, "away_team", matches["away_team"])
            block.insert(0, "home_team", matches["home_team"])
            block.insert(0, "season", season)
            odds_frames.append(block.dropna(subset=["home", "draw", "away"]))
    odds = (
        pd.concat(odds_frames, ignore_index=True)
        .drop_duplicates(["home_team", "away_team", "bookmaker", "market_time"])
        if odds_frames
        else pd.DataFrame(
            columns=["season", "home_team", "away_team", "bookmaker", "market_time", "home", "draw", "away"]
        )
    )
    return matches.reset_index(drop=True), odds
