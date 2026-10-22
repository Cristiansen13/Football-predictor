"""Match results and bookmaker odds from Football-Data.co.uk (one CSV per season).

Each file has one row per match with the final score and the pre-match odds
of several bookmakers. We keep the score, Bet365's 1X2 odds (opening and,
where published, closing) and the over/under 2.5 goals odds.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"  # E0 = Premier League
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# CSV column -> our column. Closing odds (the "C" columns) exist from 2019/20.
COLUMNS = {
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
    "B365CH": "closing_home",
    "B365CD": "closing_draw",
    "B365CA": "closing_away",
    "B365>2.5": "odds_over",
    "B365<2.5": "odds_under",
}


def season_code(season: int) -> str:
    """2023 -> '2324' (the 2023/24 season)."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def parse_csv(text: str, season: int) -> pd.DataFrame:
    """Tidy table of one season: date, teams, goals, odds; unplayed rows dropped."""
    raw = pd.read_csv(io.StringIO(text.lstrip("﻿")), on_bad_lines="skip")
    raw.columns = [c.strip() for c in raw.columns]
    present = {src: dst for src, dst in COLUMNS.items() if src in raw}
    frame = raw[list(present)].rename(columns=present)
    frame = frame.dropna(subset=["home", "away", "home_goals", "away_goals"]).copy()
    frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, format="mixed")
    frame["home_goals"] = frame["home_goals"].astype(int)
    frame["away_goals"] = frame["away_goals"].astype(int)
    for column in COLUMNS.values():
        if column not in frame:
            frame[column] = float("nan")
    frame.insert(0, "season", season)
    return frame.sort_values("date").reset_index(drop=True)


def download_season(season: int, data_dir: Path = DATA_DIR) -> Path:
    """Fetch one season's CSV into ``data_dir`` (skipped if already there)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"E0_{season}.csv"
    if not path.exists():
        response = requests.get(URL.format(code=season_code(season)), timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def load_seasons(seasons: list[int], data_dir: Path = DATA_DIR, download: bool = True) -> pd.DataFrame:
    """All matches of the given seasons, downloading files that are missing."""
    frames = []
    for season in seasons:
        path = data_dir / f"E0_{season}.csv"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path} is missing; run `python main.py fetch` first")
            download_season(season, data_dir)
        frames.append(parse_csv(path.read_text(encoding="latin-1"), season))
    return pd.concat(frames, ignore_index=True)
