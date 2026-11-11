"""Step 2: clean the raw season files into one table of matches.

Reads every ``data/raw/E0_<season>.csv`` and writes ``data/final/matches.csv``,
one row per played match with readable column names:

* identity: match_id, season, date, home, away, referee
* result: home_goals, away_goals, result (H, D or A), half-time goals
* match statistics: shots, shots on target, corners, fouls, yellow and red cards
* Bet365 odds: opening 1X2, closing 1X2 (published from 2019/20), over/under 2.5 goals

Cleaning done here: blank and unplayed rows are dropped, the two date formats
the site has used (dd/mm/yy and dd/mm/yyyy) are parsed into real dates, goals
and counts become integers, and columns a season does not have are left empty.

    python -m extraction.preprocess
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd

from .download import DEFAULT_SEASONS, RAW_DIR, download_season, parse_seasons, raw_path

FINAL_DIR = Path(__file__).resolve().parent.parent / "data" / "final"
MATCHES_FILE = FINAL_DIR / "matches.csv"

# Column in the raw file -> column in our table.
COLUMNS = {
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HTHG": "home_goals_ht",
    "HTAG": "away_goals_ht",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
    "B365CH": "closing_home",
    "B365CD": "closing_draw",
    "B365CA": "closing_away",
    "B365>2.5": "odds_over",
    "B365<2.5": "odds_under",
}
COUNT_COLUMNS = [c for c in COLUMNS.values() if c.endswith(("_ht", "_shots", "_shots_target", "_corners", "_fouls",
                                                            "_yellow", "_red"))]


def parse_csv(text: str, season: int) -> pd.DataFrame:
    """Tidy table of one season: date, teams, goals, statistics, odds; unplayed rows dropped."""
    raw = pd.read_csv(io.StringIO(text.lstrip("﻿")), on_bad_lines="skip")
    raw.columns = [c.strip() for c in raw.columns]
    present = {src: dst for src, dst in COLUMNS.items() if src in raw}
    frame = raw[list(present)].rename(columns=present)
    frame = frame.dropna(subset=["home", "away", "home_goals", "away_goals"]).copy()
    frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, format="mixed")
    for column in ("home", "away"):
        frame[column] = frame[column].astype(str).str.strip()
    frame["home_goals"] = frame["home_goals"].astype(int)
    frame["away_goals"] = frame["away_goals"].astype(int)
    for column in COLUMNS.values():
        if column not in frame:
            frame[column] = float("nan")
    for column in COUNT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    goal_difference = frame["home_goals"] - frame["away_goals"]
    frame["result"] = goal_difference.map(lambda d: "H" if d > 0 else ("D" if d == 0 else "A"))
    frame.insert(0, "season", season)
    return frame.sort_values(["date", "home"]).reset_index(drop=True)


def build_matches(seasons: list[int], raw_dir: Path = RAW_DIR, *, download: bool = True) -> pd.DataFrame:
    """Every match of ``seasons`` in date order, with a ``match_id``; missing raw files are downloaded."""
    frames = []
    for season in seasons:
        path = raw_path(season, raw_dir)
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path} is missing; run `python -m extraction.download` first")
            download_season(season, raw_dir)
        frames.append(parse_csv(path.read_text(encoding="latin-1"), season))
    matches = pd.concat(frames, ignore_index=True).sort_values(["date", "home"]).reset_index(drop=True)
    matches.insert(0, "match_id", range(len(matches)))
    return matches


def load_matches(path: Path = MATCHES_FILE) -> pd.DataFrame:
    """The clean match table written by this script."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m extraction.preprocess` first")
    return pd.read_csv(path, parse_dates=["date"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", default=DEFAULT_SEASONS, help="season start years, e.g. 2014-2025")
    args = parser.parse_args(argv)
    matches = build_matches(parse_seasons(args.seasons))
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCHES_FILE, index=False, date_format="%Y-%m-%d")
    clubs = len(set(matches["home"]) | set(matches["away"]))
    print(f"{MATCHES_FILE}: {len(matches)} matches, {matches['season'].nunique()} seasons "
          f"({matches['date'].min():%d %b %Y} to {matches['date'].max():%d %b %Y}), {clubs} clubs, "
          f"{matches.shape[1]} columns")
    missing = matches.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("Missing values per column:")
        print(missing.to_string())


if __name__ == "__main__":
    main()
