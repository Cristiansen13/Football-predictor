"""Step 1: download the raw data.

Football-Data.co.uk publishes one CSV file per league and season. This script
fetches the Premier League files (league code E0) and saves them untouched in
``data/raw``. Each file has one row per match: date, teams, full-time and
half-time score, referee, match statistics (shots, shots on target, corners,
fouls, cards) and the pre-match odds of several bookmakers.

    python -m extraction.download --seasons 2014-2025

A season is named by the year it starts in: 2023 is the 2023/24 season.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"  # E0 = Premier League
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_SEASONS = "2014-2025"
HEADERS = {"User-Agent": "premier-league-prediction (university course project)"}
PAUSE_SECONDS = 1.0  # between two requests, to be polite to the site


def season_code(season: int) -> str:
    """2023 -> '2324' (the 2023/24 season), the way the site names its folders."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def parse_seasons(text: str) -> list[int]:
    """'2014-2025' -> [2014, ..., 2025]; '2023' -> [2023]."""
    first, last = (int(x) for x in text.split("-")) if "-" in text else (int(text), int(text))
    return list(range(first, last + 1))


def raw_path(season: int, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"E0_{season}.csv"


def download_season(season: int, raw_dir: Path = RAW_DIR, *, force: bool = False) -> Path:
    """Fetch one season's CSV into ``raw_dir``. A file already there is kept unless ``force``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_path(season, raw_dir)
    if path.exists() and not force:
        return path
    response = requests.get(URL.format(code=season_code(season)), headers=HEADERS, timeout=30)
    response.raise_for_status()
    if b"HomeTeam" not in response.content[:2000]:
        raise ValueError(f"season {season}: the site did not return a match file")
    path.write_bytes(response.content)
    time.sleep(PAUSE_SECONDS)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", default=DEFAULT_SEASONS, help="season start years, e.g. 2014-2025")
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = parser.parse_args(argv)
    for season in parse_seasons(args.seasons):
        existed = raw_path(season).exists() and not args.force
        path = download_season(season, force=args.force)
        size_kb = path.stat().st_size / 1024
        print(f"{season}/{(season + 1) % 100:02d}  {path.name}  {size_kb:6.0f} KB  {'already there' if existed else 'downloaded'}")


if __name__ == "__main__":
    main()
