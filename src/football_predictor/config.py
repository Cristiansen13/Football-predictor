"""Project-wide settings and season helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path(os.environ.get("FOOTBALL_PREDICTOR_DATA", "data"))

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = DEFAULT_DATA_DIR
    # Minimum seconds between two requests to the same host. FBref enforces
    # roughly 10 requests per minute and bans aggressive clients.
    request_delay: dict[str, float] = field(
        default_factory=lambda: {
            "fbref.com": 6.5,
            "understat.com": 2.0,
            "sofifa.com": 3.0,
            "www.football-data.co.uk": 1.0,
        }
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "football.db"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


def season_label(season: int) -> str:
    """2023 -> '2023-2024' (FBref style)."""
    return f"{season}-{season + 1}"


def season_code(season: int) -> str:
    """2023 -> '2324' (Football-Data.co.uk style)."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def season_from_date(date) -> int:
    """Season start year for a match date; seasons roll over on 1 July."""
    return date.year if date.month >= 7 else date.year - 1
