"""Football-Data.co.uk: results, basic match stats and bookmaker odds (direct CSV).

Division codes are Football-Data's (:data:`DIVISIONS`): ``E0`` Premier League,
``E1`` Championship, ``D1`` Bundesliga and so on; every season file has the
same layout. ``fixtures.csv`` lists the coming week's fixtures of every
division with current prices.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from ..config import season_code, season_from_date
from .http import HttpClient

# Football-Data's main-league files, all in the same layout.
DIVISIONS = {
    "E0": "Premier League", "E1": "Championship", "E2": "League One", "E3": "League Two", "EC": "National League",
    "SC0": "Scottish Premiership", "SC1": "Scottish Championship",
    "D1": "Bundesliga", "D2": "2. Bundesliga", "SP1": "La Liga", "SP2": "Segunda Division",
    "I1": "Serie A", "I2": "Serie B", "F1": "Ligue 1", "F2": "Ligue 2", "N1": "Eredivisie",
    "B1": "Belgian Pro League", "P1": "Primeira Liga", "T1": "Super Lig", "G1": "Super League Greece",
}
# Divisions whose club names are resolved against the English register; the
# others register their Football-Data spelling as is (see Pipeline).
ENGLISH_DIVISIONS = frozenset({"E0", "E1", "E2", "E3", "EC"})
CSV_URL = "https://www.football-data.co.uk/mmz4281/{code}/{division}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
ODDS_FIELDS = ["season", "home_team", "away_team", "bookmaker", "market_time", "home", "draw", "away"]

# Football-Data's column prefix for each bookmaker (its notes.txt). A 1X2 price
# is ``<prefix>H/D/A`` before closing and ``<prefix>CH/CD/CA`` at closing; the
# other markets follow the same rule (``P>2.5`` / ``PC>2.5``, ``PAHH`` /
# ``PCAHH``). Closing prices exist from 2019/20, Pinnacle's 1X2 closing price
# from 2012/13. Pinnacle is ``PS`` on 1X2 and ``P`` elsewhere; VC Bet became
# BetVictor (``BV``) and the two codes never share a file. ``Max`` and ``Avg``
# are the market maximum and average over every bookmaker Football-Data
# follows, listed by name or not; before 2019/20 they are Betbrain's
# ``BbMx`` / ``BbAv``, pre-closing only. Which bookmakers a file names changes
# from season to season (six in 2019/20, ten in 2025/26, Pinnacle dropped in
# 2026/27), so columns that are absent are simply skipped. Order matters for
# duplicates: the first prefix of a bookmaker wins.
BOOKMAKERS = {
    "B365": "bet365",
    "PS": "pinnacle",
    "P": "pinnacle",
    "BW": "bwin",
    "IW": "interwetten",
    "WH": "william_hill",
    "VC": "betvictor",
    "BV": "betvictor",
    "LB": "ladbrokes",
    "SJ": "stan_james",
    "GB": "gamebookers",
    "BS": "blue_square",
    "SB": "sportingbet",
    "SO": "sporting_odds",
    "SY": "stanleybet",
    "BF": "betfair",
    "BFE": "betfair_exchange",
    "1XB": "1xbet",
    "BFD": "betfred",
    "BMGM": "betmgm",
    "CL": "coral",
    "PP": "paddy_power",
    "SKB": "skybet",  # not in Football-Data's notes yet; Sky Bet by its code
    "Max": "max",
    "Avg": "average",
    "BbMx": "max",
    "BbAv": "average",
}
# Bookmakers that are not a single counterparty: the market maximum (no one
# account can take every best price) and the market average.
AGGREGATES = ("max", "average")


def _is_legacy(prefix: str) -> bool:
    return prefix.startswith("Bb")


def odds_specs() -> list[tuple[str, str, tuple[str, str, str]]]:
    """(bookmaker, market time, (home, draw, away) columns) for every 1X2 price."""
    specs = []
    for prefix, bookmaker in BOOKMAKERS.items():
        specs.append((bookmaker, "opening", (f"{prefix}H", f"{prefix}D", f"{prefix}A")))
        if not _is_legacy(prefix):
            specs.append((bookmaker, "closing", (f"{prefix}CH", f"{prefix}CD", f"{prefix}CA")))
    return specs


# Over/under 2.5 goals. Before 2019/20 the files carry only Betbrain's market
# maximum and average, pre-closing only.
TOTALS_LINE = 2.5
TOTALS_FIELDS = ["season", "home_team", "away_team", "bookmaker", "market_time", "line", "over", "under"]


def totals_specs() -> list[tuple[str, str, tuple[str, str]]]:
    """(bookmaker, market time, (over, under) columns) for every over/under 2.5 price."""
    specs = []
    for prefix, bookmaker in BOOKMAKERS.items():
        specs.append((bookmaker, "opening", (f"{prefix}>2.5", f"{prefix}<2.5")))
        if not _is_legacy(prefix):
            specs.append((bookmaker, "closing", (f"{prefix}C>2.5", f"{prefix}C<2.5")))
    return specs


# Asian handicap. The line is the home side's handicap (-0.5: home gives half a
# goal); every bookmaker's price in a file refers to the same line (``AHh``
# before closing, ``AHCh`` at closing; Betbrain's ``BbAHh`` before 2019/20).
HANDICAP_FIELDS = ["season", "home_team", "away_team", "bookmaker", "market_time", "line", "home", "away"]


def handicap_specs() -> list[tuple[str, str, tuple[str, str, str]]]:
    """(bookmaker, market time, (line, home, away) columns) for every handicap price."""
    specs = []
    for prefix, bookmaker in BOOKMAKERS.items():
        line = "BbAHh" if _is_legacy(prefix) else "AHh"
        specs.append((bookmaker, "opening", (line, f"{prefix}AHH", f"{prefix}AHA")))
        if not _is_legacy(prefix):
            specs.append((bookmaker, "closing", ("AHCh", f"{prefix}CAHH", f"{prefix}CAHA")))
    return specs


STAT_COLUMNS = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
}


def fetch_season_csv(client: HttpClient, season: int, division: str = "E0", **kwargs) -> str:
    url = CSV_URL.format(code=season_code(season), division=division)
    return client.get_text(url, encoding="latin-1", **kwargs)


def fetch_fixtures_csv(client: HttpClient) -> str:
    return client.get_text(FIXTURES_URL, encoding="latin-1", use_cache=False)


def _read_csv(csv_text: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(csv_text.lstrip("﻿")), on_bad_lines="skip")
    frame.columns = frame.columns.astype(str).str.strip().str.lstrip("﻿ï»¿")
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    return frame.dropna(subset=["HomeTeam", "AwayTeam"])


def _kickoff(frame: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(frame["Date"], dayfirst=True, format="mixed")
    if "Time" in frame:
        time = pd.to_timedelta(frame["Time"].fillna("00:00").astype(str) + ":00", errors="coerce")
        date = date + time.fillna(pd.Timedelta(0))
    return date


def _prices_long(
    raw: pd.DataFrame,
    matches: pd.DataFrame,
    specs,
    names: list[str],
    fields: list[str],
    constants: dict | None = None,
) -> pd.DataFrame:
    """Stack the price columns of every (bookmaker, market time) present in ``raw``."""
    constants = constants or {}
    frames = []
    for bookmaker, market_time, columns in specs:
        if not all(c in raw for c in columns):
            continue
        block = raw[list(columns)].apply(pd.to_numeric, errors="coerce")
        block.columns = names
        # A decimal price is above 1 by definition; the files hold a handful of
        # zeros and ones (no price), which would read as a certainty.
        price_names = [n for n in names if n != "line"]
        block[price_names] = block[price_names].where(block[price_names] > 1.0)
        for column, value in constants.items():
            block.insert(0, column, value)
        block.insert(0, "market_time", market_time)
        block.insert(0, "bookmaker", bookmaker)
        block.insert(0, "away_team", matches["away_team"].to_numpy())
        block.insert(0, "home_team", matches["home_team"].to_numpy())
        block.insert(0, "season", matches["season"].to_numpy())
        frames.append(block.dropna(subset=names))
    if not frames:
        return pd.DataFrame(columns=fields)
    long = pd.concat(frames, ignore_index=True)
    keys = ["season", "home_team", "away_team", "bookmaker", "market_time"] + [c for c in ("line",) if c in constants]
    return long.drop_duplicates(keys).reset_index(drop=True)


def _odds_long(raw: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    return _prices_long(raw, matches, odds_specs(), ["home", "draw", "away"], ODDS_FIELDS)


def _totals_long(raw: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    return _prices_long(raw, matches, totals_specs(), ["over", "under"], TOTALS_FIELDS, {"line": TOTALS_LINE})


def _handicaps_long(raw: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    return _prices_long(raw, matches, handicap_specs(), ["line", "home", "away"], HANDICAP_FIELDS)


def parse_season_handicaps(csv_text: str, season: int, division: str = "E0") -> pd.DataFrame:
    """Asian handicap lines and prices of one season file, long format."""
    matches, _ = parse_season(csv_text, season, division)
    return _handicaps_long(_read_csv(csv_text).reset_index(drop=True), matches)


def parse_fixtures_handicaps(csv_text: str, divisions=("E0", "E1")) -> pd.DataFrame:
    """Current Asian handicap lines and prices for the coming week's fixtures."""
    raw = _read_csv(csv_text)
    raw = raw[raw["Div"].astype(str).str.strip().isin(divisions)].reset_index(drop=True)
    matches, _ = parse_fixtures(csv_text, divisions)
    handicaps = _handicaps_long(raw, matches)
    return handicaps[handicaps["market_time"] == "opening"].reset_index(drop=True)


def parse_season_totals(csv_text: str, season: int, division: str = "E0") -> pd.DataFrame:
    """Over/under 2.5 prices of one season file, long format like :func:`parse_season`'s odds."""
    matches, _ = parse_season(csv_text, season, division)
    return _totals_long(_read_csv(csv_text).reset_index(drop=True), matches)


def parse_fixtures_totals(csv_text: str, divisions=("E0", "E1")) -> pd.DataFrame:
    """Current over/under 2.5 prices for the coming week's fixtures."""
    raw = _read_csv(csv_text)
    raw = raw[raw["Div"].astype(str).str.strip().isin(divisions)].reset_index(drop=True)
    matches, _ = parse_fixtures(csv_text, divisions)
    totals = _totals_long(raw, matches)
    return totals[totals["market_time"] == "opening"].reset_index(drop=True)


def parse_season(csv_text: str, season: int, division: str = "E0") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one season file into (matches, odds).

    ``matches`` has one row per fixture with goals and shot counts; ``odds`` is
    long format with one row per (fixture, bookmaker, market time).
    """
    raw = _read_csv(csv_text).reset_index(drop=True)
    matches = pd.DataFrame(
        {
            "season": season,
            "division": division,
            "kickoff": _kickoff(raw),
            "home_team": raw["HomeTeam"].str.strip(),
            "away_team": raw["AwayTeam"].str.strip(),
            "home_goals": pd.to_numeric(raw["FTHG"], errors="coerce").astype("Int64"),
            "away_goals": pd.to_numeric(raw["FTAG"], errors="coerce").astype("Int64"),
        }
    )
    for column, name in STAT_COLUMNS.items():
        matches[name] = pd.to_numeric(raw[column], errors="coerce") if column in raw else np.nan
    return matches, _odds_long(raw, matches)


def parse_fixtures(csv_text: str, divisions=("E0", "E1")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Upcoming fixtures and their current (pre-closing) odds from ``fixtures.csv``."""
    raw = _read_csv(csv_text)
    raw = raw[raw["Div"].astype(str).str.strip().isin(divisions)].reset_index(drop=True)
    kickoff = _kickoff(raw)
    matches = pd.DataFrame(
        {
            "season": [season_from_date(k) for k in kickoff],
            "division": raw["Div"].str.strip(),
            "kickoff": kickoff,
            "home_team": raw["HomeTeam"].str.strip(),
            "away_team": raw["AwayTeam"].str.strip(),
            "home_goals": pd.array([pd.NA] * len(raw), dtype="Int64"),
            "away_goals": pd.array([pd.NA] * len(raw), dtype="Int64"),
        }
    )
    odds = _odds_long(raw, matches)
    return matches, odds[odds["market_time"] == "opening"].reset_index(drop=True)
