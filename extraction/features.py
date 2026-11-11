"""Step 3: build the modelling table.

Reads ``data/final/matches.csv`` and writes ``data/final/features.csv``: one
row per match with what was known *before kickoff* and the result to predict.
Every feature is computed only from matches played earlier, so a model trained
on this table never sees the match it predicts.

Features, each for the home side and the away side:

* ``elo``           Elo rating: starts at 1500, moves after every match by how
                    much the result beat the expectation. A promoted club
                    starts at the average rating of the clubs that went down.
* ``pts5``          points per game over the last 5 league matches
* ``gf5``, ``ga5``  goals scored and conceded per game over the last 5
* ``sot5``, ``sota5`` shots on target for and against per game over the last 5
* ``gf38``, ``ga38`` goals scored and conceded per game over the last 38 (a season)
* ``venue_pts5``    points per game over the last 5 matches at this venue
                    (home matches for the home side, away matches for the away side)
* ``rest``          days since the previous league match, capped at 14

plus ``elo_diff`` (home rating minus away rating).

    python -m extraction.features
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .preprocess import FINAL_DIR, MATCHES_FILE, load_matches

FEATURES_FILE = FINAL_DIR / "features.csv"

ELO_START = 1500.0
ELO_K = 20.0      # how fast a rating moves
ELO_HOME = 60.0   # rating points the home side is worth
SHORT, LONG = 5, 38
REST_CAP = 14

TEAM_FEATURES = ["elo", "pts5", "gf5", "ga5", "sot5", "sota5", "gf38", "ga38", "venue_pts5", "rest"]
FEATURE_COLUMNS = [f"{side}_{name}" for side in ("home", "away") for name in TEAM_FEATURES] + ["elo_diff"]
ID_COLUMNS = ["match_id", "season", "date", "home", "away"]


def elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """Each side's Elo rating before every match (columns ``home_elo`` and ``away_elo``)."""
    ratings: dict[str, float] = {}
    previous_clubs: set[str] = set()
    home_elo, away_elo = [], []
    for season, block in matches.groupby("season", sort=True):
        clubs = set(block["home"]) | set(block["away"])
        relegated = [ratings[c] for c in previous_clubs - clubs]
        newcomer = float(np.mean(relegated)) if relegated else ELO_START
        for club in clubs:
            ratings.setdefault(club, newcomer)
        for row in block.itertuples(index=False):
            home, away = ratings[row.home], ratings[row.away]
            home_elo.append(home)
            away_elo.append(away)
            expected = 1.0 / (1.0 + 10 ** (-(home + ELO_HOME - away) / 400.0))
            actual = 1.0 if row.home_goals > row.away_goals else (0.5 if row.home_goals == row.away_goals else 0.0)
            ratings[row.home] = home + ELO_K * (actual - expected)
            ratings[row.away] = away - ELO_K * (actual - expected)
        previous_clubs = clubs
    return pd.DataFrame({"home_elo": home_elo, "away_elo": away_elo}, index=matches.index)


def team_history(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per club per match (two per match), in date order."""
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        part = pd.DataFrame({
            "match_id": matches["match_id"],
            "date": matches["date"],
            "team": matches[side],
            "venue": side,
            "gf": matches[f"{side}_goals"],
            "ga": matches[f"{other}_goals"],
            "sot": matches[f"{side}_shots_target"].astype(float),
            "sota": matches[f"{other}_shots_target"].astype(float),
        })
        part["pts"] = np.where(part["gf"] > part["ga"], 3, np.where(part["gf"] == part["ga"], 1, 0))
        sides.append(part)
    return pd.concat(sides, ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)


def rolling_before(values: pd.Series, groups: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Mean of the previous ``window`` values within each group, the current row excluded."""
    return values.groupby(groups).transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """The modelling table: ids, pre-match features, and the result (``result``, ``outcome`` 0/1/2)."""
    matches = matches.sort_values(["date", "home"]).reset_index(drop=True)
    history = team_history(matches)
    team = history["team"]
    history["pts5"] = rolling_before(history["pts"], team, SHORT, 3)
    history["gf5"] = rolling_before(history["gf"], team, SHORT, 3)
    history["ga5"] = rolling_before(history["ga"], team, SHORT, 3)
    history["sot5"] = rolling_before(history["sot"], team, SHORT, 3)
    history["sota5"] = rolling_before(history["sota"], team, SHORT, 3)
    history["gf38"] = rolling_before(history["gf"], team, LONG, 10)
    history["ga38"] = rolling_before(history["ga"], team, LONG, 10)
    at_venue = history["team"] + "|" + history["venue"]
    history["venue_pts5"] = rolling_before(history["pts"], at_venue, SHORT, 3)
    history["rest"] = history.groupby("team")["date"].diff().dt.days.clip(upper=REST_CAP)

    table = matches[ID_COLUMNS].copy()
    elo = elo_ratings(matches)
    for side in ("home", "away"):
        rows = history[history["venue"] == side].set_index("match_id")
        table[f"{side}_elo"] = elo[f"{side}_elo"].to_numpy()
        for name in TEAM_FEATURES[1:]:
            table[f"{side}_{name}"] = table["match_id"].map(rows[name])
    table["elo_diff"] = table["home_elo"] - table["away_elo"]
    table = table[ID_COLUMNS + FEATURE_COLUMNS]
    table["result"] = matches["result"].to_numpy()
    table["outcome"] = table["result"].map({"H": 0, "D": 1, "A": 2})
    return table


def load_features(path=FEATURES_FILE) -> pd.DataFrame:
    """The modelling table written by this script."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m extraction.features` first")
    return pd.read_csv(path, parse_dates=["date"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    features = build_features(load_matches(MATCHES_FILE))
    features.to_csv(FEATURES_FILE, index=False, date_format="%Y-%m-%d", float_format="%.4f")
    complete = features[FEATURE_COLUMNS].notna().all(axis=1).sum()
    print(f"{FEATURES_FILE}: {len(features)} matches, {len(FEATURE_COLUMNS)} features; "
          f"{complete} rows have every feature (the rest are a club's first matches in the data)")


if __name__ == "__main__":
    main()
