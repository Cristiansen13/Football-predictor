"""Assemble the point-in-time feature table used by the model.

Every feature for a fixture is computed from information available before its
kickoff: EWMAs use earlier matches only, rest is measured to the previous
fixture, and lineup ratings come from the last snapshot before match day.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .ewma import time_decayed_ewma
from .fatigue import schedule_fatigue

# Metrics with home_/away_ columns in the match dataset; each yields
# <metric>_for and <metric>_against from a team's point of view.
TEAM_METRICS = ("goals", "xg", "npxg", "xa", "shots", "shots_on_target", "possession", "ppda", "deep")


def to_team_long(matches: pd.DataFrame, metrics: Sequence[str] = TEAM_METRICS) -> pd.DataFrame:
    """One row per (match, team) seen from that team's side."""
    played = matches["home_goals"].notna() & matches["away_goals"].notna()
    sides = []
    for side, other, is_home in (("home", "away", True), ("away", "home", False)):
        frame = pd.DataFrame(
            {
                "match_id": matches["match_id"].to_numpy(),
                "kickoff": pd.to_datetime(matches["kickoff"]).to_numpy(),
                "team": matches[f"{side}_team"].to_numpy(),
                "opponent": matches[f"{other}_team"].to_numpy(),
                "is_home": is_home,
                "played": played.to_numpy(),
            }
        )
        for metric in metrics:
            if f"{side}_{metric}" in matches:
                frame[f"{metric}_for"] = pd.to_numeric(matches[f"{side}_{metric}"], errors="coerce").to_numpy(
                    dtype=float
                )
                frame[f"{metric}_against"] = pd.to_numeric(
                    matches[f"{other}_{metric}"], errors="coerce"
                ).to_numpy(dtype=float)
        sides.append(frame)
    return pd.concat(sides, ignore_index=True)


def build_features(
    matches: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    halflife_days: float = 45.0,
    prior_weight: float = 2.0,
    threshold_hours: float = 72.0,
    away_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Return ``matches`` with ``home_<feature>`` / ``away_<feature>`` columns added.

    ``schedule``: all-competition fixtures (team, kickoff, venue) for fatigue.
    ``lineups``: output of :func:`lineup_strength` keyed by (match_id, team).
    """
    matches = matches.copy()
    if "match_id" not in matches:
        matches["match_id"] = np.arange(len(matches)).astype(str)
    matches["kickoff"] = pd.to_datetime(matches["kickoff"])
    long = to_team_long(matches)

    value_columns = [c for c in long.columns if c.endswith(("_for", "_against"))]
    features = [
        time_decayed_ewma(long, value_columns, halflife_days=halflife_days, prior_weight=prior_weight),
        schedule_fatigue(long, schedule, threshold_hours=threshold_hours, away_multiplier=away_multiplier),
    ]
    long = pd.concat([long] + features, axis=1)
    if lineups is not None and not lineups.empty:
        long = long.merge(
            lineups[["match_id", "team", "lineup_score", "lineup_fraction"]], on=["match_id", "team"], how="left"
        )
    else:
        long["lineup_score"] = np.nan
        long["lineup_fraction"] = np.nan

    feature_columns = [c for c in long.columns if c.startswith("ewma_")] + [
        "rest_days",
        "fatigue_index",
        "fatigue_penalty",
        "lineup_score",
        "lineup_fraction",
    ]
    for side, is_home in (("home", True), ("away", False)):
        block = long.loc[long["is_home"] == is_home, ["match_id"] + feature_columns]
        block = block.rename(columns={c: f"{side}_{c}" for c in feature_columns})
        matches = matches.merge(block, on="match_id", how="left")
    return matches
