"""Point-in-time audit: prove features only use information known before kickoff.

For a sample of played matches the feature table is rebuilt from a copy of
the data truncated at that match's kickoff, with the match's own result and
statistics blanked. Any feature that differs from the full build must have
seen the future.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .builder import TEAM_METRICS, build_features

FEATURE_MARKERS = ("ewma_", "rest_days", "fatigue_", "lineup_")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c for c in frame.columns
        if c.startswith(("home_", "away_")) and any(marker in c for marker in FEATURE_MARKERS)
    ]


def _outcome_columns(frame: pd.DataFrame) -> list[str]:
    return [f"{side}_{m}" for side in ("home", "away") for m in TEAM_METRICS if f"{side}_{m}" in frame]


def audit_point_in_time(
    matches: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    sample: int = 20,
    seed: int = 0,
    builder: Callable[..., pd.DataFrame] = build_features,
    **builder_kwargs,
) -> pd.DataFrame:
    """Return one row per (match, feature) whose value depends on the future.

    An empty frame means the audit passed. ``lineups`` (match_id, team, ...)
    is truncated to matches up to and including the audited one, since a
    starting XI is public before kickoff.
    """
    matches = matches.assign(kickoff=pd.to_datetime(matches["kickoff"]))
    full = builder(matches, schedule=schedule, lineups=lineups, **builder_kwargs)
    columns = feature_columns(full)
    played = full[full["home_goals"].notna()]
    if played.empty:
        return pd.DataFrame(columns=["match_id", "kickoff", "feature", "full", "point_in_time"])
    picks = played.sample(min(sample, len(played)), random_state=seed)

    problems = []
    for row in picks.itertuples(index=False):
        kickoff, match_id = row.kickoff, row.match_id
        past = matches[(matches["kickoff"] < kickoff) | (matches["match_id"] == match_id)].copy()
        target = past["match_id"] == match_id
        past.loc[target, ["home_goals", "away_goals"]] = pd.NA
        past.loc[target, _outcome_columns(past)] = np.nan
        past_schedule = None
        if schedule is not None and not schedule.empty:
            past_schedule = schedule[pd.to_datetime(schedule["kickoff"]) < kickoff]
        past_lineups = None
        if lineups is not None and not lineups.empty:
            past_lineups = lineups[lineups["match_id"].isin(past["match_id"])]
        rebuilt = builder(past, schedule=past_schedule, lineups=past_lineups, **builder_kwargs)
        point_in_time = rebuilt.loc[rebuilt["match_id"] == match_id, columns].iloc[0]
        reference = full.loc[full["match_id"] == match_id, columns].iloc[0]
        for column in columns:
            a, b = reference[column], point_in_time[column]
            same = (pd.isna(a) and pd.isna(b)) or (
                not pd.isna(a) and not pd.isna(b) and np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12)
            )
            if not same:
                problems.append(
                    {"match_id": match_id, "kickoff": kickoff, "feature": column, "full": a, "point_in_time": b}
                )
    return pd.DataFrame(problems, columns=["match_id", "kickoff", "feature", "full", "point_in_time"])
