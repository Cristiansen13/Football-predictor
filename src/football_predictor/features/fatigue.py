"""Schedule density and fatigue."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fatigue_penalty(rest_hours: np.ndarray, previous_away: np.ndarray, *, threshold_hours: float = 72.0,
                    away_multiplier: float = 1.5) -> np.ndarray:
    """0 with at least ``threshold_hours`` of rest, rising linearly to 1 at zero rest.

    The penalty is scaled by ``away_multiplier`` when the previous fixture was
    away from home (travel), so it can exceed 1.
    """
    rest_hours = np.asarray(rest_hours, dtype=float)
    base = np.clip((threshold_hours - rest_hours) / threshold_hours, 0.0, 1.0)
    base = np.nan_to_num(base, nan=0.0)
    return base * np.where(np.asarray(previous_away, dtype=bool), away_multiplier, 1.0)


def schedule_fatigue(
    long: pd.DataFrame,
    schedule: pd.DataFrame | None = None,
    *,
    threshold_hours: float = 72.0,
    away_multiplier: float = 1.5,
    max_rest_days: float = 14.0,
    same_fixture_hours: float = 20.0,
) -> pd.DataFrame:
    """Rest days, fatigue index (1 / rest days) and fatigue penalty at each kickoff.

    ``schedule`` should list every fixture of each team in all competitions
    (``team``, ``kickoff``, ``venue``), because a Thursday Europa League away
    trip matters for Sunday's league game. League fixtures in ``long`` are
    always included. Rest is capped at ``max_rest_days`` so the first match
    after the summer break does not dominate.

    Sources disagree on kickoff times (Football-Data had no times before
    2019, rescheduled games keep their old slot elsewhere), so the same match
    can appear twice a few hours apart. A fixture less than
    ``same_fixture_hours`` before kickoff is treated as that same match rather
    than as the previous one; no club plays twice within 20 hours.
    """
    league = long[["team", "kickoff", "is_home"]].assign(
        venue=np.where(long["is_home"], "home", "away")
    )[["team", "kickoff", "venue"]]
    fixtures = [league]
    if schedule is not None and not schedule.empty:
        fixtures.append(schedule[["team", "kickoff", "venue"]])
    calendar = (
        pd.concat(fixtures, ignore_index=True)
        .assign(kickoff=lambda f: pd.to_datetime(f["kickoff"]).astype("datetime64[ns]"))
        .drop_duplicates(["team", "kickoff"])
        .sort_values("kickoff")
        .rename(columns={"kickoff": "previous_kickoff", "venue": "previous_venue"})
    )
    calendar["_key"] = calendar["previous_kickoff"]

    rows = long[["team", "kickoff"]].assign(
        kickoff=lambda f: pd.to_datetime(f["kickoff"]).astype("datetime64[ns]"),
        _row=np.arange(len(long)),
    )
    rows["_key"] = rows["kickoff"] - pd.Timedelta(hours=same_fixture_hours)
    merged = (
        pd.merge_asof(
            rows.sort_values("_key"),
            calendar,
            on="_key",
            by="team",
            direction="backward",  # the previous fixture, never the current one
        )
        .sort_values("_row")
        .reset_index(drop=True)
    )

    rest_hours = (merged["kickoff"] - merged["previous_kickoff"]).dt.total_seconds() / 3600.0
    rest_days = np.minimum(rest_hours.fillna(max_rest_days * 24).to_numpy() / 24.0, max_rest_days)
    rest_days = np.maximum(rest_days, 0.5)
    previous_away = (merged["previous_venue"].astype(str).str.lower() == "away").to_numpy()
    return pd.DataFrame(
        {
            "rest_days": rest_days,
            "fatigue_index": 1.0 / rest_days,
            "fatigue_penalty": fatigue_penalty(
                rest_hours.to_numpy(),
                previous_away,
                threshold_hours=threshold_hours,
                away_multiplier=away_multiplier,
            ),
        },
        index=long.index,
    )
