"""Time-decayed exponentially weighted moving averages, strictly backward-looking."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def league_prior(long: pd.DataFrame, column: str) -> np.ndarray:
    """League-wide mean of ``column`` over all rows played strictly before each row's kickoff."""
    played = long[long["played"] & long[column].notna()].sort_values("kickoff")
    times = played["kickoff"].to_numpy()
    if len(times) == 0:
        return np.full(len(long), np.nan)
    cumulative = np.cumsum(played[column].to_numpy(dtype=float))
    before = np.searchsorted(times, long["kickoff"].to_numpy(), side="left")
    prior = np.full(len(long), np.nan)
    has_history = before > 0
    prior[has_history] = cumulative[before[has_history] - 1] / before[has_history]
    return prior


def time_decayed_ewma(
    long: pd.DataFrame,
    columns: Sequence[str],
    *,
    halflife_days: float = 45.0,
    prior_weight: float = 2.0,
    prefix: str = "ewma_",
) -> pd.DataFrame:
    """EWMA of each column per team, evaluated at kickoff using only earlier matches.

    A match's weight halves every ``halflife_days`` of calendar time, so a game
    four days ago counts far more than one a month ago, and the summer break
    naturally discounts the previous season. The average is shrunk towards the
    league mean with ``prior_weight`` pseudo-matches, which handles the first
    games of a team without leaking anything from the future.

    ``long`` has one row per (match, team) with ``team``, ``kickoff`` and a
    boolean ``played`` column. Unplayed fixtures receive features but never
    update the state. Returns a frame aligned to ``long.index``.
    """
    decay_rate = np.log(2.0) / (halflife_days * 86_400.0)
    out = pd.DataFrame(index=long.index)
    priors = {column: league_prior(long, column) for column in columns}
    position = pd.Series(np.arange(len(long)), index=long.index)

    for column in columns:
        result = np.full(len(long), np.nan)
        prior = priors[column]
        for _, group in long.sort_values("kickoff").groupby("team", sort=False):
            values = group[column].to_numpy(dtype=float)
            played = group["played"].to_numpy(dtype=bool)
            seconds = group["kickoff"].to_numpy().astype("datetime64[s]").astype(np.int64)
            rows = position.loc[group.index].to_numpy()
            weighted_sum = weight = 0.0
            # Results at the current timestamp are held back until every row at
            # that timestamp has its feature, so duplicated or mistimed rows
            # never see each other.
            pending_sum = pending_weight = 0.0
            last = None
            for i in range(len(group)):
                if last is not None and seconds[i] != last:
                    factor = np.exp(-decay_rate * (seconds[i] - last))
                    weighted_sum = (weighted_sum + pending_sum) * factor
                    weight = (weight + pending_weight) * factor
                    pending_sum = pending_weight = 0.0
                last = seconds[i]
                p = prior[rows[i]]
                if np.isnan(p):
                    result[rows[i]] = weighted_sum / weight if weight > 0 else np.nan
                else:
                    result[rows[i]] = (weighted_sum + prior_weight * p) / (weight + prior_weight)
                if played[i] and not np.isnan(values[i]):
                    pending_sum += values[i]
                    pending_weight += 1.0
        out[f"{prefix}{column}"] = result
    return out
