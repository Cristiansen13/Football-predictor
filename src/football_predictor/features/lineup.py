"""Point-in-time squad strength from the confirmed starting XI and player ratings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..etl.entity_resolution import FuzzyMatcher, UnresolvedEntityError, player_resolver


def lineup_strength(
    lineups: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    unknown_quantile: float = 0.25,
    xi_size: int = 11,
) -> pd.DataFrame:
    """Lineup_Score and squad fraction for every (match, team) starting XI.

    ``lineups``: match_id, team_id, player_name, is_starter, kickoff.
    ``ratings``: player_name, team_id, snapshot_date, overall.

    Each XI is rated with the newest ratings snapshot published before the
    kickoff date. ``lineup_score`` is the mean overall of the starters,
    ``best_xi_score`` the mean of the squad's top ``xi_size`` players in that
    snapshot, and ``lineup_fraction`` their ratio: 1.0 is a full-strength side,
    lower values mean injured, suspended or rested key players. Starters who
    cannot be matched to a rated player get the squad's ``unknown_quantile``
    rating (typically academy players or new signings).
    """
    columns = ["match_id", "team_id", "lineup_score", "best_xi_score", "lineup_fraction", "matched_starters"]
    starters = lineups[lineups["is_starter"].astype(bool)]
    if starters.empty or ratings.empty:
        return pd.DataFrame(columns=columns)

    snapshots = pd.DatetimeIndex(sorted(pd.to_datetime(ratings["snapshot_date"]).unique()))
    by_squad = {key: group for key, group in ratings.groupby(["team_id", "snapshot_date"])}
    by_snapshot = {key: group for key, group in ratings.groupby("snapshot_date")}
    matchers: dict[tuple, FuzzyMatcher] = {}

    def lookup(name: str, squad: pd.DataFrame, league: pd.DataFrame, key: tuple) -> float | None:
        for scope, frame, threshold in ((key, squad, 0.8), (key[1:], league, 0.9)):
            if frame is None or frame.empty:
                continue
            if scope not in matchers:
                matchers[scope] = player_resolver(frame["player_name"].unique(), threshold=threshold)
            try:
                canonical = matchers[scope].resolve(name).canonical
            except UnresolvedEntityError:
                continue
            return float(frame.loc[frame["player_name"] == canonical, "overall"].max())
        return None

    rows = []
    for (match_id, team_id), xi in starters.groupby(["match_id", "team_id"]):
        kickoff_day = pd.Timestamp(xi["kickoff"].iloc[0]).normalize()
        position = snapshots.searchsorted(kickoff_day, side="left") - 1
        if position < 0:
            continue
        snapshot = snapshots[position]
        squad = by_squad.get((team_id, snapshot))
        league = by_snapshot.get(snapshot)
        if squad is None or squad.empty:
            continue
        top = np.sort(squad["overall"].to_numpy(dtype=float))[::-1][:xi_size]
        best_xi = top.mean()
        fallback = float(np.quantile(squad["overall"], unknown_quantile))
        found = [lookup(name, squad, league, (team_id, snapshot)) for name in xi["player_name"]]
        scores = [fallback if value is None else value for value in found]
        score = float(np.mean(scores))
        rows.append(
            {
                "match_id": match_id,
                "team_id": team_id,
                "lineup_score": score,
                "best_xi_score": best_xi,
                "lineup_fraction": score / best_xi,
                "matched_starters": sum(value is not None for value in found),
            }
        )
    return pd.DataFrame(rows, columns=columns)
