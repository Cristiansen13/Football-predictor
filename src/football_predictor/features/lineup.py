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


def player_form_strength(
    lineups: pd.DataFrame,
    *,
    halflife_days: float = 365.0,
    prior_minutes: float = 600.0,
    squad_window_days: float = 365.0,
    xi_size: int = 11,
) -> pd.DataFrame:
    """Lineup strength from the starters' own production when no ratings are available.

    Each player's value is the expected xG + xA per 90 minutes from every
    earlier match in ``lineups`` (any club), with calendar-time decay and
    shrinkage towards the league rate over the first ``prior_minutes``.
    ``lineup_score`` is the sum of the eleven starters' values, so it reads as
    the chance creation per 90 of the side that actually started.
    ``best_xi_score`` is the same sum for the club's best eleven among players
    who appeared for it in the last ``squad_window_days``, and
    ``lineup_fraction`` is their ratio: 1 means the strongest available side.
    Only matches strictly before the kickoff feed a match's values, so the
    output is point-in-time; the first match of a club is still rated (every
    starter at the league rate) and ``matched_starters`` counts the starters
    with any history.

    ``lineups``: match_id, team_id, player_name, is_starter, minutes, xg, xa, kickoff.
    """
    columns = ["match_id", "team_id", "lineup_score", "best_xi_score", "lineup_fraction", "matched_starters"]
    needed = {"xg", "xa", "minutes", "kickoff"}
    if lineups.empty or not needed <= set(lineups.columns):
        return pd.DataFrame(columns=columns)
    frame = lineups.assign(kickoff=pd.to_datetime(lineups["kickoff"])).sort_values("kickoff")
    frame = frame.dropna(subset=["kickoff"])
    frame["contrib"] = frame["xg"].fillna(0.0) + frame["xa"].fillna(0.0)
    frame["minutes"] = frame["minutes"].fillna(0.0).astype(float)
    decay = np.log(2) / (halflife_days * 86400.0)
    prior_90s = prior_minutes / 90.0

    # Running decayed totals per player: (contribution, minutes, last update time).
    totals: dict[str, list[float]] = {}
    appearances: dict[str, dict[str, float]] = {}  # team -> player -> last appearance (epoch s)
    league = [0.0, 0.0, 0.0]  # decayed contribution, minutes, time

    def aged(entry: list[float], now: float) -> tuple[float, float]:
        factor = np.exp(-decay * (now - entry[2]))
        return entry[0] * factor, entry[1] * factor

    def value(player: str, now: float, league_rate: float) -> tuple[float, bool]:
        entry = totals.get(player)
        if entry is None:
            return league_rate, False
        contrib, minutes = aged(entry, now)
        nineties = minutes / 90.0
        return (contrib + prior_90s * league_rate) / (nineties + prior_90s), minutes > 0

    rows = []
    for kickoff, block in frame.groupby("kickoff", sort=True):
        now = kickoff.timestamp()
        league_contrib, league_minutes = aged(league, now)
        league_rate = league_contrib / (league_minutes / 90.0) if league_minutes > 0 else 0.14
        for (match_id, team_id), team_rows in block.groupby(["match_id", "team_id"]):
            starters = team_rows[team_rows["is_starter"].astype(bool)]["player_name"].tolist()
            if not starters:
                continue
            values = [value(name, now, league_rate) for name in starters]
            score = float(sum(v for v, _ in values))
            recent = {
                name for name, last in appearances.get(team_id, {}).items()
                if now - last <= squad_window_days * 86400.0
            } | set(starters)
            best = sorted((value(name, now, league_rate)[0] for name in recent), reverse=True)[:xi_size]
            best_xi = float(sum(best)) if len(best) >= min(xi_size, len(starters)) else score
            rows.append(
                {
                    "match_id": match_id,
                    "team_id": team_id,
                    "lineup_score": score,
                    "best_xi_score": best_xi,
                    "lineup_fraction": score / best_xi if best_xi > 0 else 1.0,
                    "matched_starters": int(sum(known for _, known in values)),
                }
            )
        # Update the running totals only after every side at this kickoff is rated.
        for row in block.itertuples(index=False):
            entry = totals.setdefault(row.player_name, [0.0, 0.0, now])
            contrib, minutes = aged(entry, now)
            entry[0], entry[1], entry[2] = contrib + row.contrib, minutes + row.minutes, now
            appearances.setdefault(row.team_id, {})[row.player_name] = now
        contrib, minutes = aged(league, now)
        league[0], league[1], league[2] = contrib + block["contrib"].sum(), minutes + block["minutes"].sum(), now
    return pd.DataFrame(rows, columns=columns)
