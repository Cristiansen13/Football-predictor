"""Data-quality report: what each source delivered, season by season."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .db import Database

# Expected clubs per division (Football-Data codes).
DIVISION_SIZE = {"E0": 20, "E1": 24}
UNDERSTAT_FIRST_SEASON = 2014


@dataclass
class Issue:
    level: str  # "error" | "warning"
    division: str
    season: int | None
    message: str

    def __str__(self) -> str:
        where = f"{self.division} {self.season}" if self.season is not None else self.division
        return f"[{self.level}] {where}: {self.message}"


def coverage_report(db: Database) -> pd.DataFrame:
    """One row per (division, season): fixtures, results and the share of played
    matches that carry each kind of data."""
    data = db.load_match_dataset()
    if data.empty:
        return pd.DataFrame()
    lineup_matches = set(db.query("SELECT DISTINCT match_id FROM lineups")["match_id"])
    snapshots = db.query("SELECT snapshot_date, COUNT(*) AS players FROM player_ratings GROUP BY snapshot_date")
    snapshots["season"] = pd.to_datetime(snapshots["snapshot_date"]).map(
        lambda d: d.year if d.month >= 7 else d.year - 1
    )
    schedule = db.load_team_schedule()
    schedule_seasons = (
        pd.to_datetime(schedule["kickoff"]).map(lambda d: d.year if d.month >= 7 else d.year - 1)
        if not schedule.empty else pd.Series(dtype=int)
    )

    rows = []
    for (division, season), group in data.groupby(["division", "season"]):
        played = group[group["home_goals"].notna()]
        n = max(len(played), 1)
        teams = set(group["home_team"]) | set(group["away_team"])

        def share(*columns: str) -> float:
            return float(played[list(columns)].notna().all(axis=1).sum() / n)

        rows.append(
            {
                "division": division,
                "season": int(season),
                "teams": len(teams),
                "fixtures": len(group),
                "played": len(played),
                "xg": share("home_xg", "away_xg"),
                "ppda": share("home_ppda", "away_ppda"),
                "possession": share("home_possession", "away_possession"),
                "shots": share("home_shots", "away_shots"),
                "closing_odds": float((played["odds_market_time"] == "closing").sum() / n),
                "any_odds": share("odds_home"),
                "lineups": float(played["match_id"].isin(lineup_matches).sum() / n),
                "schedule_rows": int((schedule_seasons == season).sum()),
                "rating_snapshots": int((snapshots["season"] == season).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["division", "season"]).reset_index(drop=True)


def validate(report: pd.DataFrame, current_season: int | None = None) -> list[Issue]:
    """Flag incomplete seasons and missing sources that the model depends on."""
    issues: list[Issue] = []
    if report.empty:
        return [Issue("error", "-", None, "database is empty; run `football-predictor ingest`")]
    for row in report.itertuples(index=False):
        size = DIVISION_SIZE.get(row.division)
        finished = current_season is None or row.season < current_season
        if size is not None:
            expected = size * (size - 1)
            if row.teams != size:
                issues.append(Issue("error", row.division, row.season,
                                    f"{row.teams} clubs, expected {size} (name resolution problem?)"))
            if finished and row.played != expected:
                issues.append(Issue("error", row.division, row.season,
                                    f"{row.played} results, expected {expected}"))
        if row.division == "E0":
            if row.season >= UNDERSTAT_FIRST_SEASON and row.xg < 0.95:
                issues.append(Issue("warning", row.division, row.season,
                                    f"xG on {row.xg:.0%} of matches (Understat)"))
            if row.any_odds < 0.95:
                issues.append(Issue("warning", row.division, row.season,
                                    f"odds on {row.any_odds:.0%} of matches"))
            if row.season >= 2012 and row.closing_odds < 0.95:
                issues.append(Issue("warning", row.division, row.season,
                                    f"closing odds on {row.closing_odds:.0%} of matches"))
    divisions = set(report["division"])
    if "E1" not in divisions:
        issues.append(Issue("warning", "E1", None,
                            "no Championship data: promoted clubs will be rated by a fallback guess"))
    e0 = report[report["division"] == "E0"]
    if not e0.empty and (e0["lineups"] == 0).all():
        issues.append(Issue("warning", "E0", None, "no lineups loaded: lineup_fraction feature is unavailable"))
    if not e0.empty and (e0["rating_snapshots"] == 0).all():
        issues.append(Issue("warning", "E0", None, "no SOFIFA ratings loaded: lineup features are unavailable"))
    if not e0.empty and (e0["schedule_rows"] == 0).all():
        issues.append(Issue("warning", "E0", None,
                            "no FBref schedules: fatigue ignores cup and European fixtures"))
    return issues
