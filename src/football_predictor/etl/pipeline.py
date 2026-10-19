"""ETL orchestration: extract from each source, resolve entities, load into SQLite."""

from __future__ import annotations

import logging
from datetime import datetime
from functools import partial
from typing import Callable, Iterable

import pandas as pd

from ..db import Database, match_uuid
from . import fbref, football_data, sofifa, understat
from .entity_resolution import FuzzyMatcher, UnresolvedEntityError, team_resolver
from .http import HttpClient

log = logging.getLogger(__name__)

SOURCES = ("football_data", "understat", "fbref", "sofifa")


class Pipeline:
    def __init__(
        self,
        db: Database,
        client: HttpClient,
        resolver: FuzzyMatcher | None = None,
        *,
        current_season: int | None = None,
    ) -> None:
        self.db = db
        self.client = client
        self.resolver = resolver or team_resolver()
        # Pages for an unfinished season change every week, so never cache them.
        self.current_season = current_season

    def _cache(self, season: int) -> dict[str, bool]:
        return {"use_cache": season != self.current_season}

    def team_id(self, name: str, source: str) -> str:
        resolution = self.resolver.resolve(name)
        return self.db.register_team(resolution, source)

    def _with_team_ids(self, frame: pd.DataFrame, source: str, columns=("home_team", "away_team")) -> pd.DataFrame:
        frame = frame.copy()
        for column in columns:
            ids = {name: self.team_id(name, source) for name in frame[column].dropna().unique()}
            frame[f"{column}_id"] = frame[column].map(ids)
        return frame

    def _with_match_ids(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(
            match_id=[
                match_uuid(int(s), h, a)
                for s, h, a in zip(frame["season"], frame["home_team_id"], frame["away_team_id"])
            ]
        )

    # -- sources ------------------------------------------------------------

    def load_football_data(self, season: int) -> None:
        csv_text = football_data.fetch_season_csv(self.client, season, **self._cache(season))
        matches, odds = football_data.parse_season(csv_text, season)
        matches = self._with_match_ids(self._with_team_ids(matches, "football_data"))
        self.db.upsert_matches(matches, "football_data")

        stats = pd.concat(
            [
                pd.DataFrame(
                    {
                        "match_id": matches["match_id"],
                        "team_id": matches[f"{side}_team_id"],
                        "shots": matches[f"{side}_shots"],
                        "shots_on_target": matches[f"{side}_shots_on_target"],
                    }
                )
                for side in ("home", "away")
            ]
        )
        self.db.upsert_match_stats(stats)

        if not odds.empty:
            odds = self._with_match_ids(self._with_team_ids(odds, "football_data"))
            self.db.upsert_odds(odds[["match_id", "bookmaker", "market_time", "home", "draw", "away"]])
        log.info("football-data %s: %d matches, %d odds rows", season, len(matches), len(odds))

    def load_understat(self, season: int, *, lineups: bool = False) -> None:
        matches, history = understat.fetch_league(self.client, season, **self._cache(season))
        matches = self._with_match_ids(self._with_team_ids(matches, "understat"))
        self.db.upsert_matches(matches, "understat")

        history = self._with_team_ids(history, "understat", columns=("team", "home_team", "away_team"))
        history = self._with_match_ids(history.assign(season=season))
        advanced = history[
            ["match_id", "team_id", "xg", "xga", "npxg", "npxga", "ppda", "ppda_allowed",
             "deep", "deep_allowed", "xpts"]
        ]
        self.db.upsert_advanced_stats(advanced)
        log.info("understat %s: %d matches, %d team rows", season, len(matches), len(advanced))

        if lineups:
            self.load_understat_lineups(matches[matches["is_result"]])

    def load_understat_lineups(self, matches: pd.DataFrame) -> None:
        """Starting XIs (and team xA) from Understat match pages; one request per match."""
        done = set(self.db.query("SELECT DISTINCT match_id FROM lineups")["match_id"])
        for match in matches.itertuples(index=False):
            if match.match_id in done:
                continue
            rosters = understat.fetch_match_rosters(self.client, match.understat_id)
            if rosters.empty:
                continue
            team_ids = {"home": match.home_team_id, "away": match.away_team_id}
            rosters["team_id"] = rosters["venue"].map(team_ids)
            rosters["match_id"] = match.match_id
            self.db.upsert_lineups(
                rosters[["match_id", "team_id", "player_name", "position", "is_starter", "minutes"]]
                .assign(is_starter=lambda f: f["is_starter"].astype(int))
                .drop_duplicates(["team_id", "player_name"])
            )
            xa = rosters.groupby("team_id", as_index=False)["xa"].sum().assign(match_id=match.match_id)
            self.db.upsert_advanced_stats(xa[["match_id", "team_id", "xa"]])

    def load_fbref(self, season: int, *, team_logs: bool = True) -> None:
        schedule = fbref.fetch_schedule(self.client, season, **self._cache(season))
        schedule = self._with_match_ids(self._with_team_ids(schedule, "fbref"))
        self.db.upsert_matches(schedule, "fbref")
        log.info("fbref %s: %d fixtures", season, len(schedule))
        if not team_logs:
            return

        for squad in fbref.squads_from_schedule(schedule).itertuples(index=False):
            schedule_log, shooting_log = fbref.fetch_team_logs(
                self.client, squad.squad_id, squad.slug, squad.team, season, **self._cache(season)
            )
            team_id = self.team_id(squad.team, "fbref")
            self.db.upsert_team_schedule(
                schedule_log.assign(team_id=team_id)[["team_id", "kickoff", "competition", "venue"]]
                .drop_duplicates("kickoff")
            )
            league = fbref.combine_team_logs(schedule_log, shooting_log)
            league = league[league["venue"].isin(["home", "away"])]
            rows = []
            for row in league.itertuples(index=False):
                try:
                    opponent_id = self.team_id(row.opponent, "fbref")
                except UnresolvedEntityError:
                    log.warning("fbref: unresolved opponent %r", row.opponent)
                    continue
                home, away = (team_id, opponent_id) if row.venue == "home" else (opponent_id, team_id)
                rows.append(
                    {
                        "match_id": match_uuid(season, home, away),
                        "team_id": team_id,
                        "possession": row.possession,
                        "shots": row.shots,
                        "shots_on_target": row.shots_on_target,
                    }
                )
            known = self.db.existing_match_ids()
            stats = pd.DataFrame(rows)
            if not stats.empty:
                self.db.upsert_match_stats(stats[stats["match_id"].isin(known)])

    def load_sofifa(self, start: datetime, end: datetime, *, every_days: int = 28, **browser_kwargs) -> None:
        with sofifa.SofifaBrowser(**browser_kwargs) as browser:
            versions = sofifa.select_snapshots(browser.roster_versions(), start, end, every_days)
            for version in versions.itertuples(index=False):
                ratings = browser.league_ratings(version.roster_id)
                ratings = ratings.dropna(subset=["overall"])
                team_ids = {}
                for team in ratings["team"].dropna().unique():
                    try:
                        team_ids[team] = self.team_id(team, "sofifa")
                    except UnresolvedEntityError:
                        log.warning("sofifa: unresolved team %r", team)
                self.db.upsert_player_ratings(
                    pd.DataFrame(
                        {
                            "player_id": "sofifa:" + ratings["sofifa_id"],
                            "player_name": ratings["player_name"],
                            "team_id": ratings["team"].map(team_ids),
                            "snapshot_date": version.snapshot_date,
                            "overall": ratings["overall"].astype(int),
                            "positions": ratings["positions"],
                        }
                    )
                )
                log.info("sofifa roster %s: %d players", version.roster_id, len(ratings))

    def run(
        self,
        seasons: Iterable[int],
        sources: Iterable[str] = ("football_data", "understat"),
        *,
        lineups: bool = False,
        browser_kwargs: dict | None = None,
    ) -> list[tuple[str, int | None, str]]:
        """Load every (source, season); returns the failures as (source, season, error).

        A failing source (blocked by the site, page layout changed) is logged
        and skipped so the remaining sources still load.
        """
        seasons, sources = list(seasons), set(sources)
        unknown = sources - set(SOURCES)
        if unknown:
            raise ValueError(f"Unknown sources: {sorted(unknown)}")
        steps: list[tuple[str, int | None, Callable[[], None]]] = []
        for season in seasons:
            # Football-Data first: it is the most authoritative source of kickoffs.
            if "football_data" in sources:
                steps.append(("football_data", season, partial(self.load_football_data, season)))
            if "fbref" in sources:
                steps.append(("fbref", season, partial(self.load_fbref, season)))
            if "understat" in sources:
                steps.append(("understat", season, partial(self.load_understat, season, lineups=lineups)))
        if "sofifa" in sources:
            window = (datetime(min(seasons), 7, 1), datetime(max(seasons) + 1, 6, 30))
            steps.append(("sofifa", None, partial(self.load_sofifa, *window, **(browser_kwargs or {}))))

        failures = []
        for source, season, step in steps:
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - keep loading the other sources
                log.exception("%s %s failed", source, season or "")
                failures.append((source, season, f"{type(exc).__name__}: {exc}"))
        return failures
