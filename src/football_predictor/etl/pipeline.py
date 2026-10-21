"""ETL orchestration: extract from each source, resolve entities, load into SQLite."""

from __future__ import annotations

import logging
from datetime import datetime
from functools import partial
from typing import Callable, Iterable

import pandas as pd

from ..config import season_from_date
from ..db import Database, match_uuid
from . import espn, fbref, football_data, sofifa, understat
from .entity_resolution import FuzzyMatcher, UnresolvedEntityError, team_resolver
from .http import HttpClient

log = logging.getLogger(__name__)

SOURCES = ("football_data", "understat", "espn", "fbref", "sofifa")
DIVISIONS = ("E0", "E1")
UNDERSTAT_FIRST_SEASON = 2014  # Understat's Premier League coverage starts in 2014/15


class Pipeline:
    def __init__(
        self,
        db: Database,
        client: HttpClient,
        resolver: FuzzyMatcher | None = None,
        *,
        current_season: int | None = None,
        divisions: Iterable[str] = DIVISIONS,
    ) -> None:
        self.db = db
        self.client = client
        self.resolver = resolver or team_resolver()
        # Pages for an unfinished season change every week, so never cache them.
        self.current_season = current_season
        self.divisions = tuple(divisions)
        self._created: set[str] = set()

    def _cache(self, season: int) -> dict[str, bool]:
        return {"use_cache": season != self.current_season}

    def team_id(self, name: str, source: str, *, create_missing: bool = False) -> str:
        resolution = self.resolver.resolve(name, create_missing=create_missing)
        if resolution.method == "created" and name not in self._created:
            self._created.add(name)
            log.info("%s: registered new club %r", source, name)
        return self.db.register_team(resolution, source)

    def _with_team_ids(
        self, frame: pd.DataFrame, source: str, columns=("home_team", "away_team"), create_missing: bool = False
    ) -> pd.DataFrame:
        frame = frame.copy()
        for column in columns:
            ids = {
                name: self.team_id(name, source, create_missing=create_missing)
                for name in frame[column].dropna().unique()
            }
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

    def _load_odds(self, odds: pd.DataFrame, create_missing: bool) -> None:
        if odds.empty:
            return
        odds = self._with_match_ids(self._with_team_ids(odds, "football_data", create_missing=create_missing))
        known = self.db.existing_match_ids()
        self.db.upsert_odds(
            odds.loc[odds["match_id"].isin(known), ["match_id", "bookmaker", "market_time", "home", "draw", "away"]]
        )

    def _load_totals(self, totals: pd.DataFrame, create_missing: bool) -> None:
        if totals.empty:
            return
        totals = self._with_match_ids(self._with_team_ids(totals, "football_data", create_missing=create_missing))
        known = self.db.existing_match_ids()
        self.db.upsert_totals_odds(
            totals.loc[totals["match_id"].isin(known),
                       ["match_id", "bookmaker", "market_time", "line", "over", "under"]]
        )

    def load_football_data(self, season: int, division: str = "E0") -> None:
        csv_text = football_data.fetch_season_csv(self.client, season, division, **self._cache(season))
        matches, odds = football_data.parse_season(csv_text, season, division)
        # Lower-division names are less curated: a club no known name resembles
        # becomes a new entity instead of failing the load.
        create = division != "E0"
        matches = self._with_match_ids(self._with_team_ids(matches, "football_data", create_missing=create))
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
        self._load_odds(odds, create)
        totals = football_data.parse_season_totals(csv_text, season, division)
        self._load_totals(totals, create)
        log.info("football-data %s %s: %d matches, %d odds rows, %d over/under rows",
                 division, season, len(matches), len(odds), len(totals))

    def load_upcoming_fixtures(self) -> None:
        """This week's fixtures with current prices (Football-Data ``fixtures.csv``)."""
        csv_text = football_data.fetch_fixtures_csv(self.client)
        matches, odds = football_data.parse_fixtures(csv_text, self.divisions)
        if matches.empty:
            log.info("football-data fixtures: none listed")
            return
        matches = self._with_match_ids(self._with_team_ids(matches, "football_data", create_missing=True))
        self.db.upsert_matches(matches, "football_data")
        self._load_odds(odds, True)
        self._load_totals(football_data.parse_fixtures_totals(csv_text, self.divisions), True)
        log.info("football-data fixtures: %d upcoming matches", len(matches))

    def load_understat(self, season: int, *, lineups: bool = False) -> None:
        if season < UNDERSTAT_FIRST_SEASON:
            log.info("understat %s: skipped, coverage starts in %s", season, UNDERSTAT_FIRST_SEASON)
            return
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
        todo = matches[~matches["match_id"].isin(done)]
        log.info("understat lineups: %d to fetch, %d already loaded", len(todo), len(matches) - len(todo))
        for count, match in enumerate(todo.itertuples(index=False), start=1):
            rosters = understat.fetch_match_rosters(self.client, match.understat_id)
            if count % 50 == 0:
                log.info("understat lineups: %d/%d", count, len(todo))
            if rosters.empty:
                continue
            team_ids = {"home": match.home_team_id, "away": match.away_team_id}
            rosters["team_id"] = rosters["venue"].map(team_ids)
            rosters["match_id"] = match.match_id
            self.db.upsert_lineups(
                rosters[["match_id", "team_id", "player_name", "position", "is_starter", "minutes", "xg", "xa"]]
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

        squads = fbref.squads_from_schedule(schedule)
        for count, squad in enumerate(squads.itertuples(index=False), start=1):
            schedule_log, shooting_log = fbref.fetch_team_logs(
                self.client, squad.squad_id, squad.slug, squad.team, season, **self._cache(season)
            )
            log.info("fbref %s: team logs %d/%d (%s)", season, count, len(squads), squad.team)
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

    def load_espn(self, season: int, division: str = "E0") -> None:
        """Possession and shots per league match, and every club's cup and European fixtures.

        One scoreboard request per match day of the division (dates come from
        the matches already loaded), then one request per club and cup
        competition. Needs Football-Data or Understat results for the season.
        """
        league = espn.LEAGUES[division]
        dates = self.db.query(
            "SELECT DISTINCT date(kickoff) AS day FROM matches WHERE season = ? AND division = ? ORDER BY day",
            (season, division),
        )["day"]
        if dates.empty:
            raise RuntimeError(f"No {division} matches for {season} in the database; load football_data first")
        boards = [espn.fetch_scoreboard(self.client, league, pd.Timestamp(day), **self._cache(season))
                  for day in dates]
        board = pd.concat(boards, ignore_index=True)
        board = board[board["is_final"]]
        if board.empty:
            raise RuntimeError(f"ESPN {league} scoreboards for {season} listed no finished matches")
        board = self._with_team_ids(board, "espn", columns=("home_team", "away_team", "team"))
        board = self._with_match_ids(board.assign(season=season))
        known = self.db.existing_match_ids()
        stats = board.loc[board["match_id"].isin(known),
                          ["match_id", "team_id", "possession", "shots", "shots_on_target"]]
        stats = stats.dropna(subset=["possession", "shots", "shots_on_target"], how="all")
        self.db.upsert_match_stats(stats)
        log.info("espn %s %s: %d match days, %d team rows with statistics", division, season, len(dates), len(stats))

        clubs = board.drop_duplicates("team")[["team", "team_id", "espn_team_id"]]
        league_rows = board[["team_id", "kickoff", "venue"]].assign(competition=espn.LEAGUE_NAMES[league])
        schedule = [league_rows]
        for club in clubs.itertuples(index=False):
            for cup, competition in espn.CUPS.items():
                fixtures = espn.fetch_team_schedule(
                    self.client, cup, club.espn_team_id, club.team, season, competition, **self._cache(season)
                )
                if not fixtures.empty:
                    schedule.append(
                        fixtures.assign(team_id=club.team_id)[["team_id", "kickoff", "venue", "competition"]]
                    )
        rows = pd.concat(schedule, ignore_index=True).drop_duplicates(["team_id", "kickoff"])
        self.db.upsert_team_schedule(rows[["team_id", "kickoff", "competition", "venue"]])
        log.info("espn %s %s: %d schedule rows for %d clubs", division, season, len(rows), len(clubs))

    def load_sofifa(self, start: datetime, end: datetime, *, every_days: int = 28, **browser_kwargs) -> None:
        """Rating snapshots roughly every ``every_days``; snapshots already loaded are skipped."""
        loaded = self.db.rating_snapshots()
        editions = sofifa.editions_for_seasons(season_from_date(start), season_from_date(end))
        with sofifa.SofifaBrowser(**browser_kwargs) as browser:
            versions = sofifa.select_snapshots(browser.roster_versions(editions), start, end, every_days)
            todo = versions[~versions["snapshot_date"].isin(loaded)]
            log.info("sofifa: %d snapshots selected, %d already loaded", len(versions), len(versions) - len(todo))
            for version in todo.itertuples(index=False):
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
                log.info("sofifa roster %s (%s): %d players", version.roster_id,
                         f"{version.snapshot_date:%Y-%m-%d}", len(ratings))

    def run(
        self,
        seasons: Iterable[int],
        sources: Iterable[str] = ("football_data", "understat"),
        *,
        lineups: bool = False,
        upcoming: bool = True,
        browser_kwargs: dict | None = None,
        sofifa_every_days: int = 28,
    ) -> list[tuple[str, int | None, str]]:
        """Load every (source, season); returns the failures as (source, season, error).

        A failing source (blocked by the site, page layout changed) is logged
        and skipped so the remaining sources still load. With ``upcoming`` and
        the current season requested, this week's fixtures and prices are
        loaded too.
        """
        seasons, sources = list(seasons), set(sources)
        unknown = sources - set(SOURCES)
        if unknown:
            raise ValueError(f"Unknown sources: {sorted(unknown)}")
        steps: list[tuple[str, int | None, Callable[[], None]]] = []
        for season in seasons:
            # Football-Data first: it is the most authoritative source of kickoffs.
            if "football_data" in sources:
                for division in self.divisions:
                    steps.append((f"football_data:{division}", season,
                                  partial(self.load_football_data, season, division)))
            if "fbref" in sources:
                steps.append(("fbref", season, partial(self.load_fbref, season)))
            if "understat" in sources:
                steps.append(("understat", season, partial(self.load_understat, season, lineups=lineups)))
            if "espn" in sources:
                # After the result sources: it looks up the season's match days in the database.
                steps.append(("espn", season, partial(self.load_espn, season)))
        if "football_data" in sources and upcoming and self.current_season in seasons:
            steps.append(("football_data:fixtures", None, self.load_upcoming_fixtures))
        if "sofifa" in sources:
            window = (datetime(min(seasons), 7, 1), datetime(max(seasons) + 1, 6, 30))
            steps.append(("sofifa", None, partial(self.load_sofifa, *window, every_days=sofifa_every_days,
                                                  **(browser_kwargs or {}))))

        failures = []
        for source, season, step in steps:
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - keep loading the other sources
                log.exception("%s %s failed", source, season or "")
                failures.append((source, season, f"{type(exc).__name__}: {exc}"))
        return failures


def probe_sources(
    client: HttpClient,
    season: int,
    sources: Iterable[str] = SOURCES,
    *,
    browser_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Fetch and parse one live page per source; reports what works before a long ingest."""
    sources = set(sources)
    results = []

    def attempt(source: str, check: Callable[[], tuple[int, str]]) -> None:
        try:
            rows, detail = check()
            results.append({"source": source, "ok": rows > 0, "rows": rows, "detail": detail})
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            results.append({"source": source, "ok": False, "rows": 0, "detail": f"{type(exc).__name__}: {exc}"[:300]})

    if "football_data" in sources:
        for division in DIVISIONS:
            def check_fd(division=division):
                csv_text = football_data.fetch_season_csv(client, season, division, use_cache=False)
                matches, odds = football_data.parse_season(csv_text, season, division)
                totals = football_data.parse_season_totals(csv_text, season, division)
                closing = (odds["market_time"] == "closing").sum() if not odds.empty else 0
                return len(matches), f"{closing} closing-odds rows, {len(totals)} over/under rows"
            attempt(f"football_data:{division}", check_fd)

    if "understat" in sources:
        def check_understat():
            matches, history = understat.fetch_league(client, season, use_cache=False)
            return len(matches), f"{matches['home_xg'].notna().sum()} matches with xG, {len(history)} team rows"
        attempt("understat", check_understat)

    if "espn" in sources:
        def check_espn():
            for day in pd.date_range(f"{season}-08-08", f"{season}-08-31"):
                board = espn.fetch_scoreboard(client, "eng.1", day, use_cache=False)
                played = board[board["is_final"]]
                if not played.empty:
                    break
            else:
                return 0, "no finished Premier League match found in August"
            club = played.iloc[0]
            cups = espn.fetch_team_schedule(client, "eng.fa", club.espn_team_id, club.team, season, "FA Cup",
                                            use_cache=False)
            return len(played) // 2, (f"{day:%Y-%m-%d}: {played['possession'].notna().sum()} team rows with "
                                      f"possession; {club.team}: {len(cups)} FA Cup fixtures")
        attempt("espn", check_espn)

    if "fbref" in sources:
        def check_fbref():
            schedule = fbref.fetch_schedule(client, season, use_cache=False)
            squad = fbref.squads_from_schedule(schedule).iloc[0]
            log_frame, shots = fbref.fetch_team_logs(client, squad.squad_id, squad.slug, squad.team, season,
                                                     use_cache=False)
            return len(schedule), (f"{squad.team}: {log_frame['possession'].notna().sum()} rows with possession, "
                                   f"{shots['shots'].notna().sum()} with shots")
        attempt("fbref", check_fbref)

    if "sofifa" in sources:
        def check_sofifa():
            with sofifa.SofifaBrowser(**(browser_kwargs or {})) as browser:
                versions = browser.roster_versions(sofifa.editions_for_seasons(season, season))
                ratings = browser.league_ratings(versions["roster_id"].iloc[-1], max_pages=1)
            return len(ratings), f"{len(versions)} roster versions; latest {versions['snapshot_date'].max():%Y-%m-%d}"
        attempt("sofifa", check_sofifa)

    return pd.DataFrame(results, columns=["source", "ok", "rows", "detail"])
