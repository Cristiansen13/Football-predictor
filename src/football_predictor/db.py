"""SQLite warehouse. Every table is keyed on resolved entity UUIDs.

A fixture's ``match_id`` is derived from (season, home team id, away team id),
which is unique within a league season, so each source can compute it
independently and the tables join without fuzzy date matching.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .etl.entity_resolution import Resolution, entity_uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id         TEXT PRIMARY KEY,
    canonical_name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS team_aliases (
    alias    TEXT NOT NULL,
    source   TEXT NOT NULL,
    team_id  TEXT NOT NULL REFERENCES teams(team_id),
    score    REAL NOT NULL,
    method   TEXT NOT NULL,
    PRIMARY KEY (alias, source)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id      TEXT PRIMARY KEY,
    season        INTEGER NOT NULL,
    division      TEXT NOT NULL DEFAULT 'E0',
    kickoff       TEXT NOT NULL,
    home_team_id  TEXT NOT NULL REFERENCES teams(team_id),
    away_team_id  TEXT NOT NULL REFERENCES teams(team_id),
    home_goals    INTEGER,
    away_goals    INTEGER,
    source        TEXT NOT NULL,
    source_rank   INTEGER NOT NULL,
    understat_id  TEXT,
    UNIQUE (season, home_team_id, away_team_id)
);

-- Basic per-team match stats (FBref match logs, Football-Data shot counts).
CREATE TABLE IF NOT EXISTS match_stats (
    match_id         TEXT NOT NULL REFERENCES matches(match_id),
    team_id          TEXT NOT NULL REFERENCES teams(team_id),
    possession       REAL,
    shots            REAL,
    shots_on_target  REAL,
    PRIMARY KEY (match_id, team_id)
);

-- Advanced per-team match metrics (Understat).
CREATE TABLE IF NOT EXISTS advanced_stats (
    match_id      TEXT NOT NULL REFERENCES matches(match_id),
    team_id       TEXT NOT NULL REFERENCES teams(team_id),
    xg            REAL,
    xga           REAL,
    npxg          REAL,
    npxga         REAL,
    xa            REAL,
    ppda          REAL,
    ppda_allowed  REAL,
    deep          REAL,
    deep_allowed  REAL,
    xpts          REAL,
    PRIMARY KEY (match_id, team_id)
);

-- Every fixture a team played in any competition: drives the fatigue index.
CREATE TABLE IF NOT EXISTS team_schedule (
    team_id      TEXT NOT NULL REFERENCES teams(team_id),
    kickoff      TEXT NOT NULL,
    competition  TEXT,
    venue        TEXT,
    PRIMARY KEY (team_id, kickoff)
);

CREATE TABLE IF NOT EXISTS odds (
    match_id     TEXT NOT NULL REFERENCES matches(match_id),
    bookmaker    TEXT NOT NULL,
    market_time  TEXT NOT NULL CHECK (market_time IN ('opening', 'closing')),
    home         REAL NOT NULL,
    draw         REAL NOT NULL,
    away         REAL NOT NULL,
    PRIMARY KEY (match_id, bookmaker, market_time)
);

CREATE TABLE IF NOT EXISTS player_ratings (
    player_id      TEXT NOT NULL,
    player_name    TEXT NOT NULL,
    team_id        TEXT REFERENCES teams(team_id),
    snapshot_date  TEXT NOT NULL,
    overall        INTEGER NOT NULL,
    positions      TEXT,
    PRIMARY KEY (player_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS lineups (
    match_id     TEXT NOT NULL REFERENCES matches(match_id),
    team_id      TEXT NOT NULL REFERENCES teams(team_id),
    player_name  TEXT NOT NULL,
    position     TEXT,
    is_starter   INTEGER NOT NULL,
    minutes      INTEGER,
    PRIMARY KEY (match_id, team_id, player_name)
);
"""

# Views are rebuilt on every connection so schema changes reach old databases.
VIEWS = """
DROP VIEW IF EXISTS v_match_dataset;
DROP VIEW IF EXISTS v_best_odds;

-- Reference price per fixture: closing before opening, sharp before soft. The
-- best-of-market "max" price is excluded: no single bettor could get it.
CREATE VIEW v_best_odds AS
SELECT match_id, bookmaker, market_time, home, draw, away
FROM (
    SELECT o.*,
           ROW_NUMBER() OVER (
               PARTITION BY match_id
               ORDER BY CASE market_time WHEN 'closing' THEN 0 ELSE 1 END,
                        CASE bookmaker WHEN 'pinnacle' THEN 0 WHEN 'average' THEN 1
                                       WHEN 'bet365' THEN 2 ELSE 3 END
           ) AS rnk
    FROM odds o
    WHERE bookmaker <> 'max'
)
WHERE rnk = 1;

CREATE VIEW v_match_dataset AS
SELECT m.match_id, m.season, m.division, m.kickoff,
       m.home_team_id, ht.canonical_name AS home_team,
       m.away_team_id, at.canonical_name AS away_team,
       m.home_goals, m.away_goals,
       ha.xg AS home_xg, aa.xg AS away_xg,
       ha.npxg AS home_npxg, aa.npxg AS away_npxg,
       ha.xa AS home_xa, aa.xa AS away_xa,
       ha.ppda AS home_ppda, aa.ppda AS away_ppda,
       ha.deep AS home_deep, aa.deep AS away_deep,
       hs.possession AS home_possession, as_.possession AS away_possession,
       hs.shots AS home_shots, as_.shots AS away_shots,
       hs.shots_on_target AS home_shots_on_target, as_.shots_on_target AS away_shots_on_target,
       o.bookmaker AS odds_bookmaker, o.market_time AS odds_market_time,
       o.home AS odds_home, o.draw AS odds_draw, o.away AS odds_away
FROM matches m
JOIN teams ht ON ht.team_id = m.home_team_id
JOIN teams at ON at.team_id = m.away_team_id
LEFT JOIN advanced_stats ha ON ha.match_id = m.match_id AND ha.team_id = m.home_team_id
LEFT JOIN advanced_stats aa ON aa.match_id = m.match_id AND aa.team_id = m.away_team_id
LEFT JOIN match_stats hs ON hs.match_id = m.match_id AND hs.team_id = m.home_team_id
LEFT JOIN match_stats as_ ON as_.match_id = m.match_id AND as_.team_id = m.away_team_id
LEFT JOIN v_best_odds o ON o.match_id = m.match_id;
"""

SOURCE_RANK = {"football_data": 0, "fbref": 1, "understat": 2, "manual": 3}


def match_uuid(season: int, home_team_id: str, away_team_id: str) -> str:
    return entity_uuid("match", f"{season}:{home_team_id}:{away_team_id}")


def _iso(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat(sep=" ")


def _none_if_na(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return _iso(value)
    # sqlite3 cannot bind numpy scalars such as int64.
    return value.item() if isinstance(value, np.generic) else value


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.executescript(VIEWS)

    def _migrate(self) -> None:
        """Bring databases created by earlier versions up to the current schema."""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(matches)")}
        if "division" not in columns:
            with self._conn:
                self._conn.execute("ALTER TABLE matches ADD COLUMN division TEXT NOT NULL DEFAULT 'E0'")

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._conn:
            yield self._conn

    def query(self, sql: str, params: Iterable = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self._conn, params=tuple(params))

    # -- master data -------------------------------------------------------

    def register_team(self, resolution: Resolution, source: str) -> str:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO teams (team_id, canonical_name) VALUES (?, ?)",
                (resolution.entity_id, resolution.canonical),
            )
            conn.execute(
                """INSERT INTO team_aliases (alias, source, team_id, score, method)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (alias, source) DO UPDATE SET
                       team_id = excluded.team_id, score = excluded.score, method = excluded.method""",
                (resolution.query, source, resolution.entity_id, resolution.score, resolution.method),
            )
        return resolution.entity_id

    # -- facts --------------------------------------------------------------

    def upsert_matches(self, frame: pd.DataFrame, source: str) -> None:
        """Rows need season, kickoff, home_team_id, away_team_id, home_goals, away_goals
        (and optionally division, understat_id).

        Goals are filled from whichever source has them; the kickoff time is
        taken from the most authoritative source seen so far.
        """
        rank = SOURCE_RANK[source]
        rows = [
            (
                match_uuid(int(r.season), r.home_team_id, r.away_team_id),
                int(r.season),
                str(getattr(r, "division", "E0")),
                _iso(r.kickoff),
                r.home_team_id,
                r.away_team_id,
                None if pd.isna(r.home_goals) else int(r.home_goals),
                None if pd.isna(r.away_goals) else int(r.away_goals),
                source,
                rank,
                _none_if_na(getattr(r, "understat_id", None)),
            )
            for r in frame.itertuples(index=False)
        ]
        with self.transaction() as conn:
            conn.executemany(
                """INSERT INTO matches (match_id, season, division, kickoff, home_team_id, away_team_id,
                                        home_goals, away_goals, source, source_rank, understat_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (match_id) DO UPDATE SET
                       home_goals = COALESCE(excluded.home_goals, matches.home_goals),
                       away_goals = COALESCE(excluded.away_goals, matches.away_goals),
                       kickoff = CASE WHEN excluded.source_rank <= matches.source_rank
                                      THEN excluded.kickoff ELSE matches.kickoff END,
                       division = CASE WHEN excluded.source_rank <= matches.source_rank
                                       THEN excluded.division ELSE matches.division END,
                       source = CASE WHEN excluded.source_rank <= matches.source_rank
                                     THEN excluded.source ELSE matches.source END,
                       source_rank = MIN(excluded.source_rank, matches.source_rank),
                       understat_id = COALESCE(excluded.understat_id, matches.understat_id)""",
                rows,
            )

    def _upsert(self, table: str, keys: list[str], frame: pd.DataFrame) -> None:
        """Insert rows; on key conflict keep existing values where the new ones are NULL."""
        if frame.empty:
            return
        columns = list(frame.columns)
        values = [c for c in columns if c not in keys]
        updates = ", ".join(f"{c} = COALESCE(excluded.{c}, {table}.{c})" for c in values)
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))}) "
            f"ON CONFLICT ({', '.join(keys)}) DO "
            + (f"UPDATE SET {updates}" if values else "NOTHING")
        )
        rows = [tuple(_none_if_na(v) for v in row) for row in frame.itertuples(index=False)]
        with self.transaction() as conn:
            conn.executemany(sql, rows)

    def upsert_match_stats(self, frame: pd.DataFrame) -> None:
        self._upsert("match_stats", ["match_id", "team_id"], frame)

    def upsert_advanced_stats(self, frame: pd.DataFrame) -> None:
        self._upsert("advanced_stats", ["match_id", "team_id"], frame)

    def upsert_team_schedule(self, frame: pd.DataFrame) -> None:
        self._upsert("team_schedule", ["team_id", "kickoff"], frame)

    def upsert_odds(self, frame: pd.DataFrame) -> None:
        self._upsert("odds", ["match_id", "bookmaker", "market_time"], frame)

    def upsert_player_ratings(self, frame: pd.DataFrame) -> None:
        self._upsert("player_ratings", ["player_id", "snapshot_date"], frame)

    def upsert_lineups(self, frame: pd.DataFrame) -> None:
        self._upsert("lineups", ["match_id", "team_id", "player_name"], frame)

    # -- reads --------------------------------------------------------------

    def rating_snapshots(self) -> set[pd.Timestamp]:
        dates = self.query("SELECT DISTINCT snapshot_date FROM player_ratings")["snapshot_date"]
        return set(pd.to_datetime(dates))

    def existing_match_ids(self) -> set[str]:
        return set(self.query("SELECT match_id FROM matches")["match_id"])

    def load_match_dataset(self) -> pd.DataFrame:
        frame = self.query("SELECT * FROM v_match_dataset ORDER BY kickoff")
        frame["kickoff"] = pd.to_datetime(frame["kickoff"])
        for column in ("home_goals", "away_goals"):
            frame[column] = frame[column].astype("Int64")
        return frame

    def load_team_schedule(self) -> pd.DataFrame:
        frame = self.query(
            """SELECT s.*, t.canonical_name AS team
               FROM team_schedule s JOIN teams t ON t.team_id = s.team_id
               ORDER BY s.team_id, s.kickoff"""
        )
        frame["kickoff"] = pd.to_datetime(frame["kickoff"])
        return frame

    def load_lineups(self) -> pd.DataFrame:
        return self.query(
            """SELECT l.*, m.kickoff FROM lineups l JOIN matches m ON m.match_id = l.match_id"""
        ).assign(kickoff=lambda f: pd.to_datetime(f["kickoff"]))

    def load_player_ratings(self) -> pd.DataFrame:
        frame = self.query("SELECT * FROM player_ratings")
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
        return frame
