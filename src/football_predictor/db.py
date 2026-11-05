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

-- Over/under prices per fixture, bookmaker, market time and goal line.
CREATE TABLE IF NOT EXISTS totals_odds (
    match_id     TEXT NOT NULL REFERENCES matches(match_id),
    bookmaker    TEXT NOT NULL,
    market_time  TEXT NOT NULL,
    line         REAL NOT NULL,
    over         REAL NOT NULL,
    under        REAL NOT NULL,
    PRIMARY KEY (match_id, bookmaker, market_time, line)
);

-- Asian handicap prices per fixture, bookmaker and market time (the line is the home handicap).
CREATE TABLE IF NOT EXISTS handicap_odds (
    match_id     TEXT NOT NULL REFERENCES matches(match_id),
    bookmaker    TEXT NOT NULL,
    market_time  TEXT NOT NULL,
    line         REAL NOT NULL,
    home         REAL NOT NULL,
    away         REAL NOT NULL,
    PRIMARY KEY (match_id, bookmaker, market_time)
);

-- Prices logged before kickoff: the live counterpart of ``odds``, with the
-- time of collection. Fixtures are keyed by the source's own id and carry the
-- warehouse match_id once both club names resolve; times are UTC.
CREATE TABLE IF NOT EXISTS price_snapshots (
    snapshot_at  TEXT NOT NULL,
    source       TEXT NOT NULL,
    division     TEXT NOT NULL,
    event_id     TEXT NOT NULL,
    kickoff_utc  TEXT NOT NULL,
    home_name    TEXT NOT NULL,
    away_name    TEXT NOT NULL,
    match_id     TEXT,
    bookmaker    TEXT NOT NULL,
    home         REAL NOT NULL,
    draw         REAL NOT NULL,
    away         REAL NOT NULL,
    PRIMARY KEY (snapshot_at, source, event_id, bookmaker)
);
CREATE INDEX IF NOT EXISTS price_snapshots_by_match ON price_snapshots (match_id, bookmaker, snapshot_at);
CREATE INDEX IF NOT EXISTS price_snapshots_by_event ON price_snapshots (source, event_id);

-- The forward test's ledger: the bet a rule picked the first time a fixture
-- qualified, as it stood then (price, fair value, EV); settled later against
-- ``matches`` and valued against the fair bookmaker's last logged price.
CREATE TABLE IF NOT EXISTS paper_bets (
    rule         TEXT NOT NULL,
    match_id     TEXT NOT NULL,
    placed_at    TEXT NOT NULL,
    source       TEXT NOT NULL,
    division     TEXT NOT NULL,
    kickoff_utc  TEXT NOT NULL,
    home_name    TEXT NOT NULL,
    away_name    TEXT NOT NULL,
    selection    TEXT NOT NULL CHECK (selection IN ('home', 'draw', 'away')),
    bookmaker    TEXT NOT NULL,
    odds         REAL NOT NULL,
    fair_book    TEXT NOT NULL,
    p_fair       REAL NOT NULL,
    ev           REAL NOT NULL,
    PRIMARY KEY (rule, match_id)
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
    xg           REAL,
    xa           REAL,
    PRIMARY KEY (match_id, team_id, player_name)
);
"""

# Views are rebuilt on every connection so schema changes reach old databases.
VIEWS = """
DROP VIEW IF EXISTS v_match_dataset;
DROP VIEW IF EXISTS v_best_odds;
DROP VIEW IF EXISTS v_best_totals;
DROP VIEW IF EXISTS v_open_odds;
DROP VIEW IF EXISTS v_open_totals;
DROP VIEW IF EXISTS v_best_handicap;
DROP VIEW IF EXISTS v_open_handicap;

CREATE VIEW v_best_handicap AS
SELECT match_id, bookmaker, market_time, line, home, away
FROM (
    SELECT h.*,
           ROW_NUMBER() OVER (
               PARTITION BY match_id
               ORDER BY CASE market_time WHEN 'closing' THEN 0 ELSE 1 END,
                        CASE bookmaker WHEN 'pinnacle' THEN 0 WHEN 'average' THEN 1
                                       WHEN 'bet365' THEN 2 ELSE 3 END
           ) AS rnk
    FROM handicap_odds h
    WHERE bookmaker <> 'max'
)
WHERE rnk = 1;

CREATE VIEW v_open_handicap AS
SELECT match_id, bookmaker, line, home, away
FROM (
    SELECT h.*,
           ROW_NUMBER() OVER (
               PARTITION BY match_id
               ORDER BY CASE bookmaker WHEN 'pinnacle' THEN 0 WHEN 'average' THEN 1
                                       WHEN 'bet365' THEN 2 ELSE 3 END
           ) AS rnk
    FROM handicap_odds h
    WHERE bookmaker <> 'max' AND market_time = 'opening'
)
WHERE rnk = 1;

-- Earliest listed price per fixture (Football-Data's pre-match column), sharp first.
CREATE VIEW v_open_odds AS
SELECT match_id, bookmaker, home, draw, away
FROM (
    SELECT o.*,
           ROW_NUMBER() OVER (
               PARTITION BY match_id
               ORDER BY CASE bookmaker WHEN 'pinnacle' THEN 0 WHEN 'average' THEN 1
                                       WHEN 'bet365' THEN 2 ELSE 3 END
           ) AS rnk
    FROM odds o
    WHERE bookmaker <> 'max' AND market_time = 'opening'
)
WHERE rnk = 1;

CREATE VIEW v_open_totals AS
SELECT match_id, bookmaker, over, under
FROM (
    SELECT t.*,
           ROW_NUMBER() OVER (
               PARTITION BY match_id
               ORDER BY CASE bookmaker WHEN 'pinnacle' THEN 0 WHEN 'average' THEN 1
                                       WHEN 'bet365' THEN 2 ELSE 3 END
           ) AS rnk
    FROM totals_odds t
    WHERE bookmaker <> 'max' AND market_time = 'opening' AND line = 2.5
)
WHERE rnk = 1;

CREATE VIEW v_best_totals AS
SELECT match_id, bookmaker, market_time, line, over, under
FROM (
    SELECT t.*,
           ROW_NUMBER() OVER (
               PARTITION BY match_id
               ORDER BY CASE market_time WHEN 'closing' THEN 0 ELSE 1 END,
                        CASE bookmaker WHEN 'pinnacle' THEN 0 WHEN 'average' THEN 1
                                       WHEN 'bet365' THEN 2 ELSE 3 END
           ) AS rnk
    FROM totals_odds t
    WHERE bookmaker <> 'max' AND line = 2.5
)
WHERE rnk = 1;

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
       o.home AS odds_home, o.draw AS odds_draw, o.away AS odds_away,
       t.bookmaker AS totals_bookmaker, t.market_time AS totals_market_time,
       t.over AS odds_over_2_5, t.under AS odds_under_2_5,
       op.home AS open_home, op.draw AS open_draw, op.away AS open_away,
       mx.home AS max_home, mx.draw AS max_draw, mx.away AS max_away,
       ot.over AS open_over_2_5, ot.under AS open_under_2_5,
       mt.over AS max_over_2_5, mt.under AS max_under_2_5,
       h.bookmaker AS handicap_bookmaker, h.market_time AS handicap_market_time,
       h.line AS ah_line, h.home AS odds_ah_home, h.away AS odds_ah_away,
       oh.line AS open_ah_line, oh.home AS open_ah_home, oh.away AS open_ah_away,
       mh.line AS max_ah_line, mh.home AS max_ah_home, mh.away AS max_ah_away
FROM matches m
JOIN teams ht ON ht.team_id = m.home_team_id
JOIN teams at ON at.team_id = m.away_team_id
LEFT JOIN advanced_stats ha ON ha.match_id = m.match_id AND ha.team_id = m.home_team_id
LEFT JOIN advanced_stats aa ON aa.match_id = m.match_id AND aa.team_id = m.away_team_id
LEFT JOIN match_stats hs ON hs.match_id = m.match_id AND hs.team_id = m.home_team_id
LEFT JOIN match_stats as_ ON as_.match_id = m.match_id AND as_.team_id = m.away_team_id
LEFT JOIN v_best_odds o ON o.match_id = m.match_id
LEFT JOIN v_best_totals t ON t.match_id = m.match_id
LEFT JOIN v_open_odds op ON op.match_id = m.match_id
LEFT JOIN odds mx ON mx.match_id = m.match_id AND mx.bookmaker = 'max' AND mx.market_time = 'closing'
LEFT JOIN v_open_totals ot ON ot.match_id = m.match_id
LEFT JOIN totals_odds mt ON mt.match_id = m.match_id AND mt.bookmaker = 'max' AND mt.market_time = 'closing'
    AND mt.line = 2.5
LEFT JOIN v_best_handicap h ON h.match_id = m.match_id
LEFT JOIN v_open_handicap oh ON oh.match_id = m.match_id
LEFT JOIN handicap_odds mh ON mh.match_id = m.match_id AND mh.bookmaker = 'max' AND mh.market_time = 'closing';
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
        # A long wait on the lock lets two ingests (say line-ups and ESPN) share one database.
        self._conn = sqlite3.connect(str(path), timeout=60.0)
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
        lineup_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(lineups)")}
        for column in ("xg", "xa"):
            if column not in lineup_columns:
                with self._conn:
                    self._conn.execute(f"ALTER TABLE lineups ADD COLUMN {column} REAL")

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

    def upsert_totals_odds(self, frame: pd.DataFrame) -> None:
        self._upsert("totals_odds", ["match_id", "bookmaker", "market_time", "line"], frame)

    def upsert_handicap_odds(self, frame: pd.DataFrame) -> None:
        self._upsert("handicap_odds", ["match_id", "bookmaker", "market_time"], frame)

    def load_prices(self, match_ids) -> dict[str, pd.DataFrame]:
        """Every stored price for the given fixtures: {'1x2', 'totals', 'handicap'} long frames."""
        ids = list(dict.fromkeys(match_ids))
        tables = {"1x2": ("odds", ""), "totals": ("totals_odds", " AND line = 2.5"), "handicap": ("handicap_odds", "")}
        out = {}
        for key, (table, extra) in tables.items():
            parts = []
            for start in range(0, len(ids), 900):  # well inside SQLite's bound-variable limit
                chunk = ids[start:start + 900]
                marks = ", ".join("?" * len(chunk))
                parts.append(self.query(f"SELECT * FROM {table} WHERE match_id IN ({marks}){extra}", chunk))
            out[key] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return out

    def bookmakers(self) -> list[str]:
        """Every bookmaker with a stored price, in any market."""
        names = set()
        for table in ("odds", "totals_odds", "handicap_odds"):
            names |= set(self.query(f"SELECT DISTINCT bookmaker FROM {table}")["bookmaker"])
        return sorted(names)

    # -- live prices and the paper-trading ledger ---------------------------

    def upsert_price_snapshots(self, frame: pd.DataFrame) -> None:
        """Rows in :data:`etl.odds_api.SNAPSHOT_FIELDS` plus ``match_id`` (None where unresolved)."""
        if frame.empty:
            return
        frame = frame.assign(snapshot_at=frame["snapshot_at"].map(_iso), kickoff_utc=frame["kickoff_utc"].map(_iso))
        columns = ["snapshot_at", "source", "division", "event_id", "kickoff_utc", "home_name", "away_name",
                   "match_id", "bookmaker", "home", "draw", "away"]
        self._upsert("price_snapshots", ["snapshot_at", "source", "event_id", "bookmaker"], frame[columns])

    def fill_snapshot_match_ids(self, source: str, event_id: str, match_id: str) -> int:
        """Attach ``match_id`` to earlier snapshots of a fixture whose names resolve only now."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE price_snapshots SET match_id = ? WHERE source = ? AND event_id = ? AND match_id IS NULL",
                (match_id, source, event_id),
            )
        return cursor.rowcount

    def load_price_snapshots(
        self,
        *,
        kickoff_from=None,
        kickoff_to=None,
        match_ids: Iterable[str] | None = None,
        bookmaker: str | None = None,
    ) -> pd.DataFrame:
        """Logged prices, timestamps parsed, oldest snapshot first."""
        clauses, params = [], []
        if kickoff_from is not None:
            clauses.append("kickoff_utc >= ?")
            params.append(_iso(kickoff_from))
        if kickoff_to is not None:
            clauses.append("kickoff_utc < ?")
            params.append(_iso(kickoff_to))
        if bookmaker is not None:
            clauses.append("bookmaker = ?")
            params.append(bookmaker)
        if match_ids is not None:
            ids = list(dict.fromkeys(match_ids))
            if not ids:
                return pd.DataFrame(columns=["snapshot_at", "source", "division", "event_id", "kickoff_utc",
                                             "home_name", "away_name", "match_id", "bookmaker", "home", "draw", "away"])
            clauses.append(f"match_id IN ({', '.join('?' * len(ids))})")
            params.extend(ids)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        frame = self.query(f"SELECT * FROM price_snapshots{where} ORDER BY snapshot_at, source, event_id, bookmaker",
                           params)
        for column in ("snapshot_at", "kickoff_utc"):
            frame[column] = pd.to_datetime(frame[column])
        return frame

    def divisions_with_kickoffs(self, start, end) -> list[str]:
        """Divisions with a logged fixture kicking off in ``[start, end)``, sorted."""
        frame = self.query(
            "SELECT DISTINCT division FROM price_snapshots WHERE kickoff_utc >= ? AND kickoff_utc < ? ORDER BY division",
            [_iso(start), _iso(end)],
        )
        return frame["division"].tolist()

    def insert_paper_bets(self, frame: pd.DataFrame) -> int:
        """Record bets; a fixture already in the ledger for the rule keeps its first bet. Returns the number added."""
        if frame.empty:
            return 0
        columns = ["rule", "match_id", "placed_at", "source", "division", "kickoff_utc", "home_name", "away_name",
                   "selection", "bookmaker", "odds", "fair_book", "p_fair", "ev"]
        frame = frame.assign(placed_at=frame["placed_at"].map(_iso), kickoff_utc=frame["kickoff_utc"].map(_iso))
        rows = [tuple(_none_if_na(v) for v in row) for row in frame[columns].itertuples(index=False)]
        with self.transaction() as conn:
            before = conn.total_changes
            conn.executemany(
                f"INSERT OR IGNORE INTO paper_bets ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})", rows
            )
            return conn.total_changes - before

    def load_paper_bets(self, rule: str | None = None) -> pd.DataFrame:
        """The ledger with the result where the match has been played (goals NULL otherwise)."""
        where, params = ("WHERE b.rule = ?", [rule]) if rule else ("", [])
        frame = self.query(
            f"""SELECT b.*, m.home_goals, m.away_goals, m.kickoff AS kickoff_local
                FROM paper_bets b LEFT JOIN matches m ON m.match_id = b.match_id
                {where} ORDER BY b.kickoff_utc, b.placed_at""",
            params,
        )
        for column in ("placed_at", "kickoff_utc"):
            frame[column] = pd.to_datetime(frame[column])
        for column in ("home_goals", "away_goals"):
            frame[column] = frame[column].astype("Int64")
        return frame

    def division_clubs(self, division: str, seasons: Iterable[int]) -> dict[str, list[str]]:
        """Every club with a fixture in ``division`` in ``seasons``: team id -> canonical name and aliases."""
        seasons = list(seasons)
        if not seasons:
            return {}
        marks = ", ".join("?" * len(seasons))
        teams = self.query(
            f"""SELECT DISTINCT t.team_id, t.canonical_name FROM matches m
                JOIN teams t ON t.team_id IN (m.home_team_id, m.away_team_id)
                WHERE m.division = ? AND m.season IN ({marks})""",
            [division, *seasons],
        )
        clubs = {row.team_id: [row.canonical_name] for row in teams.itertuples(index=False)}
        if clubs:
            ids = list(clubs)
            aliases = self.query(
                f"SELECT alias, team_id FROM team_aliases WHERE team_id IN ({', '.join('?' * len(ids))})", ids
            )
            for row in aliases.itertuples(index=False):
                if row.alias not in clubs[row.team_id]:
                    clubs[row.team_id].append(row.alias)
        return clubs

    def team_id_by_name(self, name: str) -> str | None:
        """The team whose canonical name or any stored alias is ``name``."""
        found = self.query(
            "SELECT team_id FROM teams WHERE canonical_name = ? UNION SELECT team_id FROM team_aliases WHERE alias = ?",
            [name, name],
        )
        return str(found["team_id"].iloc[0]) if len(found) else None

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
