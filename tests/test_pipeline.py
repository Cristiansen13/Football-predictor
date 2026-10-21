import sqlite3
from datetime import datetime

import pandas as pd
import pytest

from football_predictor.db import Database, match_uuid
from football_predictor.etl import pipeline as pipeline_module
from football_predictor.etl.entity_resolution import team_uuid
from football_predictor.etl.pipeline import Pipeline
from football_predictor.features import build_features, lineup_strength

from . import fake_sources as fake


@pytest.fixture()
def loaded_db(tmp_path):
    db = Database(tmp_path / "football.db")
    client = fake.FakeClient()
    pipeline = Pipeline(db, client, current_season=fake.SEASON)
    failures = pipeline.run([fake.SEASON], ["football_data", "fbref", "understat"], lineups=True)
    assert failures == []
    yield db, pipeline, client
    db.close()


def test_sources_are_merged_onto_one_entity_per_club(loaded_db):
    db, _, _ = loaded_db
    teams = set(db.query("SELECT canonical_name FROM teams")["canonical_name"])
    assert teams == set(fake.TEAMS) | set(fake.CHAMPIONSHIP_CANONICAL)
    assert team_uuid("Manchester United") in set(db.query("SELECT team_id FROM teams")["team_id"])

    aliases = db.query("SELECT alias, source, method FROM team_aliases")
    assert {"Man United", "Manchester Utd", "Manchester United"} <= set(aliases["alias"])
    # A Championship club no register knows is created, never forced onto a lookalike.
    assert aliases.loc[aliases["alias"] == "Newtown Rangers", "method"].item() == "created"
    matches = db.query("SELECT division, COUNT(*) AS n FROM matches GROUP BY division").set_index("division")["n"]
    assert matches.to_dict() == {"E0": 12, "E1": 12}


def test_match_dataset_joins_every_source(loaded_db):
    db, _, _ = loaded_db
    data = db.load_match_dataset()
    played = data[(data["division"] == "E0") & data["home_goals"].notna()]
    assert len(played) == 10
    # Understat xG and PPDA, FBref possession, Football-Data shots and closing odds on the same rows.
    assert played["home_xg"].notna().all() and played["away_ppda"].notna().all()
    assert played["home_possession"].notna().all()
    assert played["home_shots"].notna().all()
    assert (played["odds_market_time"] == "closing").all()
    assert (played["odds_bookmaker"] == "pinnacle").all()
    assert played["home_xa"].notna().all()

    first = played.iloc[0]
    assert first["home_team"] == "Manchester United" and first["away_team"] == "Manchester City"
    assert first["kickoff"] == pd.Timestamp("2023-08-12 15:00")
    assert first["home_possession"] == 40
    assert first["odds_home"] == pytest.approx(2.08)

    championship = data[data["division"] == "E1"]
    assert championship["home_goals"].notna().all() and championship["home_xg"].isna().all()
    assert (championship["odds_bookmaker"] == "average").all()


def test_upcoming_fixtures_carry_current_prices(loaded_db):
    db, _, _ = loaded_db
    data = db.load_match_dataset()
    upcoming = data[data["home_goals"].isna()]
    assert len(upcoming) == 2  # the Scottish fixture in fixtures.csv is ignored
    assert (upcoming["odds_market_time"] == "opening").all()
    wolves = upcoming[upcoming["home_team"] == "Wolverhampton Wanderers"].iloc[0]
    assert wolves["odds_home"] == pytest.approx(3.20)  # Pinnacle preferred over Bet365 and average
    assert "Celtic" not in set(db.query("SELECT canonical_name FROM teams")["canonical_name"])


def test_unplayed_fixtures_and_schedule(loaded_db):
    db, _, _ = loaded_db
    schedule = db.load_team_schedule()
    city = schedule[schedule["team"] == "Manchester City"]
    assert "EFL Cup" in set(city["competition"])


def test_reloading_is_idempotent(loaded_db):
    db, pipeline, _ = loaded_db
    tables = ("matches", "odds", "advanced_stats", "match_stats", "lineups", "teams")
    before = {t: len(db.query(f"SELECT * FROM {t}")) for t in tables}
    assert pipeline.run([fake.SEASON], ["football_data", "fbref", "understat"], lineups=True) == []
    after = {t: len(db.query(f"SELECT * FROM {t}")) for t in tables}
    assert before == after


def test_failing_source_does_not_stop_the_others(tmp_path):
    db = Database(tmp_path / "football.db")
    client = fake.FakeClient()
    del client.pages[next(url for url in client.pages if "fbref.com" in url and "schedule" in url)]
    failures = Pipeline(db, client).run([fake.SEASON], ["football_data", "fbref", "understat"])
    assert [source for source, _, _ in failures] == ["fbref"]
    assert db.load_match_dataset()["home_xg"].notna().any()  # Understat still loaded
    db.close()


def test_understat_json_endpoint_fallback(tmp_path):
    db = Database(tmp_path / "football.db")
    client = fake.FakeClient(understat_api=True)
    assert Pipeline(db, client).run([fake.SEASON], ["understat"]) == []
    data = db.load_match_dataset()
    assert data["home_xg"].notna().sum() == 10
    api_calls = [r for r in client.requested if isinstance(r, tuple)]
    assert api_calls and api_calls[0][1]["X-Requested-With"] == "XMLHttpRequest"
    assert api_calls[0][1]["Referer"].startswith("https://understat.com/league/EPL/")
    db.close()


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE teams (team_id TEXT PRIMARY KEY, canonical_name TEXT NOT NULL UNIQUE);
        CREATE TABLE matches (match_id TEXT PRIMARY KEY, season INTEGER NOT NULL, kickoff TEXT NOT NULL,
            home_team_id TEXT NOT NULL, away_team_id TEXT NOT NULL, home_goals INTEGER, away_goals INTEGER,
            source TEXT NOT NULL, source_rank INTEGER NOT NULL, understat_id TEXT,
            UNIQUE (season, home_team_id, away_team_id));
        CREATE VIEW v_match_dataset AS SELECT match_id FROM matches;
        INSERT INTO teams VALUES ('a', 'Arsenal'), ('b', 'Chelsea');
        INSERT INTO matches VALUES ('m', 2020, '2020-09-12 15:00:00', 'a', 'b', 1, 0, 'football_data', 0, NULL);
        """
    )
    conn.close()
    db = Database(path)
    data = db.load_match_dataset()
    assert data.loc[0, "division"] == "E0" and data.loc[0, "home_team"] == "Arsenal"
    db.close()


def test_sofifa_snapshots_resume(tmp_path, monkeypatch):
    calls = []

    class FakeBrowser:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def roster_versions(self, editions):
            assert editions == [24]
            return pd.DataFrame({"roster_id": ["240001", "240040"],
                                 "snapshot_date": pd.to_datetime(["2023-07-20", "2023-09-01"])})

        def league_ratings(self, roster_id):
            calls.append(roster_id)
            return pd.DataFrame({"sofifa_id": ["1", "2"], "player_name": ["Kevin De Bruyne", "Bruno Fernandes"],
                                 "team": ["Manchester City", "Manchester United"], "overall": [91, 86],
                                 "positions": ["CM", "CAM"]})

    monkeypatch.setattr(pipeline_module.sofifa, "SofifaBrowser", FakeBrowser)
    db = Database(tmp_path / "football.db")
    pipe = Pipeline(db, fake.FakeClient())
    window = (datetime(2023, 7, 1), datetime(2024, 6, 30))
    pipe.load_sofifa(*window)
    assert calls == ["240001", "240040"]
    pipe.load_sofifa(*window)
    assert calls == ["240001", "240040"]  # nothing re-scraped
    ratings = db.load_player_ratings()
    assert len(ratings) == 4 and set(ratings["team_id"]) == {team_uuid("Manchester City"), team_uuid("Manchester United")}
    db.close()


def test_lineup_strength_from_warehouse(loaded_db):
    db, _, _ = loaded_db
    city, united = team_uuid("Manchester City"), team_uuid("Manchester United")
    db.upsert_player_ratings(
        pd.DataFrame(
            {
                "player_id": ["p1", "p2", "p3", "p4", "p5"],
                "player_name": ["Ederson Moraes", "Kevin De Bruyne", "Bruno Fernandes", "André Onana", "Erling Haaland"],
                "team_id": [city, city, united, united, city],
                "snapshot_date": pd.Timestamp("2023-08-01"),
                "overall": [88, 91, 86, 84, 91],
            }
        )
    )
    strength = lineup_strength(db.load_lineups(), db.load_player_ratings())
    opener = strength[strength["match_id"] == match_uuid(fake.SEASON, united, city)].set_index("team_id")
    # City's XI is "Ederson" and "Kevin De Bruyne": the short name still matches "Ederson Moraes".
    assert opener.loc[city, "matched_starters"] == 2
    assert opener.loc[city, "lineup_score"] == pytest.approx((88 + 91) / 2)
    assert opener.loc[city, "best_xi_score"] == pytest.approx((91 + 91 + 88) / 3)
    assert opener.loc[united, "lineup_fraction"] == pytest.approx(1.0)


def test_features_from_warehouse_use_cup_fixtures(loaded_db):
    db, _, _ = loaded_db
    data = db.load_match_dataset()
    features = build_features(data, schedule=db.load_team_schedule())
    # City played an away cup tie on Wednesday 23 Aug before Saturday's 17:00 league match.
    match = features[(features["kickoff"] == pd.Timestamp("2023-08-26 17:00"))]
    side = "home" if match["home_team"].iloc[0] == "Manchester City" else "away"
    assert match[f"{side}_rest_days"].iloc[0] == pytest.approx(3 - 2.75 / 24)
    assert match[f"{side}_fatigue_penalty"].iloc[0] > 0


def test_espn_fills_possession_and_cup_schedule():
    """ESPN stands in for FBref: match statistics per league game and cup fixtures per club."""
    db = Database(":memory:")
    client = fake.FakeClient()
    pipeline = Pipeline(db, client)
    assert pipeline.run([fake.SEASON], ["football_data", "understat", "espn"], upcoming=False) == []

    stats = db.query("SELECT * FROM match_stats WHERE possession IS NOT NULL")
    assert len(stats) == 20  # ten played matches, two sides each
    assert set(stats["possession"]) <= set(range(0, 101))

    schedule = db.load_team_schedule()
    cups = schedule[schedule["competition"] == "FA Cup"]
    assert len(cups) == 4 and set(cups["team"]) == set(fake.TEAMS)
    assert (schedule["competition"] == "Premier League").sum() == 20
    assert all("site.api.espn.com" in url for url in client.requested if "espn" in url)

    dataset = db.load_match_dataset()
    assert dataset["home_possession"].notna().sum() == 10


def test_lineups_store_player_xg_and_xa_and_old_lineup_tables_are_migrated(tmp_path):
    db = Database(":memory:")
    pipeline = Pipeline(db, fake.FakeClient())
    assert pipeline.run([fake.SEASON], ["football_data", "understat"], lineups=True, upcoming=False) == []
    lineups = db.load_lineups()
    assert lineups["xg"].notna().all() and lineups["xa"].notna().all()
    assert (lineups["xg"] == 0.1).sum() > 0

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE lineups (match_id TEXT NOT NULL, team_id TEXT NOT NULL, player_name TEXT NOT NULL,
            position TEXT, is_starter INTEGER NOT NULL, minutes INTEGER, PRIMARY KEY (match_id, team_id, player_name));
        INSERT INTO lineups VALUES ('m', 'a', 'Bukayo Saka', 'AMR', 1, 90);
        """
    )
    conn.close()
    migrated = Database(path)
    row = migrated.query("SELECT xg, xa, minutes FROM lineups").iloc[0]
    assert pd.isna(row["xg"]) and pd.isna(row["xa"]) and row["minutes"] == 90
    migrated.close()


def test_over_under_prices_reach_the_dataset():
    db = Database(":memory:")
    pipeline = Pipeline(db, fake.FakeClient(), current_season=fake.SEASON)  # so fixtures.csv is loaded too
    assert pipeline.run([fake.SEASON], ["football_data"], upcoming=True) == []
    data = db.load_match_dataset()
    played = data[data["home_goals"].notna() & (data["division"] == "E0")]  # the toy E1 file has no totals
    assert played["odds_over_2_5"].notna().all() and (played["totals_bookmaker"] == "pinnacle").all()
    assert (played["totals_market_time"] == "closing").all()
    upcoming = data[data["home_goals"].isna() & (data["division"] == "E0")]
    assert upcoming["odds_over_2_5"].notna().sum() == 2 and (upcoming["totals_bookmaker"].dropna() == "average").all()
