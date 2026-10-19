import pandas as pd
import pytest

from football_predictor.db import Database, match_uuid
from football_predictor.etl.entity_resolution import team_uuid
from football_predictor.etl.pipeline import Pipeline
from football_predictor.features import build_features, lineup_strength

from . import fake_sources as fake


@pytest.fixture()
def loaded_db(tmp_path):
    db = Database(tmp_path / "football.db")
    client = fake.FakeClient()
    pipeline = Pipeline(db, client)
    pipeline.run([fake.SEASON], ["football_data", "fbref", "understat"], lineups=True)
    yield db, pipeline, client
    db.close()


def test_sources_are_merged_onto_one_entity_per_club(loaded_db):
    db, _, _ = loaded_db
    teams = db.query("SELECT * FROM teams ORDER BY canonical_name")
    assert list(teams["canonical_name"]) == sorted(fake.TEAMS)
    assert set(teams["team_id"]) == {team_uuid(t) for t in fake.TEAMS}

    aliases = db.query("SELECT alias, source FROM team_aliases")
    assert {"Man United", "Manchester Utd", "Manchester United"} <= set(aliases["alias"])
    assert len(db.query("SELECT * FROM matches")) == 12


def test_match_dataset_joins_every_source(loaded_db):
    db, _, _ = loaded_db
    data = db.load_match_dataset()
    played = data.dropna(subset=["home_goals"])
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


def test_unplayed_fixtures_and_schedule(loaded_db):
    db, _, _ = loaded_db
    data = db.load_match_dataset()
    assert data["home_goals"].isna().sum() == 2
    schedule = db.load_team_schedule()
    city = schedule[schedule["team"] == "Manchester City"]
    assert "EFL Cup" in set(city["competition"])


def test_reloading_is_idempotent(loaded_db):
    db, pipeline, _ = loaded_db
    before = {t: len(db.query(f"SELECT * FROM {t}")) for t in ("matches", "odds", "advanced_stats", "match_stats", "lineups")}
    pipeline.run([fake.SEASON], ["football_data", "fbref", "understat"], lineups=True)
    after = {t: len(db.query(f"SELECT * FROM {t}")) for t in before}
    assert before == after


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
