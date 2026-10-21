import pandas as pd

from football_predictor.db import Database
from football_predictor.etl.pipeline import Pipeline
from football_predictor.quality import coverage_report, validate

from . import fake_sources as fake


def _db(tmp_path, sources=("football_data", "understat", "fbref")):
    db = Database(tmp_path / "football.db")
    Pipeline(db, fake.FakeClient()).run([fake.SEASON], sources)
    return db


def test_coverage_report(tmp_path):
    db = _db(tmp_path)
    report = coverage_report(db).set_index("division")
    db.close()
    e0 = report.loc["E0"]
    assert (e0["teams"], e0["fixtures"], e0["played"]) == (4, 12, 10)
    assert e0["xg"] == 1.0 and e0["possession"] == 1.0 and e0["closing_odds"] == 1.0
    assert e0["lineups"] == 0.0 and e0["schedule_rows"] > 0
    assert report.loc["E1", "xg"] == 0.0


def test_validate_flags_real_problems(tmp_path):
    db = _db(tmp_path, sources=("football_data",))
    issues = validate(coverage_report(db), current_season=fake.SEASON + 1)
    db.close()
    text = "\n".join(map(str, issues))
    # The toy league has 4 clubs, not 20, and Understat was not loaded.
    assert "[error] E0 2023: 4 clubs, expected 20" in text
    assert "results, expected 380" in text
    assert "xG on 0% of matches" in text
    assert "no lineups loaded" in text and "no all-competition schedules" in text
    assert "no SOFIFA ratings" not in text  # only said once line-ups exist to rate


def test_validate_empty_database(tmp_path):
    db = Database(tmp_path / "empty.db")
    issues = validate(coverage_report(db))
    db.close()
    assert issues and issues[0].level == "error"


def test_current_season_is_not_expected_to_be_complete():
    report = pd.DataFrame(
        [{"division": "E0", "season": 2025, "teams": 20, "fixtures": 380, "played": 120, "xg": 1.0, "ppda": 1.0,
          "possession": 1.0, "shots": 1.0, "closing_odds": 1.0, "any_odds": 1.0, "lineups": 1.0,
          "schedule_rows": 900, "rating_snapshots": 4},
         {"division": "E1", "season": 2025, "teams": 24, "fixtures": 552, "played": 200, "xg": 0.0, "ppda": 0.0,
          "possession": 0.0, "shots": 1.0, "closing_odds": 1.0, "any_odds": 1.0, "lineups": 0.0,
          "schedule_rows": 0, "rating_snapshots": 0}]
    )
    assert validate(report, current_season=2025) == []
