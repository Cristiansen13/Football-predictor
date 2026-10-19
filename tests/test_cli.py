from football_predictor.cli import main, parse_seasons
from football_predictor.db import Database
from football_predictor.etl.pipeline import Pipeline

from . import fake_sources as fake


def test_parse_seasons():
    assert parse_seasons("2018-2020") == [2018, 2019, 2020]
    assert parse_seasons("2016,2019-2020") == [2016, 2019, 2020]


def test_demo_writes_reports(tmp_path, capsys):
    main(["--data-dir", str(tmp_path), "demo", "--features", "form", "--refit-days", "28"])
    out = capsys.readouterr().out
    assert "brier_model" in out and "roi" in out
    assert (tmp_path / "reports" / "pnl_2023.png").exists()
    assert (tmp_path / "reports" / "bets_2023.csv").exists()


def test_predict_from_warehouse(tmp_path, capsys):
    db = Database(tmp_path / "football.db")
    Pipeline(db, fake.FakeClient()).run([fake.SEASON], ["football_data", "understat"])
    db.close()
    main(["--data-dir", str(tmp_path), "predict", "Man City", "Wolves", "--as-of", "2023-09-30", "--l2", "2"])
    out = capsys.readouterr().out
    assert "Manchester City" in out and "Wolverhampton Wanderers" in out
    assert "Correct-score probabilities" in out

    main(["--data-dir", str(tmp_path), "predict", "--upcoming", "30", "--as-of", "2023-09-10"])
    out = capsys.readouterr().out
    assert out.count("2023-09-16") == 2  # the two unplayed fixtures of the last round
