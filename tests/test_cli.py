import json

import numpy as np
import pandas as pd
import pytest

from football_predictor import cli
from football_predictor.cli import main, parse_seasons

from . import fake_sources as fake


@pytest.fixture()
def offline(monkeypatch):
    """Route the network commands to the fake sources, with 2023 as the current season."""
    monkeypatch.setattr(cli, "make_client", lambda args: (fake.FakeClient(), None))
    monkeypatch.setattr(cli, "current_season", lambda: fake.SEASON)


def test_parse_seasons():
    assert parse_seasons("2018-2020") == [2018, 2019, 2020]
    assert parse_seasons("2016,2019-2020") == [2016, 2019, 2020]


def test_demo_writes_reports(tmp_path, capsys):
    main(["--data-dir", str(tmp_path), "demo", "--features", "form", "--xg-weight", "0.3",
          "--refit-days", "28", "--compare"])
    out = capsys.readouterr().out
    assert "Feature presets on season 2023" in out
    assert "Forecast quality" in out and "brier_skill_vs_market" in out and "p_flat_roi_le_zero" in out
    reports = tmp_path / "reports"
    for name in ("pnl_2023.png", "calibration_2023.png", "bets_2023.csv", "metrics_2023.json", "comparison_2023.csv"):
        assert (reports / name).exists(), name
    assert json.loads((reports / "metrics_2023.json").read_text())["matches"] == 380


def test_ingest_status_and_predict(tmp_path, capsys, offline):
    main(["--data-dir", str(tmp_path), "ingest", "--seasons", "2023", "--sources", "football_data",
          "understat", "fbref"])  # returns normally: no source failed
    out = capsys.readouterr().out
    assert "E0" in out and "E1" in out and "FAILED" not in out

    main(["--data-dir", str(tmp_path), "status", "--audit", "4"])
    out = capsys.readouterr().out
    assert "4 clubs, expected 20" in out
    assert "every feature reproduced from pre-kickoff data only" in out

    main(["--data-dir", str(tmp_path), "predict", "Man City", "Wolves", "--as-of", "2023-09-30", "--l2", "2"])
    out = capsys.readouterr().out
    assert "Manchester City" in out and "Wolverhampton Wanderers" in out
    assert "Correct-score probabilities" in out

    main(["--data-dir", str(tmp_path), "predict", "--upcoming", "30", "--as-of", "2023-09-10", "--xg-weight", "0.3"])
    out = capsys.readouterr().out
    assert out.count("2023-09-16") == 4  # the two unplayed fixtures of the last round, in both forecast tables
    assert "best_ev" in out and "stake_pct" in out  # priced from fixtures.csv


def test_check_sources_reports_each_source(tmp_path, capsys, offline):
    main(["--data-dir", str(tmp_path), "check-sources", "--season", "2023",
          "--sources", "football_data", "understat", "fbref"])  # returns normally: everything parsed
    out = capsys.readouterr().out
    for source in ("football_data:E0", "football_data:E1", "understat", "fbref"):
        assert f"OK   {source}" in out


def test_check_sources_exits_non_zero_on_failure(tmp_path, capsys, monkeypatch, offline):
    broken = fake.FakeClient()
    broken.pages = {url: text for url, text in broken.pages.items() if "understat" not in url}
    monkeypatch.setattr(cli, "make_client", lambda args: (broken, None))
    with pytest.raises(SystemExit) as failed:
        main(["--data-dir", str(tmp_path), "check-sources", "--season", "2023", "--sources", "understat"])
    assert failed.value.code == 1
    assert "FAIL understat" in capsys.readouterr().out


def test_default_test_season_skips_an_unfinished_season():
    from football_predictor.cli import default_test_season

    data = pd.DataFrame(
        {
            "division": ["E0"] * 4 + ["E1"],
            "season": [2024, 2024, 2025, 2025, 2026],
            "home_goals": [1.0, 0.0, 2.0, np.nan, np.nan],
        }
    )
    assert default_test_season(data, "E0") == 2024
    assert default_test_season(data.dropna(), "E0") == 2025
    assert default_test_season(data, "E1") == 2026  # nothing complete yet: latest season


def test_training_divisions_default_to_the_predicted_division(tmp_path, capsys, offline):
    from argparse import Namespace

    main(["--data-dir", str(tmp_path), "ingest", "--seasons", "2023", "--sources", "football_data"])
    capsys.readouterr()

    def divisions(train):
        args = Namespace(synthetic=False, db=tmp_path / "football.db", division="E0", train_divisions=train)
        matches, _, _ = cli.load_inputs(args)
        return set(matches["division"])

    assert divisions(None) == {"E0"}
    assert divisions(["all"]) == {"E0", "E1"}
    assert divisions(["E1"]) == {"E0", "E1"}  # the predicted division is always kept


def test_backtest_over_several_seasons_reports_each(tmp_path, capsys):
    main(["--data-dir", str(tmp_path), "demo", "--test-seasons", "2022-2023", "--refit-days", "28",
          "--markets", "totals"])
    out = capsys.readouterr().out
    assert "Test seasons 2022-2023" in out and "markets=totals" in out
    assert "Per season" in out and "brier_over_market" in out
    assert (tmp_path / "reports" / "metrics_2022-2023.json").exists()
    metrics = json.loads((tmp_path / "reports" / "metrics_2022-2023.json").read_text())
    assert metrics["matches"] == 760 and metrics["bets_1x2"] == 0


def test_predict_shows_model_and_market_anchored_forecasts(tmp_path, capsys, offline):
    main(["--data-dir", str(tmp_path), "ingest", "--seasons", "2023", "--sources", "football_data", "understat"])
    capsys.readouterr()
    main(["--data-dir", str(tmp_path), "predict", "--upcoming", "30", "--as-of", "2023-09-10"])
    out = capsys.readouterr().out
    assert "Model forecast:" in out and "Market-anchored forecast" in out
    assert "source" in out and "market" in out  # the two unplayed fixtures carry prices from fixtures.csv
