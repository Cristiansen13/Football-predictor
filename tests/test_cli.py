import argparse
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


def test_scan_lists_prices_against_fair_value(tmp_path, capsys, offline):
    main(["--data-dir", str(tmp_path), "ingest", "--seasons", "2023", "--sources", "football_data", "understat"])
    capsys.readouterr()
    main(["--data-dir", str(tmp_path), "scan", "--upcoming", "30", "--as-of", "2023-09-10", "--ev-threshold", "-1"])
    out = capsys.readouterr().out
    assert "Fair value: anchored" in out
    for word in ("1x2", "totals", "handicap", "bet365", "pinnacle", "stake_pct"):
        assert word in out, word
    assert " max " not in out  # best-of-market price hidden unless asked


def test_shop_evaluates_price_shopping_without_a_model(tmp_path, capsys, offline):
    main(["--data-dir", str(tmp_path), "ingest", "--seasons", "2023", "--sources", "football_data", "--no-upcoming"])
    capsys.readouterr()
    main(["--data-dir", str(tmp_path), "shop", "--seasons", "2023", "--books", "named", "--ev-threshold", "0",
          "--each-book", "--resamples", "20"])
    out = capsys.readouterr().out
    assert "Price shopping, E0 seasons 2023: 10 matches" in out and "closing price of pinnacle (100%" in out
    assert "Bookmakers (named): bet365, betfair_exchange, bwin, pinnacle, william_hill; exchange commission 5%" in out
    # Only William Hill's closing home price beats Pinnacle's fair value, on five of the ten fixtures.
    assert "Per bookmaker" in out and "Each bookmaker on its own" in out
    bets = pd.read_csv(tmp_path / "reports" / "shopping_2023_named.csv")
    assert len(bets) == 5 and (bets["bookmaker"] == "william_hill").all() and (bets["selection"] == "home").all()
    assert (bets["odds"] == 2.25).all() and (bets["ev"] > 0.05).all() and (bets["fair_source"] == "pinnacle").all()
    assert "market maximum against the best named price" in out
    # The exchange alone, gross of commission, finds value on the over.
    main(["--data-dir", str(tmp_path), "shop", "--seasons", "2023", "--books", "exchange", "--commission", "0",
          "--ev-threshold", "0", "--markets", "totals", "--resamples", "0"])
    out = capsys.readouterr().out
    assert "exchange commission 0%" in out
    bets = pd.read_csv(tmp_path / "reports" / "shopping_2023_exchange.csv")
    assert len(bets) == 10 and (bets["selection"] == "over").all() and (bets["odds"] == 1.95).all()
    with pytest.raises(SystemExit, match="unknown bookmaker"):
        main(["--data-dir", str(tmp_path), "shop", "--seasons", "2023", "--books", "nosuch"])


def test_log_prices_board_and_forward_test(tmp_path, capsys, offline, monkeypatch):
    from football_predictor.etl.odds_api import OddsApiClient

    main(["--data-dir", str(tmp_path), "ingest", "--seasons", "2023", "--sources", "football_data",
          "--divisions", "E0", "SC0"])
    capsys.readouterr()
    session = fake.FakeOddsSession()
    monkeypatch.setattr(cli, "make_odds_client", lambda args: OddsApiClient("test-key", session=session))
    main(["--data-dir", str(tmp_path), "log-prices", "--divisions", "E0", "SC0", "EC", "--now", "2023-09-14T10:00"])
    out = capsys.readouterr().out
    assert "Prices logged at 2023-09-14 10:00 UTC" in out
    assert "The Odds API: 3 fixtures in 2 leagues from 10 bookmakers (EC not covered); credits used 2, remaining 498" in out
    assert "Football-Data fixtures.csv: 3 fixtures; bookmakers average, bet365, pinnacle" in out
    assert "18 prices stored; 3 fixtures resolved to the warehouse, 0 listed fixture(s) not" in out
    assert "Rule favourites: every soft bookmaker against the de-margined price of pinnacle, odds up to 3" in out
    assert "3 fixtures qualify; 3 new in the ledger, 3 bets recorded in all" in out
    assert "Wolverhampton Wanderers v Manchester United" in out and "william_hill" in out and "betvictor" in out
    assert out.count(" new") >= 3

    # Near kickoff, `due` asks the API only for the leagues with a fixture coming: one credit, not two.
    main(["--data-dir", str(tmp_path), "log-prices", "--divisions", "due", "--due-hours", "2",
          "--now", "2023-09-16T14:30"])
    out = capsys.readouterr().out
    assert ("The Odds API: 2 fixtures in 1 league(s) due within 2 h (E0) from 10 bookmakers; credits used 3, "
            "remaining 497") in out
    assert "0 new in the ledger, 3 bets recorded in all" in out
    main(["--data-dir", str(tmp_path), "log-prices", "--sources", "odds_api", "--divisions", "due", "--now", "2023-10-01"])
    out = capsys.readouterr().out
    assert "The Odds API: no league has a fixture due within 12 h, nothing asked" in out and "Nothing to log" in out

    main(["--data-dir", str(tmp_path), "board", "--now", "2023-09-14T12:00", "--days", "3"])
    out = capsys.readouterr().out
    assert "Value board at 2023-09-14 12:00 UTC, fixtures in the next 3 days" in out
    assert "bet_at" in out and out.count("2023-09-14 10:00") >= 3 and "age_h" in out
    main(["--data-dir", str(tmp_path), "board", "--now", "2023-09-14T12:00", "--every-price", "--max-odds", "2"])
    assert "bet365" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="No logged prices"):
        main(["--data-dir", str(tmp_path), "board", "--now", "2023-10-01"])

    main(["--data-dir", str(tmp_path), "forward-test", "--resamples", "10"])
    out = capsys.readouterr().out
    assert "Forward test of rule favourites: 3 bets recorded from 2023-09-14 to 2023-09-14 on 2 league(s); 0 settled, 3 pending" in out
    # The due run logged BetVictor's price on Forest v City again before kickoff (the other two had kicked off).
    assert ("Price availability: 1 bet(s) have a later price from their bookmaker logged before kickoff; the price "
            "taken, or a better one, was still there in 100% of them") in out
    assert "Pending (3" in out and "Dundee v Rangers" in out
    ledger = pd.read_csv(tmp_path / "reports" / "forward_test_favourites.csv")
    assert len(ledger) == 3 and (ledger["status"] == "pending").all()
    with pytest.raises(SystemExit, match="No bets in the ledger"):
        main(["--data-dir", str(tmp_path), "forward-test", "--rule", "nosuch"])


def test_log_prices_needs_a_key_for_the_odds_api(tmp_path, capsys, offline, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ODDS_API_KEY"):
        main(["--data-dir", str(tmp_path), "log-prices"])
    # The key may sit in the data directory instead, which git ignores.
    (tmp_path / "odds_api.key").write_text("test-key\n", encoding="utf-8")
    assert cli.make_odds_client(argparse.Namespace(data_dir=tmp_path))._api_key == "test-key"
    (tmp_path / "odds_api.key").unlink()
    main(["--data-dir", str(tmp_path), "ingest", "--seasons", "2023", "--sources", "football_data"])
    capsys.readouterr()
    main(["--data-dir", str(tmp_path), "log-prices", "--sources", "football_data", "--divisions", "E0",
          "--alias", "Wolverhampton=Wolverhampton Wanderers", "--now", "2023-09-14T10:00"])
    out = capsys.readouterr().out
    # Bet365's prices in fixtures.csv never beat Pinnacle's there, so nothing is bet; the prices are logged all the same.
    assert "Football-Data fixtures.csv: 2 fixtures" in out and "No price qualifies in this snapshot" in out
    assert "6 prices stored; 2 fixtures resolved" in out
    with pytest.raises(SystemExit, match="expected NAME=CLUB"):
        main(["--data-dir", str(tmp_path), "log-prices", "--sources", "football_data", "--alias", "X=No Such Club"])


def test_shop_across_divisions_and_at_opening_prices(tmp_path, capsys, offline):
    main(["--data-dir", str(tmp_path), "ingest", "--seasons", "2023", "--sources", "football_data", "--no-upcoming",
          "--divisions", "E0", "SC0"])
    capsys.readouterr()
    # At Football-Data's earlier prices the exchange, gross of commission, beats Pinnacle on every fixture.
    main(["--data-dir", str(tmp_path), "shop", "--seasons", "2023", "--divisions", "all", "--market-time", "opening",
          "--books", "exchange", "--commission", "0", "--ev-threshold", "0", "--markets", "1x2", "--resamples", "0"])
    out = capsys.readouterr().out
    assert "Price shopping, E0 SC0 seasons 2023: 20 matches" in out and "opening price of pinnacle" in out
    assert "Per division" in out and "ev_%" in out and "clv_%" in out
    bets = pd.read_csv(tmp_path / "reports" / "shopping_2023_all_exchange_opening.csv")
    assert len(bets) == 20 and set(bets["division"]) == {"E0", "SC0"} and bets["match_id"].is_unique
    assert (bets["odds"].isin([2.16, 3.60, 3.95])).all() and (bets["clv"] != bets["ev"]).all()
