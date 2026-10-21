import json
from functools import partial

import numpy as np
import pandas as pd
import pytest

from football_predictor.evaluation import (
    BettingStrategy,
    accuracy,
    bootstrap_interval,
    brier_score,
    calibration_table,
    expected_calibration_error,
    expected_value,
    implied_probabilities,
    kelly_fraction,
    log_loss,
    outcome_index,
    plot_pnl,
    ranked_probability_score,
    run_backtest,
    save_report,
    simulate_betting,
    skill_score,
    tune_hyperparameters,
    tune_xi,
    walk_forward_predictions,
)
from football_predictor.models import DixonColesModel
from football_predictor.synthetic import simulate_league


def test_scoring_rules():
    probabilities = np.array([[1.0, 0.0, 0.0], [0.5, 0.3, 0.2]])
    outcomes = np.array([0, 2])
    assert brier_score(probabilities, outcomes) == pytest.approx((0 + 0.25 + 0.09 + 0.64) / 2)
    assert brier_score(probabilities[:1], np.array([2])) == pytest.approx(2.0)
    assert ranked_probability_score(probabilities[:1], outcomes[:1]) == 0.0
    # RPS: a draw forecast is less wrong than an away forecast when the home side wins.
    draw_heavy = ranked_probability_score(np.array([[0.0, 1.0, 0.0]]), np.array([0]))
    away_heavy = ranked_probability_score(np.array([[0.0, 0.0, 1.0]]), np.array([0]))
    assert draw_heavy < away_heavy
    assert log_loss(probabilities[1:], np.array([2])) == pytest.approx(-np.log(0.2))
    np.testing.assert_array_equal(outcome_index([2, 1, 0], [1, 1, 3]), [0, 1, 2])


def test_market_and_staking_formulas():
    implied = implied_probabilities(np.array([[2.0, 4.0, 4.0]]) / 1.05)
    np.testing.assert_allclose(implied, [[0.5, 0.25, 0.25]])
    assert expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.40, 2.0) == 0.0


def _predictions(rows):
    frame = pd.DataFrame(rows, columns=["kickoff", "p_home", "p_draw", "p_away", "odds_home", "odds_draw", "odds_away", "outcome"])
    frame["kickoff"] = pd.to_datetime(frame["kickoff"])
    frame["match_id"] = [f"m{i}" for i in range(len(frame))]
    frame["home_team"], frame["away_team"] = "H", "A"
    return frame


def test_betting_simulation_settles_and_compounds():
    predictions = _predictions(
        [
            ("2024-01-06 15:00", 0.60, 0.25, 0.15, 2.0, 3.5, 6.0, 0),  # home EV +0.20, wins
            ("2024-01-06 17:30", 0.30, 0.30, 0.40, 3.0, 3.3, 2.2, 1),  # away EV -0.12... home EV -0.10 -> no bet
            ("2024-01-13 15:00", 0.20, 0.30, 0.50, 4.0, 3.5, 2.5, 0),  # away EV +0.25, loses
        ]
    )
    strategy = BettingStrategy(ev_threshold=0.05, kelly_multiplier=0.5, max_stake_fraction=1.0)
    bets, equity = simulate_betting(predictions, strategy, initial_bankroll=100.0)
    assert list(bets["selection"]) == ["home", "away"]
    first_stake = 100 * 0.5 * (0.6 * 2 - 1) / (2 - 1)
    assert bets.loc[0, "stake"] == pytest.approx(first_stake)
    assert bets.loc[0, "profit"] == pytest.approx(first_stake)
    bankroll = 100 + first_stake
    second_stake = bankroll * 0.5 * (0.5 * 2.5 - 1) / 1.5
    assert bets.loc[1, "stake"] == pytest.approx(second_stake)
    assert equity["bankroll"].iloc[-1] == pytest.approx(bankroll - second_stake)


def test_daily_exposure_cap():
    predictions = _predictions([("2024-01-06 15:00", 0.9, 0.05, 0.05, 2.0, 10.0, 10.0, 0)] * 4)
    strategy = BettingStrategy(kelly_multiplier=1.0, max_stake_fraction=0.5, max_daily_exposure=0.4)
    bets, _ = simulate_betting(predictions, strategy, initial_bankroll=100.0)
    assert bets["stake"].sum() == pytest.approx(40.0)


@pytest.fixture(scope="module")
def league():
    return simulate_league(seasons=range(2020, 2023), seed=5)


def test_walk_forward_never_trains_on_the_future(league):
    fitted_with = []

    class RecordingModel(DixonColesModel):
        def fit(self, matches, as_of=None, warm_start=None):
            fitted_with.append(pd.Timestamp(as_of))
            return super().fit(matches, as_of=as_of, warm_start=warm_start)

    start = league.loc[league["season"] == 2022, "kickoff"].min().normalize()
    predictions = walk_forward_predictions(league, RecordingModel, start, refit_days=14)
    assert len(predictions) == 380
    assert (predictions["fitted_as_of"] <= predictions["kickoff"]).all()
    assert (predictions["kickoff"] - predictions["fitted_as_of"] < pd.Timedelta(days=14)).all()
    assert fitted_with == sorted(fitted_with)


def test_backtest_end_to_end(league, tmp_path):
    start = league.loc[league["season"] == 2022, "kickoff"].min().normalize()
    result = run_backtest(league, partial(DixonColesModel, xi=0.003), start)
    m = result.metrics
    assert m["matches"] == 380 and m["matches_with_odds"] == 380
    # The model must beat a uniform forecast (Brier 0.667) by a wide margin.
    assert m["brier_model"] < 0.62
    assert m["bets"] == len(result.bets) > 0
    assert m["final_bankroll"] == pytest.approx(1000 + result.bets["profit"].sum())
    assert (result.bets["ev"] > 0.05).all()
    path = plot_pnl(result, tmp_path / "pnl.png")
    assert path.stat().st_size > 10_000


def test_tune_xi_ranks_candidates(league):
    start = league.loc[league["season"] == 2021, "kickoff"].min().normalize()
    end = league.loc[league["season"] == 2022, "kickoff"].min().normalize()
    scores = tune_xi(league[league["kickoff"] < end], [0.0005, 0.003], start, end, refit_days=28)
    assert list(scores.columns[:2]) == ["xi", "matches"]
    assert scores["brier_model"].is_monotonic_increasing


def test_calibration_metrics():
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet([4, 2.5, 3], size=4000)
    outcomes = np.array([rng.choice(3, p=p) for p in probabilities])
    assert expected_calibration_error(probabilities, outcomes) < 0.03  # calibrated by construction
    overconfident = probabilities ** 3 / (probabilities ** 3).sum(axis=1, keepdims=True)
    assert expected_calibration_error(overconfident, outcomes) > 0.05
    table = calibration_table(probabilities, outcomes, bins=5)
    assert set(table["outcome"]) == {"home", "draw", "away"}
    assert table.groupby("outcome")["count"].sum().eq(4000).all()
    assert accuracy(np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]]), np.array([0, 1])) == 0.5


def test_bootstrap_interval_covers_the_mean():
    values = np.random.default_rng(1).normal(0.2, 1.0, 500)
    low, high, stats = bootstrap_interval(np.mean, values, n_resamples=500)
    assert low < values.mean() < high and len(stats) == 500
    assert skill_score(0.5, 0.6) == pytest.approx(1 / 6)


def test_report_metrics_and_files(league, tmp_path):
    start = league.loc[league["season"] == 2022, "kickoff"].min().normalize()
    result = run_backtest(league, partial(DixonColesModel, xi=0.003), start, refit_days=28, n_resamples=200)
    m = result.metrics
    for key in ("accuracy_model", "ece_model", "score_log_loss_model", "brier_skill_vs_base_rate",
                "brier_skill_vs_market", "brier_diff_ci_low", "flat_roi_ci_low", "p_flat_roi_le_zero",
                "bets_home", "roi_away"):
        assert key in m, key
    assert m["brier_diff_ci_low"] <= m["brier_diff_vs_market"] <= m["brier_diff_ci_high"]
    assert m["brier_skill_vs_base_rate"] > 0  # beats always forecasting the historical frequencies
    assert m["bets_home"] + m["bets_draw"] + m["bets_away"] + m["bets_over"] + m["bets_under"] == m["bets"]
    assert m["bets_1x2"] + m["bets_totals"] == m["bets"]
    paths = save_report(result, tmp_path, "2022")
    assert all(path.exists() for path in paths.values())
    assert json.loads(paths["metrics"].read_text())["matches"] == 380


def test_backtest_scores_only_requested_divisions():
    data = simulate_league(seasons=[2021, 2022], second_division_teams=20, seed=3)
    start = data.loc[data["season"] == 2022, "kickoff"].min().normalize()
    predictions = walk_forward_predictions(data, DixonColesModel, start, refit_days=28, divisions=["E0"])
    assert set(predictions["division"]) == {"E0"} and len(predictions) == 380


def test_tune_hyperparameters_grid(league):
    start = league.loc[league["season"] == 2021, "kickoff"].min().normalize()
    end = league.loc[league["season"] == 2022, "kickoff"].min().normalize()
    scores = tune_hyperparameters(league[league["kickoff"] < end], {"xi": [0.001, 0.004], "xg_weight": [0.0, 0.5]},
                                  start, end, refit_days=56)
    assert len(scores) == 4 and {"xi", "xg_weight", "brier_model"} <= set(scores.columns)
    assert scores["brier_model"].is_monotonic_increasing


def test_tune_hyperparameters_parallel_matches_serial(league):
    start = league.loc[league["season"] == 2021, "kickoff"].min().normalize()
    end = league.loc[league["season"] == 2022, "kickoff"].min().normalize()
    grid = {"xi": [0.001, 0.004], "xg_weight": [0.0, 0.5]}
    serial = tune_hyperparameters(league[league["kickoff"] < end], grid, start, end, refit_days=56)
    parallel = tune_hyperparameters(league[league["kickoff"] < end], grid, start, end, refit_days=56, n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_two_way_markets_are_scored_and_bet(league):
    from football_predictor.evaluation.backtest import BettingStrategy, evaluate_forecasts, simulate_betting

    start = league.loc[league["season"] == 2022, "kickoff"].min().normalize()
    predictions = walk_forward_predictions(league, DixonColesModel, start, refit_days=56, divisions=["E0"])
    assert set(predictions["over"].unique()) <= {0, 1} and set(predictions["btts"].unique()) <= {0, 1}
    metrics = evaluate_forecasts(predictions, n_resamples=200)
    assert 0 < metrics["brier_over_model"] < 0.25 and 0 < metrics["brier_btts_model"] < 0.25
    assert metrics["matches_with_totals_odds"] == len(predictions)
    assert "brier_over_market" in metrics and "over_diff_ci_low" in metrics

    both = simulate_betting(predictions, BettingStrategy(ev_threshold=0.0))[0]
    assert set(both["market"]) == {"1x2", "totals"}
    assert set(both.loc[both["market"] == "totals", "selection"]) <= {"over", "under"}
    totals_only = simulate_betting(predictions, BettingStrategy(ev_threshold=0.0, markets=("totals",)))[0]
    assert (totals_only["market"] == "totals").all() and len(totals_only) > 0
    # Settlement follows the goal total, not the 1X2 outcome.
    settled = totals_only.merge(predictions[["match_id", "over"]], on="match_id")
    assert (settled["won"] == ((settled["selection"] == "over") == (settled["over"] == 1))).all()
    classic = simulate_betting(predictions, BettingStrategy(ev_threshold=0.0, markets=("1x2",)))[0]
    assert (classic["market"] == "1x2").all()


def test_demargining_methods():
    odds = np.array([[1.25, 6.0, 12.0], [2.9, 3.3, 2.6]])
    for method in ("proportional", "power", "shin"):
        probs = implied_probabilities(odds, method)
        assert np.allclose(probs.sum(axis=1), 1.0) and (probs > 0).all()
    proportional, power, shin = (implied_probabilities(odds, m) for m in ("proportional", "power", "shin"))
    # Power and Shin take the margin mostly from the longshots: the favourite gains, the 12.0 shot loses.
    assert power[0, 0] > proportional[0, 0] and shin[0, 0] > proportional[0, 0]
    assert power[0, 2] < proportional[0, 2] and shin[0, 2] < proportional[0, 2]
    assert np.allclose(power, shin, atol=0.01)
    fair = np.array([[2.0, 4.0, 4.0]])  # no margin: every method returns the raw probabilities
    for method in ("proportional", "power", "shin"):
        assert np.allclose(implied_probabilities(fair, method), [[0.5, 0.25, 0.25]], atol=1e-6)


def _priced_predictions():
    """Six fixtures with closing, opening and best prices, where the model and the market disagree."""
    kickoff = pd.date_range("2024-01-06 15:00", periods=6, freq="7D")
    frame = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(6)], "kickoff": kickoff, "home_team": "A", "away_team": "B",
            "p_home": [0.60, 0.30, 0.45, 0.50, 0.40, 0.35], "p_draw": [0.22, 0.30, 0.27, 0.25, 0.28, 0.30],
            "p_away": [0.18, 0.40, 0.28, 0.25, 0.32, 0.35],
            "odds_home": [1.9, 3.2, 2.2, 2.0, 2.5, 2.8], "odds_draw": [3.6, 3.3, 3.4, 3.5, 3.3, 3.2],
            "odds_away": [4.2, 2.3, 3.4, 3.8, 2.9, 2.6],
            "open_home": [2.0, 3.0, 2.3, 2.1, 2.4, 2.9], "open_draw": [3.5, 3.4, 3.4, 3.4, 3.4, 3.2],
            "open_away": [4.0, 2.4, 3.3, 3.6, 3.0, 2.5],
            "max_home": [2.05, 3.4, 2.3, 2.1, 2.6, 2.95], "max_draw": [3.8, 3.5, 3.6, 3.7, 3.5, 3.4],
            "max_away": [4.5, 2.4, 3.6, 4.0, 3.05, 2.7],
            "p_over_2_5": [0.6, 0.5, 0.55, 0.45, 0.5, 0.52], "p_under_2_5": [0.4, 0.5, 0.45, 0.55, 0.5, 0.48],
            "odds_over_2_5": [1.8, 2.0, 1.9, 2.1, 2.0, 1.95], "odds_under_2_5": [2.0, 1.8, 1.9, 1.75, 1.8, 1.85],
            "open_over_2_5": [1.85, 2.05, 1.9, 2.1, 1.95, 2.0], "open_under_2_5": [1.95, 1.75, 1.9, 1.75, 1.85, 1.8],
            "max_over_2_5": [1.9, 2.1, 1.95, 2.15, 2.05, 2.0], "max_under_2_5": [2.1, 1.9, 1.95, 1.8, 1.9, 1.9],
            "outcome": [0, 2, 1, 0, 2, 0], "over": [1, 0, 1, 0, 1, 1],
        }
    )
    return frame


def test_fair_value_source_and_price_options():
    from football_predictor.evaluation.backtest import BettingStrategy, MARKET_DEMARGIN, simulate_betting

    frame = _priced_predictions()
    # Market as fair value at the closing reference price: the margin makes every EV negative.
    bets, _ = simulate_betting(frame, BettingStrategy(ev_threshold=0.0, fair="market", price="closing"))
    assert bets.empty
    # Market as fair value at the best price: bets exactly where the best price beats the fair one.
    bets, _ = simulate_betting(frame, BettingStrategy(ev_threshold=0.0, fair="market", price="max"))
    assert len(bets) > 0 and (bets["clv"] > 0).all()
    fair = implied_probabilities(frame[["odds_home", "odds_draw", "odds_away"]].to_numpy(), MARKET_DEMARGIN)
    best = frame[["max_home", "max_draw", "max_away"]].to_numpy()
    expected_1x2 = ((fair * best - 1) > 0).any(axis=1)
    assert set(bets.loc[bets["market"] == "1x2", "match_id"]) == set(frame.loc[expected_1x2, "match_id"])
    assert (bets["ev"] == bets["clv"]).all()  # fair value is the closing line, so EV is the closing-line value
    # Model as fair value at the opening price: settled at the opening odds.
    bets, _ = simulate_betting(frame, BettingStrategy(ev_threshold=0.0, fair="model", price="opening", markets=("1x2",)))
    home = bets[bets["selection"] == "home"].merge(frame[["match_id", "open_home"]], on="match_id")
    assert len(home) > 0 and (home["odds"] == home["open_home"]).all()
    won = home[home["won"]]
    assert np.allclose(won["profit"], won["stake"] * (won["open_home"] - 1))
    # A blend is the average of the model and the closing-line probability for the selection picked.
    blend_bets, _ = simulate_betting(frame, BettingStrategy(ev_threshold=-1.0, fair="blend", markets=("1x2",)))
    assert len(blend_bets) == 6
    closing = pd.DataFrame(fair, columns=["home", "draw", "away"], index=frame["match_id"])
    model = frame.set_index("match_id")[["p_home", "p_draw", "p_away"]].set_axis(["home", "draw", "away"], axis=1)
    for bet in blend_bets.itertuples(index=False):
        expected = 0.5 * model.loc[bet.match_id, bet.selection] + 0.5 * closing.loc[bet.match_id, bet.selection]
        assert bet.p_model == pytest.approx(expected)
    # Asking for a price the data lacks falls back to the closing reference.
    sparse = frame.drop(columns=[c for c in frame if c.startswith(("open_", "max_"))])
    fallback, _ = simulate_betting(sparse, BettingStrategy(ev_threshold=0.0, price="max", markets=("1x2",)))
    closing, _ = simulate_betting(sparse, BettingStrategy(ev_threshold=0.0, price="closing", markets=("1x2",)))
    pd.testing.assert_frame_equal(fallback, closing)


def test_walk_forward_reports_anchored_forecast_and_score_hits(league):
    from football_predictor.evaluation.backtest import evaluate_forecasts

    start = league.loc[league["season"] == 2022, "kickoff"].min().normalize()
    predictions = walk_forward_predictions(league, DixonColesModel, start, refit_days=56, divisions=["E0"])
    for column in ("anchored_p_home", "anchored_source", "modal_score_model", "modal_score_anchored",
                   "score_hit_model", "score_prob_anchored"):
        assert column in predictions, column
    assert set(predictions["anchored_source"]) == {"market"}
    assert predictions["modal_score_model"].str.match(r"^\d-\d$").all()
    metrics = evaluate_forecasts(predictions, n_resamples=0)
    assert abs(metrics["brier_anchored"] - metrics["brier_market"]) < 0.01  # anchored on the same prices
    assert 0.0 < metrics["score_hit_model"] < 0.3 and 0.0 < metrics["score_hit_anchored"] < 0.3
    assert metrics["score_log_loss_anchored"] > 0 and metrics["accuracy_anchored"] > 0.3
