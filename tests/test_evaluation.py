from functools import partial

import numpy as np
import pandas as pd
import pytest

from football_predictor.evaluation import (
    BettingStrategy,
    brier_score,
    expected_value,
    implied_probabilities,
    kelly_fraction,
    log_loss,
    outcome_index,
    plot_pnl,
    ranked_probability_score,
    run_backtest,
    simulate_betting,
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
