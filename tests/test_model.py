import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import poisson

from football_predictor.features import build_features
from football_predictor.models import DixonColesModel, market_probabilities, outcome_probabilities, score_matrices
from football_predictor.models.dixon_coles import time_decay_weights
from football_predictor.synthetic import simulate_league


@pytest.fixture(scope="module")
def league():
    return simulate_league(seasons=range(2018, 2023), seed=11, return_truth=True)


def test_score_matrix_matches_independent_poisson_without_rho():
    matrix = score_matrices(1.4, 1.1, rho=0.0, max_goals=6, normalize=False)[0]
    expected = np.outer(poisson.pmf(range(7), 1.4), poisson.pmf(range(7), 1.1))
    np.testing.assert_allclose(matrix, expected)


def test_dixon_coles_adjustment_only_touches_low_scores():
    lam, mu, rho = 1.3, 0.9, -0.12
    base = score_matrices(lam, mu, 0.0, normalize=False)[0]
    adjusted = score_matrices(lam, mu, rho, normalize=False)[0]
    ratio = adjusted / base
    assert ratio[0, 0] == pytest.approx(1 - lam * mu * rho)
    assert ratio[0, 1] == pytest.approx(1 + lam * rho)
    assert ratio[1, 0] == pytest.approx(1 + mu * rho)
    assert ratio[1, 1] == pytest.approx(1 - rho)
    ratio[:2, :2] = 1.0
    np.testing.assert_allclose(ratio, 1.0)
    # Negative rho inflates 0-0 and 1-1 draws.
    assert adjusted[0, 0] > base[0, 0] and adjusted[1, 1] > base[1, 1]


def test_outcome_aggregation():
    matrices = score_matrices(np.array([1.5, 0.7]), np.array([1.0, 1.8]), rho=-0.1)
    np.testing.assert_allclose(matrices.sum(axis=(1, 2)), 1.0)
    outcomes = outcome_probabilities(matrices)
    m = matrices[0]
    home = sum(m[x, y] for x in range(7) for y in range(7) if x > y)
    draw = sum(m[x, x] for x in range(7))
    assert outcomes[0] == pytest.approx([home, draw, 1 - home - draw])
    markets = market_probabilities(matrices)
    assert markets.loc[0, "p_over_2_5"] + markets.loc[0, "p_under_2_5"] == pytest.approx(1.0)
    assert outcomes[1, 2] > outcomes[1, 0]  # the stronger away side is favoured


def test_time_decay_weights():
    kickoff = pd.Series(pd.to_datetime(["2024-01-01", "2023-01-01"]))
    w = time_decay_weights(kickoff, pd.Timestamp("2024-01-01"), xi=0.002)
    assert w == pytest.approx([1.0, np.exp(-0.002 * 365)])


def test_fit_recovers_generating_parameters(league):
    data, truth = league
    model = DixonColesModel(xi=0.003).fit(data)
    assert model.summary_.converged
    assert model.home_advantage_ == pytest.approx(truth.home_advantage, abs=0.08)
    assert model.rho_ == pytest.approx(truth.rho, abs=0.08)
    np.testing.assert_allclose(model.attack_.sum(), 0, atol=1e-8)
    fitted = model.ratings().set_index("team")
    true_attack = truth.attack.iloc[-1].reindex(fitted.index)
    true_defence = truth.defence.iloc[-1].reindex(fitted.index)
    assert np.corrcoef(fitted["attack"], true_attack)[0, 1] > 0.85
    assert np.corrcoef(fitted["defence"], true_defence)[0, 1] > 0.85


def test_fit_only_uses_matches_before_as_of(league):
    data, _ = league
    as_of = data["kickoff"].iloc[1000]
    model = DixonColesModel().fit(data, as_of=as_of)
    assert model.summary_.n_matches == (data["kickoff"] < as_of).sum()
    tampered = data.copy()
    tampered.loc[tampered["kickoff"] >= as_of, ["home_goals", "away_goals"]] = 7
    again = DixonColesModel().fit(tampered, as_of=as_of)
    np.testing.assert_allclose(model.attack_, again.attack_)


def test_analytic_gradient_matches_finite_differences(league):
    data, _ = league
    features = build_features(data[data["season"] == 2022])
    model = DixonColesModel.from_preset("form", xi=0.004).fit(features)
    played = features.dropna(subset=["home_goals"])
    weights = time_decay_weights(played["kickoff"], model.summary_.as_of, model.xi)
    index = model.team_index_
    design = {
        "n_teams": len(index),
        "home": played["home_team"].map(index).to_numpy(),
        "away": played["away_team"].map(index).to_numpy(),
        "x": played["home_goals"].to_numpy(float),
        "y": played["away_goals"].to_numpy(float),
        "weights": weights,
        "total_weight": weights.sum(),
        "att_home": model._feature_block(played, "home", model.attack_features),
        "att_away": model._feature_block(played, "away", model.attack_features),
        "def_home": model._feature_block(played, "home", model.defence_features),
        "def_away": model._feature_block(played, "away", model.defence_features),
    }
    rng = np.random.default_rng(0)
    params = rng.normal(0, 0.15, 3 + 2 * (len(index) - 1) + 2)
    params[2] = -0.08
    numeric = approx_fprime(params, lambda p: model._objective(p, design)[0], 1e-7)
    np.testing.assert_allclose(model._objective(params, design)[1], numeric, atol=1e-5)


def test_predict_outputs_and_unseen_teams(league):
    data, _ = league
    model = DixonColesModel().fit(data)
    fixtures = pd.DataFrame({"home_team": ["Team 01", "Promoted FC"], "away_team": ["Team 02", "Team 01"]})
    predictions = model.predict(fixtures)
    np.testing.assert_allclose(predictions[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)
    assert (predictions["exp_home_goals"] > 0).all()
    ratings = model.ratings()
    weakest_attack = np.quantile(model.attack_, model.unseen_quantile)
    lam, _ = model.expected_goals(fixtures.iloc[[1]])
    assert lam[0] == pytest.approx(
        np.exp(model.intercept_ + model.home_advantage_ + weakest_attack + model.defence_[model.team_index_["Team 01"]])
    )
    assert len(ratings) == 20

    matrix = model.score_matrix("Team 01", "Team 02")
    assert matrix.shape == (7, 7)
    assert matrix.to_numpy().sum() == pytest.approx(1.0)


def test_warm_start_reaches_same_optimum(league):
    data, _ = league
    as_of = data["kickoff"].iloc[1500]
    cold = DixonColesModel().fit(data, as_of=as_of)
    previous = DixonColesModel().fit(data, as_of=data["kickoff"].iloc[1400])
    warm = DixonColesModel().fit(data, as_of=as_of, warm_start=previous)
    np.testing.assert_allclose(cold.attack_, warm.attack_, atol=1e-4)
    assert warm.rho_ == pytest.approx(cold.rho_, abs=1e-4)


def test_features_enter_the_rates(league):
    data, _ = league
    features = build_features(data)
    model = DixonColesModel.from_preset("full").fit(features)
    assert set(model.theta_attack_) == {"ewma_xg_for", "fatigue_penalty", "lineup_fraction"}
    assert model.theta_attack_["lineup_fraction"] == 0.0  # no lineup data: column is constant
    row = features.dropna(subset=["home_goals"]).tail(1)
    base, _ = model.expected_goals(row)
    boosted = row.assign(home_ewma_xg_for=row["home_ewma_xg_for"] + 0.5)
    lam, _ = model.expected_goals(boosted)
    assert np.sign(lam[0] - base[0]) == np.sign(model.theta_attack_["ewma_xg_for"])
