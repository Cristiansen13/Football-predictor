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


@pytest.mark.parametrize("xg_weight", [0.0, 0.5])
def test_analytic_gradient_matches_finite_differences(xg_weight):
    data = simulate_league(seasons=[2021, 2022], second_division_teams=20, seed=4)
    features = build_features(data)
    model = DixonColesModel.from_preset("form", xi=0.004, xg_weight=xg_weight).fit(features)
    played = features.dropna(subset=["home_goals"])
    weights = time_decay_weights(played["kickoff"], model.summary_.as_of, model.xi)
    design = model._design(played, weights)
    assert design["n_div"] == 2
    assert (design["alpha"] > 0).sum() == (xg_weight > 0) * played["home_xg"].notna().sum()
    rng = np.random.default_rng(0)
    params = rng.normal(0, 0.15, 2 * 2 + 1 + 2 * (design["n_teams"] - 1) + 2)
    params[4] = -0.08  # rho
    numeric = approx_fprime(params, lambda p: model._objective(p, design)[0], 1e-7)
    np.testing.assert_allclose(model._objective(params, design)[1], numeric, atol=1e-5)


def test_xg_weighting_tracks_true_strength_better(league):
    data, truth = league
    as_of = data["kickoff"].iloc[-1]
    goals_only = DixonColesModel(xi=0.004).fit(data, as_of=as_of)
    with_xg = DixonColesModel(xi=0.004, xg_weight=0.6).fit(data, as_of=as_of)
    true_rating = (truth.attack.iloc[-1] - truth.defence.iloc[-1])

    def correlation(model):
        fitted = model.ratings().set_index("team")["rating"]
        return np.corrcoef(fitted, true_rating.reindex(fitted.index))[0, 1]

    assert correlation(with_xg) > correlation(goals_only)


def test_divisions_share_one_strength_scale():
    data, truth = simulate_league(seasons=range(2018, 2023), second_division_teams=20, seed=8, return_truth=True)
    model = DixonColesModel(xi=0.003, xg_weight=0.3).fit(data)
    assert set(model.summary_.divisions) == {"E0", "E1"}
    assert model.default_division_ == "E0"
    # Each division keeps its own scoring level; the truth has a single one.
    assert model.division_intercepts_["E0"] == pytest.approx(model.division_intercepts_["E1"], abs=0.15)
    fitted = model.ratings().set_index("team")["rating"]
    true_rating = (truth.attack.iloc[-1] - truth.defence.iloc[-1]).reindex(fitted.index)
    assert np.corrcoef(fitted, true_rating)[0, 1] > 0.9
    # Clubs currently in E0 are rated above those in E1 on the shared scale.
    last = data[data["season"] == 2022]
    top = set(last.loc[last["division"] == "E0", "home_team"])
    assert fitted[fitted.index.isin(top)].mean() > fitted[~fitted.index.isin(top)].mean()


def test_second_division_data_rates_promoted_clubs():
    """Promoted clubs' strength relative to the incumbents is closer to the truth
    when Championship matches are in the training data."""
    errors = {}
    for name, keep in (("E0 only", ["E0"]), ("with E1", ["E0", "E1"])):
        gaps = []
        for seed in (1, 2, 3):
            data, truth = simulate_league(seasons=range(2018, 2023), second_division_teams=20, seed=seed,
                                          return_truth=True)
            start = data.loc[data["season"] == 2022, "kickoff"].min()
            now = set(data.loc[(data["season"] == 2022) & (data["division"] == "E0"), "home_team"])
            before = set(data.loc[(data["season"] == 2021) & (data["division"] == "E0"), "home_team"])
            promoted, incumbents = sorted(now - before), sorted(now & before)
            model = DixonColesModel(xi=0.003).fit(data[data["division"].isin(keep)], as_of=start)

            def fitted(teams):
                attack, defence = model._team_strengths(pd.Series(teams))
                return attack - defence

            snapshot = truth.attack.index[truth.attack.index <= start][-1]
            true = truth.attack.loc[snapshot] - truth.defence.loc[snapshot]
            fitted_gap = fitted(promoted) - fitted(incumbents).mean()
            true_gap = true[promoted].to_numpy() - true[incumbents].mean()
            gaps.append(np.abs(fitted_gap - true_gap).mean())
        errors[name] = np.mean(gaps)
    assert errors["with E1"] < errors["E0 only"]


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


def test_rates_from_market_probabilities_round_trip():
    from football_predictor.models import anchored_forecast, rates_from_probabilities
    from football_predictor.models.probability import market_probabilities, score_matrices

    lam, mu, rho = np.array([1.7, 0.9, 2.4]), np.array([1.1, 1.6, 0.6]), -0.06
    markets = market_probabilities(score_matrices(lam, mu, rho, 6))
    # Three targets, two rates: the grid can reproduce its own numbers exactly.
    lam_hat, mu_hat = rates_from_probabilities(markets["p_home"], markets["p_away"], markets["p_over_2_5"], rho=rho)
    assert np.allclose(lam_hat, lam, atol=1e-3) and np.allclose(mu_hat, mu, atol=1e-3)
    # Home and away alone pin the rates down too.
    lam_hat, mu_hat = rates_from_probabilities(markets["p_home"], markets["p_away"], rho=rho)
    assert np.allclose(lam_hat, lam, atol=1e-3) and np.allclose(mu_hat, mu, atol=1e-3)
    # A missing input gives NaN rates for that row only.
    lam_hat, _ = rates_from_probabilities([markets["p_home"][0], np.nan], [markets["p_away"][0], 0.3], rho=rho)
    assert np.isfinite(lam_hat[0]) and np.isnan(lam_hat[1])

    # Prices with a margin, one fixture without totals, one without any price.
    fixtures = pd.DataFrame(
        {
            "odds_home": [1.95, 2.6, np.nan], "odds_draw": [3.6, 3.3, np.nan], "odds_away": [4.0, 2.8, np.nan],
            "odds_over_2_5": [1.85, np.nan, np.nan], "odds_under_2_5": [2.0, np.nan, np.nan],
        }
    )
    forecast = anchored_forecast(fixtures, rho=rho, fallback=(np.array([1.5, 1.5, 1.2]), np.array([1.2, 1.2, 1.4])))
    assert list(forecast["source"]) == ["market", "market_1x2", "model"]
    assert np.allclose(forecast[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)
    assert forecast.loc[2, "exp_home_goals"] == 1.2 and forecast.loc[2, "exp_away_goals"] == 1.4
    from football_predictor.evaluation.metrics import implied_probabilities

    fair = implied_probabilities(fixtures.loc[[1], ["odds_home", "odds_draw", "odds_away"]].to_numpy(), "power")[0]
    assert np.allclose(forecast.loc[1, ["p_home", "p_draw", "p_away"]].to_numpy(dtype=float), fair, atol=1e-3)


def test_model_predict_anchored_tracks_the_market(league):
    data, _ = league
    model = DixonColesModel(xi=0.003).fit(data, as_of="2022-01-01")
    fixtures = data[(data["kickoff"] >= "2022-01-01") & (data["division"] == "E0")].head(40)
    anchored = model.predict_anchored(fixtures)
    plain = model.predict(fixtures)
    assert list(anchored.columns) == list(plain.columns) + ["source"]
    assert (anchored["source"] == "market").all()
    from football_predictor.evaluation.metrics import implied_probabilities

    fair = implied_probabilities(fixtures[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float), "power")
    # With totals prices too, two rates serve three targets: a compromise, but a close one.
    assert np.abs(anchored[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float) - fair).mean() < 0.03
    # With 1X2 prices alone the winner probabilities are reproduced exactly.
    only_1x2 = model.predict_anchored(fixtures.drop(columns=["odds_over_2_5", "odds_under_2_5"]))
    assert (only_1x2["source"] == "market_1x2").all()
    assert np.abs(only_1x2[["p_home", "p_away"]].to_numpy(dtype=float) - fair[:, [0, 2]]).max() < 1e-3
    unpriced = fixtures.drop(columns=["odds_home", "odds_draw", "odds_away", "odds_over_2_5", "odds_under_2_5"])
    fallback = model.predict_anchored(unpriced)
    assert (fallback["source"] == "model").all()
    pd.testing.assert_frame_equal(fallback.drop(columns="source"), plain)
