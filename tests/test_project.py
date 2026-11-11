"""Offline tests on a small made-up league (no downloads)."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.compare_models import season_forward  # noqa: E402
from analysis.error_analysis import confusion, value_bets as value_bets_by_model  # noqa: E402
from analysis.evaluate import (backtest, brier_per_match, implied_probabilities, paired_bootstrap, score,  # noqa: E402
                               summarize, value_bets)
from analysis.models import ML_MODELS  # noqa: E402
from analysis.poisson import PoissonModel  # noqa: E402
from extraction.download import parse_seasons, season_code  # noqa: E402
from extraction.features import FEATURE_COLUMNS, build_features, elo_ratings  # noqa: E402
from extraction.preprocess import parse_csv  # noqa: E402

CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,Referee,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR,B365H,B365D,B365A,B365>2.5,B365<2.5,B365CH,B365CD,B365CA
E0,12/08/2023,15:00,Arsenal,Chelsea,2,1,H,1,0,M Oliver,14,9,6,3,7,4,10,12,1,2,0,0,1.90,3.60,4.00,1.80,2.00,1.85,3.70,4.20
E0,19/08/23,15:00,Chelsea,Everton,1,1,D,0,1,A Taylor,11,8,4,4,5,3,9,14,2,3,0,1,1.70,3.80,5.00,1.90,1.90,1.75,3.75,4.80
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
"""


def test_download_names_seasons_the_way_the_site_does():
    assert season_code(2023) == "2324" and season_code(1999) == "9900"
    assert parse_seasons("2019-2021") == [2019, 2020, 2021] and parse_seasons("2024") == [2024]


def test_parse_csv_keeps_results_statistics_and_odds_and_drops_blank_rows():
    frame = parse_csv(CSV, 2023)
    assert len(frame) == 2 and list(frame["home"]) == ["Arsenal", "Chelsea"]
    # Both date formats the site has used are read as the same kind of date.
    assert frame["date"].tolist() == [pd.Timestamp("2023-08-12"), pd.Timestamp("2023-08-19")]
    assert frame["home_goals"].tolist() == [2, 1] and frame["result"].tolist() == ["H", "D"]
    assert frame["home_shots_target"].tolist() == [6, 4] and frame["away_red"].tolist() == [0, 1]
    assert frame["referee"].iloc[0] == "M Oliver"
    assert frame["odds_home"].iloc[0] == 1.90 and frame["closing_away"].iloc[1] == 4.80
    assert (frame["season"] == 2023).all()


def simulated_league(seasons=(2022, 2023), seed=3) -> pd.DataFrame:
    """Four clubs of known strength, double round robin per season, Poisson goals, fair odds with a 5% margin."""
    rng = np.random.default_rng(seed)
    attack = {"Strong": 1.4, "Good": 1.1, "Mid": 0.9, "Weak": 0.6}
    defence = {"Strong": 0.7, "Good": 0.9, "Mid": 1.1, "Weak": 1.3}
    rows = []
    for season in seasons:
        day = pd.Timestamp(f"{season}-08-12")
        pairs = [(h, a) for h in attack for a in attack if h != a]
        rng.shuffle(pairs)
        for k, (home, away) in enumerate(pairs * 3):  # 36 matches a season
            lam, mu = 1.5 * attack[home] * defence[away], 1.1 * attack[away] * defence[home]
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            grid = np.outer([np.exp(-lam) * lam ** i / math.factorial(i) for i in range(8)],
                            [np.exp(-mu) * mu ** j / math.factorial(j) for j in range(8)])
            i, j = np.indices(grid.shape)
            probs = np.array([grid[i > j].sum(), np.trace(grid), grid[i < j].sum()]) / grid.sum()
            odds = 1 / (probs * 1.05)
            rows.append({"season": season, "date": day + pd.Timedelta(days=7 * (k // 2)), "home": home, "away": away,
                         "home_goals": hg, "away_goals": ag, "odds_home": odds[0], "odds_draw": odds[1],
                         "odds_away": odds[2], "closing_home": odds[0], "closing_draw": odds[1], "closing_away": odds[2],
                         "odds_over": 1.9, "odds_under": 1.9,
                         "home_shots_target": hg + rng.poisson(3), "away_shots_target": ag + rng.poisson(3),
                         "result": "H" if hg > ag else ("D" if hg == ag else "A")})
    frame = pd.DataFrame(rows).sort_values(["date", "home"]).reset_index(drop=True)
    frame.insert(0, "match_id", range(len(frame)))
    return frame


def test_model_probabilities_are_coherent_and_favour_the_stronger_side():
    matches = simulated_league()
    model = PoissonModel().fit(matches)
    forecast = model.predict("Strong", "Weak")
    assert forecast["p_home"] + forecast["p_draw"] + forecast["p_away"] == pytest.approx(1.0)
    assert forecast["p_home"] > 0.6 > forecast["p_away"]
    reverse = model.predict("Weak", "Strong")
    assert reverse["p_away"] > reverse["p_home"]
    grid = model.score_grid("Strong", "Weak")
    assert grid.shape == (7, 7) and grid.to_numpy().sum() == pytest.approx(1.0)
    assert forecast["exp_home_goals"] > forecast["exp_away_goals"]
    # An unknown team is treated as average, and smoothing keeps strengths near 1 with little data.
    unknown = model.predict("Newcomer", "Mid")
    assert 0 < unknown["p_home"] < 1
    strengths = PoissonModel(prior_games=50).fit(matches).strengths
    assert (strengths[["home_attack", "away_attack"]].sub(1).abs() < 0.2).all().all()


def test_backtest_is_walk_forward_and_summary_has_both_sides():
    matches = simulated_league()
    predictions = backtest(matches, 2023, train_seasons_back=1)
    assert len(predictions) == 36 and set(predictions["outcome"]) <= {0, 1, 2}
    # No forecast may use the match's own day: the first day of the season is predicted from 2022 only.
    first_day = predictions["date"].min()
    first = predictions[predictions["date"] == first_day]
    model_2022 = PoissonModel().fit(matches[matches["season"] == 2022])
    expected = model_2022.predict_frame(first)
    assert np.allclose(first["p_home"].to_numpy(), expected["p_home"].to_numpy())
    report = summarize(predictions)
    for key in ("accuracy_model", "brier_model", "log_loss_model", "score_hit_model", "brier_bookmaker", "bets"):
        assert key in report, key
    assert 0 < report["brier_model"] < 2 and report["bookmaker_odds_used"] == "closing"
    bets = value_bets(predictions, threshold=0.0)
    assert set(bets["selection"]) <= {"home", "draw", "away"}
    assert np.allclose(implied_probabilities(np.array([[2.0, 4.0, 4.0]])), [[0.5, 0.25, 0.25]])


def test_features_only_use_earlier_matches():
    matches = simulated_league()
    features = build_features(matches)
    assert list(features["match_id"]) == list(matches["match_id"]) and set(FEATURE_COLUMNS) <= set(features.columns)
    # Nothing is known before a club's first match: ratings start level and form is empty.
    first = features.iloc[0]
    assert first["home_elo"] == first["away_elo"] == 1500.0 and first["elo_diff"] == 0.0
    assert np.isnan(first["home_pts5"]) and np.isnan(first["home_rest"])
    # Changing a result changes only the features of later matches.
    changed = matches.copy()
    row = 20
    changed.loc[row, ["home_goals", "away_goals"]] = [9, 0]
    again = build_features(changed)
    same = features[FEATURE_COLUMNS].iloc[: row + 1].to_numpy(dtype=float)
    assert np.allclose(same, again[FEATURE_COLUMNS].iloc[: row + 1].to_numpy(dtype=float), equal_nan=True)
    assert not np.allclose(features[FEATURE_COLUMNS].iloc[row + 1:].to_numpy(dtype=float),
                           again[FEATURE_COLUMNS].iloc[row + 1:].to_numpy(dtype=float), equal_nan=True)
    # A form feature is the mean of the previous five matches, the current one excluded.
    club = "Strong"
    played = matches[(matches["home"] == club) | (matches["away"] == club)].reset_index(drop=True)
    points = np.where(played["home"] == club,
                      np.select([played["home_goals"] > played["away_goals"], played["home_goals"] == played["away_goals"]], [3, 1], 0),
                      np.select([played["away_goals"] > played["home_goals"], played["home_goals"] == played["away_goals"]], [3, 1], 0))
    sixth = played.iloc[5]
    side = "home" if sixth["home"] == club else "away"
    value = features.set_index("match_id").loc[sixth["match_id"], f"{side}_pts5"]
    assert value == pytest.approx(points[:5].mean())
    # Elo: the winner's rating goes up by what the loser's goes down.
    elo = elo_ratings(matches)
    assert elo.iloc[0].tolist() == [1500.0, 1500.0]
    assert elo["home_elo"].iloc[-1] != 1500.0


def test_models_give_probabilities_and_beat_the_class_frequencies(monkeypatch):
    import analysis.compare_models as compare

    matches = simulated_league(seasons=tuple(range(2014, 2024)))
    features = build_features(matches)
    outcomes = features.set_index("match_id")["outcome"]
    monkeypatch.setattr(compare, "FIRST_TRAIN_SEASON", 2015)  # the first season only warms the features up
    test_seasons = [2021, 2022, 2023]
    scores = {}
    for key in ML_MODELS:
        forecast = season_forward(features, key, {"C": 1.0} if key == "logistic" else {}, test_seasons)
        assert len(forecast) == 108 and np.allclose(forecast.sum(axis=1), 1.0)
        scores[key] = score(forecast.to_numpy(), outcomes.loc[forecast.index].to_numpy())
        assert 0 < scores[key]["brier"] < 1 and 0 <= scores[key]["accuracy"] <= 1
    # With clubs of very different strength, the rating-based models beat the class frequencies.
    assert scores["elo"]["brier"] < scores["baseline"]["brier"]
    assert scores["logistic"]["brier"] < scores["baseline"]["brier"]


def test_scoring_helpers():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    outcomes = np.array([0, 1])
    assert brier_per_match(probabilities, outcomes) == pytest.approx([0.14, 0.78])
    assert score(probabilities, outcomes)["accuracy"] == 0.5
    mean, low, high = paired_bootstrap(np.array([0.5, 0.7, 0.6]), np.array([0.4, 0.5, 0.6]))
    assert mean == pytest.approx(0.1) and low <= mean <= high
    frame = pd.DataFrame({"outcome": [0, 1, 2, 0], "m_p_home": [0.6, 0.52, 0.2, 0.3], "m_p_draw": [0.2, 0.28, 0.2, 0.3],
                          "m_p_away": [0.2, 0.2, 0.6, 0.4], "closing_home": [2.0, 2.0, 4.0, 2.0],
                          "closing_draw": [3.5, 3.5, 3.5, 3.5], "closing_away": [4.0, 4.0, 2.0, 4.0]})
    assert confusion(frame, "m").tolist() == [[1, 0, 1], [1, 0, 0], [0, 0, 1]]
    import analysis.error_analysis as errors
    errors.SHORT_NAMES["m"] = "m"
    bets = value_bets_by_model(frame, ["m"], threshold=0.05)
    # EV picks: 0.6*2.0-1=0.2 (home, wins), 0.52*2.0-1=0.04 (below the threshold), 0.6*2.0-1=0.2 (away, wins), 0.4*4.0-1=0.6 (away, loses).
    assert bets["bets"].iloc[0] == 3 and bets["profit_units"].iloc[0] == pytest.approx(1.0 + 1.0 - 1.0)
