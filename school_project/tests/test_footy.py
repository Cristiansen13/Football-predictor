"""Offline tests on a small made-up league (no downloads)."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from footy import PoissonModel, backtest, parse_csv, summarize  # noqa: E402
from footy.evaluate import implied_probabilities, value_bets  # noqa: E402

CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,B365>2.5,B365<2.5,B365CH,B365CD,B365CA
E0,12/08/2023,15:00,Arsenal,Chelsea,2,1,H,1.90,3.60,4.00,1.80,2.00,1.85,3.70,4.20
E0,19/08/2023,15:00,Chelsea,Everton,1,1,D,1.70,3.80,5.00,1.90,1.90,1.75,3.75,4.80
,,,,,,,,,,,,,,,
"""


def test_parse_csv_keeps_results_and_odds_and_drops_blank_rows():
    frame = parse_csv(CSV, 2023)
    assert len(frame) == 2 and list(frame["home"]) == ["Arsenal", "Chelsea"]
    assert frame["home_goals"].tolist() == [2, 1] and frame["date"].iloc[0] == pd.Timestamp("2023-08-12")
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
            rows.append({"season": season, "date": day + pd.Timedelta(days=7 * (k // 6)), "home": home, "away": away,
                         "home_goals": hg, "away_goals": ag, "odds_home": odds[0], "odds_draw": odds[1],
                         "odds_away": odds[2], "closing_home": odds[0], "closing_draw": odds[1], "closing_away": odds[2],
                         "odds_over": 1.9, "odds_under": 1.9})
    return pd.DataFrame(rows)


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
