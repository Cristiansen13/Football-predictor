import numpy as np
import pandas as pd
import pytest

from football_predictor.features import (
    audit_point_in_time,
    build_features,
    fatigue_penalty,
    lineup_strength,
    schedule_fatigue,
    time_decayed_ewma,
)
from football_predictor.features.builder import to_team_long
from football_predictor.synthetic import simulate_league


def _long(values, days, played=None, team="A"):
    return pd.DataFrame(
        {
            "team": team,
            "kickoff": pd.Timestamp("2024-01-01") + pd.to_timedelta(days, unit="D"),
            "played": played if played is not None else [True] * len(values),
            "is_home": True,
            "x": values,
        }
    )


def test_ewma_is_backward_looking_and_time_weighted():
    long = _long([1.0, 3.0, 2.0], [0, 10, 20])
    out = time_decayed_ewma(long, ["x"], halflife_days=10, prior_weight=0)
    assert np.isnan(out.loc[0, "ewma_x"])  # nothing before the first match
    assert out.loc[1, "ewma_x"] == pytest.approx(1.0)  # only match 0
    # match 0 is 20 days old (weight 0.25), match 1 is 10 days old (weight 0.5)
    assert out.loc[2, "ewma_x"] == pytest.approx((0.25 * 1 + 0.5 * 3) / 0.75)


def test_ewma_recent_match_dominates():
    long = _long([0.0, 3.0, np.nan], [0, 26, 30], played=[True, True, False])
    out = time_decayed_ewma(long, ["x"], halflife_days=7, prior_weight=0)
    assert out.loc[2, "ewma_x"] > 2.5  # four-day-old match outweighs the 30-day-old one


def test_ewma_shrinks_towards_league_prior():
    a = _long([2.0, 2.0], [0, 7], team="A")
    b = _long([0.0], [3], team="B")
    long = pd.concat([a, b], ignore_index=True)
    out = time_decayed_ewma(long, ["x"], halflife_days=1e9, prior_weight=1.0)
    # Team B at day 3 has no history; the league mean of earlier matches is 2.0.
    assert out.loc[2, "ewma_x"] == pytest.approx(2.0)
    # Team A at day 7: own mean 2.0 (n=1), league prior (2 + 0) / 2 = 1.0, weight 1.
    assert out.loc[1, "ewma_x"] == pytest.approx((2.0 + 1.0) / 2)


def test_features_do_not_leak_future_results():
    data = simulate_league(seasons=[2022], seed=3)
    cut = data["kickoff"].iloc[200]
    altered = data.copy()
    future = altered["kickoff"] >= cut
    altered.loc[future, ["home_goals", "away_goals"]] = 9
    altered.loc[future, ["home_xg", "away_xg"]] = 9.0

    original = build_features(data)
    changed = build_features(altered)
    feature_columns = [c for c in original.columns if "ewma" in c or "rest" in c or "fatigue" in c]
    up_to_cut = original["kickoff"] <= cut
    pd.testing.assert_frame_equal(
        original.loc[up_to_cut, feature_columns], changed.loc[up_to_cut, feature_columns]
    )
    assert not original.loc[~up_to_cut, feature_columns].equals(changed.loc[~up_to_cut, feature_columns])


def test_fatigue_penalty_function():
    penalty = fatigue_penalty(np.array([96.0, 72.0, 36.0, 36.0]), np.array([False, False, False, True]))
    assert penalty == pytest.approx([0.0, 0.0, 0.5, 0.75])


def test_schedule_fatigue_uses_all_competitions():
    long = pd.DataFrame(
        {
            "team": ["A", "A"],
            "kickoff": pd.to_datetime(["2024-01-06 15:00", "2024-01-13 15:00"]),
            "is_home": [True, True],
        }
    )
    schedule = pd.DataFrame(
        {"team": ["A"], "kickoff": pd.to_datetime(["2024-01-11 20:00"]), "venue": ["away"]}
    )
    out = schedule_fatigue(long, schedule)
    assert out.loc[0, "rest_days"] == 14.0  # no earlier fixture: capped
    assert out.loc[0, "fatigue_penalty"] == 0.0
    assert out.loc[1, "rest_days"] == pytest.approx(43 / 24)
    assert out.loc[1, "fatigue_index"] == pytest.approx(24 / 43)
    assert out.loc[1, "fatigue_penalty"] == pytest.approx((72 - 43) / 72 * 1.5)


def test_lineup_strength_imputes_unknown_players():
    lineups = pd.DataFrame(
        {
            "match_id": "m1",
            "team_id": "t1",
            "player_name": ["Known Player", "Mystery Youngster"],
            "is_starter": [1, 1],
            "kickoff": pd.Timestamp("2024-02-01 15:00"),
        }
    )
    ratings = pd.DataFrame(
        {
            "player_name": ["Known Player", "Star A", "Star B", "Squad C"],
            "team_id": "t1",
            "snapshot_date": pd.Timestamp("2024-01-15"),
            "overall": [80, 90, 86, 70],
        }
    )
    out = lineup_strength(lineups, ratings, xi_size=2, unknown_quantile=0.0).iloc[0]
    assert out["matched_starters"] == 1
    assert out["lineup_score"] == pytest.approx((80 + 70) / 2)
    assert out["lineup_fraction"] == pytest.approx(75 / 88)


def test_lineup_ratings_snapshot_must_predate_match_day():
    lineups = pd.DataFrame(
        {"match_id": "m1", "team_id": "t1", "player_name": ["P"], "is_starter": [1], "kickoff": pd.Timestamp("2024-02-01 15:00")}
    )
    ratings = pd.DataFrame(
        {"player_name": ["P"], "team_id": "t1", "snapshot_date": pd.Timestamp("2024-02-01"), "overall": [80]}
    )
    assert lineup_strength(lineups, ratings).empty


def test_long_format_mirrors_sides():
    data = simulate_league(seasons=[2022], seed=1).head(3)
    long = to_team_long(data)
    assert len(long) == 6
    home = long[long["is_home"]].reset_index(drop=True)
    away = long[~long["is_home"]].reset_index(drop=True)
    np.testing.assert_array_equal(home["goals_for"], away["goals_against"])
    np.testing.assert_array_equal(home["xg_against"], away["xg_for"])


def test_same_match_listed_twice_is_not_the_previous_fixture():
    # Football-Data says 15:00, FBref's all-competition log says 12:30 for the same game.
    long = pd.DataFrame(
        {"team": ["A", "A"], "kickoff": pd.to_datetime(["2024-01-06 15:00", "2024-01-13 15:00"]), "is_home": [True, False]}
    )
    schedule = pd.DataFrame(
        {"team": ["A", "A"], "kickoff": pd.to_datetime(["2024-01-06 12:30", "2024-01-13 12:30"]),
         "venue": ["home", "away"]}
    )
    out = schedule_fatigue(long, schedule)
    assert out.loc[1, "rest_days"] == pytest.approx(7.0)  # previous *distinct* match, not 2.5 h earlier
    assert out.loc[1, "fatigue_penalty"] == 0.0


def test_point_in_time_audit_passes_on_real_features():
    data = simulate_league(seasons=[2021, 2022], second_division_teams=20, seed=2)
    schedule = to_team_long(data)[["team", "kickoff"]].assign(
        kickoff=lambda f: f["kickoff"] - pd.Timedelta(days=3), venue="away"
    )  # a midweek cup tie before every league match
    assert audit_point_in_time(data, schedule=schedule, sample=8).empty


def test_point_in_time_audit_catches_leaks():
    data = simulate_league(seasons=[2022], seed=2)

    def leaky_builder(matches, **kwargs):
        out = build_features(matches, **kwargs)
        out["home_ewma_xg_for"] = out["home_xg"]  # the match's own xG
        return out

    problems = audit_point_in_time(data, sample=5, builder=leaky_builder)
    assert set(problems["feature"]) == {"home_ewma_xg_for"}
    assert problems["match_id"].nunique() == 5


def test_ewma_ignores_rows_at_the_same_timestamp():
    # A duplicated or mistimed row must not feed the other row at the same kickoff.
    long = _long([1.0, 5.0, 2.0], [0, 7, 7])
    out = time_decayed_ewma(long, ["x"], halflife_days=10, prior_weight=0)
    assert out.loc[1, "ewma_x"] == pytest.approx(1.0)
    assert out.loc[2, "ewma_x"] == pytest.approx(1.0)
