"""Price shopping without a model: best prices over bookmaker sets, fair values and the study itself."""

import numpy as np
import pandas as pd
import pytest

from football_predictor.evaluation import (
    aggregate_gap,
    best_prices,
    implied_probabilities,
    net_odds,
    price_shopping,
    reference_probabilities,
    resolve_books,
    settle_units,
    with_best_prices,
)
from football_predictor.evaluation.backtest import MARKET_DEMARGIN
from football_predictor.evaluation.shopping import net_exchange


def _prices():
    """Four fixtures: Pinnacle and the average close every market; a soft book and the exchange beat them sometimes."""
    rows_1x2 = []
    pinnacle = [(2.00, 3.50, 4.00), (1.50, 4.20, 7.00), (3.00, 3.40, 2.40), (2.20, 3.30, 3.40)]
    soft = [(2.20, 3.40, 3.80), (1.45, 4.00, 6.50), (3.30, 3.30, 2.30), (2.10, 3.30, 3.80)]
    exchange = [(2.10, 3.65, 4.20), (1.56, 4.40, 7.40), (3.15, 3.55, 2.50), (2.28, 3.45, 3.55)]
    for i, (p, s, e) in enumerate(zip(pinnacle, soft, exchange)):
        match = f"m{i}"
        if i != 3:  # the last fixture has no Pinnacle price: the average stands in
            rows_1x2.append((match, "pinnacle", "closing", *p))
            rows_1x2.append((match, "pinnacle", "opening", p[0] + 0.1, p[1], p[2] - 0.1))  # the home side shortened
        rows_1x2.append((match, "average", "closing", p[0] * 0.98, p[1] * 0.98, p[2] * 0.98))
        rows_1x2.append((match, "softbook", "closing", *s))
        rows_1x2.append((match, "betfair_exchange", "closing", *e))
        rows_1x2.append((match, "max", "closing", max(p[0], s[0], e[0]) * 1.02, max(p[1], s[1], e[1]), max(p[2], s[2], e[2])))
        rows_1x2.append((match, "softbook", "opening", s[0] + 0.3, s[1], s[2]))
    totals = [(f"m{i}", b, "closing", 2.5, o, u) for i in range(4)
              for b, (o, u) in (("pinnacle", (1.90, 2.00)), ("softbook", (2.00, 1.90)), ("betfair_exchange", (1.96, 2.06)))]
    handicap = [(f"m{i}", b, "closing", -0.25, h, a) for i in range(4)
                for b, (h, a) in (("pinnacle", (1.95, 1.95)), ("softbook", (2.05, 1.85)))]
    handicap.append(("m0", "softbook", "closing", -0.5, 2.20, 1.70))  # a different line: must not be mixed in
    return {
        "1x2": pd.DataFrame(rows_1x2, columns=["match_id", "bookmaker", "market_time", "home", "draw", "away"]),
        "totals": pd.DataFrame(totals, columns=["match_id", "bookmaker", "market_time", "line", "over", "under"]),
        "handicap": pd.DataFrame(handicap, columns=["match_id", "bookmaker", "market_time", "line", "home", "away"]),
    }


def _matches():
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(4)], "season": [2023, 2023, 2024, 2024],
            "kickoff": pd.date_range("2023-09-02 15:00", periods=4, freq="7D"),
            "home_team": "A", "away_team": "B",
            "home_goals": [2, 1, 0, 1], "away_goals": [0, 1, 1, 3],
        }
    )


def test_resolve_books_sets_and_lists():
    available = ["average", "bet365", "betfair_exchange", "bwin", "max", "pinnacle"]
    assert resolve_books("max", available) == ["max"]
    assert resolve_books("named", available) == ["bet365", "betfair_exchange", "bwin", "pinnacle"]
    assert resolve_books("soft", available) == ["bet365", "bwin"]
    assert resolve_books("exchange", available) == ["betfair_exchange"]
    assert resolve_books("bwin, bet365", available) == ["bwin", "bet365"]
    with pytest.raises(ValueError, match="unknown bookmaker"):
        resolve_books("nosuch", available)


def test_best_prices_pick_the_top_price_and_name_the_bookmaker():
    prices = _prices()
    best = best_prices(prices, ["pinnacle", "softbook"])
    assert best.loc["m0", ["max_home", "book_home"]].tolist() == [2.20, "softbook"]
    assert best.loc["m0", ["max_away", "book_away"]].tolist() == [4.00, "pinnacle"]
    assert best.loc["m3", "book_home"] == "softbook"  # no Pinnacle row on m3: softbook stands alone
    assert best.loc["m1", "book_draw"] == "pinnacle"  # ties go to the earlier bookmaker in the list
    assert best.loc["m0", ["max_over_2_5", "book_over_2_5"]].tolist() == [2.00, "softbook"]
    # Handicaps are compared at the quoted line; the stray -0.5 row is left out.
    assert best.loc["m0", ["max_ah_line", "max_ah_home", "book_ah_home"]].tolist() == [-0.25, 2.05, "softbook"]
    # The exchange's price is netted of commission before it competes.
    netted = best_prices(prices, ["softbook", "betfair_exchange"], commission=0.05)
    assert netted.loc["m0", "max_away"] == pytest.approx(net_odds(4.20, 0.05)) and netted.loc["m0", "book_away"] == "betfair_exchange"
    assert netted.loc["m0", "max_draw"] == pytest.approx(net_odds(3.65, 0.05))
    gross = best_prices(prices, ["softbook", "betfair_exchange"], commission=0.0)
    assert gross.loc["m0", "max_away"] == 4.20
    # Opening prices are a separate market time.
    opening = best_prices(prices, ["softbook"], market_time="opening")
    assert opening.loc["m0", "max_home"] == pytest.approx(2.50) and "max_over_2_5" not in opening
    assert best_prices({"1x2": prices["1x2"].iloc[0:0]}, ["pinnacle"]).empty


def test_reference_probabilities_fall_back_and_name_their_source():
    fair = reference_probabilities(_prices(), book="pinnacle")
    expected = implied_probabilities(np.array([[2.00, 3.50, 4.00]]), MARKET_DEMARGIN)[0]
    assert np.allclose(fair.loc["m0", ["p_home", "p_draw", "p_away"]].to_numpy(dtype=float), expected)
    assert fair.loc["m0", "ref_1x2"] == "pinnacle" and fair.loc["m3", "ref_1x2"] == "average"
    assert fair.loc["m0", "ref_ah_line"] == -0.25 and fair.loc["m0", "p_ah_home"] == pytest.approx(0.5)
    exchange = reference_probabilities(_prices(), book="betfair_exchange")
    assert (exchange["ref_1x2"] == "betfair_exchange").all()


def test_price_shopping_bets_where_a_bookmaker_beats_the_fair_value():
    matches, prices = _matches(), _prices()
    result = price_shopping(matches, prices, books=["softbook"], ev_threshold=0.0, markets=("1x2",), n_resamples=50)
    bets = result.bets
    fair = reference_probabilities(prices)
    assert (bets["ev"] == bets["p_fair"] * bets["odds"] - 1).all() and (bets["ev"] > 0).all()
    # One bet per fixture, the highest EV, and only where softbook tops the de-margined Pinnacle price.
    assert bets["match_id"].is_unique
    for bet in bets.itertuples(index=False):
        row = prices["1x2"].set_index(["match_id", "bookmaker", "market_time"]).loc[(bet.match_id, "softbook", "closing")]
        ev = fair.loc[bet.match_id, ["p_home", "p_draw", "p_away"]].to_numpy(dtype=float) * row[["home", "draw", "away"]].to_numpy(dtype=float) - 1
        assert bet.ev == pytest.approx(ev.max()) and ev.max() > 0
    assert (bets["bookmaker"] == "softbook").all() and (bets["market"] == "1x2").all()
    # Settlement: m0 ended 2-0, so a home bet there wins odds - 1; m2 ended 0-1, so its home bet loses the stake.
    settled = bets.set_index("match_id")
    assert set(settled.index) == {"m0", "m2", "m3"}
    assert settled.loc["m0", "selection"] == "home" and settled.loc["m0", "units"] == pytest.approx(1.20)
    assert settled.loc["m2", "selection"] == "home" and settled.loc["m2", "units"] == -1.0
    assert settled.loc["m3", "selection"] == "away" and settled.loc["m3", "units"] == pytest.approx(2.80)
    assert result.metrics["bets"] == len(bets) and result.metrics["flat_stake_roi"] == pytest.approx(bets["units"].mean())
    assert "flat_roi_ci_low" in result.metrics and 0 <= result.metrics["p_flat_roi_le_zero"] <= 1
    assert result.by_bookmaker["bets"].sum() == len(bets) and result.by_season["bets"].sum() == len(bets)
    assert result.coverage["matches"] == 4 and result.coverage["share_fair_from_book"] == 0.75
    # The fixture without a Pinnacle price is judged by the average, or skipped when strict.
    assert "average" in set(bets["fair_source"]) or len(bets[bets["match_id"] == "m3"]) == 0
    strict = price_shopping(matches, prices, books=["softbook"], ev_threshold=0.0, markets=("1x2",), strict_fair=True,
                            n_resamples=0)
    assert "m3" not in set(strict.bets["match_id"]) and (strict.bets["fair_source"] == "pinnacle").all()
    # Every selection above the threshold: at least as many bets, never fewer.
    every = price_shopping(matches, prices, books=["softbook"], ev_threshold=0.0, markets=("1x2",), one_per_match=False,
                           n_resamples=0)
    assert len(every.bets) >= len(bets)
    # A threshold no price clears means no bets and NaN metrics rather than an error.
    none = price_shopping(matches, prices, books=["softbook"], ev_threshold=0.5, n_resamples=0)
    assert none.metrics["bets"] == 0 and np.isnan(none.metrics["flat_stake_roi"]) and none.by_season.empty


def test_price_shopping_handles_totals_handicaps_and_the_exchange():
    matches, prices = _matches(), _prices()
    totals = price_shopping(matches, prices, books=["softbook"], ev_threshold=0.0, markets=("totals",), n_resamples=0)
    assert (totals.bets["selection"] == "over").all() and len(totals.bets) == 4  # 2.00 beats Pinnacle's 1.90 over
    assert totals.bets.set_index("match_id").loc["m0", "units"] == pytest.approx(-1.0)  # 2-0 is under 2.5
    handicap = price_shopping(matches, prices, books=["softbook"], ev_threshold=0.0, markets=("handicap",), n_resamples=0)
    assert (handicap.bets["selection"] == "ah_home").all() and (handicap.bets["line"] == -0.25).all()
    m1 = handicap.bets.set_index("match_id").loc["m1"]  # 1-1 at -0.25: half the stake lost, half pushed
    assert m1["units"] == pytest.approx(-0.5)
    # The exchange at 5% commission beats Pinnacle less often than gross.
    gross = price_shopping(matches, prices, books=["betfair_exchange"], ev_threshold=0.0, commission=0.0, n_resamples=0)
    net = price_shopping(matches, prices, books=["betfair_exchange"], ev_threshold=0.0, commission=0.05, n_resamples=0)
    assert len(net.bets) <= len(gross.bets) and (net.bets["bookmaker"] == "betfair_exchange").all()
    netted = net_exchange(prices, 0.05)["1x2"].set_index(["match_id", "bookmaker", "market_time"]).sort_index()
    assert netted.loc[("m0", "betfair_exchange", "closing"), "home"] == pytest.approx(net_odds(2.10, 0.05))
    assert netted.loc[("m0", "pinnacle", "closing"), "home"] == 2.00  # every other bookmaker is left as quoted


def test_price_shopping_at_opening_prices_values_bets_at_closing():
    matches, prices = _matches(), _prices()
    opening = price_shopping(matches, prices, books=["softbook"], ev_threshold=0.0, markets=("1x2",),
                             market_time="opening", n_resamples=0)
    bets = opening.bets.set_index("match_id")
    # Softbook's opening home price (closing + 0.3) beats Pinnacle's opening price on the three fixtures it prices.
    assert set(bets.index) == {"m0", "m1", "m2"} and (bets["selection"] == "home").all()
    assert bets.loc["m0", "odds"] == pytest.approx(2.50) and (bets["fair_source"] == "pinnacle").all()
    fair_open = implied_probabilities(np.array([[2.10, 3.50, 3.90]]), MARKET_DEMARGIN)[0]
    fair_close = implied_probabilities(np.array([[2.00, 3.50, 4.00]]), MARKET_DEMARGIN)[0]
    assert bets.loc["m0", "ev"] == pytest.approx(fair_open[0] * 2.50 - 1)
    # The closing-line value judges the same bet by Pinnacle's closing price, which moved towards the home side.
    assert bets.loc["m0", "clv"] == pytest.approx(fair_close[0] * 2.50 - 1) and bets.loc["m0", "clv"] > bets.loc["m0", "ev"]
    assert opening.metrics["mean_ev"] == pytest.approx(bets["ev"].mean())
    assert opening.metrics["mean_clv"] == pytest.approx(bets["clv"].mean()) and opening.metrics["mean_clv"] != opening.metrics["mean_ev"]
    # Settled at the opening price taken: m0 ended 2-0.
    assert bets.loc["m0", "units"] == pytest.approx(1.50)
    # At closing the closing-line value is the EV itself.
    closing = price_shopping(matches, prices, books=["softbook"], ev_threshold=0.0, markets=("1x2",), n_resamples=0)
    assert np.allclose(closing.bets["clv"], closing.bets["ev"]) and closing.metrics["mean_clv"] == closing.metrics["mean_ev"]
    # Bets carry the division when the matches do, and the result then has a per-division table.
    divided = price_shopping(matches.assign(division=["E0", "E0", "SC0", "SC0"]), prices, books=["softbook"],
                             ev_threshold=0.0, markets=("1x2",), n_resamples=0)
    assert set(divided.bets["division"]) == {"E0", "SC0"} and divided.by_division["bets"].sum() == len(divided.bets)
    assert closing.by_division.empty


def test_aggregate_gap_and_best_price_injection():
    prices = _prices()
    gap = aggregate_gap(prices, ["pinnacle", "softbook", "betfair_exchange"])
    assert gap["selections"] == 12 and gap["share_unlisted_best"] == pytest.approx(4 / 12)  # the home max is 2% above
    assert gap["mean_gap"] > 0 and gap["share_gap_over_1pct"] == pytest.approx(4 / 12)
    data = pd.DataFrame({"match_id": ["m0", "m1", "zz"], "max_home": [9.0, 9.0, 9.0], "max_over_2_5": [9.0, 9.0, 9.0],
                         "odds_home": [2.0, 1.5, 3.0]})
    injected = with_best_prices(data, best_prices(prices, ["softbook"]))
    assert injected["max_home"].tolist()[:2] == [2.20, 1.45] and np.isnan(injected["max_home"].iloc[2])
    assert injected["book_home"].tolist()[:2] == ["softbook", "softbook"] and injected["max_ah_line"].iloc[0] == -0.25
    assert injected["odds_home"].tolist() == [2.0, 1.5, 3.0]  # the reference price is untouched


def test_settle_units_from_goals_or_outcomes():
    units = settle_units([0, 1, 2, 3, 4], [2.0, 3.0, 4.0, 1.9, 2.1], home_goals=[1, 1, 0, 2, 0], away_goals=[0, 1, 2, 1, 0])
    assert np.allclose(units, [1.0, 2.0, 3.0, 0.9, 1.1])
    same = settle_units([0, 1, 2, 3, 4], [2.0, 3.0, 4.0, 1.9, 2.1], outcome=[0, 1, 2, 0, 1], over=[1, 1, 1, 1, 0])
    assert np.allclose(same, [1.0, 2.0, 3.0, 0.9, 1.1])
    lost = settle_units([0, 3], [2.0, 1.9], home_goals=[0, 1], away_goals=[1, 0])
    assert lost.tolist() == [-1.0, -1.0]
    pushed = settle_units([5, 6], [1.9, 1.9], home_goals=[1, 1], away_goals=[0, 0], line=[-1.0, -1.0])
    assert pushed.tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        settle_units([5], [1.9], outcome=[0])
