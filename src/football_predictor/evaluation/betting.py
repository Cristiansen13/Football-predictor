"""Expected value, settlement and Kelly staking."""

from __future__ import annotations

import numpy as np

# Index of every selection the engine can bet, across the three markets.
SELECTION_NAMES = ("home", "draw", "away", "over", "under", "ah_home", "ah_away")


def expected_value(probability, decimal_odds):
    """EV per unit staked: P_model * odds - 1."""
    return np.asarray(probability, dtype=float) * np.asarray(decimal_odds, dtype=float) - 1.0


def net_odds(decimal_odds, commission: float):
    """Decimal odds after an exchange takes ``commission`` of the winnings."""
    return 1.0 + (np.asarray(decimal_odds, dtype=float) - 1.0) * (1.0 - commission)


def handicap_profit(home_goals, away_goals, line, odds, side) -> np.ndarray:
    """Profit per unit stake of an Asian handicap bet once the match is played.

    ``line`` is the home handicap, ``side`` +1 for a bet on the home side and
    -1 for the away side. Full win pays ``odds - 1``, a push returns the stake,
    a loss costs it; a quarter line settles half the stake on each neighbouring
    half-goal line.
    """
    home_goals = np.asarray(home_goals, dtype=float)
    away_goals = np.asarray(away_goals, dtype=float)
    line = np.asarray(line, dtype=float)
    odds = np.asarray(odds, dtype=float)
    side = np.asarray(side, dtype=float)
    quarter = (np.abs(np.round(line * 4) - line * 4) < 1e-9) & ((np.round(line * 4).astype(int) % 2) != 0)
    total = np.zeros(np.broadcast(home_goals, line).shape)
    for offset in (-0.25, 0.25):
        part_line = np.where(quarter, line + offset, line)
        margin = side * (home_goals - away_goals + part_line)
        result = np.where(margin > 1e-9, odds - 1.0, np.where(margin < -1e-9, -1.0, 0.0))
        total = total + 0.5 * result
    return total


def settle_units(
    selection_index,
    odds,
    *,
    home_goals=None,
    away_goals=None,
    outcome=None,
    over=None,
    line=None,
    total_line: float = 2.5,
) -> np.ndarray:
    """Profit per unit stake of settled bets on the selections in :data:`SELECTION_NAMES`.

    1X2 and over/under bets settle from the goals (or from ``outcome`` 0/1/2
    and ``over`` 0/1 when given); handicaps need the goals and the ``line``,
    and follow :func:`handicap_profit` (pushes return the stake, quarter lines
    split it).
    """
    selection_index = np.asarray(selection_index, dtype=int)
    odds = np.asarray(odds, dtype=float)
    n = len(selection_index)
    if home_goals is not None and away_goals is not None:
        home_goals = np.asarray(home_goals, dtype=float)
        away_goals = np.asarray(away_goals, dtype=float)
        if outcome is None:
            outcome = np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))
        if over is None:
            over = (home_goals + away_goals > total_line).astype(int)
    if outcome is None:
        raise ValueError("settlement needs the goals or the outcome")
    outcome = np.asarray(outcome, dtype=int)
    over = np.zeros(n, dtype=int) if over is None else np.asarray(over, dtype=int)
    won_1x2 = selection_index == outcome
    won_totals = np.where(selection_index == 3, over == 1, over == 0)
    units = np.where(np.where(selection_index < 3, won_1x2, won_totals), odds - 1.0, -1.0).astype(float)
    is_handicap = selection_index >= 5
    if is_handicap.any():
        if home_goals is None or away_goals is None or line is None:
            raise ValueError("handicap settlement needs the goals and the line")
        rows = np.nonzero(is_handicap)[0]
        side = np.where(selection_index[rows] == 5, 1.0, -1.0)
        units[rows] = handicap_profit(home_goals[rows], away_goals[rows], np.asarray(line, dtype=float)[rows],
                                      odds[rows], side)
    return units


def kelly_fraction(probability, decimal_odds):
    """Full-Kelly bankroll fraction (p * o - 1) / (o - 1), floored at zero."""
    probability = np.asarray(probability, dtype=float)
    decimal_odds = np.asarray(decimal_odds, dtype=float)
    edge = probability * decimal_odds - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(decimal_odds > 1.0, edge / (decimal_odds - 1.0), 0.0)
    return np.clip(fraction, 0.0, 1.0)
