"""Expected value and Kelly staking."""

from __future__ import annotations

import numpy as np


def expected_value(probability, decimal_odds):
    """EV per unit staked: P_model * odds - 1."""
    return np.asarray(probability, dtype=float) * np.asarray(decimal_odds, dtype=float) - 1.0


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


def kelly_fraction(probability, decimal_odds):
    """Full-Kelly bankroll fraction (p * o - 1) / (o - 1), floored at zero."""
    probability = np.asarray(probability, dtype=float)
    decimal_odds = np.asarray(decimal_odds, dtype=float)
    edge = probability * decimal_odds - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(decimal_odds > 1.0, edge / (decimal_odds - 1.0), 0.0)
    return np.clip(fraction, 0.0, 1.0)
