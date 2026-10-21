"""Expected value and Kelly staking."""

from __future__ import annotations

import numpy as np


def expected_value(probability, decimal_odds):
    """EV per unit staked: P_model * odds - 1."""
    return np.asarray(probability, dtype=float) * np.asarray(decimal_odds, dtype=float) - 1.0


def kelly_fraction(probability, decimal_odds):
    """Full-Kelly bankroll fraction (p * o - 1) / (o - 1), floored at zero."""
    probability = np.asarray(probability, dtype=float)
    decimal_odds = np.asarray(decimal_odds, dtype=float)
    edge = probability * decimal_odds - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(decimal_odds > 1.0, edge / (decimal_odds - 1.0), 0.0)
    return np.clip(fraction, 0.0, 1.0)
