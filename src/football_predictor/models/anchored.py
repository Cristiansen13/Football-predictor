"""Market-anchored forecasts: the expected goals that reproduce the market's prices.

The closing line is the best pre-match forecast of the result there is, but
it says nothing directly about the score. The Dixon-Coles grid is a bridge:
find the home and away scoring rates whose grid gives the market's home, away
and over-2.5 probabilities, and the whole correct-score distribution follows.
Where a fixture has no prices the model's own rates are used instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .probability import market_probabilities, outcome_probabilities, score_matrices

ANCHORED_COLUMNS = ["exp_home_goals", "exp_away_goals", "p_home", "p_draw", "p_away", "p_over_2_5", "p_under_2_5",
                    "p_btts", "source"]


def _over_probability(matrices: np.ndarray) -> np.ndarray:
    goals = np.arange(matrices.shape[1])
    return (matrices * ((goals[:, None] + goals[None, :]) > 2.5)).sum(axis=(1, 2))


def rates_from_probabilities(
    p_home,
    p_away,
    p_over=None,
    *,
    rho: float = 0.0,
    max_goals: int = 6,
    initial: tuple[float, float] = (1.5, 1.2),
) -> tuple[np.ndarray, np.ndarray]:
    """Scoring rates (λ, μ) whose Dixon-Coles grid reproduces the given probabilities.

    With ``p_over`` three targets (home win, away win, over 2.5) are fitted in
    least squares, since two rates cannot hit three numbers exactly; without
    it, home and away win are matched exactly and the draw follows from the
    grid. Rows with a missing input get NaN rates.
    """
    p_home = np.atleast_1d(np.asarray(p_home, dtype=float))
    p_away = np.atleast_1d(np.asarray(p_away, dtype=float))
    p_over = None if p_over is None else np.atleast_1d(np.asarray(p_over, dtype=float))
    lam, mu = np.full(len(p_home), np.nan), np.full(len(p_home), np.nan)
    x0 = np.log(np.asarray(initial, dtype=float))
    for i in range(len(p_home)):
        targets = [p_home[i], p_away[i]]
        if p_over is not None and np.isfinite(p_over[i]):
            targets.append(p_over[i])
        if not np.all(np.isfinite(targets[:2])):
            continue
        targets = np.asarray(targets)

        def residuals(log_rates):
            grid = score_matrices(np.exp(log_rates[0]), np.exp(log_rates[1]), rho, max_goals)
            outcomes = outcome_probabilities(grid)[0]
            fitted = [outcomes[0], outcomes[2]]
            if len(targets) == 3:
                fitted.append(_over_probability(grid)[0])
            return np.asarray(fitted) - targets

        result = least_squares(residuals, x0, method="lm", xtol=1e-10, ftol=1e-10)
        lam[i], mu[i] = np.exp(result.x)
    return lam, mu


def anchored_forecast(
    fixtures: pd.DataFrame,
    *,
    rho: float,
    max_goals: int = 6,
    fallback: tuple[np.ndarray, np.ndarray] | None = None,
    price_columns: tuple[str, str, str] = ("odds_home", "odds_draw", "odds_away"),
    totals_columns: tuple[str, str] = ("odds_over_2_5", "odds_under_2_5"),
    demargin: str = "power",
) -> pd.DataFrame:
    """Winner, totals and both-teams-to-score probabilities anchored on the market.

    ``source`` says what each row rests on: ``market`` (1X2 and over/under
    prices), ``market_1x2`` (1X2 prices only) or ``model`` (no prices; the
    ``fallback`` rates, normally the model's expected goals).
    """
    from ..evaluation.metrics import implied_probabilities  # local import: evaluation imports models

    n = len(fixtures)
    lam = np.full(n, np.nan)
    mu = np.full(n, np.nan)
    source = np.full(n, "model", dtype=object)
    have_1x2 = all(c in fixtures for c in price_columns)
    have_totals = all(c in fixtures for c in totals_columns)
    if have_1x2:
        odds = fixtures[list(price_columns)].to_numpy(dtype=float)
        priced = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
        if priced.any():
            probs = implied_probabilities(odds[priced], demargin)
            p_over = np.full(priced.sum(), np.nan)
            if have_totals:
                totals = fixtures[list(totals_columns)].to_numpy(dtype=float)[priced]
                with_totals = np.isfinite(totals).all(axis=1) & (totals > 1.0).all(axis=1)
                if with_totals.any():
                    p_over[with_totals] = implied_probabilities(totals[with_totals], demargin)[:, 0]
            lam[priced], mu[priced] = rates_from_probabilities(probs[:, 0], probs[:, 2], p_over, rho=rho, max_goals=max_goals)
            source[priced] = np.where(np.isfinite(p_over), "market", "market_1x2")
    missing = ~np.isfinite(lam) | ~np.isfinite(mu)
    if missing.any():
        if fallback is None:
            raise ValueError("fixtures without prices need fallback rates")
        lam[missing], mu[missing] = np.asarray(fallback[0], dtype=float)[missing], np.asarray(fallback[1], dtype=float)[missing]
        source[missing] = "model"
    markets = market_probabilities(score_matrices(lam, mu, rho, max_goals))
    markets.insert(0, "exp_away_goals", mu)
    markets.insert(0, "exp_home_goals", lam)
    markets["source"] = source
    markets.index = fixtures.index
    return markets[ANCHORED_COLUMNS]
