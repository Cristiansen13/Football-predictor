"""Correct-score probability matrices and the markets aggregated from them."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


def tau_matrix(lam: np.ndarray, mu: np.ndarray, rho: float, max_goals: int) -> np.ndarray:
    """Dixon-Coles low-score correction factors, shape (n, G+1, G+1); rows are home goals."""
    lam, mu = np.atleast_1d(lam), np.atleast_1d(mu)
    tau = np.ones((lam.size, max_goals + 1, max_goals + 1))
    tau[:, 0, 0] = 1.0 - lam * mu * rho
    if max_goals >= 1:
        tau[:, 0, 1] = 1.0 + lam * rho
        tau[:, 1, 0] = 1.0 + mu * rho
        tau[:, 1, 1] = 1.0 - rho
    return tau


def score_matrices(
    lam: np.ndarray | float,
    mu: np.ndarray | float,
    rho: float = 0.0,
    max_goals: int = 6,
    normalize: bool = True,
) -> np.ndarray:
    """Joint probability of every score from 0-0 to G-G, shape (n, G+1, G+1).

    ``matrix[i, x, y]`` is P(home = x, away = y) for fixture ``i``. With
    ``normalize`` the mass beyond ``max_goals`` (a fraction of a percent for
    G = 6) is redistributed so every matrix sums to one.
    """
    lam, mu = np.atleast_1d(np.asarray(lam, dtype=float)), np.atleast_1d(np.asarray(mu, dtype=float))
    goals = np.arange(max_goals + 1)
    home = poisson.pmf(goals[None, :], lam[:, None])
    away = poisson.pmf(goals[None, :], mu[:, None])
    matrices = home[:, :, None] * away[:, None, :] * tau_matrix(lam, mu, rho, max_goals)
    matrices = np.clip(matrices, 0.0, None)
    if normalize:
        matrices /= matrices.sum(axis=(1, 2), keepdims=True)
    return matrices


def outcome_probabilities(matrices: np.ndarray) -> np.ndarray:
    """(n, 3) array of P(home win), P(draw), P(away win)."""
    matrices = np.asarray(matrices)
    if matrices.ndim == 2:
        matrices = matrices[None]
    home = np.tril(np.ones(matrices.shape[1:]), k=-1)  # x > y
    away = np.triu(np.ones(matrices.shape[1:]), k=1)  # x < y
    p_home = (matrices * home).sum(axis=(1, 2))
    p_away = (matrices * away).sum(axis=(1, 2))
    p_draw = np.trace(matrices, axis1=1, axis2=2)
    return np.column_stack([p_home, p_draw, p_away])


def market_probabilities(matrices: np.ndarray) -> pd.DataFrame:
    """1X2, over/under 2.5 goals and both-teams-to-score probabilities."""
    matrices = np.asarray(matrices)
    if matrices.ndim == 2:
        matrices = matrices[None]
    outcomes = outcome_probabilities(matrices)
    goals = np.arange(matrices.shape[1])
    total = goals[:, None] + goals[None, :]
    over = (matrices * (total > 2.5)).sum(axis=(1, 2))
    btts = matrices[:, 1:, 1:].sum(axis=(1, 2))
    return pd.DataFrame(
        {
            "p_home": outcomes[:, 0],
            "p_draw": outcomes[:, 1],
            "p_away": outcomes[:, 2],
            "p_over_2_5": over,
            "p_under_2_5": matrices.sum(axis=(1, 2)) - over,
            "p_btts": btts,
        }
    )
