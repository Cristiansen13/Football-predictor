"""Probabilistic scoring rules for 1X2 forecasts (columns: home, draw, away)."""

from __future__ import annotations

import numpy as np

HOME, DRAW, AWAY = 0, 1, 2


def outcome_index(home_goals, away_goals) -> np.ndarray:
    home_goals, away_goals = np.asarray(home_goals, dtype=float), np.asarray(away_goals, dtype=float)
    return np.where(home_goals > away_goals, HOME, np.where(home_goals == away_goals, DRAW, AWAY))


def _one_hot(outcomes: np.ndarray, k: int = 3) -> np.ndarray:
    return np.eye(k)[np.asarray(outcomes, dtype=int)]


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean over matches of sum_k (p_k - o_k)^2; 0 is perfect, 2 is maximally wrong."""
    probabilities = np.asarray(probabilities, dtype=float)
    return float(np.mean(np.sum((probabilities - _one_hot(outcomes)) ** 2, axis=1)))


def ranked_probability_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """RPS respects the ordering home < draw < away (a draw is 'closer' to a home win)."""
    probabilities = np.asarray(probabilities, dtype=float)
    cumulative = np.cumsum(probabilities, axis=1)[:, :-1]
    observed = np.cumsum(_one_hot(outcomes), axis=1)[:, :-1]
    return float(np.mean(np.sum((cumulative - observed) ** 2, axis=1) / (probabilities.shape[1] - 1)))


def log_loss(probabilities: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0)
    picked = probabilities[np.arange(len(probabilities)), np.asarray(outcomes, dtype=int)]
    return float(-np.mean(np.log(picked)))


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Bookmaker probabilities with the overround removed proportionally."""
    raw = 1.0 / np.asarray(odds, dtype=float)
    return raw / raw.sum(axis=1, keepdims=True)


def overround(odds: np.ndarray) -> np.ndarray:
    return (1.0 / np.asarray(odds, dtype=float)).sum(axis=1) - 1.0
