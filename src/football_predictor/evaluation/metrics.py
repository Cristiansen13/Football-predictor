"""Probabilistic scoring rules for 1X2 forecasts (columns: home, draw, away)."""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def accuracy(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Share of matches whose most likely outcome happened (secondary, coarse metric)."""
    return float(np.mean(np.argmax(np.asarray(probabilities), axis=1) == np.asarray(outcomes, dtype=int)))


def brier_per_match(probabilities: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    return np.sum((np.asarray(probabilities, dtype=float) - _one_hot(outcomes)) ** 2, axis=1)


def skill_score(score: float, reference: float) -> float:
    """1 - score / reference: > 0 means better than the reference forecast."""
    return float(1.0 - score / reference) if reference else float("nan")


def calibration_table(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability table: per outcome and probability bin, mean forecast vs observed frequency."""
    probabilities = np.asarray(probabilities, dtype=float)
    observed = _one_hot(outcomes)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for k, name in enumerate(("home", "draw", "away")):
        which = np.clip(np.digitize(probabilities[:, k], edges[1:-1]), 0, bins - 1)
        for b in range(bins):
            mask = which == b
            if mask.any():
                rows.append(
                    {
                        "outcome": name,
                        "bin_low": edges[b],
                        "bin_high": edges[b + 1],
                        "count": int(mask.sum()),
                        "mean_forecast": float(probabilities[mask, k].mean()),
                        "observed": float(observed[mask, k].mean()),
                    }
                )
    return pd.DataFrame(rows)


def expected_calibration_error(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    """Count-weighted mean |forecast - observed| over all outcome bins (0 = perfectly calibrated)."""
    table = calibration_table(probabilities, outcomes, bins)
    if table.empty:
        return float("nan")
    gap = (table["mean_forecast"] - table["observed"]).abs()
    return float(np.average(gap, weights=table["count"]))


# -- two-way markets (over/under, both teams to score) -------------------------


def binary_brier(probability, outcome) -> float:
    """Mean squared error of a yes/no forecast; 0 is perfect, 0.25 is the coin flip."""
    p, y = np.asarray(probability, dtype=float), np.asarray(outcome, dtype=float)
    return float(np.mean((p - y) ** 2))


def binary_brier_per_match(probability, outcome) -> np.ndarray:
    p, y = np.asarray(probability, dtype=float), np.asarray(outcome, dtype=float)
    return (p - y) ** 2


def binary_log_loss(probability, outcome, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(probability, dtype=float), eps, 1 - eps)
    y = np.asarray(outcome, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def two_way_probabilities(over_odds, under_odds) -> np.ndarray:
    """De-margined (p_over, p_under) from a pair of decimal prices."""
    return implied_probabilities(np.column_stack([np.asarray(over_odds, dtype=float),
                                                  np.asarray(under_odds, dtype=float)]))


def bootstrap_interval(
    statistic,
    *arrays: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Percentile bootstrap over matched rows of ``arrays``.

    Returns (low, high, resampled statistics).
    """
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        stats[i] = statistic(*(np.asarray(a)[idx] for a in arrays))
    tail = (1.0 - level) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail)), stats
