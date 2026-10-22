"""Honest evaluation: refit before every match day, then score the forecasts.

Walk-forward means the model only ever sees matches played before the day it
predicts, exactly as it would in real use. Scores:

* accuracy      how often the most likely result (home/draw/away) happened
* Brier score   mean squared error of the three probabilities (0 perfect, 2 worst)
* log loss      average -log(probability given to what happened)
* score hit     how often the most likely exact score happened

The same numbers are computed for the bookmaker (Bet365) from its odds, so the
model can be judged against a strong reference, and a simple flat-stake
"value bet" rule shows whether beating the odds is realistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import PoissonModel

OUTCOMES = ["p_home", "p_draw", "p_away"]


def outcome_index(home_goals, away_goals) -> np.ndarray:
    h, a = np.asarray(home_goals), np.asarray(away_goals)
    return np.where(h > a, 0, np.where(h == a, 1, 2))


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Bookmaker odds -> probabilities, with the margin removed proportionally."""
    raw = 1.0 / np.asarray(odds, dtype=float)
    return raw / raw.sum(axis=1, keepdims=True)


def brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    one_hot = np.eye(3)[outcomes]
    return float(((probabilities - one_hot) ** 2).sum(axis=1).mean())


def log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    picked = probabilities[np.arange(len(outcomes)), outcomes]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def backtest(
    matches: pd.DataFrame,
    test_season: int,
    *,
    train_seasons_back: int = 2,
    prior_games: float = 5.0,
) -> pd.DataFrame:
    """Walk-forward forecasts for every match of ``test_season``.

    Before each match day the model is fitted on the matches of the previous
    ``train_seasons_back`` seasons plus the test season so far. Returns the
    matches with the model's probabilities and most likely score attached.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    training = matches[matches["season"].between(test_season - train_seasons_back, test_season)]
    test = training[training["season"] == test_season]
    blocks = []
    for day, block in test.groupby(test["date"].dt.normalize()):
        history = training[training["date"] < day]
        if history.empty:
            continue
        model = PoissonModel(prior_games=prior_games).fit(history)
        blocks.append(pd.concat([block, model.predict_frame(block)], axis=1))
    if not blocks:
        raise ValueError(f"no matches to predict in season {test_season}")
    predictions = pd.concat(blocks)
    predictions["outcome"] = outcome_index(predictions["home_goals"], predictions["away_goals"])
    predictions["actual_score"] = predictions["home_goals"].astype(str) + "-" + predictions["away_goals"].astype(str)
    return predictions


def value_bets(predictions: pd.DataFrame, threshold: float = 0.05, odds_prefix: str = "odds") -> pd.DataFrame:
    """Flat 1-unit bets on the result whose expected value ``p * odds - 1`` is highest and above ``threshold``."""
    columns = [f"{odds_prefix}_home", f"{odds_prefix}_draw", f"{odds_prefix}_away"]
    priced = predictions.dropna(subset=columns)
    odds = priced[columns].to_numpy(dtype=float)
    probs = priced[OUTCOMES].to_numpy(dtype=float)
    ev = probs * odds - 1.0
    pick = ev.argmax(axis=1)
    rows = np.arange(len(priced))
    take = ev[rows, pick] > threshold
    bets = priced[take].copy()
    bets["selection"] = np.array(["home", "draw", "away"])[pick[take]]
    bets["odds"] = odds[rows, pick][take]
    bets["ev"] = ev[rows, pick][take]
    bets["won"] = pick[take] == bets["outcome"].to_numpy()
    bets["profit"] = np.where(bets["won"], bets["odds"] - 1.0, -1.0)
    return bets


def summarize(predictions: pd.DataFrame, threshold: float = 0.05) -> dict:
    """Model vs bookmaker scores and the flat-stake value-betting result."""
    outcomes = predictions["outcome"].to_numpy()
    probs = predictions[OUTCOMES].to_numpy(dtype=float)
    result = {
        "matches": len(predictions),
        "accuracy_model": float((probs.argmax(axis=1) == outcomes).mean()),
        "brier_model": brier(probs, outcomes),
        "log_loss_model": log_loss(probs, outcomes),
        "score_hit_model": float((predictions["likely_score"] == predictions["actual_score"]).mean()),
        "always_1_1_hit": float((predictions["actual_score"] == "1-1").mean()),
        "always_home_accuracy": float((outcomes == 0).mean()),
    }
    # The bookmaker's closing odds are the best available reference; fall back to opening odds.
    prefix = "closing" if predictions["closing_home"].notna().any() else "odds"
    columns = [f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]
    priced = predictions.dropna(subset=columns)
    if not priced.empty:
        market = implied_probabilities(priced[columns].to_numpy(dtype=float))
        market_outcomes = priced["outcome"].to_numpy()
        result.update(
            {
                "bookmaker_odds_used": prefix,
                "accuracy_bookmaker": float((market.argmax(axis=1) == market_outcomes).mean()),
                "brier_bookmaker": brier(market, market_outcomes),
                "log_loss_bookmaker": log_loss(market, market_outcomes),
            }
        )
    bets = value_bets(predictions, threshold, odds_prefix=prefix)
    result.update(
        {
            "bets": len(bets),
            "bets_won": int(bets["won"].sum()) if len(bets) else 0,
            "flat_stake_roi": float(bets["profit"].mean()) if len(bets) else float("nan"),
            "flat_stake_profit_units": float(bets["profit"].sum()) if len(bets) else 0.0,
        }
    )
    return result
