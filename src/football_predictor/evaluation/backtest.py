"""Walk-forward forecasting, evaluation and betting simulation on a temporal test set."""

from __future__ import annotations

import itertools
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..models.dixon_coles import DixonColesModel
from .betting import expected_value, kelly_fraction
from .metrics import (
    accuracy,
    binary_brier,
    binary_brier_per_match,
    binary_log_loss,
    two_way_probabilities,
    bootstrap_interval,
    brier_per_match,
    brier_score,
    calibration_table,
    expected_calibration_error,
    implied_probabilities,
    log_loss,
    outcome_index,
    overround,
    ranked_probability_score,
    skill_score,
)

log = logging.getLogger(__name__)

SELECTIONS = np.array(["home", "draw", "away", "over", "under"])
PROB_COLUMNS = ["p_home", "p_draw", "p_away"]
ODDS_COLUMNS = ["odds_home", "odds_draw", "odds_away"]
TOTALS_PROB_COLUMNS = ["p_over_2_5", "p_under_2_5"]
TOTALS_ODDS_COLUMNS = ["odds_over_2_5", "odds_under_2_5"]
MARKETS = {"1x2": (0, 1, 2), "totals": (3, 4)}
# Prices a bet can be struck at: the closing reference, the earliest listed price,
# or the best price across bookmakers at closing.
PRICE_COLUMNS = {
    "closing": (ODDS_COLUMNS, TOTALS_ODDS_COLUMNS),
    "opening": (["open_home", "open_draw", "open_away"], ["open_over_2_5", "open_under_2_5"]),
    "max": (["max_home", "max_draw", "max_away"], ["max_over_2_5", "max_under_2_5"]),
}
# Market probabilities are de-margined with the power method: the proportional
# method hands longshots too much probability and makes them look like value.
MARKET_DEMARGIN = "power"


@dataclass
class BettingStrategy:
    ev_threshold: float = 0.05
    kelly_multiplier: float = 0.25
    max_stake_fraction: float = 0.05
    max_daily_exposure: float = 0.5
    min_odds: float = 1.01
    max_odds: float = 15.0
    markets: tuple[str, ...] = ("1x2", "totals")
    # Probability the EV is judged with: the model, the de-margined closing reference
    # price (the sharp line as fair value), or a blend of the two.
    fair: str = "model"
    blend_weight: float = 0.5  # weight on the market in a blend
    # Price the bet is struck and settled at (see PRICE_COLUMNS).
    price: str = "closing"


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    bets: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    initial_bankroll: float = 1000.0


def _played(data: pd.DataFrame) -> pd.Series:
    return data["home_goals"].notna() & data["away_goals"].notna()


def _in_divisions(data: pd.DataFrame, divisions: Iterable[str] | None) -> pd.Series:
    if divisions is None or "division" not in data:
        return pd.Series(True, index=data.index)
    return data["division"].isin(list(divisions))


def walk_forward_predictions(
    data: pd.DataFrame,
    model_factory: Callable[[], DixonColesModel],
    start: pd.Timestamp | str,
    end: pd.Timestamp | str | None = None,
    *,
    refit_days: int = 7,
    divisions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Refit every ``refit_days`` on matches before the block, predict the block.

    The model never sees a result from on or after the block start, so every
    forecast is one that could have been made at the time. It trains on every
    division in ``data`` but only matches in ``divisions`` are predicted.
    """
    data = data.assign(kickoff=pd.to_datetime(data["kickoff"])).sort_values("kickoff")
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else data["kickoff"].max() + pd.Timedelta(seconds=1)
    window = (data["kickoff"] >= start) & (data["kickoff"] < end)
    test = data[_played(data) & window & _in_divisions(data, divisions)]

    boundaries = list(pd.date_range(start, end, freq=f"{refit_days}D")) + [end]
    model, blocks = None, []
    for block_start, block_end in zip(boundaries[:-1], boundaries[1:]):
        block = test[(test["kickoff"] >= block_start) & (test["kickoff"] < block_end)]
        if block.empty:
            continue
        model = model_factory().fit(data, as_of=block_start, warm_start=model)
        predicted = model.predict(block)
        predicted["score_loglik"] = model.log_likelihood(block)
        blocks.append(pd.concat([block, predicted], axis=1).assign(fitted_as_of=block_start))
    if not blocks:
        raise ValueError(f"No played matches between {start} and {end}")
    predictions = pd.concat(blocks).reset_index(drop=True)
    predictions["outcome"] = outcome_index(predictions["home_goals"], predictions["away_goals"])
    goals_home = predictions["home_goals"].to_numpy(dtype=float)
    goals_away = predictions["away_goals"].to_numpy(dtype=float)
    predictions["over"] = (goals_home + goals_away > 2.5).astype(int)
    predictions["btts"] = ((goals_home > 0) & (goals_away > 0)).astype(int)
    return predictions


def base_rates(data: pd.DataFrame, before: pd.Timestamp, divisions: Iterable[str] | None = None) -> np.ndarray:
    """Home/draw/away frequencies of matches before ``before``: the 'climatology' forecast."""
    past = data[_played(data) & (pd.to_datetime(data["kickoff"]) < before) & _in_divisions(data, divisions)]
    if past.empty:
        return np.array([0.45, 0.25, 0.30])
    counts = np.bincount(outcome_index(past["home_goals"], past["away_goals"]), minlength=3)
    return counts / counts.sum()


def evaluate_forecasts(
    predictions: pd.DataFrame,
    reference_rates: np.ndarray | None = None,
    *,
    n_resamples: int = 2000,
) -> dict:
    """Scoring rules, calibration and skill against the base rate and the market.

    ``brier_diff_vs_market`` is model minus market per match (negative = model
    better), with a paired bootstrap interval when ``n_resamples`` > 0.
    """
    outcomes = predictions["outcome"].to_numpy()
    model = predictions[PROB_COLUMNS].to_numpy(dtype=float)
    metrics = {
        "matches": len(predictions),
        "brier_model": brier_score(model, outcomes),
        "rps_model": ranked_probability_score(model, outcomes),
        "log_loss_model": log_loss(model, outcomes),
        "accuracy_model": accuracy(model, outcomes),
        "ece_model": expected_calibration_error(model, outcomes),
    }
    if "score_loglik" in predictions:
        metrics["score_log_loss_model"] = float(-predictions["score_loglik"].mean())
    if reference_rates is not None:
        climatology = np.tile(reference_rates, (len(predictions), 1))
        metrics["brier_base_rate"] = brier_score(climatology, outcomes)
        metrics["brier_skill_vs_base_rate"] = skill_score(metrics["brier_model"], metrics["brier_base_rate"])

    if all(c in predictions for c in ODDS_COLUMNS):
        priced = predictions.dropna(subset=ODDS_COLUMNS)
        if not priced.empty:
            odds = priced[ODDS_COLUMNS].to_numpy(dtype=float)
            market = implied_probabilities(odds, MARKET_DEMARGIN)
            priced_outcomes = priced["outcome"].to_numpy()
            priced_model = priced[PROB_COLUMNS].to_numpy(dtype=float)
            diff = brier_per_match(priced_model, priced_outcomes) - brier_per_match(market, priced_outcomes)
            metrics.update(
                {
                    "matches_with_odds": len(priced),
                    "brier_model_priced": brier_score(priced_model, priced_outcomes),
                    "brier_market": brier_score(market, priced_outcomes),
                    "rps_market": ranked_probability_score(market, priced_outcomes),
                    "log_loss_market": log_loss(market, priced_outcomes),
                    "accuracy_market": accuracy(market, priced_outcomes),
                    "ece_market": expected_calibration_error(market, priced_outcomes),
                    "mean_overround": float(overround(odds).mean()),
                    "brier_diff_vs_market": float(diff.mean()),
                }
            )
            metrics["brier_skill_vs_market"] = skill_score(metrics["brier_model_priced"], metrics["brier_market"])
            if n_resamples:
                low, high, _ = bootstrap_interval(np.mean, diff, n_resamples=n_resamples)
                metrics["brier_diff_ci_low"], metrics["brier_diff_ci_high"] = low, high
    metrics.update(evaluate_two_way_markets(predictions, n_resamples=n_resamples))
    return metrics


def evaluate_two_way_markets(predictions: pd.DataFrame, *, n_resamples: int = 2000) -> dict:
    """Over/under 2.5 (against the market where priced) and both teams to score."""
    metrics: dict = {}
    if "over" in predictions and "p_over_2_5" in predictions:
        over = predictions["over"].to_numpy(dtype=float)
        p_over = predictions["p_over_2_5"].to_numpy(dtype=float)
        metrics.update(
            {
                "over_rate": float(over.mean()),
                "brier_over_model": binary_brier(p_over, over),
                "log_loss_over_model": binary_log_loss(p_over, over),
            }
        )
        if all(c in predictions for c in TOTALS_ODDS_COLUMNS):
            priced = predictions.dropna(subset=TOTALS_ODDS_COLUMNS)
            if not priced.empty:
                market = two_way_probabilities(priced["odds_over_2_5"], priced["odds_under_2_5"], MARKET_DEMARGIN)[:, 0]
                y = priced["over"].to_numpy(dtype=float)
                model = priced["p_over_2_5"].to_numpy(dtype=float)
                diff = binary_brier_per_match(model, y) - binary_brier_per_match(market, y)
                metrics.update(
                    {
                        "matches_with_totals_odds": len(priced),
                        "brier_over_model_priced": binary_brier(model, y),
                        "brier_over_market": binary_brier(market, y),
                        "brier_over_diff_vs_market": float(diff.mean()),
                        "mean_overround_totals": float(
                            overround(priced[TOTALS_ODDS_COLUMNS].to_numpy(dtype=float)).mean()
                        ),
                    }
                )
                if n_resamples:
                    low, high, _ = bootstrap_interval(np.mean, diff, n_resamples=n_resamples)
                    metrics["over_diff_ci_low"], metrics["over_diff_ci_high"] = low, high
    if "btts" in predictions and "p_btts" in predictions:
        btts = predictions["btts"].to_numpy(dtype=float)
        p_btts = predictions["p_btts"].to_numpy(dtype=float)
        metrics.update(
            {
                "btts_rate": float(btts.mean()),
                "brier_btts_model": binary_brier(p_btts, btts),
                "log_loss_btts_model": binary_log_loss(p_btts, btts),
            }
        )
    return metrics


def _price_columns(predictions: pd.DataFrame, strategy: BettingStrategy) -> tuple[list[str], list[str]]:
    """Columns of the price to bet at, falling back to the closing reference when absent."""
    three, two = PRICE_COLUMNS[strategy.price]
    if strategy.price != "closing" and not (all(c in predictions for c in three) or all(c in predictions for c in two)):
        log.warning("no %s prices in the data; betting at the closing reference price instead", strategy.price)
        return PRICE_COLUMNS["closing"]
    return three, two


def _selection_frame(
    predictions: pd.DataFrame, strategy: BettingStrategy
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows with at least one allowed, priced selection, and per selection: the
    probability the EV is judged with, the odds taken, the de-margined closing
    probability, and the closing-line value ``p_closing × odds − 1``."""
    n = len(predictions)
    shape = (n, len(SELECTIONS))
    model = np.full(shape, np.nan)
    closing = np.full(shape, np.nan)
    odds = np.full(shape, np.nan)
    three, two = _price_columns(predictions, strategy)
    if "1x2" in strategy.markets and all(c in predictions for c in ODDS_COLUMNS) and all(c in predictions for c in three):
        model[:, :3] = predictions[PROB_COLUMNS].to_numpy(dtype=float)
        odds[:, :3] = predictions[three].to_numpy(dtype=float)
        reference = predictions[ODDS_COLUMNS].to_numpy(dtype=float)
        priced = ~np.isnan(reference).any(axis=1)
        closing[priced, :3] = implied_probabilities(reference[priced], MARKET_DEMARGIN)
    if ("totals" in strategy.markets and all(c in predictions for c in TOTALS_ODDS_COLUMNS + TOTALS_PROB_COLUMNS)
            and all(c in predictions for c in two)):
        model[:, 3:] = predictions[TOTALS_PROB_COLUMNS].to_numpy(dtype=float)
        odds[:, 3:] = predictions[two].to_numpy(dtype=float)
        reference = predictions[TOTALS_ODDS_COLUMNS].to_numpy(dtype=float)
        priced = ~np.isnan(reference).any(axis=1)
        closing[priced, 3:] = implied_probabilities(reference[priced], MARKET_DEMARGIN)
    if strategy.fair == "model":
        fair = model
    elif strategy.fair == "market":
        fair = closing
    elif strategy.fair == "blend":
        fair = strategy.blend_weight * closing + (1 - strategy.blend_weight) * model
    else:
        raise ValueError(f"unknown fair-value source {strategy.fair!r}")
    allowed = (np.isfinite(odds) & np.isfinite(fair) & np.isfinite(closing)
               & (odds >= strategy.min_odds) & (odds <= strategy.max_odds))
    keep = allowed.any(axis=1)
    odds = np.where(allowed, odds, np.nan)
    clv = closing * odds - 1.0
    return predictions[keep].reset_index(drop=True), fair[keep], odds[keep], closing[keep], clv[keep]


def _won(selection_index: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    outcome = frame["outcome"].to_numpy()
    over = frame["over"].to_numpy() if "over" in frame else np.zeros(len(frame), dtype=int)
    return np.where(selection_index < 3, selection_index == outcome,
                    np.where(selection_index == 3, over == 1, over == 0))


def simulate_betting(
    predictions: pd.DataFrame,
    strategy: BettingStrategy | None = None,
    initial_bankroll: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bet the highest-EV selection of each match when EV clears the threshold.

    Selections span the markets in ``strategy.markets``: home/draw/away at the
    1X2 prices and over/under 2.5 goals at the totals prices. The EV is judged
    with ``strategy.fair`` (the model, the de-margined closing line, or a
    blend) at the price in ``strategy.price`` (closing reference, opening, or
    best price across bookmakers). Stakes are fractional Kelly on the bankroll
    at the start of each match day, capped per bet and per day, and settled at
    the price taken. Every bet also records its closing-line value: the return
    the de-margined closing price expects for that bet, the yardstick that
    separates an edge from luck. Returns (bets, equity curve by day).
    """
    strategy = strategy or BettingStrategy()
    ordered = predictions.sort_values("kickoff").reset_index(drop=True)
    priced, probabilities, odds, market, clv = _selection_frame(ordered, strategy)
    ev = np.where(np.isnan(odds), -np.inf, expected_value(np.nan_to_num(probabilities), np.nan_to_num(odds, nan=1.0)))
    pick = ev.argmax(axis=1) if len(priced) else np.array([], dtype=int)
    rows = np.arange(len(priced))
    candidates = priced.assign(
        selection_index=pick,
        selection=SELECTIONS[pick],
        market=np.where(pick < 3, "1x2", "totals"),
        odds=odds[rows, pick],
        p_model=probabilities[rows, pick],
        p_market=market[rows, pick],
        ev=ev[rows, pick],
        clv=clv[rows, pick],
        kelly=kelly_fraction(probabilities[rows, pick], odds[rows, pick]),
    )
    candidates = candidates[candidates["ev"] > strategy.ev_threshold]

    bankroll = initial_bankroll
    equity = [{"date": ordered["kickoff"].min().normalize() - pd.Timedelta(days=1), "bankroll": bankroll}] if len(
        ordered
    ) else []
    bets = []
    for day, day_bets in candidates.groupby(candidates["kickoff"].dt.normalize()):
        fractions = np.minimum(strategy.kelly_multiplier * day_bets["kelly"].to_numpy(), strategy.max_stake_fraction)
        stakes = bankroll * fractions
        cap = strategy.max_daily_exposure * bankroll
        if stakes.sum() > cap:
            stakes *= cap / stakes.sum()
        won = _won(day_bets["selection_index"].to_numpy(), day_bets)
        profit = np.where(won, stakes * (day_bets["odds"].to_numpy() - 1.0), -stakes)
        bankroll += profit.sum()
        bets.append(day_bets.assign(stake=stakes, won=won, profit=profit, bankroll_after=bankroll))
        equity.append({"date": day, "bankroll": bankroll})

    columns = [
        "match_id", "kickoff", "home_team", "away_team", "market", "selection", "odds", "p_model",
        "p_market", "ev", "clv", "kelly", "stake", "won", "profit", "bankroll_after",
    ]
    bets_frame = pd.concat(bets, ignore_index=True)[columns] if bets else pd.DataFrame(columns=columns)
    equity_frame = pd.DataFrame(equity, columns=["date", "bankroll"])
    equity_frame["cumulative_pnl"] = equity_frame["bankroll"] - initial_bankroll
    return bets_frame, equity_frame


def betting_metrics(
    bets: pd.DataFrame, equity: pd.DataFrame, initial_bankroll: float, *, n_resamples: int = 2000
) -> dict:
    """Profitability of the simulated bets.

    Significance is measured on flat 1-unit stakes, which do not depend on the
    bankroll path: ``p_flat_roi_le_zero`` is the bootstrap probability that the
    true flat-stake ROI is zero or negative.
    """
    if bets.empty:
        return {"bets": 0, "staked": 0.0, "profit": 0.0, "roi": float("nan"), "final_bankroll": initial_bankroll}
    peak = equity["bankroll"].cummax()
    flat_profit = np.where(bets["won"], bets["odds"] - 1.0, -1.0).astype(float)
    metrics = {
        "bets": len(bets),
        "staked": float(bets["stake"].sum()),
        "profit": float(bets["profit"].sum()),
        "roi": float(bets["profit"].sum() / bets["stake"].sum()),
        "final_bankroll": float(equity["bankroll"].iloc[-1]),
        "bankroll_growth": float(equity["bankroll"].iloc[-1] / initial_bankroll - 1.0),
        "max_drawdown": float(((peak - equity["bankroll"]) / peak).max()),
        "hit_rate": float(bets["won"].mean()),
        "mean_odds": float(bets["odds"].mean()),
        "mean_ev": float(bets["ev"].mean()),
        "flat_stake_profit_units": float(flat_profit.sum()),
        "flat_stake_roi": float(flat_profit.mean()),
        "mean_clv": float(bets["clv"].mean()) if "clv" in bets else float("nan"),
    }
    if n_resamples and len(bets) > 1:
        low, high, stats = bootstrap_interval(np.mean, flat_profit, n_resamples=n_resamples)
        metrics.update(
            {"flat_roi_ci_low": low, "flat_roi_ci_high": high, "p_flat_roi_le_zero": float((stats <= 0).mean())}
        )
    for selection in SELECTIONS:
        picked = bets[bets["selection"] == selection]
        metrics[f"bets_{selection}"] = len(picked)
        metrics[f"roi_{selection}"] = (
            float(picked["profit"].sum() / picked["stake"].sum()) if len(picked) else float("nan")
        )
    for market in MARKETS:
        picked = bets[bets["market"] == market] if "market" in bets else bets.iloc[0:0]
        metrics[f"bets_{market}"] = len(picked)
        flat = np.where(picked["won"], picked["odds"] - 1.0, -1.0).astype(float)
        metrics[f"flat_stake_roi_{market}"] = float(flat.mean()) if len(picked) else float("nan")
        metrics[f"clv_{market}"] = float(picked["clv"].mean()) if len(picked) and "clv" in picked else float("nan")
    return metrics


def run_backtest(
    data: pd.DataFrame,
    model_factory: Callable[[], DixonColesModel],
    start: pd.Timestamp | str,
    end: pd.Timestamp | str | None = None,
    *,
    refit_days: int = 7,
    strategy: BettingStrategy | None = None,
    initial_bankroll: float = 1000.0,
    divisions: Iterable[str] | None = None,
    n_resamples: int = 2000,
) -> BacktestResult:
    predictions = walk_forward_predictions(
        data, model_factory, start, end, refit_days=refit_days, divisions=divisions
    )
    bets, equity = simulate_betting(predictions, strategy, initial_bankroll)
    rates = base_rates(data, pd.Timestamp(start), divisions)
    metrics = evaluate_forecasts(predictions, rates, n_resamples=n_resamples)
    metrics.update(betting_metrics(bets, equity, initial_bankroll, n_resamples=n_resamples))
    return BacktestResult(predictions, bets, equity, metrics, initial_bankroll)


def tune_hyperparameters(
    data: pd.DataFrame,
    grid: dict[str, Sequence],
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    *,
    model_kwargs: dict | None = None,
    refit_days: int = 14,
    criterion: str = "brier_model",
    divisions: Iterable[str] | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Out-of-sample score of every combination in ``grid`` on a validation window.

    Use a window that precedes the test set so the choice cannot leak test
    information; a window of two seasons is far more stable than one, because
    the ranking of settings changes a lot from season to season. Rows are
    sorted best first by ``criterion``. Combinations run in ``n_jobs``
    processes.
    """
    model_kwargs = model_kwargs or {}
    names = list(grid)
    combinations = [dict(zip(names, values)) for values in itertools.product(*(grid[name] for name in names))]
    score = partial(_score_combination, data=data, model_kwargs=model_kwargs, start=start, end=end,
                    refit_days=refit_days, divisions=list(divisions) if divisions is not None else None)
    if n_jobs > 1 and len(combinations) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(combinations))) as pool:
            rows = list(pool.map(score, combinations))
    else:
        rows = [score(params) for params in combinations]
    for row in rows:
        log.info("%s -> %s=%.5f", {k: row[k] for k in names}, criterion, row[criterion])
    return pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)


def _score_combination(params: dict, *, data, model_kwargs, start, end, refit_days, divisions) -> dict:
    factory = partial(DixonColesModel, **{**model_kwargs, **params})
    predictions = walk_forward_predictions(data, factory, start, end, refit_days=refit_days, divisions=divisions)
    return {**params, **evaluate_forecasts(predictions, n_resamples=0)}


def tune_xi(
    data: pd.DataFrame,
    candidates: Iterable[float],
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    **kwargs,
) -> pd.DataFrame:
    """:func:`tune_hyperparameters` over the time-decay rate only."""
    return tune_hyperparameters(data, {"xi": list(candidates)}, start, end, **kwargs)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pnl(result: BacktestResult, path: str | Path, title: str = "Cumulative profit & loss") -> Path:
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[2, 1])

    equity = result.equity
    top.plot(
        equity["date"], equity["cumulative_pnl"], color="#1f6feb", linewidth=1.8,
        drawstyle="steps-post", label="Fractional Kelly",
    )
    top.axhline(0, color="#888888", linewidth=0.8)
    top.set_ylabel("Cumulative PnL")
    m = result.metrics
    top.set_title(
        f"{title}\n{m.get('bets', 0)} bets · ROI {m.get('roi', float('nan')):+.1%} · "
        f"bankroll {result.initial_bankroll:,.0f} → {m.get('final_bankroll', result.initial_bankroll):,.0f}"
    )
    top.legend(loc="upper left")

    if not result.bets.empty:
        bets = result.bets
        units = pd.Series(np.where(bets["won"], bets["odds"] - 1.0, -1.0), index=bets["kickoff"].dt.normalize())
        flat = units.groupby(level=0).sum().cumsum()
        bottom.plot(
            flat.index, flat.to_numpy(), color="#d29922", linewidth=1.4,
            drawstyle="steps-post", label="Flat 1-unit stakes",
        )
        bottom.legend(loc="upper left")
    bottom.axhline(0, color="#888888", linewidth=0.8)
    bottom.set_ylabel("Units")
    bottom.set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_calibration(predictions: pd.DataFrame, path: str | Path, bins: int = 10) -> Path:
    """Reliability diagram per outcome: forecast probability vs observed frequency."""
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    outcomes = predictions["outcome"].to_numpy()
    sources = {"Model": calibration_table(predictions[PROB_COLUMNS].to_numpy(dtype=float), outcomes, bins)}
    priced = predictions.dropna(subset=ODDS_COLUMNS) if all(c in predictions for c in ODDS_COLUMNS) else None
    if priced is not None and not priced.empty:
        market = implied_probabilities(priced[ODDS_COLUMNS].to_numpy(dtype=float), MARKET_DEMARGIN)
        sources["Market"] = calibration_table(market, priced["outcome"].to_numpy(), bins)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    styles = {"Model": ("#1f6feb", "o"), "Market": ("#d29922", "s")}
    for ax, outcome in zip(axes, ("home", "draw", "away")):
        ax.plot([0, 1], [0, 1], color="#888888", linewidth=0.8, linestyle="--")
        for name, table in sources.items():
            rows = table[table["outcome"] == outcome]
            color, marker = styles[name]
            ax.plot(rows["mean_forecast"], rows["observed"], color=color, marker=marker, linewidth=1.2,
                    markersize=4, label=name)
        ax.set_title(f"{outcome.capitalize()} win" if outcome != "draw" else "Draw")
        ax.set_xlabel("Forecast probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_report(result: BacktestResult, directory: str | Path, label: str, title: str | None = None) -> dict[str, Path]:
    """Write metrics, predictions, bets, calibration and plots for one backtest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": directory / f"metrics_{label}.json",
        "predictions": directory / f"predictions_{label}.csv",
        "bets": directory / f"bets_{label}.csv",
        "calibration": directory / f"calibration_{label}.csv",
        "pnl_plot": directory / f"pnl_{label}.png",
        "calibration_plot": directory / f"calibration_{label}.png",
    }
    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in result.metrics.items()}
    paths["metrics"].write_text(json.dumps(clean, indent=2, default=float))
    result.predictions.to_csv(paths["predictions"], index=False)
    result.bets.to_csv(paths["bets"], index=False)
    calibration_table(result.predictions[PROB_COLUMNS].to_numpy(dtype=float),
                      result.predictions["outcome"].to_numpy()).to_csv(paths["calibration"], index=False)
    plot_pnl(result, paths["pnl_plot"], title=title or "Cumulative profit & loss")
    plot_calibration(result.predictions, paths["calibration_plot"])
    return paths
