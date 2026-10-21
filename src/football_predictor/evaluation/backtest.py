"""Walk-forward forecasting, evaluation and betting simulation on a temporal test set."""

from __future__ import annotations

import itertools
import json
import logging
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

SELECTIONS = np.array(["home", "draw", "away"])
PROB_COLUMNS = ["p_home", "p_draw", "p_away"]
ODDS_COLUMNS = ["odds_home", "odds_draw", "odds_away"]


@dataclass
class BettingStrategy:
    ev_threshold: float = 0.05
    kelly_multiplier: float = 0.25
    max_stake_fraction: float = 0.05
    max_daily_exposure: float = 0.5
    min_odds: float = 1.01
    max_odds: float = 15.0


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
            market = implied_probabilities(odds)
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
    return metrics


def simulate_betting(
    predictions: pd.DataFrame,
    strategy: BettingStrategy | None = None,
    initial_bankroll: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bet the highest-EV outcome of each match when EV clears the threshold.

    Stakes are fractional Kelly on the bankroll at the start of each match day,
    capped per bet and per day, and settled at the published decimal odds.
    Returns (bets, equity curve by day).
    """
    strategy = strategy or BettingStrategy()
    priced = predictions.dropna(subset=ODDS_COLUMNS).sort_values("kickoff").reset_index(drop=True)
    probabilities = priced[PROB_COLUMNS].to_numpy(dtype=float)
    odds = priced[ODDS_COLUMNS].to_numpy(dtype=float)
    ev = expected_value(probabilities, odds)
    allowed = (odds >= strategy.min_odds) & (odds <= strategy.max_odds)
    ev_allowed = np.where(allowed, ev, -np.inf)
    pick = ev_allowed.argmax(axis=1)
    rows = np.arange(len(priced))
    candidates = priced.assign(
        selection_index=pick,
        selection=SELECTIONS[pick],
        odds=odds[rows, pick],
        p_model=probabilities[rows, pick],
        p_market=implied_probabilities(odds)[rows, pick] if len(priced) else [],
        ev=ev_allowed[rows, pick],
        kelly=kelly_fraction(probabilities[rows, pick], odds[rows, pick]),
    )
    candidates = candidates[candidates["ev"] > strategy.ev_threshold]

    bankroll = initial_bankroll
    equity = [{"date": priced["kickoff"].min().normalize() - pd.Timedelta(days=1), "bankroll": bankroll}] if len(
        priced
    ) else []
    bets = []
    for day, day_bets in candidates.groupby(candidates["kickoff"].dt.normalize()):
        fractions = np.minimum(strategy.kelly_multiplier * day_bets["kelly"].to_numpy(), strategy.max_stake_fraction)
        stakes = bankroll * fractions
        cap = strategy.max_daily_exposure * bankroll
        if stakes.sum() > cap:
            stakes *= cap / stakes.sum()
        won = day_bets["selection_index"].to_numpy() == day_bets["outcome"].to_numpy()
        profit = np.where(won, stakes * (day_bets["odds"].to_numpy() - 1.0), -stakes)
        bankroll += profit.sum()
        bets.append(day_bets.assign(stake=stakes, won=won, profit=profit, bankroll_after=bankroll))
        equity.append({"date": day, "bankroll": bankroll})

    columns = [
        "match_id", "kickoff", "home_team", "away_team", "selection", "odds", "p_model",
        "p_market", "ev", "kelly", "stake", "won", "profit", "bankroll_after",
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
) -> pd.DataFrame:
    """Out-of-sample score of every combination in ``grid`` on a validation window.

    Use a window that precedes the test set so the choice cannot leak test
    information. Rows are sorted best first by ``criterion``.
    """
    model_kwargs = model_kwargs or {}
    names = list(grid)
    rows = []
    for values in itertools.product(*(grid[name] for name in names)):
        params = dict(zip(names, values))
        factory = partial(DixonColesModel, **{**model_kwargs, **params})
        predictions = walk_forward_predictions(
            data, factory, start, end, refit_days=refit_days, divisions=divisions
        )
        rows.append({**params, **evaluate_forecasts(predictions, n_resamples=0)})
        log.info("%s -> %s=%.5f", params, criterion, rows[-1][criterion])
    return pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)


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
        market = implied_probabilities(priced[ODDS_COLUMNS].to_numpy(dtype=float))
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
