"""Train every model, predict three test seasons walk-forward and compare with the bookmaker.

Protocol (nothing is ever predicted with information from its own day or later):

* The machine-learning models are refitted before each test season on every
  earlier season; the features themselves (ratings, form) keep updating match
  by match. Season 2014/15 only warms up the ratings and the form features.
* Their settings are chosen on two validation seasons (2021/22, 2022/23) that
  come before every test season, by log loss.
* The Poisson model is refitted before every match day on the two previous
  seasons plus the test season so far.
* The bookmaker's forecast is Bet365's closing odds with the margin removed.

Writes ``analysis/results/predictions.csv`` (every forecast of every model),
``model_comparison.csv``, ``model_comparison_by_season.csv``, ``tuning.csv``,
``feature_importance.csv`` and two figures in ``docs/figures``.

    python -m analysis.compare_models
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from extraction.features import FEATURE_COLUMNS, load_features
from extraction.preprocess import load_matches

from . import style
from .evaluate import backtest, brier_per_match, implied_probabilities, log_loss, paired_bootstrap, score
from .models import GRIDS, ML_MODELS, MODEL_NAMES, build, columns_for, fit_predict

TEST_SEASONS = [2023, 2024, 2025]
VALIDATION_SEASONS = [2021, 2022]
FIRST_TRAIN_SEASON = 2015
PROBABILITIES = ["p_home", "p_draw", "p_away"]
ORDER = ["baseline", "elo", "poisson", "logistic", "forest", "boosting", "bookmaker"]


def season_forward(features: pd.DataFrame, key: str, params: dict, seasons: list[int]) -> pd.DataFrame:
    """Forecasts of model ``key`` for ``seasons``, refitted before each one on the seasons before it."""
    blocks = []
    for season in seasons:
        train = features[features["season"].between(FIRST_TRAIN_SEASON, season - 1)]
        test = features[features["season"] == season]
        blocks.append(pd.DataFrame(fit_predict(key, params, train, test), columns=PROBABILITIES,
                                   index=test["match_id"]))
    return pd.concat(blocks)


def tune(features: pd.DataFrame) -> tuple[dict[str, dict], pd.DataFrame]:
    """Pick each model's setting by log loss on the validation seasons."""
    outcomes = features.set_index("match_id")["outcome"]
    chosen, rows = {}, []
    for key in ML_MODELS:
        best = None
        for params in GRIDS[key]:
            forecast = season_forward(features, key, params, VALIDATION_SEASONS)
            loss = log_loss(forecast.to_numpy(), outcomes.loc[forecast.index].to_numpy())
            rows.append({"model": key, "setting": str(params) if params else "-", "validation_log_loss": loss})
            if best is None or loss < best[0]:
                best = (loss, params)
        chosen[key] = best[1]
    table = pd.DataFrame(rows)
    table["chosen"] = [str(chosen[m]) == s or (not chosen[m] and s == "-") for m, s in zip(table["model"], table["setting"])]
    return chosen, table


def poisson_forward(matches: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """The Poisson model's forecasts, refitted before every match day."""
    blocks = [backtest(matches, season) for season in seasons]
    forecast = pd.concat(blocks).set_index("match_id")
    return forecast[PROBABILITIES + ["exp_home_goals", "exp_away_goals", "likely_score", "p_likely_score"]]


def bookmaker_forecast(matches: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Bet365's closing odds turned into probabilities (matches without closing odds are left out)."""
    columns = ["closing_home", "closing_draw", "closing_away"]
    priced = matches[matches["season"].isin(seasons)].dropna(subset=columns)
    return pd.DataFrame(implied_probabilities(priced[columns].to_numpy(dtype=float)), columns=PROBABILITIES,
                        index=priced["match_id"])


def comparison(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores per model over all test seasons (with the gap to the bookmaker) and the Brier score per season."""
    outcomes = predictions["outcome"].to_numpy()
    book = brier_per_match(predictions[[f"bookmaker_{p}" for p in PROBABILITIES]].to_numpy(), outcomes)
    rows, by_season = [], []
    for key in ORDER:
        probabilities = predictions[[f"{key}_{p}" for p in PROBABILITIES]].to_numpy()
        row = {"model": key, "name": MODEL_NAMES[key], "matches": len(predictions), **score(probabilities, outcomes)}
        difference, low, high = paired_bootstrap(brier_per_match(probabilities, outcomes), book)
        row.update({"brier_minus_bookmaker": difference, "ci_low": low, "ci_high": high})
        rows.append(row)
        for season, block in predictions.groupby("season"):
            part = score(block[[f"{key}_{p}" for p in PROBABILITIES]].to_numpy(), block["outcome"].to_numpy())
            by_season.append({"model": key, "name": MODEL_NAMES[key], "season": season, **part})
    return pd.DataFrame(rows), pd.DataFrame(by_season)


def feature_importance(features: pd.DataFrame, key: str, params: dict) -> pd.DataFrame:
    """How much the test log loss worsens when one feature is shuffled (model fitted on the seasons before)."""
    train = features[features["season"].between(FIRST_TRAIN_SEASON, TEST_SEASONS[0] - 1)]
    test = features[features["season"].isin(TEST_SEASONS)]
    columns = columns_for(key)
    model = build(key, **params).fit(train[columns], train["outcome"])
    result = permutation_importance(model, test[columns], test["outcome"], scoring="neg_log_loss",
                                    n_repeats=30, random_state=0)
    return (pd.DataFrame({"feature": columns, "importance": result.importances_mean, "std": result.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def plot_comparison(table: pd.DataFrame) -> None:
    """Each model's Brier score relative to the bookmaker's, with its bootstrap interval."""
    rows = table[table["model"] != "bookmaker"].sort_values("brier_minus_bookmaker", ascending=False)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(rows))
    colors = [style.REFERENCE if m == "baseline" else style.BLUE for m in rows["model"]]
    ax.hlines(y, rows["ci_low"], rows["ci_high"], color=colors, linewidth=2)
    ax.scatter(rows["brier_minus_bookmaker"], y, s=70, color=colors, edgecolor=style.SURFACE, linewidth=2, zorder=3)
    ax.axvline(0, color=style.INK, linewidth=1)
    for position, (_, row) in zip(y, rows.iterrows()):
        ax.text(row["ci_high"] + 0.002, position, f"{row['brier']:.3f}", va="center", color=style.INK_SECONDARY, fontsize=9)
    ax.text(0.001, len(rows) - 0.35, f"bookmaker {table.set_index('model').loc['bookmaker', 'brier']:.3f}",
            color=style.INK_SECONDARY, fontsize=9, va="center")
    ax.set_yticks(y, rows["name"])
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Brier score minus the bookmaker's (0 = as good as the bookmaker; right = worse)")
    ax.set_title(f"No model reaches the bookmaker: Brier score over {int(table['matches'].iloc[0]):,} test matches\n"
                 "dot = model, line = 95% bootstrap interval, number = the model's Brier score", fontsize=11)
    style.save(fig, "model_comparison.png")


def plot_importance(importance: pd.DataFrame, name: str) -> None:
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.barh(top["feature"], top["importance"], height=0.55, color=style.BLUE)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Increase in test log loss when the feature is shuffled")
    ax.set_title(f"What the {name.lower()} relies on (ten most important features)")
    style.save(fig, "feature_importance.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    style.apply_style()
    style.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    matches, features = load_matches(), load_features()

    chosen, tuning = tune(features)
    tuning.to_csv(style.RESULTS_DIR / "tuning.csv", index=False)
    print("Settings chosen on the validation seasons 2021/22 and 2022/23:")
    print(tuning[tuning["chosen"]][["model", "setting", "validation_log_loss"]].round(4).to_string(index=False))

    keep = ["match_id", "season", "date", "home", "away", "home_goals", "away_goals", "result",
            "odds_home", "odds_draw", "odds_away", "closing_home", "closing_draw", "closing_away"]
    predictions = matches[matches["season"].isin(TEST_SEASONS)][keep].set_index("match_id")
    predictions["outcome"] = predictions["result"].map({"H": 0, "D": 1, "A": 2})
    for key in ML_MODELS:
        forecast = season_forward(features, key, chosen[key], TEST_SEASONS)
        for column in PROBABILITIES:
            predictions[f"{key}_{column}"] = forecast[column]
    poisson = poisson_forward(matches, TEST_SEASONS)
    for column in poisson.columns:
        predictions[f"poisson_{column}"] = poisson[column]
    book = bookmaker_forecast(matches, TEST_SEASONS)
    for column in PROBABILITIES:
        predictions[f"bookmaker_{column}"] = book[column]
    predictions = predictions.dropna(subset=["bookmaker_p_home"]).reset_index()
    predictions.to_csv(style.RESULTS_DIR / "predictions.csv", index=False, date_format="%Y-%m-%d", float_format="%.5f")

    table, by_season = comparison(predictions)
    table.to_csv(style.RESULTS_DIR / "model_comparison.csv", index=False, float_format="%.5f")
    by_season.to_csv(style.RESULTS_DIR / "model_comparison_by_season.csv", index=False, float_format="%.5f")
    print(f"\nTest seasons 2023/24 to 2025/26, {len(predictions)} matches:")
    shown = table[["name", "accuracy", "brier", "log_loss", "brier_minus_bookmaker", "ci_low", "ci_high"]]
    print(shown.round(4).to_string(index=False))
    print("\nBrier score per season:")
    print(by_season.pivot(index="name", columns="season", values="brier").loc[[MODEL_NAMES[k] for k in ORDER]]
          .round(3).to_string())

    best = table[table["model"].isin(["logistic", "forest", "boosting"])].sort_values("brier")["model"].iloc[0]
    importance = feature_importance(features, best, chosen[best])
    importance.to_csv(style.RESULTS_DIR / "feature_importance.csv", index=False, float_format="%.5f")
    print(f"\nFeature importance of the {MODEL_NAMES[best].lower()} (top 8):")
    print(importance.head(8).round(4).to_string(index=False))
    plot_comparison(table)
    plot_importance(importance, MODEL_NAMES[best])
    print(f"\nTables in {style.RESULTS_DIR}, figures in {style.FIGURES_DIR}")


if __name__ == "__main__":
    main()
