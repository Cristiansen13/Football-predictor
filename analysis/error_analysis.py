"""Error analysis: where the forecasts go wrong, and whether they are wrong in the same places as the bookmaker.

Reads ``analysis/results/predictions.csv`` (written by ``compare_models``) and
looks at the best machine-learning model, the Poisson model and the bookmaker:

* confusion matrices and recall per result: which results are never predicted
* calibration: do predicted probabilities match observed frequencies
* errors by kind of match (even match, clear favourite, strong favourite)
* errors by club: which clubs the model over- and under-rates
* errors by stage of the season
* disagreement with the bookmaker: who is right when the two differ, and what
  betting on the model's "value" would have returned
* the matches the model got most wrong

Tables go to ``analysis/results/error_*.csv``, figures to ``docs/figures/error_*.png``.

    python -m analysis.error_analysis
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style
from .evaluate import brier_per_match, score
from .models import MODEL_NAMES

PROBABILITIES = ["p_home", "p_draw", "p_away"]
RESULTS = ["Home win", "Draw", "Away win"]
OUR_MODELS = ["elo", "poisson", "logistic", "forest", "boosting"]
SHORT_NAMES = {"logistic": "Logistic regression", "poisson": "Poisson", "bookmaker": "Bookmaker", "elo": "Elo",
               "forest": "Random forest", "boosting": "Gradient boosting", "baseline": "Class frequencies"}


def probabilities(predictions: pd.DataFrame, key: str) -> np.ndarray:
    return predictions[[f"{key}_{p}" for p in PROBABILITIES]].to_numpy(dtype=float)


def confusion(predictions: pd.DataFrame, key: str) -> np.ndarray:
    """Rows: what happened. Columns: what the model called most likely."""
    predicted = probabilities(predictions, key).argmax(axis=1)
    table = np.zeros((3, 3), dtype=int)
    np.add.at(table, (predictions["outcome"].to_numpy(), predicted), 1)
    return table


def recall_table(predictions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key in keys:
        table = confusion(predictions, key)
        probs = probabilities(predictions, key)
        row = {"model": SHORT_NAMES[key]}
        for k, result in enumerate(RESULTS):
            label = result.lower().replace(" ", "_")
            row[f"predicted_{label}"] = table[:, k].sum() / table.sum()
            row[f"recall_{label}"] = table[k, k] / table[k].sum()
        row["highest_draw_probability"] = probs[:, 1].max()
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(predictions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Predicted probability against observed frequency, the three results pooled, in bins a tenth wide."""
    outcomes = np.eye(3)[predictions["outcome"].to_numpy()]
    edges = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0001])
    rows = []
    for key in keys:
        p, hit = probabilities(predictions, key).ravel(), outcomes.ravel()
        bins = np.digitize(p, edges) - 1
        for b in range(len(edges) - 1):
            inside = bins == b
            if inside.sum() >= 20:
                rows.append({"model": SHORT_NAMES[key], "bin": f"{edges[b]:.1f}-{min(edges[b + 1], 1):.1f}",
                             "forecasts": int(inside.sum()), "predicted": p[inside].mean(), "observed": hit[inside].mean()})
    return pd.DataFrame(rows)


def by_group(predictions: pd.DataFrame, groups: pd.Series, keys: list[str], name: str) -> pd.DataFrame:
    rows = []
    for value in groups.cat.categories:  # in the order the groups were defined
        block = predictions[groups == value]
        if block.empty:
            continue
        row = {name: value, "matches": len(block)}
        for key in keys:
            part = score(probabilities(block, key), block["outcome"].to_numpy())
            row[f"accuracy_{key}"], row[f"brier_{key}"] = part["accuracy"], part["brier"]
        rows.append(row)
    return pd.DataFrame(rows)


def match_type(predictions: pd.DataFrame) -> pd.Series:
    """How clear the favourite is, by the bookmaker's highest probability."""
    favourite = probabilities(predictions, "bookmaker").max(axis=1)
    labels = ["Even match (favourite below 45%)", "Clear favourite (45% to 65%)", "Strong favourite (above 65%)"]
    return pd.Series(pd.cut(favourite, [0, 0.45, 0.65, 1.0], labels=labels), index=predictions.index)


def season_stage(predictions: pd.DataFrame) -> pd.Series:
    """Rounds 1-5, 6-19 and 20-38, a round being ten consecutive matches of a season."""
    order = predictions.groupby("season")["date"].rank(method="first")
    rounds = np.ceil(order / 10)
    labels = ["Rounds 1 to 5", "Rounds 6 to 19", "Rounds 20 to 38"]
    return pd.Series(pd.cut(rounds, [0, 5, 19, 38], labels=labels), index=predictions.index)


def by_team(predictions: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per club: points per game won against the points the model expected."""
    probs = probabilities(predictions, key)
    outcome = predictions["outcome"].to_numpy()
    per_match = brier_per_match(probs, outcome)
    home = pd.DataFrame({"team": predictions["home"], "points": np.select([outcome == 0, outcome == 1], [3, 1], 0),
                         "expected": 3 * probs[:, 0] + probs[:, 1], "brier": per_match})
    away = pd.DataFrame({"team": predictions["away"], "points": np.select([outcome == 2, outcome == 1], [3, 1], 0),
                         "expected": 3 * probs[:, 2] + probs[:, 1], "brier": per_match})
    table = pd.concat([home, away]).groupby("team").agg(matches=("points", "size"), points_per_game=("points", "mean"),
                                                        expected_per_game=("expected", "mean"), brier=("brier", "mean"))
    table["surprise"] = table["points_per_game"] - table["expected_per_game"]
    return table.sort_values("surprise", ascending=False).reset_index()


def disagreement(predictions: pd.DataFrame, key: str) -> pd.DataFrame:
    """Both forecasts' scores by how far the model's probabilities sit from the bookmaker's."""
    gap = np.abs(probabilities(predictions, key) - probabilities(predictions, "bookmaker")).max(axis=1)
    labels = ["Within 5 points", "5 to 10 points apart", "More than 10 points apart"]
    groups = pd.Series(pd.cut(gap, [0, 0.05, 0.10, 1.0], labels=labels, include_lowest=True), index=predictions.index)
    return by_group(predictions, groups, [key, "bookmaker"], "gap_to_bookmaker")


def value_bets(predictions: pd.DataFrame, keys: list[str], threshold: float = 0.05) -> pd.DataFrame:
    """One unit on the result with the highest expected value ``p * odds - 1``, when it is above the threshold."""
    odds = predictions[["closing_home", "closing_draw", "closing_away"]].to_numpy(dtype=float)
    outcome = predictions["outcome"].to_numpy()
    rows = []
    for key in keys:
        ev = probabilities(predictions, key) * odds - 1.0
        pick = ev.argmax(axis=1)
        every = np.arange(len(predictions))
        take = ev[every, pick] > threshold
        won = pick[take] == outcome[take]
        profit = np.where(won, odds[every, pick][take] - 1.0, -1.0)
        rows.append({"model": SHORT_NAMES[key], "bets": int(take.sum()), "won": float(won.mean()) if take.any() else np.nan,
                     "mean_odds": float(odds[every, pick][take].mean()) if take.any() else np.nan,
                     "profit_units": float(profit.sum()), "roi": float(profit.mean()) if take.any() else np.nan})
    return pd.DataFrame(rows)


def biggest_misses(predictions: pd.DataFrame, key: str, count: int = 10) -> pd.DataFrame:
    """The matches where the model gave the actual result the least chance."""
    every = np.arange(len(predictions))
    outcome = predictions["outcome"].to_numpy()
    table = predictions[["date", "home", "away", "home_goals", "away_goals"]].copy()
    table["model_probability"] = probabilities(predictions, key)[every, outcome]
    table["bookmaker_probability"] = probabilities(predictions, "bookmaker")[every, outcome]
    return table.sort_values("model_probability").head(count).reset_index(drop=True)


def plot_confusion(predictions: pd.DataFrame, keys: list[str]) -> None:
    fig, axes = plt.subplots(1, len(keys), figsize=(9.2, 3.3))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blues", [style.SURFACE] + style.BLUES)
    for ax, key in zip(axes, keys):
        table = confusion(predictions, key)
        share = table / table.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{table[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 0.55 else style.INK)
        ax.set_xticks(range(3), ["Home", "Draw", "Away"])
        ax.set_yticks(range(3), ["Home", "Draw", "Away"] if ax is axes[0] else ["", "", ""])
        ax.set_xlabel("Predicted")
        ax.set_title(SHORT_NAMES[key], fontsize=10.5)
        ax.grid(visible=False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Actual result")
    fig.suptitle("Draws are almost never the most likely result (count and share of each actual result)",
                 x=0.07, ha="left", fontsize=12, fontweight="semibold", y=1.04)
    style.save(fig, "error_confusion.png")


def plot_calibration(table: pd.DataFrame, keys: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.plot([0, 0.85], [0, 0.85], color=style.AXIS, linewidth=1, label="Perfect calibration")
    for key, color in zip(keys, (style.BLUE, style.ORANGE, style.AQUA)):
        rows = table[table["model"] == SHORT_NAMES[key]]
        ax.plot(rows["predicted"], rows["observed"], color=color, marker="o",
                markeredgecolor=style.SURFACE, markeredgewidth=2, label=SHORT_NAMES[key])
    ticks = np.arange(0, 0.9, 0.2)
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_xlim(0, 0.88)
    ax.set_ylim(0, 0.88)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("How often it happened")
    ax.legend(loc="upper left")
    ax.set_title("Predicted probabilities match observed frequencies")
    style.save(fig, "error_calibration.png")


def plot_match_type(table: pd.DataFrame, key: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(table))
    width = 0.26
    for offset, column, color, label in ((-width / 2 - 0.01, f"brier_{key}", style.BLUE, SHORT_NAMES[key]),
                                         (width / 2 + 0.01, "brier_bookmaker", style.REFERENCE, "Bookmaker")):
        bars = ax.bar(x + offset, table[column], width=width, color=color, label=label)
        for bar, value in zip(bars, table[column]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=9,
                    color=style.INK_SECONDARY)
    ax.set_xticks(x, [f"{name}\n{count} matches" for name, count in zip(table["match_type"], table["matches"])])
    ax.set_ylim(0, table[[f"brier_{key}", "brier_bookmaker"]].to_numpy().max() * 1.2)
    ax.set_ylabel("Brier score (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Even matches are the hard ones, and the model trails the bookmaker in every kind of match")
    style.save(fig, "error_by_match_type.png")


def plot_team(table: pd.DataFrame, key: str) -> None:
    shown = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.barh(shown["team"], shown["surprise"], height=0.55, color=style.BLUE)
    ax.axvline(0, color=style.INK, linewidth=1)
    limit = shown["surprise"].abs().max() * 1.12
    ax.set_xlim(-limit, limit)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Points per game won minus points per game the model expected")
    ax.set_title(f"Clubs the {SHORT_NAMES[key].lower()} under-rated (right) and over-rated (left)\n"
                 "test seasons 2023/24 to 2025/26", fontsize=11)
    style.save(fig, "error_by_team.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    style.apply_style()
    path = style.RESULTS_DIR / "predictions.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `python -m analysis.compare_models` first")
    predictions = pd.read_csv(path, parse_dates=["date"])
    scores = {key: score(probabilities(predictions, key), predictions["outcome"].to_numpy())["brier"]
              for key in ("logistic", "forest", "boosting")}
    best = min(scores, key=scores.get)
    keys = [best, "poisson", "bookmaker"]
    print(f"{len(predictions)} test matches; best machine-learning model: {MODEL_NAMES[best]}\n")

    recall = recall_table(predictions, OUR_MODELS + ["bookmaker"])
    recall.to_csv(style.RESULTS_DIR / "error_recall.csv", index=False, float_format="%.4f")
    print("Share of matches each result is predicted in, recall per result, highest draw probability ever given:")
    print(recall.round(3).to_string(index=False))
    actual = predictions["outcome"].value_counts(normalize=True).sort_index()
    print(f"Actual shares: home {actual[0]:.3f}, draw {actual[1]:.3f}, away {actual[2]:.3f}")

    calibration = calibration_table(predictions, keys)
    calibration.to_csv(style.RESULTS_DIR / "error_calibration.csv", index=False, float_format="%.4f")
    print("\nCalibration (three results pooled):")
    print(calibration.round(3).to_string(index=False))

    kinds = by_group(predictions, match_type(predictions), keys, "match_type")
    kinds.to_csv(style.RESULTS_DIR / "error_by_match_type.csv", index=False, float_format="%.4f")
    print("\nBy kind of match:")
    print(kinds.round(3).to_string(index=False))

    stages = by_group(predictions, season_stage(predictions), keys, "stage")
    stages.to_csv(style.RESULTS_DIR / "error_by_stage.csv", index=False, float_format="%.4f")
    print("\nBy stage of the season:")
    print(stages.round(3).to_string(index=False))

    teams = by_team(predictions, best)
    teams.to_csv(style.RESULTS_DIR / "error_by_team.csv", index=False, float_format="%.4f")
    print(f"\nClubs the {MODEL_NAMES[best].lower()} under-rated most, then over-rated most (points per game):")
    print(pd.concat([teams.head(4), teams.tail(4)]).round(3).to_string(index=False))

    gaps = disagreement(predictions, best)
    gaps.to_csv(style.RESULTS_DIR / "error_disagreement.csv", index=False, float_format="%.4f")
    print("\nWhen the model and the bookmaker disagree:")
    print(gaps.round(3).to_string(index=False))

    bets = value_bets(predictions, OUR_MODELS)
    bets.to_csv(style.RESULTS_DIR / "error_value_bets.csv", index=False, float_format="%.4f")
    print("\nBetting one unit wherever a model sees more than 5% expected value at the closing odds:")
    print(bets.round(3).to_string(index=False))

    hit = (predictions["poisson_likely_score"] == predictions["home_goals"].astype(str) + "-"
           + predictions["away_goals"].astype(str)).mean()
    one_one = ((predictions["home_goals"] == 1) & (predictions["away_goals"] == 1)).mean()
    print(f"\nExact score: the Poisson model's most likely score was right in {hit:.1%} of matches; "
          f"always saying 1-1 would be right in {one_one:.1%}")

    misses = biggest_misses(predictions, best)
    misses.to_csv(style.RESULTS_DIR / "error_biggest_misses.csv", index=False, date_format="%Y-%m-%d",
                  float_format="%.4f")
    print("\nThe ten results the model thought least likely:")
    print(misses.assign(date=misses["date"].dt.strftime("%Y-%m-%d")).round(3).to_string(index=False))

    plot_confusion(predictions, keys)
    plot_calibration(calibration, keys)
    plot_match_type(kinds, best)
    plot_team(teams, best)
    print(f"\nTables in {style.RESULTS_DIR}, figures in {style.FIGURES_DIR}")


if __name__ == "__main__":
    main()
