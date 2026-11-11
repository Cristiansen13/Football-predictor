"""Exploratory data analysis of the match table.

Prints the dataset's descriptive statistics, writes them as tables in
``analysis/results`` and draws the figures used in the presentations
(``docs/figures/eda_*.png``):

* results by season: how often the home side wins, draws, loses
* goals per team against the Poisson distribution the first model assumes
* the most frequent scores
* how the result depends on the Elo rating difference
* which pre-match features are related to the result

    python -m analysis.eda
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extraction.features import FEATURE_COLUMNS, TEAM_FEATURES, load_features
from extraction.preprocess import load_matches

from . import style

LOWER_IS_BETTER = {"ga5", "sota5", "ga38"}  # conceding less is the good direction
FEATURE_LABELS = {
    "elo": "Elo rating", "pts5": "Points per game, last 5", "gf5": "Goals scored, last 5",
    "ga5": "Goals conceded, last 5", "sot5": "Shots on target, last 5", "sota5": "Shots on target conceded, last 5",
    "gf38": "Goals scored, last 38", "ga38": "Goals conceded, last 38",
    "venue_pts5": "Points per game at this venue, last 5", "rest": "Rest days",
}


def season_table(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per season: matches, clubs, goals, result shares, shots, cards, bookmaker margin."""
    rows = []
    for season, block in matches.groupby("season"):
        margin = (1 / block[["odds_home", "odds_draw", "odds_away"]]).sum(axis=1).mean() - 1
        rows.append({
            "season": style.season_label(season),
            "matches": len(block),
            "clubs": len(set(block["home"]) | set(block["away"])),
            "goals_per_match": (block["home_goals"] + block["away_goals"]).mean(),
            "home_goals": block["home_goals"].mean(),
            "away_goals": block["away_goals"].mean(),
            "home_win": (block["result"] == "H").mean(),
            "draw": (block["result"] == "D").mean(),
            "away_win": (block["result"] == "A").mean(),
            "shots_per_match": (block["home_shots"] + block["away_shots"]).mean(),
            "yellow_per_match": (block["home_yellow"] + block["away_yellow"]).mean(),
            "bet365_margin": margin,
        })
    return pd.DataFrame(rows)


def column_table(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per column of the match table: type, missing values, range."""
    rows = []
    for column in matches.columns:
        values = matches[column]
        numeric = pd.api.types.is_numeric_dtype(values)
        rows.append({
            "column": column,
            "type": "number" if numeric else ("date" if column == "date" else "text"),
            "missing": int(values.isna().sum()),
            "distinct": int(values.nunique()),
            "min": values.min() if numeric else "",
            "mean": round(float(values.mean()), 2) if numeric else "",
            "max": values.max() if numeric else "",
        })
    return pd.DataFrame(rows)


def plot_outcomes_by_season(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(table))
    bottom = np.zeros(len(table))
    for label, column in (("Home win", "home_win"), ("Draw", "draw"), ("Away win", "away_win")):
        share = table[column].to_numpy() * 100
        ax.bar(x, share, bottom=bottom, width=0.55, color=style.OUTCOME_COLORS[label], label=label,
               edgecolor=style.SURFACE, linewidth=1.5)
        for position, value, base in zip(x, share, bottom):
            ax.text(position, base + value / 2, f"{value:.0f}", ha="center", va="center", fontsize=8,
                    color="white")
        bottom += share
    low = int(table["home_win"].to_numpy().argmin())
    ax.annotate(f"{table['season'].iloc[low]}, played mostly without crowds: lowest home share",
                xy=(low, 101), xytext=(low, 107), ha="center", fontsize=8.5, color=style.INK_SECONDARY,
                arrowprops={"arrowstyle": "-", "color": style.AXIS})
    ax.set_xticks(x, table["season"], rotation=45, ha="right")
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Share of matches (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.01, 1.0))
    ax.set_title("Results by season: the home side wins 44% of matches", pad=30)
    style.save(fig, "eda_outcomes_by_season.png")


def plot_goals_distribution(matches: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    goals = np.arange(0, 7)
    for ax, column, label, color in ((axes[0], "home_goals", "Home side", style.BLUE),
                                     (axes[1], "away_goals", "Away side", style.ORANGE)):
        observed = matches[column].clip(upper=6).value_counts(normalize=True).reindex(goals, fill_value=0) * 100
        mean = matches[column].mean()
        poisson = np.array([math.exp(-mean) * mean ** k / math.factorial(k) for k in goals]) * 100
        poisson[-1] = 100 - poisson[:-1].sum()  # 6 or more
        ax.bar(goals, observed, width=0.5, color=color, label="Observed")
        ax.plot(goals, poisson, color=style.INK, linewidth=1.5, marker="o", markersize=5,
                markeredgecolor=style.SURFACE, markeredgewidth=1.5, label=f"Poisson, mean {mean:.2f}")
        ax.set_xticks(goals, [str(k) for k in goals[:-1]] + ["6+"])
        ax.set_xlabel("Goals in a match")
        ax.set_title(label, fontsize=10.5)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Share of matches (%)")
    fig.suptitle("Goals per team follow a Poisson distribution closely", x=0.065, ha="left", fontsize=12,
                 fontweight="semibold", y=1.02)
    style.save(fig, "eda_goals_distribution.png")


def plot_scorelines(matches: pd.DataFrame) -> None:
    grid = (pd.crosstab(matches["home_goals"].clip(upper=5), matches["away_goals"].clip(upper=5), normalize=True)
            .reindex(index=range(6), columns=range(6), fill_value=0) * 100)
    fig, ax = plt.subplots(figsize=(4.9, 4.1))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blues", [style.SURFACE] + style.BLUES)
    ax.imshow(grid.to_numpy(), cmap=cmap, vmin=0, vmax=grid.to_numpy().max())
    for i in range(6):
        for j in range(6):
            value = grid.iat[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if value > 0.55 * grid.to_numpy().max() else style.INK)
    labels = [str(k) for k in range(5)] + ["5+"]
    ax.set_xticks(range(6), labels)
    ax.set_yticks(range(6), labels)
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("How often each score happens (% of matches)", y=1.14)
    style.save(fig, "eda_scorelines.png")


def elo_bins(features: pd.DataFrame) -> pd.DataFrame:
    """Result shares by Elo rating difference (home minus away), seasons after the warm-up one."""
    data = features[features["season"] > features["season"].min()]
    edges = [-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf]
    labels = ["below -200", "-200 to -100", "-100 to -50", "-50 to 0", "0 to 50", "50 to 100", "100 to 200", "above 200"]
    data = data.assign(bin=pd.cut(data["elo_diff"], edges, labels=labels))
    table = data.groupby("bin", observed=True)["result"].value_counts(normalize=True).unstack(fill_value=0)
    table = table.rename(columns={"H": "home_win", "D": "draw", "A": "away_win"})[["home_win", "draw", "away_win"]]
    table.insert(0, "matches", data.groupby("bin", observed=True).size())
    return table.reset_index()


def plot_elo(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    x = np.arange(len(table))
    for label, column in (("Home win", "home_win"), ("Draw", "draw"), ("Away win", "away_win")):
        values = table[column].to_numpy() * 100
        ax.plot(x, values, color=style.OUTCOME_COLORS[label], marker="o", markeredgecolor=style.SURFACE,
                markeredgewidth=2, label=label)
        ax.text(x[-1] + 0.15, values[-1], f"{label} {values[-1]:.0f}%", va="center", fontsize=9,
                color=style.INK_SECONDARY)
    ax.set_xticks(x, table["bin"], rotation=30, ha="right")
    ax.set_xlim(-0.3, len(table) + 0.55)
    ax.set_ylim(0, 85)
    ax.set_ylabel("Share of matches (%)")
    ax.set_xlabel("Elo rating of the home side minus the away side, before kickoff")
    ax.legend(ncol=3, loc="upper center")
    ax.set_title("The rating difference separates the results; draws stay near one in four")
    style.save(fig, "eda_elo_vs_result.png")


def feature_signal(features: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation of each feature (home value minus away value) with the home side's goal difference."""
    matches = load_matches().set_index("match_id")
    data = features[features["season"] > features["season"].min()].set_index("match_id")
    goal_difference = (matches["home_goals"] - matches["away_goals"]).loc[data.index]
    rows = []
    for name in TEAM_FEATURES:
        gap = data[f"home_{name}"] - data[f"away_{name}"]
        if name in LOWER_IS_BETTER:
            gap = -gap
        rows.append({"feature": name, "label": FEATURE_LABELS[name],
                     "correlation": gap.corr(goal_difference, method="spearman")})
    return pd.DataFrame(rows).sort_values("correlation", ascending=False).reset_index(drop=True)


def plot_feature_signal(table: pd.DataFrame) -> None:
    shown = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.barh(shown["label"], shown["correlation"], height=0.5, color=style.BLUE)
    for value, label in zip(shown["correlation"], shown["label"]):
        ax.text(value + 0.006, label, f"{value:.2f}", va="center", fontsize=9, color=style.INK_SECONDARY)
    ax.set_xlim(min(0, shown["correlation"].min() - 0.02), shown["correlation"].max() + 0.06)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rank correlation with the home side's goal difference")
    ax.set_title("Long-run strength carries the signal; recent form adds little\n"
                 "feature = home side's advantage over the away side before kickoff", fontsize=11)
    style.save(fig, "eda_feature_signal.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    style.apply_style()
    style.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    matches, features = load_matches(), load_features()

    clubs = len(set(matches["home"]) | set(matches["away"]))
    total_goals = matches["home_goals"] + matches["away_goals"]
    print(f"Matches: {len(matches)} in {matches['season'].nunique()} seasons, "
          f"{matches['date'].min():%d %b %Y} to {matches['date'].max():%d %b %Y}; {clubs} clubs; "
          f"{matches.shape[1]} columns, {len(FEATURE_COLUMNS)} engineered features")
    print(f"Results: home {(matches['result'] == 'H').mean():.1%}, draw {(matches['result'] == 'D').mean():.1%}, "
          f"away {(matches['result'] == 'A').mean():.1%}")
    print(f"Goals per match: {total_goals.mean():.2f} (home {matches['home_goals'].mean():.2f}, "
          f"away {matches['away_goals'].mean():.2f}); variance of home goals {matches['home_goals'].var():.2f}, "
          f"of away goals {matches['away_goals'].var():.2f}; over 2.5 goals in {(total_goals > 2.5).mean():.1%}")
    top = (matches["home_goals"].astype(str) + "-" + matches["away_goals"].astype(str)).value_counts(normalize=True)
    print("Most frequent scores: " + ", ".join(f"{score} {share:.1%}" for score, share in top.head(5).items()))

    seasons = season_table(matches)
    seasons.to_csv(style.RESULTS_DIR / "eda_seasons.csv", index=False, float_format="%.4f")
    print("\nBy season:")
    print(seasons.round(3).to_string(index=False))
    columns = column_table(matches)
    columns.to_csv(style.RESULTS_DIR / "eda_columns.csv", index=False)
    print("\nColumns with missing values:")
    print(columns[columns["missing"] > 0][["column", "missing"]].to_string(index=False))
    elo = elo_bins(features)
    elo.to_csv(style.RESULTS_DIR / "eda_elo_bins.csv", index=False, float_format="%.4f")
    print("\nResult shares by Elo difference:")
    print(elo.round(3).to_string(index=False))
    signal = feature_signal(features)
    signal.to_csv(style.RESULTS_DIR / "eda_feature_signal.csv", index=False, float_format="%.4f")
    print("\nFeature signal:")
    print(signal[["label", "correlation"]].round(3).to_string(index=False))

    plot_outcomes_by_season(seasons)
    plot_goals_distribution(matches)
    plot_scorelines(matches)
    plot_elo(elo)
    plot_feature_signal(signal)
    print(f"\nFigures in {style.FIGURES_DIR}")


if __name__ == "__main__":
    main()
