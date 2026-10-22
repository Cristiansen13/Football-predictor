"""Poisson goal model built from team scoring averages (the classic Maher approach).

The number of goals a team scores is treated as a Poisson variable. Its mean
is the league's average for that venue, scaled up or down by the team's
attack strength and the opponent's defence weakness:

    home goals ~ Poisson(home_attack * away_defence * league_home_average)
    away goals ~ Poisson(away_attack * home_defence * league_away_average)

A team's attack strength is its goals scored per game divided by the league
average; its defence weakness is goals conceded per game divided by the
league average. 1.0 means average, 1.3 means 30% better (attack) or 30%
worse (defence). The two Poisson variables are treated as independent, which
is what makes every probability a simple product.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def poisson_pmf(k: np.ndarray, mean: float) -> np.ndarray:
    """P(X = k) for a Poisson variable with the given mean, for each k."""
    k = np.asarray(k)
    factorials = np.array([math.factorial(int(i)) for i in k], dtype=float)
    return np.exp(-mean) * mean ** k / factorials


class PoissonModel:
    def __init__(self, max_goals: int = 6, prior_games: float = 5.0):
        """
        ``prior_games`` smooths the strengths of teams with few matches: each
        team is treated as if it had played this many extra games at exactly
        the league average, so three lucky games do not make a side look
        unbeatable (a newly promoted club starts at 1.0 and moves from there).
        """
        self.max_goals = max_goals
        self.prior_games = prior_games
        self.strengths: pd.DataFrame | None = None
        self.home_average = self.away_average = float("nan")

    # -- fitting ------------------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        """Compute league averages and every team's attack/defence strengths."""
        played = matches.dropna(subset=["home_goals", "away_goals"])
        if played.empty:
            raise ValueError("no played matches to fit on")
        self.home_average = float(played["home_goals"].mean())
        self.away_average = float(played["away_goals"].mean())
        k = self.prior_games
        teams = sorted(set(played["home"]) | set(played["away"]))

        # Goals scored and conceded, and games played, at each venue.
        at_home = played.groupby("home").agg(scored=("home_goals", "sum"), conceded=("away_goals", "sum"),
                                             games=("home_goals", "size")).reindex(teams, fill_value=0)
        away = played.groupby("away").agg(scored=("away_goals", "sum"), conceded=("home_goals", "sum"),
                                          games=("away_goals", "size")).reindex(teams, fill_value=0)

        def strength(goals, games, average):
            # (goals + k * average) / (games + k) per game, relative to the average:
            # with few games the ratio is pulled towards 1.0.
            return (goals + k * average) / ((games + k) * average)

        self.strengths = pd.DataFrame(
            {
                "home_attack": strength(at_home["scored"], at_home["games"], self.home_average),
                "home_defence": strength(at_home["conceded"], at_home["games"], self.away_average),
                "away_attack": strength(away["scored"], away["games"], self.away_average),
                "away_defence": strength(away["conceded"], away["games"], self.home_average),
                "games": at_home["games"] + away["games"],
            },
            index=pd.Index(teams, name="team"),
        )
        return self

    def _strength(self, team: str, column: str) -> float:
        if self.strengths is not None and team in self.strengths.index:
            return float(self.strengths.loc[team, column])
        return 1.0  # unknown team: assume average

    # -- prediction ---------------------------------------------------------

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        lam = self._strength(home, "home_attack") * self._strength(away, "away_defence") * self.home_average
        mu = self._strength(away, "away_attack") * self._strength(home, "home_defence") * self.away_average
        return lam, mu

    def score_grid(self, home: str, away: str) -> pd.DataFrame:
        """P(home scores i, away scores j) for i, j in 0..max_goals, renormalised to sum to one."""
        lam, mu = self.expected_goals(home, away)
        goals = np.arange(self.max_goals + 1)
        grid = np.outer(poisson_pmf(goals, lam), poisson_pmf(goals, mu))
        grid /= grid.sum()
        return pd.DataFrame(grid, index=pd.Index(goals, name=f"{home} goals"),
                            columns=pd.Index(goals, name=f"{away} goals"))

    def predict(self, home: str, away: str) -> dict:
        """Winner, totals and both-teams-to-score probabilities plus the most likely score."""
        grid = self.score_grid(home, away).to_numpy()
        lam, mu = self.expected_goals(home, away)
        i, j = np.indices(grid.shape)
        best = np.unravel_index(grid.argmax(), grid.shape)
        return {
            "exp_home_goals": lam,
            "exp_away_goals": mu,
            "p_home": float(grid[i > j].sum()),
            "p_draw": float(np.trace(grid)),
            "p_away": float(grid[i < j].sum()),
            "p_over_2_5": float(grid[i + j > 2.5].sum()),
            "p_btts": float(grid[1:, 1:].sum()),
            "likely_score": f"{best[0]}-{best[1]}",
            "p_likely_score": float(grid[best]),
        }

    def predict_frame(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """``predict`` for every row of a frame with ``home`` and ``away`` columns."""
        rows = [self.predict(row.home, row.away) for row in fixtures.itertuples(index=False)]
        return pd.DataFrame(rows, index=fixtures.index)
