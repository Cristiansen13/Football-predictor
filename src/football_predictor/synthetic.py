"""Synthetic league generator for offline demos and tests.

Matches are drawn from a known Dixon-Coles process whose team strengths
drift over time, with noisy xG and bookmaker odds that carry a margin. The
output has the same columns as ``Database.load_match_dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models.probability import outcome_probabilities, score_matrices


@dataclass
class SyntheticTruth:
    teams: list[str]
    attack: pd.DataFrame  # kickoff-date x team
    defence: pd.DataFrame
    home_advantage: float
    rho: float
    intercept: float


def round_robin(teams: list[str], rng: np.random.Generator) -> list[list[tuple[str, str]]]:
    """Double round robin (circle method); second half mirrors the first."""
    order = list(teams)
    rng.shuffle(order)
    if len(order) % 2:
        order.append(None)
    n = len(order)
    first_half = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = order[i], order[n - 1 - i]
            if a is not None and b is not None:
                pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        first_half.append(pairs)
        order = [order[0]] + [order[-1]] + order[1:-1]
    second_half = [[(b, a) for a, b in rnd] for rnd in first_half]
    return first_half + second_half


def simulate_league(
    seasons: range | list[int] = range(2018, 2024),
    n_teams: int = 20,
    *,
    second_division_teams: int = 0,
    promoted: int = 3,
    division_gap: float = 0.35,
    intercept: float = 0.12,
    home_advantage: float = 0.25,
    rho: float = -0.1,
    strength_sd: float = 0.3,
    weekly_drift_sd: float = 0.015,
    xg_noise_sd: float = 0.35,
    bookmaker_margin: float = 0.05,
    bookmaker_noise_sd: float = 0.15,
    seed: int = 7,
    return_truth: bool = False,
):
    """Simulate a top division ("E0") and optionally a second division ("E1").

    With a second division, clubs there are ``division_gap`` weaker in both
    attack and defence, the bottom ``promoted`` E0 clubs swap places with the
    top ``promoted`` E1 clubs after every season, and E1 matches have no xG
    (as on Understat).
    """
    rng = np.random.default_rng(seed)
    total = n_teams + second_division_teams
    teams = [f"Team {i + 1:02d}" for i in range(total)]
    attack = rng.normal(0, strength_sd, total)
    defence = -0.7 * attack + rng.normal(0, strength_sd * 0.6, total)  # good attacks tend to defend well
    attack[n_teams:] -= division_gap
    defence[n_teams:] += division_gap
    attack -= attack.mean()
    defence -= defence.mean()
    index = {t: i for i, t in enumerate(teams)}
    members = {"E0": teams[:n_teams], "E1": teams[n_teams:]} if second_division_teams else {"E0": teams}

    rows, attack_path, defence_path = [], {}, {}
    match_number = 0
    for season in seasons:
        schedules = {division: round_robin(clubs, rng) for division, clubs in members.items()}
        points = {t: [0, 0] for t in teams}  # points, goal difference
        day = pd.Timestamp(f"{season}-08-10")
        day += pd.Timedelta(days=(5 - day.dayofweek) % 7)  # first Saturday
        for r in range(max(len(rounds) for rounds in schedules.values())):
            # Every fifth round is a Tuesday fixture between two Saturdays.
            if r > 0:
                midweek = r % 5 == 0
                after_midweek = r > 1 and (r - 1) % 5 == 0
                day += pd.Timedelta(days=3 if midweek else 4 if after_midweek else 7)
            attack += rng.normal(0, weekly_drift_sd, total)
            defence += rng.normal(0, weekly_drift_sd, total)
            attack -= attack.mean()
            defence -= defence.mean()
            attack_path[day] = attack.copy()
            defence_path[day] = defence.copy()

            for division, rounds in schedules.items():
                if r >= len(rounds):
                    continue
                fixtures = rounds[r]
                h = np.array([index[a] for a, _ in fixtures])
                a = np.array([index[b] for _, b in fixtures])
                lam = np.exp(intercept + home_advantage + attack[h] + defence[a])
                mu = np.exp(intercept + attack[a] + defence[h])
                matrices = score_matrices(lam, mu, rho, max_goals=10)
                probs = outcome_probabilities(matrices)

                # Bookmaker: true probabilities with log-odds noise and a proportional margin.
                noisy = np.exp(np.log(probs) + rng.normal(0, bookmaker_noise_sd, probs.shape))
                book = noisy / noisy.sum(axis=1, keepdims=True)
                odds = np.round(1.0 / (book * (1.0 + bookmaker_margin)), 2)

                for k, (home, away) in enumerate(fixtures):
                    flat = matrices[k].ravel()
                    score = rng.choice(flat.size, p=flat / flat.sum())
                    hg, ag = divmod(score, 11)
                    kickoff = day + pd.Timedelta(hours=15 if k % 3 else 17, minutes=30 if k % 3 == 2 else 0)
                    xg_h = float(lam[k] * np.exp(rng.normal(-xg_noise_sd**2 / 2, xg_noise_sd)))
                    xg_a = float(mu[k] * np.exp(rng.normal(-xg_noise_sd**2 / 2, xg_noise_sd)))
                    has_xg = division == "E0"
                    rows.append(
                        {
                            "match_id": f"m{match_number:05d}",
                            "division": division,
                            "season": season,
                            "kickoff": kickoff,
                            "home_team": home,
                            "away_team": away,
                            "home_goals": int(hg),
                            "away_goals": int(ag),
                            "home_xg": round(xg_h, 2) if has_xg else np.nan,
                            "away_xg": round(xg_a, 2) if has_xg else np.nan,
                            "odds_home": odds[k, 0],
                            "odds_draw": odds[k, 1],
                            "odds_away": odds[k, 2],
                        }
                    )
                    match_number += 1
                    result = 3 if hg > ag else 1 if hg == ag else 0
                    points[home][0] += result
                    points[away][0] += 3 if ag > hg else 1 if hg == ag else 0
                    points[home][1] += hg - ag
                    points[away][1] += ag - hg

        if second_division_teams:
            table = {div: sorted(clubs, key=lambda t: points[t], reverse=True) for div, clubs in members.items()}
            down, up = table["E0"][-promoted:], table["E1"][:promoted]
            members = {
                "E0": [t for t in members["E0"] if t not in down] + up,
                "E1": [t for t in members["E1"] if t not in up] + down,
            }

    frame = pd.DataFrame(rows).sort_values("kickoff", kind="stable").reset_index(drop=True)
    frame[["home_goals", "away_goals"]] = frame[["home_goals", "away_goals"]].astype("Int64")
    if not return_truth:
        return frame
    truth = SyntheticTruth(
        teams=teams,
        attack=pd.DataFrame(attack_path, index=teams).T,
        defence=pd.DataFrame(defence_path, index=teams).T,
        home_advantage=home_advantage,
        rho=rho,
        intercept=intercept,
    )
    return frame, truth
