"""Dixon-Coles bivariate Poisson model fitted by time-weighted maximum likelihood.

Scoring rates are log-linear::

    log λ = c + γ + attack[home] + defence[away] + θ_att·A(home) + θ_def·D(away)
    log μ = c     + attack[away] + defence[home] + θ_att·A(away) + θ_def·D(home)

where ``defence`` is a weakness (higher concedes more), γ is the home
advantage, and A / D are optional point-in-time feature vectors (form,
fatigue, lineup strength) of the scoring and the conceding side. Attack and
defence are constrained to sum to zero. Low scores are corrected by the
Dixon-Coles ρ term and every match is weighted by exp(-ξ · days before the
fit date), so the coefficients describe the teams as they are now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .probability import market_probabilities, score_matrices

# attack features describe the scoring side, defence features the conceding side.
FEATURE_PRESETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "none": ((), ()),
    "form": (("ewma_xg_for",), ("ewma_xg_against",)),
    "full": (
        ("ewma_xg_for", "fatigue_penalty", "lineup_fraction"),
        ("ewma_xg_against", "fatigue_penalty", "lineup_fraction"),
    ),
}

_TAU_FLOOR = 1e-10


@dataclass
class FitSummary:
    converged: bool
    message: str
    iterations: int
    log_likelihood: float
    n_matches: int
    effective_matches: float
    as_of: pd.Timestamp


def time_decay_weights(kickoff: pd.Series, as_of: pd.Timestamp, xi: float) -> np.ndarray:
    """exp(-ξ · Δt) with Δt in days between the match and the fit date."""
    days = (as_of - pd.to_datetime(kickoff)).dt.total_seconds().to_numpy() / 86_400.0
    return np.exp(-xi * np.clip(days, 0.0, None))


def dixon_coles_loglik_terms(x, y, lam, mu, rho):
    """Per-match log-likelihood and its derivatives w.r.t. log λ, log μ and ρ."""
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau = np.ones_like(lam)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    # τ <= 0 is outside the admissible ρ range; the floor makes the likelihood
    # flat (and very poor) there, so its derivatives are zeroed to match.
    valid = tau > _TAU_FLOOR
    tau = np.where(valid, tau, _TAU_FLOOR)
    m00, m01, m10, m11 = m00 & valid, m01 & valid, m10 & valid, m11 & valid

    loglik = np.log(tau) + x * np.log(lam) - lam + y * np.log(mu) - mu - gammaln(x + 1) - gammaln(y + 1)

    # d log τ / d log λ = λ · d log τ / dλ, and likewise for μ.
    dtau_lam = np.zeros_like(lam)
    dtau_mu = np.zeros_like(lam)
    dtau_rho = np.zeros_like(lam)
    dtau_lam[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
    dtau_mu[m00] = dtau_lam[m00]
    dtau_rho[m00] = -lam[m00] * mu[m00] / tau[m00]
    dtau_lam[m01] = lam[m01] * rho / tau[m01]
    dtau_rho[m01] = lam[m01] / tau[m01]
    dtau_mu[m10] = mu[m10] * rho / tau[m10]
    dtau_rho[m10] = mu[m10] / tau[m10]
    dtau_rho[m11] = -1.0 / tau[m11]

    g_lam = x - lam + dtau_lam
    g_mu = y - mu + dtau_mu
    return loglik, g_lam, g_mu, dtau_rho


class DixonColesModel:
    def __init__(
        self,
        xi: float = 0.0019,
        *,
        l2: float = 0.5,
        attack_features: Sequence[str] = (),
        defence_features: Sequence[str] = (),
        max_goals: int = 6,
        rho_bounds: tuple[float, float] = (-0.3, 0.3),
        unseen_quantile: float = 0.2,
    ) -> None:
        """
        Parameters
        ----------
        xi:
            Time-decay rate per day. 0.0019 gives a half-life of about a year.
        l2:
            Ridge penalty on team strengths and feature coefficients, in units of
            one fully weighted match. Keeps promoted teams with few matches sane.
        attack_features, defence_features:
            Names of point-in-time features (without the ``home_``/``away_``
            prefix) of the scoring side and of the conceding side.
        unseen_quantile:
            A team absent from the training data (usually a promoted club) is
            rated at this quantile of attack and the mirrored quantile of defence.
        """
        self.xi = xi
        self.l2 = l2
        self.attack_features = tuple(attack_features)
        self.defence_features = tuple(defence_features)
        self.max_goals = max_goals
        self.rho_bounds = rho_bounds
        self.unseen_quantile = unseen_quantile

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "DixonColesModel":
        attack, defence = FEATURE_PRESETS[preset]
        return cls(attack_features=attack, defence_features=defence, **kwargs)

    # -- design -------------------------------------------------------------

    def _feature_block(self, frame: pd.DataFrame, side: str, names: tuple[str, ...]) -> np.ndarray:
        if not names:
            return np.zeros((len(frame), 0))
        columns = []
        for name in names:
            column = f"{side}_{name}"
            values = frame[column].to_numpy(dtype=float) if column in frame else np.full(len(frame), np.nan)
            mean, std = self.feature_stats_[name]
            columns.append(np.nan_to_num((values - mean) / std, nan=0.0))
        return np.column_stack(columns)

    def _fit_feature_stats(self, frame: pd.DataFrame, weights: np.ndarray) -> None:
        self.feature_stats_: dict[str, tuple[float, float]] = {}
        for name in set(self.attack_features) | set(self.defence_features):
            values, w = [], []
            for side in ("home", "away"):
                column = f"{side}_{name}"
                if column in frame:
                    values.append(frame[column].to_numpy(dtype=float))
                    w.append(weights)
            values = np.concatenate(values) if values else np.array([np.nan])
            w = np.concatenate(w) if w else np.array([0.0])
            ok = ~np.isnan(values) & (w > 0)
            if ok.sum() < 2:
                self.feature_stats_[name] = (0.0, 1.0)
                continue
            mean = np.average(values[ok], weights=w[ok])
            std = np.sqrt(np.average((values[ok] - mean) ** 2, weights=w[ok]))
            self.feature_stats_[name] = (float(mean), float(std) if std > 1e-9 else 1.0)

    # -- likelihood ---------------------------------------------------------

    def _unpack(self, params: np.ndarray, n_teams: int):
        ka, kd = len(self.attack_features), len(self.defence_features)
        c, gamma, rho = params[:3]
        offset = 3
        attack_free = params[offset : offset + n_teams - 1]
        offset += n_teams - 1
        defence_free = params[offset : offset + n_teams - 1]
        offset += n_teams - 1
        theta_a = params[offset : offset + ka]
        theta_d = params[offset + ka : offset + ka + kd]
        attack = np.append(attack_free, -attack_free.sum())
        defence = np.append(defence_free, -defence_free.sum())
        return c, gamma, rho, attack, defence, theta_a, theta_d

    def _objective(self, params: np.ndarray, data: dict) -> tuple[float, np.ndarray]:
        n = data["n_teams"]
        c, gamma, rho, attack, defence, theta_a, theta_d = self._unpack(params, n)
        h, a, w = data["home"], data["away"], data["weights"]
        eta_lam = c + gamma + attack[h] + defence[a] + data["att_home"] @ theta_a + data["def_away"] @ theta_d
        eta_mu = c + attack[a] + defence[h] + data["att_away"] @ theta_a + data["def_home"] @ theta_d
        lam, mu = np.exp(eta_lam), np.exp(eta_mu)

        loglik, g_lam, g_mu, g_rho = dixon_coles_loglik_terms(data["x"], data["y"], lam, mu, rho)
        wl, wm = w * g_lam, w * g_mu

        grad_attack = np.bincount(h, wl, n) + np.bincount(a, wm, n) - 2 * self.l2 * attack
        grad_defence = np.bincount(a, wl, n) + np.bincount(h, wm, n) - 2 * self.l2 * defence
        grad = np.concatenate(
            [
                [wl.sum() + wm.sum(), wl.sum(), (w * g_rho).sum()],
                grad_attack[:-1] - grad_attack[-1],  # chain rule through the sum-to-zero constraint
                grad_defence[:-1] - grad_defence[-1],
                data["att_home"].T @ wl + data["att_away"].T @ wm - 2 * self.l2 * theta_a,
                data["def_away"].T @ wl + data["def_home"].T @ wm - 2 * self.l2 * theta_d,
            ]
        )
        penalty = self.l2 * (attack @ attack + defence @ defence + theta_a @ theta_a + theta_d @ theta_d)
        total = data["total_weight"]
        return -((w * loglik).sum() - penalty) / total, -grad / total

    # -- fitting ------------------------------------------------------------

    def fit(
        self,
        matches: pd.DataFrame,
        as_of: pd.Timestamp | str | None = None,
        warm_start: "DixonColesModel | None" = None,
    ) -> "DixonColesModel":
        """Fit on played matches strictly before ``as_of`` (default: after the last one)."""
        played = matches.dropna(subset=["home_goals", "away_goals"])
        kickoff = pd.to_datetime(played["kickoff"])
        if as_of is None:
            as_of = kickoff.max() + pd.Timedelta(seconds=1)
        as_of = pd.Timestamp(as_of)
        played = played[kickoff < as_of]
        if played.empty:
            raise ValueError(f"No played matches before {as_of}")

        weights = time_decay_weights(played["kickoff"], as_of, self.xi)
        keep = weights > 1e-6
        played, weights = played[keep], weights[keep]

        teams = np.array(sorted(set(played["home_team"]) | set(played["away_team"])))
        index = {team: i for i, team in enumerate(teams)}
        self._fit_feature_stats(played, weights)
        data = {
            "n_teams": len(teams),
            "home": played["home_team"].map(index).to_numpy(),
            "away": played["away_team"].map(index).to_numpy(),
            "x": played["home_goals"].to_numpy(dtype=float),
            "y": played["away_goals"].to_numpy(dtype=float),
            "weights": weights,
            "total_weight": weights.sum(),
            "att_home": self._feature_block(played, "home", self.attack_features),
            "att_away": self._feature_block(played, "away", self.attack_features),
            "def_home": self._feature_block(played, "home", self.defence_features),
            "def_away": self._feature_block(played, "away", self.defence_features),
        }

        x0 = self._initial_params(teams, data, warm_start)
        bounds = [(None, None), (None, None), self.rho_bounds] + [(None, None)] * (len(x0) - 3)
        result = minimize(
            self._objective,
            x0,
            args=(data,),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-7},
        )

        c, gamma, rho, attack, defence, theta_a, theta_d = self._unpack(result.x, len(teams))
        self.teams_ = teams
        self.team_index_ = index
        self.intercept_ = float(c)
        self.home_advantage_ = float(gamma)
        self.rho_ = float(rho)
        self.attack_ = attack
        self.defence_ = defence
        self.theta_attack_ = dict(zip(self.attack_features, theta_a))
        self.theta_defence_ = dict(zip(self.defence_features, theta_d))
        self.summary_ = FitSummary(
            converged=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
            log_likelihood=float(-result.fun * data["total_weight"]),
            n_matches=len(played),
            effective_matches=float(weights.sum()),
            as_of=as_of,
        )
        return self

    def _initial_params(self, teams: np.ndarray, data: dict, warm_start) -> np.ndarray:
        n, ka, kd = len(teams), len(self.attack_features), len(self.defence_features)
        attack, defence = np.zeros(n), np.zeros(n)
        if warm_start is not None and hasattr(warm_start, "teams_"):
            c, gamma, rho = warm_start.intercept_, warm_start.home_advantage_, warm_start.rho_
            for i, team in enumerate(teams):
                j = warm_start.team_index_.get(team)
                if j is not None:
                    attack[i], defence[i] = warm_start.attack_[j], warm_start.defence_[j]
            attack -= attack.mean()
            defence -= defence.mean()
            theta_a = [warm_start.theta_attack_.get(f, 0.0) for f in self.attack_features]
            theta_d = [warm_start.theta_defence_.get(f, 0.0) for f in self.defence_features]
        else:
            w = data["weights"]
            home_rate = np.average(data["x"], weights=w)
            away_rate = np.average(data["y"], weights=w)
            c = np.log(max(away_rate, 0.1))
            gamma = np.log(max(home_rate, 0.1) / max(away_rate, 0.1))
            rho = 0.0
            theta_a, theta_d = np.zeros(ka), np.zeros(kd)
        rho = float(np.clip(rho, *self.rho_bounds))
        return np.concatenate([[c, gamma, rho], attack[:-1], defence[:-1], theta_a, theta_d])

    # -- prediction ---------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "teams_"):
            raise RuntimeError("Model is not fitted")

    def _team_strengths(self, teams: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        q = self.unseen_quantile
        unseen_attack = np.quantile(self.attack_, q)
        unseen_defence = np.quantile(self.defence_, 1 - q)
        idx = teams.map(self.team_index_)
        known = idx.notna().to_numpy()
        positions = idx.fillna(0).astype(int).to_numpy()
        attack = np.where(known, self.attack_[positions], unseen_attack)
        defence = np.where(known, self.defence_[positions], unseen_defence)
        return attack, defence

    def expected_goals(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Scoring rates (λ, μ) for each fixture row (home_team, away_team, features)."""
        self._check_fitted()
        att_h, def_h = self._team_strengths(fixtures["home_team"])
        att_a, def_a = self._team_strengths(fixtures["away_team"])
        theta_a = np.array([self.theta_attack_[f] for f in self.attack_features])
        theta_d = np.array([self.theta_defence_[f] for f in self.defence_features])
        eta_lam = (
            self.intercept_ + self.home_advantage_ + att_h + def_a
            + self._feature_block(fixtures, "home", self.attack_features) @ theta_a
            + self._feature_block(fixtures, "away", self.defence_features) @ theta_d
        )
        eta_mu = (
            self.intercept_ + att_a + def_h
            + self._feature_block(fixtures, "away", self.attack_features) @ theta_a
            + self._feature_block(fixtures, "home", self.defence_features) @ theta_d
        )
        return np.exp(eta_lam), np.exp(eta_mu)

    def score_matrices(self, fixtures: pd.DataFrame) -> np.ndarray:
        lam, mu = self.expected_goals(fixtures)
        return score_matrices(lam, mu, self.rho_, self.max_goals)

    def score_matrix(self, home_team: str, away_team: str, **features: float) -> pd.DataFrame:
        """Correct-score grid for one fixture; ``features`` are ``home_x=..``/``away_x=..`` values."""
        fixture = pd.DataFrame([{"home_team": home_team, "away_team": away_team, **features}])
        matrix = self.score_matrices(fixture)[0]
        goals = range(self.max_goals + 1)
        return pd.DataFrame(
            matrix,
            index=pd.Index(goals, name=f"{home_team} goals"),
            columns=pd.Index(goals, name=f"{away_team} goals"),
        )

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Expected goals and market probabilities, one row per fixture."""
        lam, mu = self.expected_goals(fixtures)
        markets = market_probabilities(score_matrices(lam, mu, self.rho_, self.max_goals))
        markets.insert(0, "exp_away_goals", mu)
        markets.insert(0, "exp_home_goals", lam)
        markets.index = fixtures.index
        return markets

    def log_likelihood(self, matches: pd.DataFrame) -> np.ndarray:
        """Per-match log-likelihood of observed scores (out-of-sample evaluation)."""
        lam, mu = self.expected_goals(matches)
        loglik, *_ = dixon_coles_loglik_terms(
            matches["home_goals"].to_numpy(dtype=float),
            matches["away_goals"].to_numpy(dtype=float),
            lam,
            mu,
            self.rho_,
        )
        return loglik

    def ratings(self) -> pd.DataFrame:
        """Team strengths: expected goals for/against against an average side on neutral ground."""
        self._check_fitted()
        frame = pd.DataFrame({"team": self.teams_, "attack": self.attack_, "defence": self.defence_})
        frame["goals_for_vs_avg"] = np.exp(self.intercept_ + self.home_advantage_ / 2 + frame["attack"])
        frame["goals_against_vs_avg"] = np.exp(self.intercept_ + self.home_advantage_ / 2 + frame["defence"])
        frame["rating"] = frame["attack"] - frame["defence"]
        return frame.sort_values("rating", ascending=False).reset_index(drop=True)
