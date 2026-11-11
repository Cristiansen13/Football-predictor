"""The classifiers compared with the Poisson model.

Each one predicts P(home win), P(draw), P(away win) from the pre-match
features in ``data/final/features.csv``:

* ``baseline``  class frequencies of the training matches (no information about the two clubs)
* ``elo``       multinomial logistic regression on the Elo rating difference alone
* ``logistic``  multinomial logistic regression on every feature
* ``forest``    random forest
* ``boosting``  gradient boosted trees

The Poisson model (``analysis/poisson.py``) and the bookmaker are added by
``analysis/compare_models.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from extraction.features import FEATURE_COLUMNS

MODEL_NAMES = {
    "baseline": "Class frequencies",
    "elo": "Elo + logistic regression",
    "poisson": "Poisson (team averages)",
    "logistic": "Logistic regression",
    "forest": "Random forest",
    "boosting": "Gradient boosting",
    "bookmaker": "Bookmaker (Bet365 closing odds)",
}
ML_MODELS = ["baseline", "elo", "logistic", "forest", "boosting"]

# Candidate settings, chosen on validation seasons that come before every test season.
GRIDS = {
    "baseline": [{}],
    "elo": [{}],
    "logistic": [{"C": c} for c in (0.003, 0.01, 0.03, 0.1, 1.0)],
    "forest": [{"min_samples_leaf": leaf} for leaf in (10, 30, 60)],
    "boosting": [{"learning_rate": rate, "max_iter": trees, "max_depth": depth}
                 for rate, trees in ((0.03, 100), (0.03, 200), (0.05, 100)) for depth in (2, 3)],
}


def columns_for(key: str) -> list[str]:
    return ["elo_diff"] if key == "elo" else FEATURE_COLUMNS


def build(key: str, **params):
    """An unfitted scikit-learn estimator for model ``key``."""
    if key == "baseline":
        return DummyClassifier(strategy="prior")
    if key == "elo":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if key == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=params.get("C", 0.03), max_iter=2000))
    if key == "forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=400, min_samples_leaf=params.get("min_samples_leaf", 30),
                                                    random_state=0, n_jobs=-1))
    if key == "boosting":
        return HistGradientBoostingClassifier(learning_rate=params.get("learning_rate", 0.03),
                                              max_iter=params.get("max_iter", 100),
                                              max_depth=params.get("max_depth", 2),
                                              early_stopping=False, random_state=0)
    raise ValueError(f"unknown model {key!r}")


def fit_predict(key: str, params: dict, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit on ``train`` and return the three probabilities for every row of ``test``."""
    columns = columns_for(key)
    model = build(key, **params).fit(train[columns], train["outcome"])
    return model.predict_proba(test[columns])
