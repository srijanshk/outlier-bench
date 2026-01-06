from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def fit_predict_scores(
    X: np.ndarray,
    seed: int,
    contamination_hint: float | None,
    n_estimators: int = 300,
):
    """
    Returns anomaly scores (higher = more anomalous).
    IsolationForest.score_samples returns higher = more normal, so we negate it.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        random_state=seed,
        contamination=contamination_hint if contamination_hint is not None else "auto",
    )
    model.fit(X)
    scores = -model.score_samples(X)
    return scores
