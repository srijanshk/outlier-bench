from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def fit_predict_scores(
    X: np.ndarray,
    seed: int,
    contamination_hint: float | None,
    nu: float | None = None,
    kernel: str = "rbf",
    gamma: str | float = "scale",
):
    """
    Returns anomaly scores (higher = more anomalous).
    OneClassSVM.decision_function: higher => more inlier, so we negate it.

    nu is an upper bound on the fraction of anomalies; if not provided:
      - use contamination_hint if available,
      - else default to 0.05.
    """
    if nu is None:
        nu = float(contamination_hint) if contamination_hint is not None else 0.05

    # guardrails
    nu = min(max(nu, 1e-4), 0.5)

    pipe = make_pipeline(
        StandardScaler(),
        OneClassSVM(nu=nu, kernel=kernel, gamma=gamma),
    )
    pipe.fit(X)

    scores = -pipe.decision_function(X).ravel()
    return scores
