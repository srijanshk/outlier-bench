from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def fit_predict_scores(
    X: np.ndarray,
    seed: int,
    contamination_hint: float | None,
    n_neighbors: int = 35,
):
    """
    Returns anomaly scores (higher = more anomalous).
    LOF provides negative_outlier_factor_: lower => more abnormal, so we negate it.
    Note: LOF doesn't use seed; included for API consistency.
    """
    # Scaling improves distance-based methods like LOF
    Xs = StandardScaler().fit_transform(X)

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination_hint if contamination_hint is not None else "auto",
        novelty=False,  # allows use of negative_outlier_factor_ on the fitted data
    )
    model.fit(Xs)

    # negative_outlier_factor_ is negative; more negative => more outlier
    scores = -model.negative_outlier_factor_
    return scores
