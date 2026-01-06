from __future__ import annotations
import numpy as np


def fit_predict_scores(X: np.ndarray, seed: int, contamination_hint: float | None):
    """
    Returns anomaly scores (higher = more anomalous), shape (n_samples,).
    seed/contamination_hint included to match a common model API.
    """
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + 1e-9

    z = np.abs((X - med) / mad)
    scores = np.mean(z, axis=1)
    return scores
