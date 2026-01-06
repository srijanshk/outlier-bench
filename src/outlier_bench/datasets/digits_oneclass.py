from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits


def make_digits_oneclass(seed: int, normal_digit: int = 1):
    """
    Create a one-class anomaly detection dataset:
    - normal class: a single digit (default 1)
    - anomalies: all other digits (sampled to match normal count)
    Returns X, y where y=1 indicates anomaly.
    """
    X, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(seed)

    normal_mask = (y == normal_digit)
    anomaly_mask = ~normal_mask

    Xn = X[normal_mask]
    Xa = X[anomaly_mask]

    n_norm = Xn.shape[0]
    # sample anomalies to match normals for stable evaluation
    Xa_s = Xa[rng.choice(Xa.shape[0], size=n_norm, replace=False)]

    X_all = np.vstack([Xn, Xa_s])
    y_all = np.hstack([np.zeros(n_norm, dtype=int), np.ones(n_norm, dtype=int)])

    idx = rng.permutation(X_all.shape[0])
    return X_all[idx], y_all[idx]
