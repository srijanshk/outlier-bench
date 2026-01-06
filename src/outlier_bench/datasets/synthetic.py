from __future__ import annotations
import numpy as np


def make_synthetic_gaussian(
    n_samples: int,
    n_features: int,
    contamination: float,
    seed: int,
    outlier_mode: str = "clustered",
):
    """
    Returns X, y where y=1 indicates anomaly.

    outlier_mode:
      - "clustered": outliers form a separate dense cluster (hard for LOF sometimes)
      - "scattered": outliers are dispersed (fairer for LOF / distance-based methods)
    """
    rng = np.random.default_rng(seed)
    n_out = int(n_samples * contamination)
    n_in = n_samples - n_out

    X_in = rng.normal(loc=0.0, scale=1.0, size=(n_in, n_features))

    if outlier_mode == "clustered":
        shift = rng.uniform(4.0, 8.0, size=(n_features,))
        X_out = rng.normal(loc=shift, scale=2.5, size=(n_out, n_features))

    elif outlier_mode == "scattered":
        # dispersed outliers: uniform in a wide box, avoiding the dense inlier region
        X_out = rng.uniform(low=-10.0, high=10.0, size=(n_out, n_features))

    else:
        raise ValueError(f"Unknown outlier_mode: {outlier_mode}")

    X = np.vstack([X_in, X_out])
    y = np.hstack([np.zeros(n_in, dtype=int), np.ones(n_out, dtype=int)])

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]
