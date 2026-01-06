from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }
