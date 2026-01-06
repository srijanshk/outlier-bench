from __future__ import annotations
import numpy as np


def precision_recall_f1_at_top_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    n = int(y_true.shape[0])
    k = max(1, min(int(k), n))

    idx = np.argsort(scores)[::-1][:k]
    y_pred = np.zeros_like(y_true)
    y_pred[idx] = 1

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "k": k,
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
        "f1_at_k": float(f1),
    }
