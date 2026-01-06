from __future__ import annotations

from typing import Callable, Tuple
import numpy as np

from outlier_bench.datasets.synthetic import make_synthetic_gaussian
from outlier_bench.datasets.digits_oneclass import make_digits_oneclass

from outlier_bench.models.robust_zscore import fit_predict_scores as zscore_scores
from outlier_bench.models.isolation_forest import fit_predict_scores as iforest_scores
from outlier_bench.models.lof import fit_predict_scores as lof_scores
from outlier_bench.models.ocsvm import fit_predict_scores as ocsvm_scores

DatasetLoader = Callable[..., Tuple[np.ndarray, np.ndarray]]
ModelScorer = Callable[..., np.ndarray]


DATASETS: dict[str, DatasetLoader] = {
    "synthetic_gaussian": make_synthetic_gaussian,
    "digits_oneclass": make_digits_oneclass,
}

MODELS: dict[str, ModelScorer] = {
    "robust_zscore": zscore_scores,
    "isolation_forest": iforest_scores,
    "lof": lof_scores,
    "ocsvm": ocsvm_scores,
}

