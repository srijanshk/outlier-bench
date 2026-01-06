import math
import pytest

from outlier_bench.config import RunConfig


@pytest.mark.parametrize("model_name", ["robust_zscore", "isolation_forest", "lof", "ocsvm"])
def test_smoke_synthetic_all_models(model_name):
    cfg = RunConfig(
        seed=1,
        dataset={
            "name": "synthetic_gaussian",
            "n_samples": 600,
            "n_features": 8,
            "contamination": 0.05,
            "outlier_mode": "scattered",
        },
        model={"name": model_name, "contamination_hint": 0.05},
        report={"path": "reports/test_leaderboard.csv"},
    )

    from outlier_bench.eval.runner import run_once

    row = run_once(cfg)

    # required keys
    for k in ["roc_auc", "pr_auc", "precision_at_k", "recall_at_k", "f1_at_k"]:
        assert k in row

    # valid numeric ranges
    for k in ["roc_auc", "pr_auc", "precision_at_k", "recall_at_k", "f1_at_k"]:
        v = row[k]
        assert 0.0 <= v <= 1.0
        assert not math.isnan(v)
