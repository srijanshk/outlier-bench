from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class SweepConfig(BaseModel):
    seeds: List[int] = Field(default_factory=lambda: [42])

    datasets: List[dict] = Field(
        default_factory=lambda: [
            {"name": "synthetic_gaussian", "n_samples": 2000, "n_features": 12, "contamination": 0.03, "outlier_mode": "scattered"},
            {"name": "digits_oneclass", "normal_digit": 1},
        ]
    )

    models: List[dict] = Field(
        default_factory=lambda: [
            {"name": "robust_zscore"},
            {"name": "isolation_forest", "n_estimators": 300},
            {"name": "lof", "n_neighbors": 35},
            {"name": "ocsvm", "nu": 0.05, "kernel": "rbf", "gamma": "scale"},
        ]
    )

    report_path: str = "reports/leaderboard.csv"
    write_markdown: bool = True
    markdown_path: str = "reports/leaderboard.md"
    sort_by: str = "pr_auc"
    top_n: int = 50

    reset_report: bool = False
    run_tag: Optional[str] = None
