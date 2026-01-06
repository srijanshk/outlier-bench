from __future__ import annotations

import csv
import os
import time

from outlier_bench.config import RunConfig
from outlier_bench.eval.metrics import compute_metrics
from outlier_bench.registry import DATASETS, MODELS
from outlier_bench.eval.thresholding import precision_recall_f1_at_top_k



def run_once(cfg: RunConfig) -> dict:
    # --- Load dataset
    if cfg.dataset.name == "synthetic_gaussian":
        X, y = DATASETS[cfg.dataset.name](
            n_samples=cfg.dataset.n_samples,
            n_features=cfg.dataset.n_features,
            contamination=cfg.dataset.contamination,
            seed=cfg.seed,
            outlier_mode=cfg.dataset.outlier_mode,
        )

        contamination = float(cfg.dataset.contamination)
    elif cfg.dataset.name == "digits_oneclass":
        X, y = DATASETS[cfg.dataset.name](
            seed=cfg.seed,
            normal_digit=cfg.dataset.normal_digit,
        )
        contamination = None
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

    # --- Score with model
    scorer = MODELS.get(cfg.model.name)
    if scorer is None:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    t0 = time.perf_counter()
    if cfg.model.name == "isolation_forest":
        scores = scorer(
            X,
            seed=cfg.seed,
            contamination_hint=cfg.model.contamination_hint,
            n_estimators=cfg.model.n_estimators,
        )
    else:
        scores = scorer(X, seed=cfg.seed, contamination_hint=cfg.model.contamination_hint)
    t1 = time.perf_counter()

    # --- Metrics
    m = compute_metrics(y, scores)
        # Threshold metric at top-k
    n = int(X.shape[0])
    if contamination is not None:
        k = int(round(contamination * n))
    else:
        k = int(y.sum())

    m.update(precision_recall_f1_at_top_k(y, scores, k=k))
    m.update(
        {
            "dataset": cfg.dataset.name,
            "model": cfg.model.name,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "contamination": contamination,
            "latency_sec": float(t1 - t0),
            "seed": int(cfg.seed),
        }
    )
    return m


def append_report(row: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return

    # Read existing rows + header
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        existing_rows = list(reader)

    # Union of columns (preserve existing order, append new fields at end)
    new_fields = existing_fields + [k for k in row.keys() if k not in existing_fields]

    if new_fields != existing_fields:
        # Rewrite file with expanded schema
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=new_fields)
            w.writeheader()
            for r in existing_rows:
                w.writerow({k: r.get(k, "") for k in new_fields})
            w.writerow({k: row.get(k, "") for k in new_fields})
    else:
        # Append with existing schema
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=existing_fields)
            w.writerow({k: row.get(k, "") for k in existing_fields})

