# Outlier Bench
Config-driven benchmarking toolkit for classical outlier / anomaly detection methods (tabular + one-class).  
Designed for reproducibility: one command runs a sweep and writes a CSV + Markdown leaderboard.

## Features
- Datasets: synthetic Gaussian outliers, `digits_oneclass` (sklearn)
- Models: Robust Z-Score, Isolation Forest, LOF, One-Class SVM
- Metrics:
  - Ranking: ROC-AUC, PR-AUC
  - Decision-style: Precision@k / Recall@k / F1@k (top-k by anomaly score)
- Reproducible experiment definitions via YAML
- CLI + automated leaderboard generation
- Tests + CI + linting

## Installation

Using `uv`:

```bash
uv sync
```

## Quickstart (recommended)

Run a small sweep and generate the leaderboard:

```bash
uv run outlier-bench sweep --preset quick --reset --run-tag quick_v1
cat reports/leaderboard.md
```

## Run a single benchmark

Example (synthetic + Isolation Forest):

```bash
uv run outlier-bench run --config configs/synthetic_iforest.yaml
```

## Output artifacts

* `reports/leaderboard.csv`: appendable experiment log (CSV)
* `reports/leaderboard.md`: Markdown table generated from the CSV

## Notes on interpretation

* **LOF can fail on clustered outliers**: LOF is density-based and can treat a separate dense outlier cluster as “normal”.
  Use `outlier_mode: scattered` in the synthetic dataset for a regime where LOF is more appropriate.
* In anomaly detection, AUC metrics do not choose a decision threshold.
  That is why this repo reports **F1@top-k** (and Precision/Recall@k) as a practical decision-style metric.

## Development

Run tests:

```bash
uv run pytest -q
```

Run lint:

```bash
uv run ruff check .
```

