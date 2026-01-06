from __future__ import annotations

import os
from datetime import datetime, timezone

from outlier_bench.config import RunConfig
from outlier_bench.eval.runner import run_once, append_report
from outlier_bench.reporting.leaderboard import csv_to_markdown
from outlier_bench.sweep_config import SweepConfig


def run_sweep(scfg: SweepConfig) -> int:
    if scfg.reset_report and os.path.exists(scfg.report_path):
        os.remove(scfg.report_path)

    run_ts = datetime.now(timezone.utc).isoformat()
    n_runs = 0

    for seed in scfg.seeds:
        for dcfg in scfg.datasets:
            for mcfg in scfg.models:
                cfg = RunConfig(
                    seed=seed,
                    dataset=dcfg,
                    model=mcfg,
                    report={"path": scfg.report_path},
                )

                row = run_once(cfg)

                # Attach sweep metadata (helps trace duplicates later)
                row["run_ts_utc"] = run_ts
                row["run_tag"] = scfg.run_tag or ""

                append_report(row, scfg.report_path)
                n_runs += 1

    if scfg.write_markdown:
        csv_to_markdown(
            scfg.report_path,
            scfg.markdown_path,
            sort_by=scfg.sort_by,
            top_n=scfg.top_n,
        )

    return n_runs
