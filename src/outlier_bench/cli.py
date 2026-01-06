from __future__ import annotations

import typer
import yaml

from outlier_bench.config import RunConfig
from outlier_bench.eval.runner import run_once, append_report
from outlier_bench.reporting.leaderboard import csv_to_markdown
from outlier_bench.sweep_config import SweepConfig
from outlier_bench.eval.sweep import run_sweep

app = typer.Typer(no_args_is_help=True)


@app.command()
def hello():
    """Sanity check command."""
    typer.echo("outlier-bench is wired correctly.")


@app.command()
def run(config: str = typer.Option(..., help="Path to YAML config file.")):
    """Run a single benchmark from a YAML config and append to the leaderboard CSV."""
    with open(config, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cfg = RunConfig(**data)
    row = run_once(cfg)
    append_report(row, cfg.report.path)

    typer.echo("Run complete. Appended result:")
    for k, v in row.items():
        typer.echo(f"  {k}: {v}")
    typer.echo(f"\nLeaderboard: {cfg.report.path}")

@app.command()
def leaderboard(
    csv_path: str = typer.Option("reports/leaderboard.csv", help="Path to leaderboard CSV."),
    md_path: str = typer.Option("reports/leaderboard.md", help="Output markdown path."),
    sort_by: str = typer.Option("pr_auc", help="Column to sort by."),
    top_n: int = typer.Option(20, help="Number of rows to include."),
):
    """Generate a markdown leaderboard table from the CSV."""
    csv_to_markdown(csv_path, md_path, sort_by=sort_by, top_n=top_n)
    typer.echo(f"Wrote {md_path}")


@app.command()
def sweep(
    config: str = typer.Option(None, help="Path to YAML sweep config."),
    preset: str = typer.Option("quick", help="Preset: quick or full (used if --config is not given)."),
    reset: bool = typer.Option(False, help="If set, delete report CSV before running."),
    run_tag: str = typer.Option("", help="Optional tag stored in the report rows."),
):
    """Run multiple dataset/model/seed combinations and write a leaderboard."""
    if config:
        with open(config, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        scfg = SweepConfig(**data)
    else:
        # presets...
        if preset == "quick":
            scfg = SweepConfig(
                seeds=[42],
                datasets=[
                    {"name": "synthetic_gaussian", "n_samples": 2000, "n_features": 12, "contamination": 0.03, "outlier_mode": "scattered"},
                    {"name": "digits_oneclass", "normal_digit": 1},
                ],
                models=[
                    {"name": "robust_zscore"},
                    {"name": "isolation_forest", "n_estimators": 300},
                    {"name": "lof", "n_neighbors": 35},
                    {"name": "ocsvm", "nu": 0.05, "kernel": "rbf", "gamma": "scale"},
                ],
            )
        elif preset == "full":
            scfg = SweepConfig(
                seeds=[1, 2, 3, 42],
                datasets=[
                    {"name": "synthetic_gaussian", "n_samples": 8000, "n_features": 12, "contamination": 0.03, "outlier_mode": "clustered"},
                    {"name": "synthetic_gaussian", "n_samples": 8000, "n_features": 12, "contamination": 0.03, "outlier_mode": "scattered"},
                    {"name": "digits_oneclass", "normal_digit": 1},
                ],
                models=[
                    {"name": "robust_zscore"},
                    {"name": "isolation_forest", "n_estimators": 300},
                    {"name": "lof", "n_neighbors": 35},
                    {"name": "ocsvm", "nu": 0.05, "kernel": "rbf", "gamma": "scale"},
                ],
            )
        else:
            raise typer.BadParameter("preset must be 'quick' or 'full'")

    scfg.reset_report = reset
    scfg.run_tag = run_tag

    n = run_sweep(scfg)
    typer.echo(f"Sweep complete. Runs: {n}")
    typer.echo(f"CSV: {scfg.report_path}")
    if scfg.write_markdown:
        typer.echo(f"MD: {scfg.markdown_path}")



if __name__ == "__main__":
    app()
