"""Extract per_step_trace.csv, per_epoch_eval.csv, and diagnostic_extract.json
from a finished wandb run, into outputs/runs/<name>/.

Usage:
    python scripts/extract_run_diagnostics.py --name aux_h256_cohom_diag

Looks up the wandb run by display name in the configured project, pulls
its full history, then writes the standard artifact set the project's
run-record convention expects (CLAUDE.md "Every completed training run
gets recorded artifacts" rule).

The narrative ``diagnostics_report.md`` is intentionally NOT generated
here; it's a hand-written summary of the numbers this script produces.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-step columns we record (everything the train loop writes for the
# per-step log_metrics call). Columns missing from history are dropped
# silently.
_DEFAULT_PROJECT = "jepa-equational-implication"


def _resolve_run(api: wandb.Api, project: str, name: str) -> "wandb.apis.public.Run":
    runs = list(api.runs(project, filters={"displayName": name}))
    if not runs:
        raise SystemExit(f"no wandb run found with displayName={name!r} in {project!r}")
    if len(runs) > 1:
        # Pick the most recent finished one; warn the rest.
        runs.sort(key=lambda r: r.created_at, reverse=True)
        for r in runs[1:]:
            print(f"  (ignoring earlier run {r.id} from {r.created_at})")
    return runs[0]


def _step_aggregates(df: pd.DataFrame, key: str) -> dict[str, float]:
    """Compute summary stats for a per-step series within one epoch."""
    s = df[key].dropna().to_numpy()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "p05": float(np.percentile(s, 5)),
        "p95": float(np.percentile(s, 95)),
        "std": float(np.std(s)),
    }


def extract(name: str, project: str, output_dir: Path | None = None) -> Path:
    api = wandb.Api()
    run = _resolve_run(api, project, name)
    print(f"resolved run: id={run.id} name={run.name} state={run.state}")

    out_dir = output_dir or (_REPO_ROOT / "outputs/runs" / name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full history (samples=10000 keeps everything for short runs; bump
    # if your runs log >10K steps).
    history = run.history(samples=10000, pandas=True)
    print(f"history rows: {len(history)}, columns: {len(history.columns)}")

    # --- per-step trace: rows that have train/loss logged. -----------------
    train_step_cols = [c for c in history.columns if c.startswith("train/")]
    per_step = history.dropna(subset=["train/loss"], how="any") if "train/loss" in history.columns else history.iloc[0:0]
    per_step_df = per_step[["_step"] + train_step_cols].rename(columns={"_step": "step"})
    # Synthesise an epoch column from monotone elapsed_sec breakpoints
    # by aligning to per-epoch boundaries (computed from val rows below).
    val_rows = history.dropna(subset=["val/balanced_accuracy"], how="any") if "val/balanced_accuracy" in history.columns else history.iloc[0:0]
    epoch_boundaries = val_rows["_step"].to_numpy() if len(val_rows) else np.array([], dtype=int)
    # Each per-step row's epoch = number of epoch_boundaries with step <= boundary that the row precedes.
    # Equivalent: epoch = #boundaries that are >= this step's first epoch end. Simpler: rank by boundaries.
    if len(epoch_boundaries) > 0:
        # For each per-step row, count how many boundaries are < its step.
        per_step_df.insert(0, "epoch", np.searchsorted(epoch_boundaries, per_step_df["step"].to_numpy(), side="right"))
    else:
        per_step_df.insert(0, "epoch", 0)

    per_step_path = out_dir / "per_step_trace.csv"
    per_step_df.to_csv(per_step_path, index=False)
    print(f"wrote {per_step_path} ({len(per_step_df)} rows)")

    # --- per-epoch eval: rows that have val/balanced_accuracy. ------------
    val_cols = [c for c in val_rows.columns if c.startswith("val/") or c in ("epoch", "elapsed_sec")]
    train_avg_cols = [c for c in val_rows.columns if c.startswith("train/avg") or c.startswith("train/clipped_step_fraction")]
    keep = sorted(set(val_cols + train_avg_cols + ["elapsed_sec", "epoch"]))
    per_epoch_df = val_rows[[c for c in keep if c in val_rows.columns]].copy()
    per_epoch_path = out_dir / "per_epoch_eval.csv"
    per_epoch_df.to_csv(per_epoch_path, index=False)
    print(f"wrote {per_epoch_path} ({len(per_epoch_df)} rows)")

    # --- diagnostic_extract.json ------------------------------------------
    step_aggregates_by_epoch: dict[str, dict[str, dict[str, float]]] = {}
    for ep in sorted(per_step_df["epoch"].unique()):
        ep_df = per_step_df[per_step_df["epoch"] == ep]
        step_aggregates_by_epoch[str(int(ep))] = {
            col: _step_aggregates(ep_df, col)
            for col in train_step_cols
            if col in ep_df.columns
        }

    if "val/balanced_accuracy" in per_epoch_df.columns:
        best_idx = per_epoch_df["val/balanced_accuracy"].idxmax()
        best_epoch_val = per_epoch_df.loc[best_idx]
        best_epoch = {
            "epoch": int(best_epoch_val["epoch"]) if "epoch" in best_epoch_val else None,
            "bal_acc": float(best_epoch_val["val/balanced_accuracy"]),
        }
    else:
        best_epoch = {"epoch": None, "bal_acc": None}

    # Locate the on-disk wandb dir for this run by scanning for a path
    # ending in the run id (timestamp prefix differs by timezone).
    wandb_root = _REPO_ROOT / "wandb"
    matching = [p.name for p in wandb_root.iterdir() if p.name.endswith(f"-{run.id}")] if wandb_root.exists() else []
    wandb_dir = f"wandb/{matching[0]}" if matching else f"wandb/run-?-{run.id}"

    extract_data = {
        "run_id": run.id,
        "wandb_dir": wandb_dir,
        "per_step_count": int(len(per_step_df)),
        "per_epoch_eval": [
            {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v) else v)
             for k, v in row.items() if not pd.isna(v)}
            for _, row in per_epoch_df.iterrows()
        ],
        "step_aggregates_by_epoch": step_aggregates_by_epoch,
        "best_epoch": best_epoch,
    }
    extract_path = out_dir / "diagnostic_extract.json"
    extract_path.write_text(json.dumps(extract_data, indent=2, default=str))
    print(f"wrote {extract_path}")
    print(f"\nbest_epoch: {best_epoch}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="wandb run displayName")
    parser.add_argument("--project", default=_DEFAULT_PROJECT, help="wandb project")
    parser.add_argument("--output-dir", default=None, help="override output dir")
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir) if args.output_dir else None
    extract(args.name, args.project, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
