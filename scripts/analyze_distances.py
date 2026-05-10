"""Analysis 5 of testing_plan_phase2.md: distance distribution by outcome.

The pairwise distance encodes how confident the model is about each
prediction. A pair just below the threshold is borderline; one far
below it is confidently TRUE in the model's view. This analysis splits
the corpus into the four (truth, prediction) buckets and reports the
distance distribution of each:

  correct TRUE   — truth=1, predicted=1   (distance below threshold)
  correct FALSE  — truth=0, predicted=0   (distance above threshold)
  wrong TRUE     — truth=1, predicted=0   (distance above threshold,
                                           but should have been below)
  wrong FALSE    — truth=0, predicted=1   (distance below threshold,
                                           but should have been above)

For the two wrong buckets we additionally report:
  - "borderline wrong"  : within ±0.05 of threshold
  - "confidently wrong" : > 0.20 away from threshold on the wrong side

These two are useful because the borderline pairs are essentially
threshold-calibration noise (a small shift in threshold flips them);
the confidently-wrong pairs are the ones the model has a firm wrong
belief about, and they're the right population for downstream analyses
to focus on.

Outputs:
  outputs/failure_analysis/analysis/distance_distribution.json
  outputs/failure_analysis/figures/distance_distributions.png
  outputs/failure_analysis/figures/wrong_distance_zoom.png

Usage:
  python scripts/analyze_distances.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--predictions',
        default='outputs/failure_analysis/predictions.parquet',
    )
    p.add_argument(
        '--summary',
        default='outputs/failure_analysis/failure_summary.json',
    )
    p.add_argument(
        '--output-analysis',
        default='outputs/failure_analysis/analysis/distance_distribution.json',
    )
    p.add_argument(
        '--output-figure',
        default='outputs/failure_analysis/figures/distance_distributions.png',
    )
    p.add_argument(
        '--output-zoom-figure',
        default='outputs/failure_analysis/figures/wrong_distance_zoom.png',
    )
    p.add_argument('--borderline-half-width', type=float, default=0.05)
    p.add_argument('--confident-margin', type=float, default=0.20)
    args = p.parse_args(argv)

    print(f'reading {args.predictions}')
    tbl = pq.read_table(_abs(args.predictions))
    gt = tbl['ground_truth'].to_numpy()
    pred = tbl['predicted'].to_numpy()
    dist = tbl['distance'].to_numpy()
    print(f'  {len(gt):,} pairs')

    print(f'reading {args.summary}')
    with open(_abs(args.summary)) as f:
        summary = json.load(f)
    threshold = float(summary['threshold'])
    print(f'  threshold = {threshold:.4f}')

    # --- bucket masks ----------------------------------------------------
    correct_true = (pred == 1) & (gt == 1)
    correct_false = (pred == 0) & (gt == 0)
    wrong_true = (pred == 0) & (gt == 1)   # truth TRUE, predicted FALSE
    wrong_false = (pred == 1) & (gt == 0)  # truth FALSE, predicted TRUE

    def stats(mask: np.ndarray) -> dict:
        d = dist[mask]
        if d.size == 0:
            return {'n': 0}
        return {
            'n': int(d.size),
            'mean': float(d.mean()),
            'std': float(d.std()),
            'median': float(np.median(d)),
            'p05': float(np.quantile(d, 0.05)),
            'p25': float(np.quantile(d, 0.25)),
            'p75': float(np.quantile(d, 0.75)),
            'p95': float(np.quantile(d, 0.95)),
            'min': float(d.min()),
            'max': float(d.max()),
        }

    bw = args.borderline_half_width
    cm = args.confident_margin

    # Borderline = within bw of threshold (regardless of side)
    near_threshold = np.abs(dist - threshold) <= bw

    wrong_true_borderline = wrong_true & near_threshold
    wrong_false_borderline = wrong_false & near_threshold

    # Confidently wrong: at least cm into the wrong side.
    # wrong_true means d > threshold (predicted FALSE) — confident if d > threshold + cm
    # wrong_false means d < threshold (predicted TRUE)  — confident if d < threshold - cm
    wrong_true_confident = wrong_true & (dist > threshold + cm)
    wrong_false_confident = wrong_false & (dist < threshold - cm)

    record = {
        'threshold': threshold,
        'borderline_half_width': bw,
        'confident_margin': cm,
        'overall': {
            'correct_true': stats(correct_true),
            'correct_false': stats(correct_false),
            'wrong_true': stats(wrong_true),
            'wrong_false': stats(wrong_false),
        },
        'borderline_count': {
            'wrong_true': int(wrong_true_borderline.sum()),
            'wrong_false': int(wrong_false_borderline.sum()),
        },
        'confidently_wrong_count': {
            'wrong_true': int(wrong_true_confident.sum()),
            'wrong_false': int(wrong_false_confident.sum()),
        },
    }

    print()
    print(f'=== distance distribution by outcome ===')
    for label, mask in (
        ('correct TRUE ', correct_true),
        ('correct FALSE', correct_false),
        ('wrong TRUE   ', wrong_true),
        ('wrong FALSE  ', wrong_false),
    ):
        s = stats(mask)
        print(f'  {label}  n={s["n"]:>10,}  median={s["median"]:.4f}  '
              f'mean={s["mean"]:.4f}  '
              f'IQR={s["p25"]:.4f}–{s["p75"]:.4f}')

    print()
    print(f'=== wrong-pair concentration ===')
    n_wt = int(wrong_true.sum())
    n_wf = int(wrong_false.sum())
    print(f'  wrong TRUE  ({n_wt:,}):')
    print(f'    borderline (within ±{bw} of threshold)    : '
          f'{int(wrong_true_borderline.sum()):>9,}  '
          f'({100 * wrong_true_borderline.sum() / max(n_wt, 1):.1f}%)')
    print(f'    confidently wrong (> {cm} past threshold): '
          f'{int(wrong_true_confident.sum()):>9,}  '
          f'({100 * wrong_true_confident.sum() / max(n_wt, 1):.1f}%)')
    print(f'  wrong FALSE ({n_wf:,}):')
    print(f'    borderline (within ±{bw} of threshold)    : '
          f'{int(wrong_false_borderline.sum()):>9,}  '
          f'({100 * wrong_false_borderline.sum() / max(n_wf, 1):.1f}%)')
    print(f'    confidently wrong (> {cm} past threshold): '
          f'{int(wrong_false_confident.sum()):>9,}  '
          f'({100 * wrong_false_confident.sum() / max(n_wf, 1):.1f}%)')

    # --- write JSON ------------------------------------------------------
    out_json = _abs(args.output_analysis)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(record, f, indent=2, sort_keys=False)
    print(f'\nwrote {out_json}')

    # --- figures ---------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_fig = _abs(args.output_figure)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    bins = np.linspace(0, 2.0, 121)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    panels = [
        ('correct TRUE',  correct_true,  'C2', axes[0, 0]),
        ('correct FALSE', correct_false, 'C0', axes[0, 1]),
        ('wrong TRUE\n(predicted FALSE)',  wrong_true,    'C1', axes[1, 0]),
        ('wrong FALSE\n(predicted TRUE)',  wrong_false,   'C3', axes[1, 1]),
    ]
    for name, mask, color, ax in panels:
        d = dist[mask]
        ax.hist(d, bins=bins, color=color, alpha=0.85)
        ax.axvline(threshold, color='black', linestyle='--',
                   label=f'threshold = {threshold:.3f}')
        ax.set_title(f'{name}  (n={int(mask.sum()):,})')
        ax.set_xlabel('pairwise distance')
        ax.set_ylabel('count')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f'wrote {out_fig}')

    # Zoom: wrong-only, sharing axes, with confidently-wrong shaded.
    out_zoom = _abs(args.output_zoom_figure)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    ax = axes[0]
    ax.hist(dist[wrong_true], bins=bins, color='C1', alpha=0.85)
    ax.axvline(threshold, color='black', linestyle='--', label='threshold')
    ax.axvline(threshold + cm, color='red', linestyle=':', alpha=0.7,
               label=f'confidently wrong cutoff (+{cm})')
    ax.set_title(f'wrong TRUE (predicted FALSE)\n'
                 f'n={n_wt:,}, confidently wrong={int(wrong_true_confident.sum()):,}')
    ax.set_xlabel('pairwise distance')
    ax.set_ylabel('count')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(dist[wrong_false], bins=bins, color='C3', alpha=0.85)
    ax.axvline(threshold, color='black', linestyle='--', label='threshold')
    ax.axvline(threshold - cm, color='red', linestyle=':', alpha=0.7,
               label=f'confidently wrong cutoff (-{cm})')
    ax.set_title(f'wrong FALSE (predicted TRUE)\n'
                 f'n={n_wf:,}, confidently wrong={int(wrong_false_confident.sum()):,}')
    ax.set_xlabel('pairwise distance')
    ax.set_ylabel('count')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_zoom, dpi=300)
    plt.close(fig)
    print(f'wrote {out_zoom}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
