"""Analysis 1 of testing_plan_phase2.md: Austin-pair failure analysis.

Cross-reference the 820 curated Austin implications (FALSE in general,
TRUE for every finite magma) against the model's per-pair predictions
from embeddings/outputs/failure_analysis/predictions.parquet. Reports:

  - Austin overall accuracy + comparison to corpus FALSE accuracy
  - Whether Austin pairs are over-represented in the failure set
  - Each misclassified Austin pair, sorted by confidence (most
    confidently wrong first), with equation text printed
  - Distance distribution within Austin pairs (vs the corpus FALSE
    distribution and the calibrated threshold)

Outputs:
  embeddings/outputs/failure_analysis/analysis/austin_analysis.json
  embeddings/outputs/failure_analysis/figures/austin_distance_distribution.png

Usage:
  python embeddings/scripts/analyze_austin.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _load_austin_pairs(path: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    with open(path) as f:
        for line in f:
            m = re.match(r'Equation(\d+)\s*→\s*Equation(\d+)', line.strip())
            if m:
                out.add((int(m.group(1)), int(m.group(2))))
    return out


def _load_equations(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--predictions',
        default='embeddings/outputs/failure_analysis/predictions.parquet',
        help='input predictions parquet (default: %(default)s)',
    )
    p.add_argument(
        '--summary',
        default='embeddings/outputs/failure_analysis/failure_summary.json',
        help='input failure summary JSON for the threshold (default: %(default)s)',
    )
    p.add_argument(
        '--austin',
        default='data/raw/etp/Austin_implications.txt',
        help='Austin-implications file (default: %(default)s)',
    )
    p.add_argument(
        '--equations',
        default='data/raw/etp/equations.txt',
        help='equations.txt (default: %(default)s)',
    )
    p.add_argument(
        '--output-analysis',
        default='embeddings/outputs/failure_analysis/analysis/austin_analysis.json',
        help='output JSON path',
    )
    p.add_argument(
        '--output-figure',
        default='embeddings/outputs/failure_analysis/figures/austin_distance_distribution.png',
        help='output PNG path',
    )
    args = p.parse_args(argv)

    # --- load -------------------------------------------------------------
    print(f'reading {args.predictions}')
    tbl = pq.read_table(_abs(args.predictions))
    eq1 = tbl['eq1_id'].to_numpy()
    eq2 = tbl['eq2_id'].to_numpy()
    gt = tbl['ground_truth'].to_numpy()
    pred = tbl['predicted'].to_numpy()
    dist = tbl['distance'].to_numpy()
    print(f'  {len(eq1):,} pairs')

    print(f'reading {args.summary}')
    with open(_abs(args.summary)) as f:
        summary = json.load(f)
    threshold = float(summary['threshold'])
    print(f'  threshold = {threshold:.4f}')

    print(f'reading {args.austin}')
    austin = _load_austin_pairs(_abs(args.austin))
    print(f'  {len(austin):,} austin pairs')

    print(f'reading {args.equations}')
    eqs = _load_equations(_abs(args.equations))

    # --- find Austin rows in the predictions table ------------------------
    # Build a (eq1, eq2) → row-index lookup once.
    n = len(eq1)
    pair_to_row: dict[tuple[int, int], int] = {}
    for i in range(n):
        pair_to_row[(int(eq1[i]), int(eq2[i]))] = i
    austin_rows = np.array(
        [pair_to_row[p] for p in austin if p in pair_to_row],
        dtype=np.int64,
    )
    print(f'  matched {len(austin_rows):,} Austin pairs in predictions')

    # --- headline stats ---------------------------------------------------
    a_gt = gt[austin_rows]
    a_pred = pred[austin_rows]
    a_dist = dist[austin_rows]

    # All Austin pairs are ground-truth FALSE. Sanity-check.
    if not (a_gt == 0).all():
        n_unexpected = int((a_gt != 0).sum())
        print(f'WARNING: {n_unexpected} Austin pairs have non-FALSE ground '
              f'truth in the predictions table — investigate.')

    n_austin = len(austin_rows)
    n_correct = int((a_pred == a_gt).sum())
    n_wrong = n_austin - n_correct
    austin_accuracy = n_correct / max(n_austin, 1)

    # Corpus stats for comparison.
    corpus_false_mask = gt == 0
    n_corpus_false = int(corpus_false_mask.sum())
    n_corpus_false_correct = int(((pred == 0) & corpus_false_mask).sum())
    corpus_false_accuracy = n_corpus_false_correct / max(n_corpus_false, 1)

    n_failures = int((pred != gt).sum())
    austin_failures = n_wrong
    base_rate = n_austin / max(n, 1)
    failure_share = austin_failures / max(n_failures, 1)
    over_rep_factor = (
        (failure_share / base_rate) if base_rate > 0 else float('nan')
    )

    print()
    print(f'=== Austin analysis ===')
    print(f'  n austin pairs     : {n_austin}')
    print(f'  austin accuracy    : {austin_accuracy:.4f}  ({n_correct}/{n_austin})')
    print(f'  corpus FALSE acc   : {corpus_false_accuracy:.4f}  '
          f'({n_corpus_false_correct:,}/{n_corpus_false:,})')
    print(f'  gap (corpus - aust): {corpus_false_accuracy - austin_accuracy:+.4f}')
    print(f'  base rate          : {100*base_rate:.4f}% of pairs are Austin')
    print(f'  failure share      : {100*failure_share:.4f}% of failures are Austin')
    print(f'  over-rep factor    : {over_rep_factor:.2f}x  '
          f'(>1 = Austin is over-represented in failures)')

    # --- list misclassified Austin pairs, most-confidently-wrong first ---
    # Wrong = predicted TRUE while truth is FALSE → distance < threshold.
    # "Most confidently wrong" = smallest distance (model thinks they're
    # very TRUE-like).
    wrong_mask = a_pred != a_gt
    wrong_rows = austin_rows[wrong_mask]
    wrong_dist = a_dist[wrong_mask]
    order = np.argsort(wrong_dist)
    misclassified = []
    for j in order:
        r = int(wrong_rows[j])
        misclassified.append({
            'eq1_id': int(eq1[r]),
            'eq2_id': int(eq2[r]),
            'eq1': eqs[int(eq1[r]) - 1],
            'eq2': eqs[int(eq2[r]) - 1],
            'distance': float(dist[r]),
            'margin_under_threshold': float(threshold - dist[r]),
        })
    print()
    print(f'=== misclassified Austin pairs ({n_wrong}) — sorted by confidence ===')
    for k, m in enumerate(misclassified[:20]):
        print(f'  {k+1:2d}. d={m["distance"]:.4f}  '
              f'Eq{m["eq1_id"]} → Eq{m["eq2_id"]}')
        print(f'      [{m["eq1"]}]  ⇒  [{m["eq2"]}]')
    if len(misclassified) > 20:
        print(f'  ... and {len(misclassified) - 20} more (full list in JSON output)')

    # --- write JSON -------------------------------------------------------
    out_json = _abs(args.output_analysis)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    record = {
        'n_austin': n_austin,
        'n_correct': n_correct,
        'n_wrong': n_wrong,
        'austin_accuracy': austin_accuracy,
        'corpus_false_accuracy': corpus_false_accuracy,
        'gap_corpus_minus_austin': corpus_false_accuracy - austin_accuracy,
        'base_rate': base_rate,
        'failure_share': failure_share,
        'over_representation_factor': over_rep_factor,
        'threshold': threshold,
        'misclassified': misclassified,
    }
    with open(out_json, 'w') as f:
        json.dump(record, f, indent=2, sort_keys=False)
    print(f'\nwrote {out_json}')

    # --- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_fig = _abs(args.output_figure)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    corpus_false_dist = dist[corpus_false_mask]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 2.0, 81)
    ax.hist(
        corpus_false_dist, bins=bins, density=True, alpha=0.45,
        color='C0', label=f'all corpus FALSE (n={n_corpus_false:,})',
    )
    ax.hist(
        a_dist, bins=bins, density=True, alpha=0.7,
        color='C3', label=f'austin (n={n_austin:,})',
    )
    ax.axvline(threshold, color='black', linestyle='--',
               label=f'threshold = {threshold:.4f}')
    ax.set_xlabel('pairwise distance')
    ax.set_ylabel('density')
    ax.set_title(f'Austin vs corpus FALSE distance distribution\n'
                 f'austin acc {austin_accuracy:.3f} vs corpus FALSE acc {corpus_false_accuracy:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f'wrote {out_fig}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
