"""Analysis 6 of testing_plan_phase2.md: per-equation failure concentration.

Each of the 4,694 equations participates in 2*(N-1) = 9,386 pairs (it
can appear as eq1 against every other equation, and as eq2 against
every other equation). For every equation, count:

  appearances : number of pairs it appears in (off-diagonal corpus)
  failures    : number of those pairs the model gets wrong
  failure_rate: failures / appearances

If failures cluster on a few equations, the failure mode has a clear
structural locus. If they're roughly uniform, every equation is hard
in roughly the same way and the failure mode is interaction-dependent.

Outputs:
  outputs/failure_analysis/analysis/equation_concentration.parquet
    one row per equation: eq_id, appearances, failures, failure_rate
  outputs/failure_analysis/analysis/top20_failure_equations.txt
    human-readable report — top 20 by failure rate, with equation text
  outputs/failure_analysis/figures/equation_failure_rate_distribution.png
    histogram + cumulative curve

Usage:
  python scripts/analyze_equation_concentration.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _load_equations(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--predictions',
        default='outputs/failure_analysis/predictions.parquet',
    )
    p.add_argument(
        '--equations',
        default='data/raw/etp/equations.txt',
    )
    p.add_argument(
        '--output-parquet',
        default='outputs/failure_analysis/analysis/equation_concentration.parquet',
    )
    p.add_argument(
        '--output-text',
        default='outputs/failure_analysis/analysis/top20_failure_equations.txt',
    )
    p.add_argument(
        '--output-figure',
        default='outputs/failure_analysis/figures/equation_failure_rate_distribution.png',
    )
    args = p.parse_args(argv)

    print(f'reading {args.predictions}')
    tbl = pq.read_table(_abs(args.predictions))
    eq1 = tbl['eq1_id'].to_numpy()
    eq2 = tbl['eq2_id'].to_numpy()
    gt = tbl['ground_truth'].to_numpy()
    pred = tbl['predicted'].to_numpy()
    print(f'  {len(eq1):,} pairs')

    eqs = _load_equations(_abs(args.equations))
    n_eqs = len(eqs)
    print(f'  {n_eqs:,} equations')

    wrong = pred != gt
    print(f'  {int(wrong.sum()):,} wrong pairs '
          f'({100 * wrong.sum() / len(wrong):.2f}%)')

    # --- count appearances + failures per equation ----------------------
    # Each pair contributes 1 appearance to eq1 and 1 to eq2 (so
    # appearances are counted as "rows in the table where this equation
    # appears on either side"). Vectorize via bincount.
    eq_max = max(int(eq1.max()), int(eq2.max()))
    n_bins = eq_max + 1

    app_e1 = np.bincount(eq1, minlength=n_bins)
    app_e2 = np.bincount(eq2, minlength=n_bins)
    appearances = (app_e1 + app_e2).astype(np.int64)

    fail_e1 = np.bincount(eq1[wrong], minlength=n_bins)
    fail_e2 = np.bincount(eq2[wrong], minlength=n_bins)
    failures = (fail_e1 + fail_e2).astype(np.int64)

    # Drop the unused 0-index slot.
    eq_ids = np.arange(1, n_bins, dtype=np.int32)
    appearances = appearances[1:]
    failures = failures[1:]
    rates = failures / np.maximum(appearances, 1)

    print()
    print(f'=== per-equation failure rate ===')
    print(f'  median rate     : {np.median(rates):.4f}')
    print(f'  mean rate       : {rates.mean():.4f}')
    print(f'  std             : {rates.std():.4f}')
    print(f'  min / max       : {rates.min():.4f} / {rates.max():.4f}')
    print(f'  90th / 99th pct : {np.quantile(rates, 0.9):.4f} / '
          f'{np.quantile(rates, 0.99):.4f}')

    # Concentration: what share of total failures is in the top-K
    # equations by absolute failure count?
    order_by_count = np.argsort(failures)[::-1]
    cum_failures = np.cumsum(failures[order_by_count])
    total_failures = int(failures.sum())
    for k in (10, 50, 100, 500, 1000):
        if k <= len(failures):
            share = cum_failures[k - 1] / total_failures
            print(f'  top-{k:>4} equations cover {100 * share:.2f}% of all '
                  f'failure-pair endpoints')

    n_perfect = int((failures == 0).sum())
    print(f'  equations with 0 failure pairs: {n_perfect}  '
          f'({100 * n_perfect / len(rates):.2f}%)')

    # --- write parquet ---------------------------------------------------
    out_pq = _abs(args.output_parquet)
    os.makedirs(os.path.dirname(out_pq), exist_ok=True)
    out_table = pa.table({
        'eq_id': pa.array(eq_ids, type=pa.int32()),
        'appearances': pa.array(appearances, type=pa.int64()),
        'failures': pa.array(failures, type=pa.int64()),
        'failure_rate': pa.array(rates.astype(np.float32), type=pa.float32()),
    })
    pq.write_table(out_table, out_pq, compression='zstd')
    print(f'\nwrote {out_pq}')

    # --- top-20 text -----------------------------------------------------
    order_by_rate = np.argsort(rates)[::-1][:20]
    out_txt = _abs(args.output_text)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    lines = []
    lines.append(f'Top 20 equations by failure rate.')
    lines.append(f'(participation as eq1 OR eq2 across the '
                 f'{len(eq1):,}-pair off-diagonal corpus)')
    lines.append('')
    lines.append(
        f'{"rank":>4}  {"eq_id":>5}  {"failures":>10}  {"appearances":>11}  '
        f'{"rate":>6}  equation'
    )
    lines.append('-' * 110)
    for rank, idx in enumerate(order_by_rate, start=1):
        eid = int(eq_ids[idx])
        lines.append(
            f'{rank:>4}  {eid:>5}  {int(failures[idx]):>10,}  '
            f'{int(appearances[idx]):>11,}  {rates[idx]:>6.4f}  '
            f'{eqs[eid - 1]}'
        )
    body = '\n'.join(lines)
    print()
    print(body)
    with open(out_txt, 'w') as f:
        f.write(body + '\n')
    print(f'\nwrote {out_txt}')

    # --- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_fig = _abs(args.output_figure)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram of per-equation failure rate.
    ax = axes[0]
    ax.hist(rates, bins=80, color='C3', alpha=0.85)
    ax.axvline(rates.mean(), color='black', linestyle='--',
               label=f'mean = {rates.mean():.3f}')
    ax.set_xlabel('per-equation failure rate')
    ax.set_ylabel('# equations')
    ax.set_title('distribution of per-equation failure rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: cumulative failure share across sorted-descending equations.
    ax = axes[1]
    sorted_failures = np.sort(failures)[::-1]
    cum = np.cumsum(sorted_failures) / max(int(sorted_failures.sum()), 1)
    ks = np.arange(1, len(sorted_failures) + 1)
    ax.plot(ks, cum, color='C0')
    ax.set_xlabel('top-K equations (sorted by absolute failure count)')
    ax.set_ylabel('cumulative share of all failure-pair endpoints')
    ax.set_title('failure-mass concentration')
    for k in (10, 100, 1000):
        if k <= len(cum):
            ax.axvline(k, color='gray', linestyle=':', alpha=0.5)
            ax.text(k, 0.05, f' k={k}\n {100*cum[k-1]:.0f}%',
                    rotation=0, fontsize=9, va='bottom')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f'wrote {out_fig}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
