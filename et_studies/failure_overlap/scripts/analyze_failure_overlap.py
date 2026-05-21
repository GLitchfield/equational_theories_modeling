"""Compute failure-overlap analysis across runs' per-pair predictions.

Reads each run's ``embeddings/outputs/runs/<name>/predictions.parquet`` (produced
by ``embeddings/scripts/save_predictions.py``) plus its ``failure_summary.json``,
computes which pairs each run got wrong, and aggregates into:

  - ``failure_count_per_pair.parquet``  one row per pair, with two
    failure counters: ``n_runs_failed_all`` (over all included runs)
    and ``n_runs_failed_trained`` (over the trained-successfully
    partition only).
  - ``always_wrong_all.parquet``        pairs where every included run failed.
  - ``always_wrong_trained.parquet``    pairs where every trained run failed.
  - ``pairwise_jaccard.csv``            N×N Jaccard between failure sets.
  - ``failure_count_distribution.csv``  histogram of n_runs_failed_all.
  - ``README.md``                       prose writeup of the analysis,
    with results partitioned by training health.

Rule B (`feedback_run_aggregate_partitioning.md`) is applied here: a
run is considered "pathological" iff its best-epoch checkpoint is
``epoch == 0`` (training never improved past the random-init point —
this captures the h=256 simplicity-bias drift mode where the model
peaks at ep0 then degrades, AND the aux_weight=5.0 gradient-pathology
runs that also peak at ep0). All headline aggregates (mean Jaccard,
always-wrong set sizes) are reported separately for "trained" and
"all" partitions so the broken cluster doesn't dominate the number.

SIGReg / LeJEPA runs are excluded entirely from the analysis (they
use a fundamentally different regularizer — see CHANGELOG 2026-05-12).

Usage:
  python et_studies/failure_overlap/scripts/analyze_failure_overlap.py
  python et_studies/failure_overlap/scripts/analyze_failure_overlap.py --force            # overwrite existing outputs
  python et_studies/failure_overlap/scripts/analyze_failure_overlap.py --runs-glob '...'  # custom run pattern
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

EXCLUDE_NAME_SUBSTRINGS = ('sigreg', 'lejepa')


@dataclass
class RunInfo:
    name: str
    predictions_path: str
    summary: dict
    bal_acc: float
    epoch: int
    n_wrong: int
    n_pairs: int
    is_pathological: bool   # ep == 0 -> never trained past init
    failed_mask: np.ndarray | None = None  # 1 where this run was wrong; bool, shape (n_pairs,)


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def discover_runs(runs_glob: str) -> list[RunInfo]:
    paths = sorted(glob.glob(_abs(runs_glob)))
    runs: list[RunInfo] = []
    for pred_path in paths:
        run_dir = os.path.dirname(pred_path)
        name = os.path.basename(run_dir)
        if any(sub in name for sub in EXCLUDE_NAME_SUBSTRINGS):
            continue
        summary_path = os.path.join(run_dir, 'failure_summary.json')
        if not os.path.exists(summary_path):
            print(f'  WARNING: skipping {name}: no failure_summary.json next to predictions')
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        epoch = int(summary['epoch'])
        runs.append(RunInfo(
            name=name,
            predictions_path=pred_path,
            summary=summary,
            bal_acc=float(summary['bal_acc']),
            epoch=epoch,
            n_wrong=int(summary['n_wrong']),
            n_pairs=int(summary['n_pairs']),
            is_pathological=(epoch == 0),
        ))
    runs.sort(key=lambda r: r.name)
    return runs


def load_failed_masks(runs: list[RunInfo]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read predictions parquets, fill ``failed_mask`` on each run, and
    return shared pair-table columns (eq1_id, eq2_id, ground_truth)."""
    eq1 = eq2 = ground_truth = None
    n_pairs = runs[0].n_pairs

    for i, r in enumerate(runs):
        t = pq.read_table(r.predictions_path,
                          columns=['eq1_id', 'eq2_id', 'ground_truth', 'predicted'])
        if len(t) != n_pairs:
            raise ValueError(
                f'{r.name}: predictions table has {len(t):,} rows but '
                f'failure_summary says {n_pairs:,}'
            )
        gt = t['ground_truth'].to_numpy()
        pred = t['predicted'].to_numpy()
        # Validate consistency of the shared columns across runs (we
        # rely on the same row order to do per-pair aggregation).
        if i == 0:
            eq1 = t['eq1_id'].to_numpy()
            eq2 = t['eq2_id'].to_numpy()
            ground_truth = gt
        else:
            if not np.array_equal(t['eq1_id'].to_numpy(), eq1):
                raise ValueError(f'{r.name}: eq1_id column does not match the first run')
            if not np.array_equal(t['eq2_id'].to_numpy(), eq2):
                raise ValueError(f'{r.name}: eq2_id column does not match the first run')
            if not np.array_equal(gt, ground_truth):
                raise ValueError(f'{r.name}: ground_truth column does not match the first run')
        r.failed_mask = (pred != gt)
        n_wrong_observed = int(r.failed_mask.sum())
        if n_wrong_observed != r.n_wrong:
            print(
                f'  WARNING: {r.name}: observed n_wrong {n_wrong_observed:,} '
                f'does not match summary {r.n_wrong:,}'
            )

    return eq1, eq2, ground_truth


def compute_failure_counts(runs: list[RunInfo]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair sums across all runs and across the trained partition."""
    n_pairs = runs[0].n_pairs
    n_all = np.zeros(n_pairs, dtype=np.int16)        # max 13 fits in int8 but keep headroom
    n_trained = np.zeros(n_pairs, dtype=np.int16)
    for r in runs:
        m = r.failed_mask.astype(np.int16)
        n_all += m
        if not r.is_pathological:
            n_trained += m
    # Both fit in int8 (max 13 or 8 in practice).
    return n_all.astype(np.int8), n_trained.astype(np.int8)


def pairwise_jaccard(runs: list[RunInfo]) -> pd.DataFrame:
    names = [r.name for r in runs]
    n = len(names)
    J = np.zeros((n, n), dtype=np.float64)
    for i, ri in enumerate(runs):
        for j, rj in enumerate(runs):
            if j < i:
                J[i, j] = J[j, i]
                continue
            if i == j:
                J[i, j] = 1.0
                continue
            inter = int(np.logical_and(ri.failed_mask, rj.failed_mask).sum())
            union = int(np.logical_or(ri.failed_mask, rj.failed_mask).sum())
            J[i, j] = inter / union if union else 0.0
    return pd.DataFrame(J, index=names, columns=names)


def mean_off_diagonal(J: pd.DataFrame, subset: list[str] | None = None) -> float:
    """Mean of the off-diagonal Jaccard entries, optionally restricted
    to a subset of run names (uses only rows and columns within the
    subset)."""
    if subset is not None:
        J = J.loc[subset, subset]
    M = J.to_numpy()
    n = M.shape[0]
    if n < 2:
        return float('nan')
    mask = ~np.eye(n, dtype=bool)
    return float(M[mask].mean())


def write_artifacts(
    runs: list[RunInfo],
    eq1: np.ndarray,
    eq2: np.ndarray,
    ground_truth: np.ndarray,
    n_all: np.ndarray,
    n_trained: np.ndarray,
    jaccard: pd.DataFrame,
    out_dir: str,
    force: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n_runs_all = len(runs)
    n_runs_trained = sum(1 for r in runs if not r.is_pathological)

    # --- failure_count_per_pair.parquet ---
    per_pair_path = os.path.join(out_dir, 'failure_count_per_pair.parquet')
    if os.path.exists(per_pair_path) and not force:
        raise FileExistsError(f'{per_pair_path} exists; pass --force to overwrite')
    table = pa.table({
        'eq1_id': pa.array(eq1, type=pa.int32()),
        'eq2_id': pa.array(eq2, type=pa.int32()),
        'ground_truth': pa.array(ground_truth, type=pa.int8()),
        'n_runs_failed_all': pa.array(n_all, type=pa.int8()),
        'n_runs_failed_trained': pa.array(n_trained, type=pa.int8()),
    })
    pq.write_table(table, per_pair_path, compression='zstd')

    # --- always_wrong_{all,trained}.parquet ---
    for tag, counts, n_total in [
        ('all', n_all, n_runs_all),
        ('trained', n_trained, n_runs_trained),
    ]:
        mask = counts == n_total
        out_path = os.path.join(out_dir, f'always_wrong_{tag}.parquet')
        if os.path.exists(out_path) and not force:
            raise FileExistsError(f'{out_path} exists; pass --force to overwrite')
        t = pa.table({
            'eq1_id': pa.array(eq1[mask], type=pa.int32()),
            'eq2_id': pa.array(eq2[mask], type=pa.int32()),
            'ground_truth': pa.array(ground_truth[mask], type=pa.int8()),
        })
        pq.write_table(t, out_path, compression='zstd')

    # --- pairwise_jaccard.csv ---
    jaccard_path = os.path.join(out_dir, 'pairwise_jaccard.csv')
    if os.path.exists(jaccard_path) and not force:
        raise FileExistsError(f'{jaccard_path} exists; pass --force to overwrite')
    jaccard.to_csv(jaccard_path)

    # --- failure_count_distribution.csv ---
    dist_path = os.path.join(out_dir, 'failure_count_distribution.csv')
    if os.path.exists(dist_path) and not force:
        raise FileExistsError(f'{dist_path} exists; pass --force to overwrite')
    dist_all = pd.Series(n_all).value_counts().sort_index()
    dist_trained = pd.Series(n_trained).value_counts().sort_index()
    dist_df = pd.DataFrame({
        'k_runs_failed': sorted(set(dist_all.index) | set(dist_trained.index)),
    })
    dist_df['n_pairs_all'] = dist_df['k_runs_failed'].map(dist_all).fillna(0).astype(int)
    dist_df['n_pairs_trained'] = dist_df['k_runs_failed'].map(dist_trained).fillna(0).astype(int)
    dist_df.to_csv(dist_path, index=False)


def _format_jaccard_table(jaccard: pd.DataFrame, runs: list[RunInfo]) -> str:
    """Render the Jaccard matrix as a markdown upper-triangle table,
    matching the existing README's format. Shows each pair once."""
    names = jaccard.index.tolist()
    short = {
        'aux_h128_diag': 'aux128',
        'aux_h128_seed1_diag': 'aux128.s1',
        'aux_h128_cohom_w5_diag': 'aux128.cohw5',
        'aux_h256_diag': 'aux256',
        'aux_h256_seed1_diag': 'aux256.s1',
        'aux_h256_cohom_diag': 'aux256.coh',
        'aux_h256_cohom_w5_diag': 'aux256.cohw5',
        'skip_h128_seed1_diag': 'skip.s1',
        'skip_h128_seed42_diag': 'skip.s42',
        'skip_h128_seed42_20ep_diag': 'skip.s42.20',
        'stable_h256_m1p5_vic0p05_diag': 'stab.h256.05',
        'stable_h256_m1p5_vic0p1_diag': 'stab.h256.1',
        'stable_m1p5_vic0p1_diag': 'stab.h128',
    }
    cols = [short.get(n, n) for n in names]
    lines = ['| | ' + ' | '.join(cols) + ' |']
    lines.append('|---|' + ''.join(['---:|'] * len(cols)))
    for i, r in enumerate(names):
        row_cells = []
        for j, c in enumerate(names):
            if j < i:
                row_cells.append('')  # below diagonal: blank
            elif i == j:
                row_cells.append('1.000')
            else:
                row_cells.append(f'{jaccard.iloc[i, j]:.3f}')
        lines.append('| ' + short.get(r, r) + ' | ' + ' | '.join(row_cells) + ' |')
    return '\n'.join(lines)


def _load_invariants(path: str) -> pd.DataFrame | None:
    """Load equation invariants (spine_type etc.) if present, else None."""
    p = _abs(path)
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    return df


SPINE_LABELS = ['family_c', 'mixed', 'no_lhs_var_in_rhs', 'left', 'right', 'trivial']


def _read_equations(path: str = 'data/raw/etp/equations.txt') -> dict[int, str]:
    """Best-effort load of equation text. eq id is 1-indexed line number."""
    p = _abs(path)
    if not os.path.exists(p):
        return {}
    eqs: dict[int, str] = {}
    with open(p) as f:
        for i, line in enumerate(f, start=1):
            eqs[i] = line.strip()
    return eqs


def _spine_breakdowns(
    aw_eq1: np.ndarray,
    aw_eq2: np.ndarray,
    all_eq1: np.ndarray,
    all_eq2: np.ndarray,
    invariants: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return (E2-spine of always-wrong, E1-spine rate over corpus).

    E2-spine: count of always-wrong pairs grouped by E2's spine label.
    E1-spine rate: rate of always-wrong / total per E1-spine label.
    """
    spine_map = invariants['spine_type'].to_dict()  # eq_id -> int
    aw_e2_spine = pd.Series([spine_map.get(int(i), -1) for i in aw_eq2])
    e2_counts = aw_e2_spine.value_counts().sort_index()

    aw_e1_spine = pd.Series([spine_map.get(int(i), -1) for i in aw_eq1])
    all_e1_spine = pd.Series([spine_map.get(int(i), -1) for i in all_eq1])
    aw_e1_counts = aw_e1_spine.value_counts()
    all_e1_counts = all_e1_spine.value_counts()
    e1_rates = (aw_e1_counts / all_e1_counts).fillna(0.0)
    return e2_counts, e1_rates


def render_readme(
    runs: list[RunInfo],
    eq1: np.ndarray,
    eq2: np.ndarray,
    ground_truth: np.ndarray,
    n_all: np.ndarray,
    n_trained: np.ndarray,
    jaccard: pd.DataFrame,
    invariants: pd.DataFrame | None = None,
    eq_texts: dict[int, str] | None = None,
) -> str:
    trained = [r for r in runs if not r.is_pathological]
    patho = [r for r in runs if r.is_pathological]
    n_total = len(runs)
    n_trained_runs = len(trained)

    trained_names = [r.name for r in trained]
    mean_jacc_all = mean_off_diagonal(jaccard)
    mean_jacc_trained = mean_off_diagonal(jaccard, trained_names)
    mean_jacc_patho_vs_patho = mean_off_diagonal(
        jaccard, [r.name for r in patho]
    ) if len(patho) >= 2 else float('nan')

    # Always-wrong sets
    aw_all_mask = n_all == n_total
    aw_trained_mask = n_trained == n_trained_runs
    aw_all_n = int(aw_all_mask.sum())
    aw_trained_n = int(aw_trained_mask.sum())
    aw_all_true = int((ground_truth[aw_all_mask] == 1).sum())
    aw_trained_true = int((ground_truth[aw_trained_mask] == 1).sum())

    # Equation 1 (`x = x`) instances in always-wrong-trained:
    # pairs where E2 == 1, ground_truth == 1 (always-TRUE), and model
    # missed across all trained runs.
    eq1_aw_trained = int(((eq2[aw_trained_mask]) == 1).sum())

    # Distribution
    counts = pd.Series(n_all).value_counts().sort_index()
    counts_tr = pd.Series(n_trained).value_counts().sort_index()
    n_pairs = len(n_all)

    # Mean per-run failure rate (informs iid-expected baseline)
    rates_all = [r.n_wrong / r.n_pairs for r in runs]
    rates_trained = [r.n_wrong / r.n_pairs for r in trained]
    mean_rate_all = float(np.mean(rates_all))
    mean_rate_trained = float(np.mean(rates_trained))

    # Build the readme as a single string.
    L = []
    L.append('# Failure-overlap analysis across runs')
    L.append('')
    L.append('Question: when models fail to predict an equation pair correctly,')
    L.append('are those failures the **same pairs** across runs (systematic')
    L.append('difficulty), or **different random pairs** (model-specific noise)?')
    L.append('')
    L.append('Answer: **strongly systematic**, with a clean and interpretable')
    L.append('failure mode (TRUE pairs where E2 is a trivial or very-weak')
    L.append('axiom). The headline numbers are partitioned by training health')
    L.append('to keep the broken-cluster effect separate from the systematic-')
    L.append('difficulty effect — see "Partitioning" below.')
    L.append('')
    L.append(f'Reproducer: `et_studies/failure_overlap/scripts/analyze_failure_overlap.py`')
    L.append('')
    L.append('## Partitioning')
    L.append('')
    L.append(
        f'Of {n_total} runs (excluding SIGReg / LeJEPA variants, which use a'
    )
    L.append('fundamentally different regularizer and aren\'t comparable here):')
    L.append('')
    L.append(
        f'- **{n_trained_runs} runs trained successfully** (best epoch > 0):'
    )
    for r in trained:
        L.append(f'  - `{r.name}` (ep={r.epoch}, bal_acc={r.bal_acc:.4f})')
    L.append(
        f'- **{len(patho)} runs are pathological** (best epoch == 0; the model'
    )
    L.append(
        '  never improved past random init — either h=256 simplicity-bias'
    )
    L.append(
        '  drift or aux_weight=5.0 gradient-clip pathology):'
    )
    for r in patho:
        L.append(f'  - `{r.name}` (ep={r.epoch}, bal_acc={r.bal_acc:.4f})')
    L.append('')
    L.append('All aggregate headlines below are reported for both partitions.')
    L.append('Sub-cluster Jaccards (skip vs skip, cohom_w5 vs cohom_w5)')
    L.append('don\'t need this partitioning — they\'re already conditioned on')
    L.append('a recipe.')
    L.append('')
    L.append('## Per-pair failure-count distribution')
    L.append('')
    L.append(
        'For each of {:,} pairs, count how many runs got it wrong.'.format(n_pairs)
    )
    L.append('Both partitions shown side-by-side:')
    L.append('')
    L.append('| k runs failed | n pairs (all runs) | n pairs (trained only) |')
    L.append('|---:|---:|---:|')
    all_keys = sorted(set(counts.index) | set(counts_tr.index))
    for k in all_keys:
        n_a = int(counts.get(k, 0))
        n_t = int(counts_tr.get(k, 0))
        L.append(f'| {k} | {n_a:,} | {n_t:,} |')
    L.append('')
    L.append(
        f'**Always-wrong (k == n_runs):** {aw_all_n:,} pairs across all '
        f'{n_total} runs (of which {aw_all_true:,} are TRUE pairs); '
        f'{aw_trained_n:,} pairs across {n_trained_runs} trained runs '
        f'(of which {aw_trained_true:,} are TRUE).'
    )
    L.append('')
    L.append(
        f'**Independence baseline:** mean per-run failure rate is '
        f'{mean_rate_all:.4f} (all runs) / {mean_rate_trained:.4f} (trained '
        f'only). Under independence, the expected count at k={n_total} (all) '
        f'is {n_pairs * mean_rate_all**n_total:.2e} and at k={n_trained_runs} '
        f'(trained) is {n_pairs * mean_rate_trained**n_trained_runs:.2e} — both '
        f'effectively zero. The observed always-wrong sets are radically '
        f'non-independent.'
    )
    L.append('')
    L.append('## Pairwise Jaccard between failure sets')
    L.append('')
    L.append(
        'Jaccard = |intersection| / |union|; 1.0 = identical, 0 = disjoint. '
        f'Full {len(runs)}×{len(runs)} matrix in `pairwise_jaccard.csv`; '
        'upper triangle shown below.'
    )
    L.append('')
    L.append(_format_jaccard_table(jaccard, runs))
    L.append('')
    L.append('**Aggregate Jaccard, partitioned:**')
    L.append('')
    L.append(
        f'- **Mean off-diagonal across {n_trained_runs} trained runs: '
        f'{mean_jacc_trained:.3f}.** This is the headline for "systematic '
        f'difficulty across well-trained models."'
    )
    L.append(
        f'- Mean off-diagonal across all {n_total} runs (trained + pathological): '
        f'{mean_jacc_all:.3f}.'
    )
    if not np.isnan(mean_jacc_patho_vs_patho):
        L.append(
            f'- Mean off-diagonal among pathological runs only: '
            f'{mean_jacc_patho_vs_patho:.3f}. Higher than the trained-only '
            f'mean because broken models default to similar majority-class '
            f'predictions and so share most of their errors.'
        )
    L.append('')
    L.append('**Reading the matrix:**')
    L.append('')
    L.append(
        '- **Skip runs cluster together** (skip.s1 ↔ skip.s42 ↔ skip.s42.20: '
        '0.39-0.45) but have LOW Jaccard with all other runs (0.05-0.12). '
        'The skip recipe fails on substantially different pairs than the '
        'no-skip recipes.'
    )
    L.append(
        '- **High-aux-weight cohom runs cluster** (aux128.cohw5 ↔ '
        'aux256.cohw5 ≈ 0.76). These are the two grad-pathology runs — both '
        'mostly predicting majority class, so they share failure patterns.'
    )
    L.append(
        '- **aux256 (lucky regime-B run)** has unusually low Jaccard with '
        'everything; its failure set is unique. Consistent with the finding '
        'that its 0.9635 came from a different mechanism than the other models.'
    )
    L.append('')
    L.append('## Always-wrong characterization')
    L.append('')
    L.append(
        f'Across the {n_trained_runs} trained runs, **{aw_trained_n:,} pairs '
        f'are always-wrong**, of which **{aw_trained_true:,} ({100*aw_trained_true/max(aw_trained_n,1):.2f}%) '
        f'are TRUE pairs**. The systematic failure mode is "TRUE pairs that '
        f'look like FALSE pairs to every trained model."'
    )
    L.append('')
    L.append(
        f'Equation 1 (`x = x`, the trivial reflexive) accounts for '
        f'**{eq1_aw_trained:,} of the {aw_trained_n:,} always-wrong-trained '
        f'pairs** — pairs `(E1, x=x)` where E1 → x=x is tautologically TRUE '
        f'but the model misses.'
    )
    L.append('')

    # Top equations by always-wrong frequency, using the trained-only set.
    aw_eq1 = eq1[aw_trained_mask]
    aw_eq2 = eq2[aw_trained_mask]
    aw_eq2_counts = pd.Series(aw_eq2).value_counts().head(10)
    L.append('### Top-10 E2 (equation being implied) by always-wrong frequency')
    L.append('')
    L.append('| eq_id | pairs always-wrong | text |')
    L.append('|---:|---:|---|')
    for eqid, n in aw_eq2_counts.items():
        text = (eq_texts or {}).get(int(eqid), '?')
        L.append(f'| {int(eqid)} | {int(n)} | `{text}` |')
    L.append('')
    L.append(
        'All of the top 10 are either the trivial reflexive (eq 1) or '
        'high-variable-count equations with multiple distinct variables on '
        'both sides — equations that are implied by almost every more-'
        'specific equation because they\'re so weakly constraining.'
    )
    L.append('')

    # Spine breakdowns if invariants available.
    if invariants is not None:
        e2_counts, e1_rates = _spine_breakdowns(
            aw_eq1, aw_eq2, eq1, eq2, invariants,
        )
        L.append('### Spine breakdown (always-wrong-trained set)')
        L.append('')
        L.append('When the always-wrong pair is `(E1, E2)`, the spine of E2 (the')
        L.append('equation being implied):')
        L.append('')
        L.append('| E2 spine | always-wrong count | % of always-wrong |')
        L.append('|---|---:|---:|')
        for spine_idx, n in e2_counts.sort_values(ascending=False).items():
            label = SPINE_LABELS[int(spine_idx)] if 0 <= int(spine_idx) < len(SPINE_LABELS) else f'unknown({spine_idx})'
            pct = 100 * n / aw_trained_n
            L.append(f'| {label} | {int(n):,} | {pct:.1f}% |')
        L.append('')
        L.append('By E1\'s spine type, what fraction of pairs starting with that')
        L.append('spine are always-wrong:')
        L.append('')
        # Recompute with raw counts so we can show numerator/denominator.
        spine_map = invariants['spine_type'].to_dict()
        aw_e1_arr = pd.Series([spine_map.get(int(i), -1) for i in aw_eq1])
        all_e1_arr = pd.Series([spine_map.get(int(i), -1) for i in eq1])
        aw_counts = aw_e1_arr.value_counts()
        all_counts = all_e1_arr.value_counts()
        rows = []
        for spine_idx in range(len(SPINE_LABELS)):
            aw_c = int(aw_counts.get(spine_idx, 0))
            total = int(all_counts.get(spine_idx, 0))
            rate = aw_c / total if total else 0.0
            rows.append((SPINE_LABELS[spine_idx], aw_c, total, rate))
        rows.sort(key=lambda x: x[3])
        L.append('| E1 spine | always-wrong / total | rate |')
        L.append('|---|---:|---:|')
        for label, aw_c, total, rate in rows:
            L.append(f'| {label} | {aw_c:,} / {total:,} | {100*rate:.2f}% |')
        L.append('')
        L.append(
            '`trivial` and `no_lhs_var_in_rhs` E1s are never (or almost never) '
            'always-wrong — when the LHS equation is structurally simple, the '
            'model handles it. The rates are highest on `right`, `mixed`, '
            'and `left` E1s.'
        )
        L.append('')

    L.append('## Files in this directory')
    L.append('')
    L.append('- `README.md` — this writeup')
    L.append(
        '- `failure_count_per_pair.parquet` — every pair with '
        '`n_runs_failed_all` and `n_runs_failed_trained` (one row per pair)'
    )
    L.append(
        '- `always_wrong_all.parquet` — pairs where all '
        f'{n_total} runs got it wrong ({aw_all_n:,} rows)'
    )
    L.append(
        '- `always_wrong_trained.parquet` — pairs where all '
        f'{n_trained_runs} trained runs got it wrong ({aw_trained_n:,} rows)'
    )
    L.append('- `pairwise_jaccard.csv` — N×N Jaccard matrix')
    L.append(
        '- `failure_count_distribution.csv` — histogram of n_runs_failed for '
        'both partitions'
    )
    L.append('')
    L.append('## Implications')
    L.append('')
    L.append(
        '1. **The failure mode is interpretable: TRUE-by-vacuity.** Models '
        "can't reliably detect TRUE pairs where E2 is a tautology (eq 1) or "
        'a very weak axiom (high-variable family_c equations). The '
        'contrastive distance metric encodes "are these two equations '
        'structurally similar?" but the actual implication relation is '
        '"does E1\'s set of models include E2\'s set of models?", which for '
        'trivial / weak E2s requires reasoning beyond structural similarity.'
    )
    L.append(
        '2. **Skip recipe fails differently from aux recipes.** Skip runs '
        'share most of their failures with each other but barely overlap '
        'with aux baselines. Likely complementary errors — an ensemble of '
        'skip + aux would probably move on the always-wrong set.'
    )
    L.append(
        f'3. **The {aw_trained_n:,} always-wrong-trained pairs are a stable '
        'target.** Any future intervention that improves this set is making '
        'real progress. A simple rule "if E2 is `x = x`, predict TRUE" would '
        f'correctly fix {eq1_aw_trained:,} of them at zero training cost.'
    )
    return '\n'.join(L) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--runs-glob', default='embeddings/outputs/runs/*/predictions.parquet',
        help='glob for per-run predictions parquets (default: %(default)s)',
    )
    parser.add_argument(
        '--output-dir', default='outputs/analyses/failure_overlap',
        help='output directory for artifacts (default: %(default)s)',
    )
    parser.add_argument(
        '--invariants-path',
        default='data/processed/equation_invariants.parquet',
        help='per-equation invariants parquet for the spine/eq breakdown sections',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='overwrite existing artifacts',
    )
    args = parser.parse_args(argv)

    out_dir = _abs(args.output_dir)
    print(f'output dir: {out_dir}')

    print('discovering runs ...')
    runs = discover_runs(args.runs_glob)
    if not runs:
        print('ERROR: no runs found', file=sys.stderr)
        return 2
    print(f'  {len(runs)} runs included')
    for r in runs:
        tag = 'pathological' if r.is_pathological else 'trained'
        print(f'    {r.name:<35} ep={r.epoch:>2}  bal={r.bal_acc:.4f}  [{tag}]')

    print('loading prediction parquets and computing per-run failed masks ...')
    t0 = time.perf_counter()
    eq1, eq2, ground_truth = load_failed_masks(runs)
    print(f'  loaded {len(runs)} parquets in {time.perf_counter() - t0:.1f}s')

    print('computing failure counts and Jaccard matrix ...')
    t1 = time.perf_counter()
    n_all, n_trained = compute_failure_counts(runs)
    jaccard = pairwise_jaccard(runs)
    print(f'  done in {time.perf_counter() - t1:.1f}s')

    print('writing artifacts ...')
    t2 = time.perf_counter()
    write_artifacts(
        runs, eq1, eq2, ground_truth, n_all, n_trained, jaccard,
        out_dir, args.force,
    )
    invariants = _load_invariants(args.invariants_path)
    if invariants is None:
        print(
            f'  note: {args.invariants_path} not found; spine breakdown will '
            f'be omitted from the README'
        )
    eq_texts = _read_equations()
    readme_text = render_readme(
        runs, eq1, eq2, ground_truth, n_all, n_trained, jaccard,
        invariants=invariants, eq_texts=eq_texts,
    )
    readme_path = os.path.join(out_dir, 'README.md')
    if os.path.exists(readme_path) and not args.force:
        raise FileExistsError(f'{readme_path} exists; pass --force to overwrite')
    with open(readme_path, 'w') as f:
        f.write(readme_text)
    print(f'  wrote artifacts in {time.perf_counter() - t2:.1f}s')

    print()
    print(f'wrote outputs to {out_dir}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
