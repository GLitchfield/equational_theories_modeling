"""Phase 6 (Task 3) of `rewrite_chain_study_plan.md` — intermediate-state
characterization.

Two passes:

  1. Cheap whole-corpus pass on the existing traces parquet:
       classify each closed pair as forward_only / backward_only /
       both_move from (forward_depth, backward_depth). No engine
       re-run needed — the depths are already in the trace.

  2. Sample pass with engine re-run: for N=500 closed pairs sampled
     proportionally across the total_depth distribution, run
     `find_path` to reconstruct one shortest proof and compute the
     term-size trajectory along the chain.

The "intermediate states are equations" claim from the survey is
verified by construction: the engine tracks (current_lhs, current_rhs)
as a pair, so every intermediate state IS an equation. The reported
metric here is whether the rewrite holds the equation's term-size
bounded vs blows it up.

Reads:
  et_studies/rewrite_chain_study/data/rewrite_engine_traces.parquet
  et_studies/rewrite_chain_study/data/rewrite_chain_dataset.parquet  (for e1,e2)
  data/raw/etp/equations.txt

Writes:
  outputs/analyses/rewrite_chain_study/task3_intermediate_states.json
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

from lib.data.parser import parse_equation  # noqa: E402
from et_studies.rewrite_chain_study.code.engine import default_size_cap, find_path  # noqa: E402
from lib.rewrite.matching import term_size  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def load_equations(path: str) -> list:
    with open(_abs(path)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    return [parse_equation(line) for line in lines]


def classify_direction(fd: int, bd: int) -> str:
    if fd == 0 and bd == 0:
        return 'trivial'
    if fd == 0 and bd > 0:
        return 'backward_only'
    if fd > 0 and bd == 0:
        return 'forward_only'
    return 'both_move'


def percentiles(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {'n': 0}
    return {
        'n': int(len(arr)),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--traces',
                    default='et_studies/rewrite_chain_study/data/rewrite_engine_traces.parquet')
    ap.add_argument('--dataset',
                    default='et_studies/rewrite_chain_study/data/rewrite_chain_dataset.parquet')
    ap.add_argument('--output',
                    default='outputs/analyses/rewrite_chain_study/task3_intermediate_states.json')
    ap.add_argument('--sample-size', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--d-max', type=int, default=6)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    out_path = _abs(args.output)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force.', file=sys.stderr)
        return 2

    # --- 1. load ---
    print(f'loading equations from {args.equations}')
    eqs = load_equations(args.equations)
    print(f'  {len(eqs):,} equations parsed')

    print(f'loading traces from {args.traces}')
    tr = pq.read_table(_abs(args.traces)).to_pandas()
    print(f'  {len(tr):,} rows (closed={int(tr["closed"].sum()):,})')

    # --- 2. PASS 1: whole-corpus direction classification ---
    closed = tr[tr['closed']].copy()
    closed['direction'] = [classify_direction(fd, bd) for fd, bd in
                           zip(closed['forward_depth'], closed['backward_depth'])]
    direction_counts = closed['direction'].value_counts().to_dict()
    direction_pct = {k: float(v) / len(closed) for k, v in direction_counts.items()}

    print()
    print('=== Direction of motion (closed pairs) ===')
    for k in ['trivial', 'forward_only', 'backward_only', 'both_move']:
        c = direction_counts.get(k, 0)
        p = direction_pct.get(k, 0.0)
        print(f'  {k:>14}: {c:>5,} ({100*p:5.2f}%)')

    # Direction × total_depth cross-tab
    print()
    print('=== Direction × total_depth ===')
    cross = (
        closed.groupby(['total_depth', 'direction']).size().unstack(fill_value=0)
    )
    print(cross.to_string())
    cross_dict = {int(d): {k: int(v) for k, v in row.items()}
                  for d, row in cross.iterrows()}

    # --- 3. PASS 2: sample-level trajectory reconstruction ---
    print()
    print(f'=== Sample pass: re-running find_path on {args.sample_size} pairs ===')
    rng = np.random.default_rng(args.seed)
    if len(closed) <= args.sample_size:
        sample = closed
    else:
        # Stratify by total_depth so each depth is represented.
        sample_parts = []
        for d, grp in closed.groupby('total_depth'):
            frac = len(grp) / len(closed)
            n_take = max(1, int(round(args.sample_size * frac)))
            n_take = min(n_take, len(grp))
            idx = rng.choice(len(grp), size=n_take, replace=False)
            sample_parts.append(grp.iloc[idx])
        sample = (
            __import__('pandas').concat(sample_parts).sample(
                n=args.sample_size, random_state=args.seed)
            if sum(len(p) for p in sample_parts) > args.sample_size
            else __import__('pandas').concat(sample_parts)
        )
    print(f'  sampled {len(sample):,} pairs')

    trajectories = []
    for _, row in sample.iterrows():
        e1_id, e2_id = int(row['e1']), int(row['e2'])
        eq1, eq2 = eqs[e1_id - 1], eqs[e2_id - 1]
        cap = default_size_cap(eq2.lhs, eq2.rhs)
        p = find_path(eq2.lhs, eq2.rhs, eq1.lhs, eq1.rhs,
                      d_max=args.d_max, size_cap=cap)
        if not p.closed:
            continue
        fwd_sizes = [term_size(t) for t in p.forward_path]
        bwd_sizes = [term_size(t) for t in p.backward_path]
        start_lhs_size = fwd_sizes[0]
        start_rhs_size = bwd_sizes[0]
        meet_size = fwd_sizes[-1]  # same as bwd_sizes[-1]
        all_sizes = fwd_sizes + bwd_sizes
        trajectories.append({
            'e1': e1_id, 'e2': e2_id,
            'forward_depth': p.forward_depth,
            'backward_depth': p.backward_depth,
            'total_depth': p.total_depth,
            'direction': classify_direction(p.forward_depth, p.backward_depth),
            'start_lhs_size': start_lhs_size,
            'start_rhs_size': start_rhs_size,
            'meet_size': meet_size,
            'max_size_along_path': int(max(all_sizes)),
            'fwd_size_trajectory': fwd_sizes,
            'bwd_size_trajectory': bwd_sizes,
        })

    # --- 4. aggregate trajectory stats ---
    starts_max = np.array([max(t['start_lhs_size'], t['start_rhs_size'])
                           for t in trajectories])
    maxes = np.array([t['max_size_along_path'] for t in trajectories])
    meets = np.array([t['meet_size'] for t in trajectories])
    blowup = (maxes - starts_max).clip(min=0)
    # how often does the max-size-along-path exceed the larger of the two
    # endpoints? (i.e., did the proof have to "blow up" intermediate terms
    # beyond what the endpoints already required?)
    blew_up = blowup > 0

    summary = {
        'config': {
            'n_traces_total': int(len(tr)),
            'n_closed_total': int(len(closed)),
            'sample_size_requested': args.sample_size,
            'sample_size_actual': int(len(trajectories)),
            'seed': args.seed,
        },
        'whole_corpus_direction_classification': {
            'counts': {k: int(v) for k, v in direction_counts.items()},
            'fractions': {k: float(v) for k, v in direction_pct.items()},
        },
        'whole_corpus_direction_by_depth': cross_dict,
        'sample_trajectory_summary': {
            'start_max_size': percentiles(starts_max),
            'max_size_along_path': percentiles(maxes),
            'meet_term_size': percentiles(meets),
            'size_blowup_above_starts': percentiles(blowup),
            'pct_with_blowup': float(blew_up.mean()),
            'note': ('intermediate states are equations by construction '
                     '(engine tracks (lhs, rhs) pair); the term-size '
                     'trajectory measures whether intermediates blow up '
                     'beyond the goal endpoints'),
        },
        'sample_trajectories_first_20': trajectories[:20],
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Cast trajectory size lists to plain ints so JSON serializes cleanly.
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    with open(out_path, 'w') as f:
        json.dump(_clean(summary), f, indent=2, sort_keys=False)
    print(f'wrote {out_path}')

    # --- 5. headline ---
    print()
    print('=== Sample trajectory summary ===')
    print(f'  start max-side size   : median {np.median(starts_max):.1f}, '
          f'p95 {np.percentile(starts_max, 95):.1f}, max {starts_max.max()}')
    print(f'  meeting term size     : median {np.median(meets):.1f}, '
          f'p95 {np.percentile(meets, 95):.1f}, max {meets.max()}')
    print(f'  max-along-path size   : median {np.median(maxes):.1f}, '
          f'p95 {np.percentile(maxes, 95):.1f}, max {maxes.max()}')
    print(f'  pairs whose path goes above start-max: '
          f'{int(blew_up.sum()):,} / {len(maxes):,} '
          f'({100*blew_up.mean():.1f}%)')
    print(f'  blowup amount (when nonzero):  median '
          f'{np.median(blowup[blew_up]) if blew_up.any() else 0:.1f}, '
          f'max {blowup.max() if len(blowup) else 0}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
