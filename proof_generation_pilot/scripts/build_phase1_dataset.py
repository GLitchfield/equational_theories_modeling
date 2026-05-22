"""Phase 1 orchestration: run 3 tools (matcher, Vampire, grind) per pair,
capture per-tactic transitions via PyPantograph, write two parquets.

Per pair, per tool:
  - matcher: wrap precomputed `lean_proof_tactic` in `def submission := by
             intro G _ h; <tactic>`; drive PyPantograph through both tactics
  - vampire: call ETP's Vampire (subprocess), leanify the trace, drive
             PyPantograph through each emitted tactic line
  - grind:   `intro G _ h; grind`; let Lean's grind try to close the goal

For each pair, record:
  proofs.parquet      : one row per (pair, tool) — proof text + status + timing
  transitions.parquet : one row per (state_before, tactic, state_after) step

A single PyPantograph Server is reused across all pairs (start cost ~10s,
amortized).

Usage:
    python proof_generation_pilot/scripts/build_phase1_dataset.py \\
        --sample proof_generation_pilot/data/processed/phase1_sample_100.parquet \\
        --out-dir proof_generation_pilot/outputs/phase1_smoke

Outputs:
    <out-dir>/proofs.parquet
    <out-dir>/transitions.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from proof_generation_pilot.methods.matcher.to_lean import (
    make_submission as matcher_sub,
    expected_tactics as matcher_tacs,
)
from proof_generation_pilot.methods.grind.to_lean import (
    make_submission as grind_sub,
    expected_tactics as grind_tacs,
)
from proof_generation_pilot.methods.vampire.to_lean import (
    make_submission as vampire_sub,
)


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def _goal_expr(e1: int, e2: int) -> str:
    return f"∀ (G : Type) [Magma G], Equation{e1} G → Equation{e2} G"


def _split_vampire_body(lean_source: str) -> list[str]:
    """Split a Vampire `def submission` body into per-line tactics."""
    body = lean_source.split(':= by\n', 1)[1]
    return [ln.strip() for ln in body.split('\n') if ln.strip()]


def _drive_proof(server, goal_expr, tactics):
    """Apply tactics one by one. Returns (transitions, status, error).
    Each transition is a dict {state_before, tactic, state_after, is_terminal}."""
    transitions = []
    try:
        state = server.goal_start(goal_expr)
    except Exception as e:
        return transitions, 'error', f'goal_start failed: {e}'

    for tac in tactics:
        state_before_str = str(state)
        try:
            new_state = server.goal_tactic(state, tactic=tac)
        except Exception as e:
            transitions.append({
                'state_before': state_before_str,
                'tactic': tac,
                'state_after': None,
                'is_terminal': False,
            })
            return transitions, 'error', f'goal_tactic failed: {e}'
        state_after_str = str(new_state)
        is_terminal = new_state.is_solved
        transitions.append({
            'state_before': state_before_str,
            'tactic': tac,
            'state_after': state_after_str,
            'is_terminal': is_terminal,
        })
        state = new_state
        if is_terminal:
            break

    status = 'proved' if state.is_solved else 'no_proof'
    return transitions, status, None


def run_matcher(server, e1, e2, lean_proof_tactic):
    """Returns (lean_source, status, transitions, wall_ms, error)."""
    t0 = time.perf_counter()
    lean_src = matcher_sub(e1, e2, lean_proof_tactic)
    tactics = matcher_tacs(lean_proof_tactic)
    transitions, status, err = _drive_proof(server, _goal_expr(e1, e2), tactics)
    return lean_src, status, transitions, int((time.perf_counter() - t0) * 1000), err


def run_grind(server, e1, e2):
    t0 = time.perf_counter()
    lean_src = grind_sub(e1, e2)
    tactics = grind_tacs()
    transitions, status, err = _drive_proof(server, _goal_expr(e1, e2), tactics)
    return lean_src, status, transitions, int((time.perf_counter() - t0) * 1000), err


def run_vampire(server, e1, e2):
    t0 = time.perf_counter()
    vam_result = vampire_sub(e1, e2)
    if vam_result['status'] != 'proved':
        # Vampire itself failed (no_proof / timeout / error); record without driving
        return (vam_result['lean'] or '',
                vam_result['status'], [],
                int((time.perf_counter() - t0) * 1000),
                vam_result['error'])
    tactics = _split_vampire_body(vam_result['lean'])
    transitions, status, err = _drive_proof(server, _goal_expr(e1, e2), tactics)
    return (vam_result['lean'], status, transitions,
            int((time.perf_counter() - t0) * 1000),
            err)


TOOL_RUNNERS = {
    'matcher': lambda server, row: run_matcher(server, int(row['eq1_id']),
                                                int(row['eq2_id']),
                                                row['lean_proof_tactic']),
    'vampire': lambda server, row: run_vampire(server, int(row['eq1_id']),
                                                int(row['eq2_id'])),
    'grind':   lambda server, row: run_grind(server, int(row['eq1_id']),
                                              int(row['eq2_id'])),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--sample', required=True,
                    help='path to phase1_sample_*.parquet')
    ap.add_argument('--out-dir', required=True,
                    help='output dir for proofs.parquet + transitions.parquet')
    ap.add_argument('--tools', nargs='+', default=['matcher', 'vampire', 'grind'],
                    choices=list(TOOL_RUNNERS.keys()))
    ap.add_argument('--limit', type=int, default=None,
                    help='only process the first N pairs (debug)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    sample_path = _abs(args.sample)
    out_dir = _abs(args.out_dir)
    proofs_path = os.path.join(out_dir, 'proofs.parquet')
    trans_path = os.path.join(out_dir, 'transitions.parquet')
    for p in [proofs_path, trans_path]:
        if os.path.exists(p) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    print(f'loading sample from {args.sample} ...')
    sample = pq.read_table(sample_path).to_pandas()
    if args.limit:
        sample = sample.head(args.limit).copy()
    print(f'  {len(sample):,} pairs; tools={args.tools}')

    from pantograph.server import Server
    project_path = Path('proof_generation_pilot/lean_project').resolve()
    print(f'\nstarting PyPantograph server at {project_path} ...')
    t0 = time.perf_counter()
    server = Server(project_path=str(project_path), imports=['PilotShim'])
    print(f'  ready in {time.perf_counter() - t0:.1f}s\n')

    proofs_rows = []
    trans_rows = []
    t_total = time.perf_counter()
    for i, row in sample.iterrows():
        pair_id = int(row['pair_id'])
        e1, e2 = int(row['eq1_id']), int(row['eq2_id'])
        for tool in args.tools:
            lean_src, status, transitions, wall_ms, err = TOOL_RUNNERS[tool](server, row)
            proofs_rows.append({
                'pair_id': pair_id,
                'eq1_id': e1, 'eq2_id': e2,
                'tool': tool,
                'proof_lean': lean_src or '',
                'status': status,
                'n_transitions': len(transitions),
                'wall_clock_ms': wall_ms,
                'validated': (status == 'proved'),
                'validation_log': err or '',
            })
            for step_idx, t in enumerate(transitions):
                trans_rows.append({
                    'pair_id': pair_id,
                    'eq1_id': e1, 'eq2_id': e2,
                    'tool': tool,
                    'step_idx': step_idx,
                    'state_before': t['state_before'],
                    'tactic': t['tactic'],
                    'state_after': t['state_after'] or '',
                    'is_terminal': t['is_terminal'],
                })
        if (i + 1) % 10 == 0 or (i + 1) == len(sample):
            elapsed = time.perf_counter() - t_total
            print(f'  [{i+1:>4}/{len(sample)}] proofs={len(proofs_rows):>4} '
                  f'transitions={len(trans_rows):>4} elapsed={elapsed:.1f}s')

    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(proofs_rows), proofs_path, compression='zstd')
    pq.write_table(pa.Table.from_pylist(trans_rows), trans_path, compression='zstd')
    print(f'\nwrote {proofs_path}  ({len(proofs_rows):,} rows)')
    print(f'wrote {trans_path}  ({len(trans_rows):,} rows)')

    # Per-tool summary
    print(f'\nper-tool status counts:')
    from collections import Counter
    by_tool_status = Counter((r['tool'], r['status']) for r in proofs_rows)
    for tool in args.tools:
        statuses = {s: c for (t, s), c in by_tool_status.items() if t == tool}
        n_proved = statuses.get('proved', 0)
        n_total = sum(statuses.values())
        print(f'  {tool:>8}: {n_proved}/{n_total} proved  '
              f'({100*n_proved/max(1,n_total):.1f}%)  '
              f'{statuses}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
