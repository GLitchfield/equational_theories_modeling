"""Phase 0 of `completion_exploration_plan.md` — variable-inclusion
pre-check on the 4,694 ETP equations.

For each equation `lhs = rhs`, count how many of the two directions
satisfy `Vars(rhs') ⊆ Vars(lhs')` — the variable-inclusion condition
for a valid rewrite rule. Classifies each as:

  both         : Vars(lhs) == Vars(rhs)  — both orientations valid
  lhs_to_rhs   : Vars(rhs) ⊊ Vars(lhs)   — only lhs → rhs valid (rule drops vars)
  rhs_to_lhs   : Vars(lhs) ⊊ Vars(rhs)   — only rhs → lhs valid
  incomparable : neither subset holds    — zero directions valid
                  (sub-case: disjoint var sets)

Zero-direction equations are structurally outside ALL rewriting-based
methods, including ordered completion. They put a ceiling on what any
completion-based decision procedure can decide.

Cross-method check: do the zero-direction equations overlap with the
"always-wrong-trained" cluster from outputs/analyses/failure_overlap/?
If yes, the same equations resist contrastive-embedding training AND
rewriting-based decision procedures — a hard cluster across methods.

Writes:
  outputs/analyses/completion_study/phase0_variable_inclusion.json
  et_studies/completion_exploration/data/equation_orientability.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.parser import Op, Term, Var, parse_equation  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def collect_vars(t: Term) -> frozenset[str]:
    if isinstance(t, Var):
        return frozenset({t.name})
    if isinstance(t, Op):
        return collect_vars(t.left) | collect_vars(t.right)
    raise TypeError(t)


def classify(vlhs: frozenset[str], vrhs: frozenset[str]) -> str:
    if vlhs == vrhs:
        return 'both'
    if vrhs < vlhs:
        return 'lhs_to_rhs'
    if vlhs < vrhs:
        return 'rhs_to_lhs'
    return 'incomparable'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--always-wrong',
                    default='outputs/analyses/failure_overlap/always_wrong_trained.parquet')
    ap.add_argument('--output-json',
                    default='outputs/analyses/completion_study/phase0_variable_inclusion.json')
    ap.add_argument('--output-parquet',
                    default='et_studies/completion_exploration/data/equation_orientability.parquet')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    out_json = _abs(args.output_json)
    out_parquet = _abs(args.output_parquet)
    for p in [out_json, out_parquet]:
        if os.path.exists(p) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    print(f'loading equations from {args.equations}')
    with open(_abs(args.equations)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    eqs = [parse_equation(l) for l in lines]
    print(f'  {len(eqs):,} equations parsed')

    rows = []
    for i, eq in enumerate(eqs, start=1):
        vlhs = collect_vars(eq.lhs)
        vrhs = collect_vars(eq.rhs)
        cls = classify(vlhs, vrhs)
        # Sub-classify "incomparable" as "disjoint" if no shared vars.
        sub = ''
        if cls == 'incomparable':
            sub = 'disjoint' if not (vlhs & vrhs) else 'overlapping'
        rows.append({
            'eq_id': i,
            'lhs_n_vars': len(vlhs),
            'rhs_n_vars': len(vrhs),
            'shared_n_vars': len(vlhs & vrhs),
            'classification': cls,
            'sub_classification': sub,
            'lhs_is_var': isinstance(eq.lhs, Var),
            'rhs_is_var': isinstance(eq.rhs, Var),
        })

    counts = Counter(r['classification'] for r in rows)
    sub_counts = Counter((r['classification'], r['sub_classification']) for r in rows
                         if r['classification'] == 'incomparable')

    print()
    print('=== Variable-inclusion classification (n=4,694) ===')
    for k in ['both', 'lhs_to_rhs', 'rhs_to_lhs', 'incomparable']:
        c = counts[k]
        print(f'  {k:>14}: {c:>5,}  ({100*c/len(eqs):5.2f}%)')
    print()
    print('  Incomparable sub-classification:')
    for (cl, sub), c in sub_counts.most_common():
        print(f'    {sub:>14}: {c:>5,}')

    # Family_c (LHS is a product) cross-tab.
    n_family_c = sum(1 for r in rows if not r['lhs_is_var'])
    print()
    print(f'=== Family classification ===')
    print(f'  family_c (LHS is Op, per parser docstring): {n_family_c:,} ({100*n_family_c/len(eqs):.2f}%)')
    print(f'  family_var (LHS is Var):                    {len(eqs)-n_family_c:,}')
    fam_c_vs_class = Counter()
    for r in rows:
        fam = 'family_c' if not r['lhs_is_var'] else 'family_var'
        fam_c_vs_class[(fam, r['classification'])] += 1
    print('  family × classification cross-tab:')
    for k, v in sorted(fam_c_vs_class.items()):
        print(f'    {k}: {v}')

    # Cross-method check: failure_overlap always-wrong-trained.
    print()
    print('=== Cross-method check: always-wrong-trained pairs ===')
    if os.path.exists(_abs(args.always_wrong)):
        aw = pq.read_table(_abs(args.always_wrong)).to_pandas()
        print(f'  {len(aw):,} always-wrong-trained pairs')
        cls_by_eq = {r['eq_id']: r['classification'] for r in rows}
        # For each role (eq1 = antecedent, eq2 = consequent), count classifications.
        aw_eq1_cls = Counter(cls_by_eq.get(int(e), 'unknown') for e in aw['eq1_id'])
        aw_eq2_cls = Counter(cls_by_eq.get(int(e), 'unknown') for e in aw['eq2_id'])
        # baseline: each equation appears in ~(N-1) pairs as either role
        baseline = {k: v/len(eqs) for k, v in counts.items()}
        print(f'  baseline classification fractions: '
              f"both={baseline['both']:.3f} "
              f"lhs_to_rhs={baseline['lhs_to_rhs']:.3f} "
              f"rhs_to_lhs={baseline['rhs_to_lhs']:.3f} "
              f"incomparable={baseline['incomparable']:.3f}")
        for role, ctr in [('eq1 (antecedent)', aw_eq1_cls), ('eq2 (consequent)', aw_eq2_cls)]:
            print(f'  role={role:>20}:')
            tot = sum(ctr.values())
            for k in ['both', 'lhs_to_rhs', 'rhs_to_lhs', 'incomparable']:
                c = ctr[k]
                pct = 100*c/tot if tot else 0
                base_pct = 100*baseline[k]
                delta = pct - base_pct
                marker = ' ↑' if delta > 5 else ' ↓' if delta < -5 else '  '
                print(f'    {k:>14}: {c:>6,} ({pct:5.2f}% vs {base_pct:5.2f}% baseline){marker}')

        # Unique eq2 (consequent) classifications: of the unique equations that
        # appear as the consequent in always-wrong-trained, what's their classification distribution?
        unique_eq2 = set(int(e) for e in aw['eq2_id'])
        unique_eq2_cls = Counter(cls_by_eq.get(e, 'unknown') for e in unique_eq2)
        print(f'  Unique consequents (eq2): {len(unique_eq2):,}')
        for k in ['both', 'lhs_to_rhs', 'rhs_to_lhs', 'incomparable']:
            c = unique_eq2_cls[k]
            print(f'    {k:>14}: {c:>4} ({100*c/len(unique_eq2):5.2f}%)')

        # Same for unique eq1 (antecedents).
        unique_eq1 = set(int(e) for e in aw['eq1_id'])
        unique_eq1_cls = Counter(cls_by_eq.get(e, 'unknown') for e in unique_eq1)
        print(f'  Unique antecedents (eq1): {len(unique_eq1):,}')
        for k in ['both', 'lhs_to_rhs', 'rhs_to_lhs', 'incomparable']:
            c = unique_eq1_cls[k]
            print(f'    {k:>14}: {c:>4} ({100*c/len(unique_eq1):5.2f}%)')
    else:
        print(f'  (skipped: {args.always_wrong} not found)')
        aw_eq1_cls = aw_eq2_cls = unique_eq1_cls = unique_eq2_cls = None

    # --- write outputs ---
    summary = {
        'config': {
            'n_equations': len(eqs),
            'equations_path': args.equations,
        },
        'classification_counts': dict(counts),
        'classification_fractions': {k: v/len(eqs) for k, v in counts.items()},
        'incomparable_sub_counts': {sub: c for (_, sub), c in sub_counts.items()},
        'family_classification_counts': {
            'family_c': n_family_c, 'family_var': len(eqs) - n_family_c
        },
        'family_vs_classification': {f'{k[0]}::{k[1]}': v
                                     for k, v in fam_c_vs_class.items()},
    }
    if aw_eq1_cls is not None:
        summary['always_wrong_trained_cross_method'] = {
            'n_pairs': int(len(aw)),
            'eq1_role_classification_counts': dict(aw_eq1_cls),
            'eq2_role_classification_counts': dict(aw_eq2_cls),
            'unique_eq1_classification_counts': dict(unique_eq1_cls),
            'unique_eq2_classification_counts': dict(unique_eq2_cls),
            'n_unique_eq1': len(set(int(e) for e in aw['eq1_id'])),
            'n_unique_eq2': len(set(int(e) for e in aw['eq2_id'])),
        }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'\nwrote {out_json}')

    table = pa.Table.from_pylist(rows)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    pq.write_table(table, out_parquet, compression='zstd')
    print(f'wrote {out_parquet}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
