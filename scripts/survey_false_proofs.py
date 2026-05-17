"""Task 2 of `proof_corpus_survey_plan.md` — FALSE proof characterization.

Walks every `.lean` file under the ETP repo, finds every counterexample
theorem (declarations whose type is `∃ (G : Type) (_ : Magma G), Equation_X G ∧ ¬ Equation_Y G [∧ ...]`),
extracts the proof body, and classifies it by *surface template*.

The unit of analysis is **the source theorem**, not the registry
entry. The registry (`full_entries.json`) sometimes expands a single
"family theorem" into many entries (e.g., `LiftingMagmaFamily.establishNonimplication`
in `InvariantMetatheoremNonimplications/*` registers one source
theorem per (satisfied, refuted) pair, often with synthetic names
that don't appear in the source). For surveying the *shapes* of
proofs the FALSE generator must produce, the source theorem is the
right unit.

Templates we detect:

  - **angle_tuple**: `⟨carrier, ⟨op⟩, sat_proof, refuted_proof⟩` —
    the classical hand-written counterexample style.
    Subdivided by carrier into `Fin n`, `Nat`, `Bool`, `other`.
  - **lifting_family**: `LiftingMagmaFamily.establishNonimplication (...)` —
    auto-generated proofs from invariant-metatheorem lifting families.
  - **establish_***: any `<Module>.establishNonimplication (...)` or
    `<Module>.establish ...` style that's not a lifting family.
  - **direct_term**: term-mode but not matching either template above
    (a raw λ or constructor expression).
  - **tactic_block**: `by ...` body.
  - **irregular**: anything else.

Output:
  - `outputs/analyses/proof_survey/false_characterization.json` —
    aggregate counts (committed).
  - `data/processed/proof_corpus/facts_corpus.parquet` — per-theorem
    structured records (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


# Counterexample theorem shape — the existential statement after `:`.
# We accept any number of `¬ Equation_Y G` conjuncts. The Lean Unicode
# `∃`, `¬` and the ASCII `Exists`, `Not` forms are both possible (we
# only handle the Unicode form for now; check if ASCII form appears).
COUNTEREX_THEOREM_RE = re.compile(
    r'(?:^|\n)theorem\s+(\w+)\s*'         # theorem name
    r':\s*∃\s*\([^)]*Type[^)]*\)\s*'      # ∃ (G : Type) (or similar)
    r'\([^)]*Magma[^)]*\)\s*,\s*'         # (_ : Magma G), (or similar)
    r'Equation\d+\s*G\s*'                  # Equation_X G
    r'(?:∧\s*¬\s*Equation\d+\s*G\s*)+'     # ∧ ¬ Equation_Y G (one or more)
    r':=\s*([\s\S]*?)'                     # := body
    r'(?=\n(?:@\[|theorem |lemma |def |instance |end |namespace )|\Z)',
    re.MULTILINE,
)

# Less strict pattern: name matches *_not_implies_* (used to count theorems
# we should have but didn't, as a sanity check).
NOT_IMPLIES_NAME_RE = re.compile(r'(?:^|\n)theorem\s+(\w+_not_implies_\w+)\s')


def classify_template(body: str) -> tuple[str, dict]:
    """Return (template_name, extra_fields) for a FALSE-proof body."""
    b = body.strip()
    extra: dict = {}

    # tactic block
    if b.startswith('by ') or b.startswith('by\n') or b == 'by':
        extra['first_tactic'] = _first_tactic(b)
        return 'tactic_block', extra

    # angle-tuple
    if b.startswith('⟨'):
        parts = _split_angle_top(b)
        if parts is not None and len(parts) >= 2:
            carrier = parts[0].strip()
            extra['carrier_text'] = carrier
            extra['n_tuple_parts'] = len(parts)
            # Classify carrier
            m = re.match(r'^Fin\s+(\d+)$', carrier)
            if m:
                extra['carrier_kind'] = 'Fin'
                extra['carrier_n'] = int(m.group(1))
                return 'angle_tuple_Fin', extra
            if carrier in ('ℕ', 'Nat'):
                extra['carrier_kind'] = 'Nat'
                return 'angle_tuple_Nat', extra
            if carrier in ('ℤ', 'Int'):
                extra['carrier_kind'] = 'Int'
                return 'angle_tuple_Int', extra
            if carrier == 'Bool':
                extra['carrier_kind'] = 'Bool'
                return 'angle_tuple_Bool', extra
            extra['carrier_kind'] = 'other'
            return 'angle_tuple_other', extra
        return 'angle_tuple_malformed', extra

    # Lifting / establish-style
    m = re.match(r'(\w+(?:\.\w+)*)\s*\.\s*(establishNonimplication|establish_nonimplication|establish_non_implication)\b', b)
    if m or 'establishNonimplication' in b[:120]:
        # capture the module path before `.establishNonimplication`
        m2 = re.match(r'(\w+(?:\.\w+)*)\.establishNonimplication\b', b)
        extra['module'] = m2.group(1) if m2 else None
        if extra.get('module', '').startswith('LiftingMagmaFamily'):
            return 'lifting_family', extra
        return 'establish_other', extra

    # Generic term mode
    return 'direct_term', extra


def _first_tactic(b: str) -> str:
    m = re.match(r'by\s+(\w+)', b)
    return m.group(1) if m else '?'


def _split_angle_top(b: str) -> list[str] | None:
    """Split `⟨a, b, c⟩` into [a, b, c] respecting nested ⟨⟩ and (...)."""
    if not b.startswith('⟨'):
        return None
    depth_angle = 0
    depth_paren = 0
    parts: list[str] = []
    cur: list[str] = []
    started = False
    for i, ch in enumerate(b):
        if ch == '⟨':
            depth_angle += 1
            if depth_angle == 1:
                started = True
                continue
            cur.append(ch)
        elif ch == '⟩':
            depth_angle -= 1
            if depth_angle == 0:
                if cur:
                    parts.append(''.join(cur).strip())
                return parts
            cur.append(ch)
        elif ch == '(':
            depth_paren += 1
            cur.append(ch)
        elif ch == ')':
            depth_paren -= 1
            cur.append(ch)
        elif ch == ',' and depth_angle == 1 and depth_paren == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            if started:
                cur.append(ch)
    return None


def survey_files(lean_root: str) -> tuple[dict, pd.DataFrame]:
    print(f'walking {lean_root}')
    files = []
    for root, _dirs, fs in os.walk(lean_root):
        for f in fs:
            if f.endswith('.lean'):
                files.append(os.path.join(root, f))
    print(f'  {len(files):,} Lean files')

    records: list[dict] = []
    n_unfound = 0
    n_unparsed_names: dict[str, list[str]] = {}
    for path in files:
        try:
            text = open(path, errors='ignore').read()
        except OSError:
            continue
        # Counterexample theorems with proper type signature
        cx_found = list(COUNTEREX_THEOREM_RE.finditer(text))
        # Sanity: how many _not_implies_ names in the file vs how many we matched
        name_only_found = list(NOT_IMPLIES_NAME_RE.finditer(text))
        if len(name_only_found) > len(cx_found):
            # The pattern missed some — store names for diagnostics
            cx_names = {m.group(1) for m in cx_found}
            missed = [m.group(1) for m in name_only_found if m.group(1) not in cx_names]
            if missed:
                n_unparsed_names.setdefault(path.split('equational_theories/')[-1], []).extend(missed[:5])
                n_unfound += len(missed)
        for m in cx_found:
            name = m.group(1)
            body = m.group(2).strip()
            template, extra = classify_template(body)
            records.append({
                'name': name,
                'file': path.split('/equational_theories/equational_theories/')[-1] if '/equational_theories/equational_theories/' in path else path,
                'body_len_chars': len(body),
                'body_first_120': body[:120].replace('\n', ' '),
                'template': template,
                **extra,
            })

    df = pd.DataFrame(records)
    print(f'  extracted {len(df):,} counterexample theorems from source')
    print(f'  (additional {n_unfound:,} theorems with _not_implies_ names not captured by the type-signature pattern)')

    # Aggregate
    template_dist = df['template'].value_counts().to_dict()
    carrier_dist = (
        df.loc[df['template'].str.startswith('angle_tuple_'), 'carrier_kind']
          .value_counts()
          .to_dict()
        if 'carrier_kind' in df.columns else {}
    )
    fin_size_dist = (
        df.loc[df['template'] == 'angle_tuple_Fin', 'carrier_n']
          .value_counts()
          .sort_index()
          .to_dict()
        if 'carrier_n' in df.columns else {}
    )
    fin_size_dist = {int(k): int(v) for k, v in fin_size_dist.items()}

    # Top modules for establish_*
    if 'module' in df.columns:
        top_modules = (
            df.loc[df['template'].str.startswith(('lifting_family', 'establish_other')), 'module']
              .value_counts()
              .head(15)
              .to_dict()
        )
    else:
        top_modules = {}

    # Top first tactics for tactic_block
    if 'first_tactic' in df.columns:
        top_tactics = (
            df.loc[df['template'] == 'tactic_block', 'first_tactic']
              .value_counts()
              .head(15)
              .to_dict()
        )
    else:
        top_tactics = {}

    # Per-file top
    per_file_top = (
        df.groupby('file').size().sort_values(ascending=False).head(15).to_dict()
    )

    summary = {
        'lean_root': lean_root,
        'n_counterex_theorems_extracted': int(len(df)),
        'n_not_implies_names_not_captured': int(n_unfound),
        'template_distribution': template_dist,
        'angle_tuple_carrier_distribution': carrier_dist,
        'angle_tuple_Fin_size_distribution': fin_size_dist,
        'top_establish_modules': top_modules,
        'top_first_tactics_for_tactic_blocks': top_tactics,
        'top_files_by_counterex_count': per_file_top,
    }
    return summary, df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--etp-root', default='/home/gavin/git/equational_theories/equational_theories')
    parser.add_argument('--output-summary',
                        default='outputs/analyses/proof_survey/false_characterization.json')
    parser.add_argument('--output-corpus',
                        default='data/processed/proof_corpus/facts_corpus.parquet')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    summary_path = _abs(args.output_summary)
    corpus_path = _abs(args.output_corpus)
    for p in [summary_path, corpus_path]:
        if os.path.exists(p) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    summary, df = survey_files(args.etp_root)

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'\nwrote summary: {summary_path}')

    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    df.to_parquet(corpus_path, compression='zstd')
    print(f'wrote per-theorem corpus: {corpus_path} ({len(df):,} rows)')

    print()
    print('=== FALSE-proof characterization ===')
    print(f'  Counterexample theorems extracted: {summary["n_counterex_theorems_extracted"]:,}')
    print(f'  _not_implies_ names not captured by type pattern: {summary["n_not_implies_names_not_captured"]:,}')
    print()
    print('  Template distribution:')
    total = summary['n_counterex_theorems_extracted']
    for k, v in sorted(summary['template_distribution'].items(), key=lambda kv: -kv[1]):
        print(f'    {k:<28} {v:>6,}  ({100*v/max(total,1):5.1f}%)')
    if summary['angle_tuple_Fin_size_distribution']:
        print()
        print('  angle_tuple Fin-size distribution:')
        for k in sorted(summary['angle_tuple_Fin_size_distribution']):
            v = summary['angle_tuple_Fin_size_distribution'][k]
            print(f'    Fin {k:>3}                       {v:>6,}')
    if summary['top_establish_modules']:
        print()
        print('  Top establish_* modules:')
        for k, v in summary['top_establish_modules'].items():
            print(f'    {k:<32} {v:>6,}')
    if summary['top_files_by_counterex_count']:
        print()
        print('  Top files (by counterex theorem count):')
        for k, v in summary['top_files_by_counterex_count'].items():
            print(f'    {k:<60} {v:>6,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
