"""Task 4 of `proof_corpus_survey_plan.md` — proof clustering.

Apply normalization transformations to every TRUE proof body in
order, then exact-match cluster the normalized strings. Report
cluster counts at several normalization levels so any wild swings
(>2× as cluster spec changes) are visible.

Normalization pipeline (Methodology Resolution 6), applied
cumulatively in order:

  N1. Whitespace collapsed (multiple spaces/newlines → single space,
      trim leading/trailing).
  N2. + Object-level variable names → `v1, v2, ...` (in order of
      first appearance). Pattern: `\\b[a-z]\\b` single-letter
      identifiers (the corpus uses x, y, z, w, u, v).
  N3. + Equation references (`Equation\\d+` and `Eq\\d+`) → `E1, E2, ...`
      in order of first appearance.
  N4. + Local hypothesis names (`h\\d*`, `eq\\d+`, `nh`, `sK\\d+`,
      `X\\d+`) → `h1, h2, ...` in order of first appearance.
  N5. + Implicit-argument numbers replaced with `?` (rare in this
      corpus but documented for completeness).

Inputs:
  - data/processed/proof_corpus/true_corpus.parquet (10,676 proofs)
  - (re-extract bodies from .lean files via the same regex used by
    survey_true_proofs.py, since the parquet stores only metadata)

Output:
  - outputs/analyses/proof_survey/clustering_results.json
  - data/processed/proof_corpus/normalized_proofs.parquet (gitignored,
    one row per proof with the N1..N5 normalized strings).
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


IMPL_THEOREM_RE = re.compile(
    r'(?:^|\n)theorem\s+Equation(\d+)_implies_Equation(\d+)\s*'
    r'\([^)]*\)\s*\[[^]]*\]\s*\([^)]*\)\s*:\s*Equation\d+\s*G\s*'
    r':=\s*([\s\S]*?)'
    r'(?=\n(?:@\[|theorem |lemma |def |instance |end |namespace )|\Z)',
    re.MULTILINE,
)


def extract_all_bodies(lean_root: str) -> list[dict]:
    """Yield {e1, e2, file, body} for every implication theorem."""
    out: list[dict] = []
    for r, _d, fs in os.walk(lean_root):
        for f in fs:
            if not f.endswith('.lean'):
                continue
            path = os.path.join(r, f)
            try:
                text = open(path, errors='ignore').read()
            except OSError:
                continue
            for m in IMPL_THEOREM_RE.finditer(text):
                out.append({
                    'e1': int(m.group(1)),
                    'e2': int(m.group(2)),
                    'file': path.split('/equational_theories/equational_theories/')[-1]
                            if '/equational_theories/equational_theories/' in path else path,
                    'body': m.group(3).strip(),
                })
    return out


def normalize_n1(body: str) -> str:
    """Whitespace collapsed."""
    # Strip trailing /-- ... -/ block comments first
    b = re.sub(r'/-[\s\S]*?-/\s*$', '', body).strip()
    b = re.sub(r'--[^\n]*$', '', b).strip()
    b = re.sub(r'\s+', ' ', b)
    return b


# Single-letter lowercase identifier that's a "variable name" in this corpus.
# Lean's parse-tree variables are x, y, z, w, u, v. We map by first appearance.
SINGLE_LETTER_RE = re.compile(r'\b([uvwxyz])\b')


def normalize_n2(s: str) -> str:
    """+ object-level variable names → v1, v2, ..."""
    seen: dict[str, str] = {}
    counter = [0]
    def replace(m: re.Match) -> str:
        name = m.group(1)
        if name not in seen:
            counter[0] += 1
            seen[name] = f'v{counter[0]}'
        return seen[name]
    return SINGLE_LETTER_RE.sub(replace, s)


EQ_REF_RE = re.compile(r'\b(?:Equation|Eq|Law)(\d+)\b')


def normalize_n3(s: str) -> str:
    """+ equation references → E1, E2, ..."""
    seen: dict[str, str] = {}
    counter = [0]
    def replace(m: re.Match) -> str:
        full = m.group(0)
        if full not in seen:
            counter[0] += 1
            seen[full] = f'E{counter[0]}'
        return seen[full]
    return EQ_REF_RE.sub(replace, s)


# Hypothesis-name patterns: `h`, `h0`-`h99`, `eq9`-`eq99`, `nh`, `sK0`-`sK99`, `X0`-`X99`
HYP_NAME_RES = [
    re.compile(r'\b(h\d*)\b'),
    re.compile(r'\b(eq\d+)\b'),
    re.compile(r'\b(nh)\b'),
    re.compile(r'\b(sK\d+)\b'),
    re.compile(r'\b(X\d+)\b'),
]


def normalize_n4(s: str) -> str:
    """+ local hypothesis names → h1, h2, ..."""
    seen: dict[str, str] = {}
    counter = [0]
    def make_replace():
        def replace(m: re.Match) -> str:
            name = m.group(1)
            if name not in seen:
                counter[0] += 1
                seen[name] = f'h{counter[0]}'
            return seen[name]
        return replace
    out = s
    for pat in HYP_NAME_RES:
        out = pat.sub(make_replace(), out)
    return out


def normalize_n5(s: str) -> str:
    """+ implicit-argument numbers → ?"""
    # Heuristic: integer literals that appear after `:=` or in arg positions
    # (but not inside Equation_N etc., which were replaced above).
    return re.sub(r'\b\d+\b', '?', s)


def cluster_and_report(strings: list[str], top_k_list: list[int] = (10, 20, 50, 100)) -> dict:
    ct = Counter(strings)
    n = len(strings)
    total_distinct = len(ct)
    sorted_counts = [c for _, c in ct.most_common()]
    coverage = {}
    cum = 0
    for k in top_k_list:
        cum_k = sum(sorted_counts[:k])
        coverage[f'top_{k}_coverage'] = cum_k / n if n else 0.0
    # Coverage at 50%, 80%, 95% thresholds: how many clusters?
    cum = 0
    thresholds = {0.5: None, 0.8: None, 0.95: None}
    for i, c in enumerate(sorted_counts, start=1):
        cum += c
        frac = cum / n
        for t in thresholds:
            if thresholds[t] is None and frac >= t:
                thresholds[t] = i
    return {
        'total_proofs': n,
        'distinct_clusters': total_distinct,
        'top_k_coverage': coverage,
        'clusters_to_cover_pct': {
            f'>={int(t*100)}pct': thresholds[t] for t in thresholds
        },
        'top_10_clusters': [(s[:120], c) for s, c in ct.most_common(10)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--etp-root', default='/home/gavin/git/equational_theories/equational_theories')
    parser.add_argument('--output-summary',
                        default='outputs/analyses/proof_survey/clustering_results.json')
    parser.add_argument('--output-corpus',
                        default='data/processed/proof_corpus/normalized_proofs.parquet')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    summary_path = _abs(args.output_summary)
    corpus_path = _abs(args.output_corpus)
    for p in [summary_path, corpus_path]:
        if os.path.exists(p) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    print(f'extracting bodies from {args.etp_root}')
    bodies = extract_all_bodies(args.etp_root)
    print(f'  {len(bodies):,} implication theorems')

    # Apply N1..N5 cumulatively
    print('normalizing under N1..N5 ...')
    records: list[dict] = []
    for b in bodies:
        n1 = normalize_n1(b['body'])
        n2 = normalize_n2(n1)
        n3 = normalize_n3(n2)
        n4 = normalize_n4(n3)
        n5 = normalize_n5(n4)
        records.append({
            'e1': b['e1'],
            'e2': b['e2'],
            'file': b['file'],
            'n1_whitespace': n1,
            'n2_plus_vars': n2,
            'n3_plus_eqrefs': n3,
            'n4_plus_hyps': n4,
            'n5_plus_implicits': n5,
        })

    df = pd.DataFrame(records)

    print('clustering under each normalization level ...')
    levels = ['n1_whitespace', 'n2_plus_vars', 'n3_plus_eqrefs', 'n4_plus_hyps', 'n5_plus_implicits']
    summary = {
        'total_proofs': len(df),
        'levels': {},
    }
    for lvl in levels:
        info = cluster_and_report(df[lvl].tolist())
        summary['levels'][lvl] = info
        cov = info['top_k_coverage']
        thresh = info['clusters_to_cover_pct']
        print(f'  {lvl:<22} clusters={info["distinct_clusters"]:>5,}'
              f'  top10={cov["top_10_coverage"]*100:5.1f}%'
              f'  top20={cov["top_20_coverage"]*100:5.1f}%'
              f'  top50={cov["top_50_coverage"]*100:5.1f}%'
              f'  >=80pct={thresh[">=80pct"]:>4}'
              f'  >=95pct={thresh[">=95pct"]:>4}')

    print(f'\ntop 10 clusters at {levels[-1]}:')
    for s, c in summary['levels'][levels[-1]]['top_10_clusters']:
        print(f'  {c:>5,}  {s}')

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'\nwrote {summary_path}')

    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    df.to_parquet(corpus_path, compression='zstd')
    print(f'wrote {corpus_path} ({len(df):,} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
