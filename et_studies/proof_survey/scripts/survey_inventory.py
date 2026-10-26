"""Task 1 of `proof_corpus_survey_plan.md` — full-corpus proof inventory.

Reads the ETP `full_entries.json` registry plus the actual Lean
source files to produce a structured inventory of:

  - Implication-proof counts (total, by source file, by syntactic mode)
  - Facts-entry counts and the total FALSE-pair coverage they provide
  - Unconditional entries (just `Equation1` reflexive)
  - Magma-size distribution from the facts entries
  - Corpus size: number of .lean files, total LOC under
    equational_theories/
  - Lean toolchain version

Output: ``et_studies/proof_survey/outputs/inventory.json`` plus a
human-readable summary printed to stdout.

This is the formal counting pass behind `orientation_notes.md`.
Run after `et_studies/proof_survey/scripts/survey_inventory.py` updates if the ETP repo
changes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


# Match `theorem NAME ... := BODY`. The signature spans multiple
# lines and contains `:` (in type ascriptions); we just look for the
# first `:=` after `theorem NAME`. The body runs until the next
# top-level Lean declaration start. Heuristic regex — full Lean
# parsing would need a real parser — but adequate for the regular
# ETP corpus.
THEOREM_BODY_RE = re.compile(
    r'\btheorem\s+(\w+)[\s\S]*?:=\s*([\s\S]*?)'
    r'(?=\n(?:@\[|theorem |lemma |def |instance |end |namespace |\Z))',
    re.MULTILINE,
)


def detect_mode(body: str) -> str:
    """Return 'tactic' if the proof body opens a `by` block, else 'term'."""
    # Strip leading whitespace and comments.
    b = body.lstrip()
    # Remove leading line comments
    while b.startswith('--'):
        nl = b.find('\n')
        if nl == -1:
            return 'term'
        b = b[nl + 1:].lstrip()
    # Lean tactic-mode bodies start with `by` (possibly followed by `;`, newline, or whitespace).
    if b.startswith('by ') or b.startswith('by\n') or b == 'by' or b.startswith('by{') or b.startswith('by '):
        return 'tactic'
    return 'term'


def parse_proof_bodies(source_text: str) -> dict[str, str]:
    """Return a dict mapping theorem name -> proof body for a Lean file.

    Heuristic regex-based parsing — adequate for the regular ETP files
    where each `theorem` block is well-formed and on its own logical
    statement, but may miss exotic cases (nested theorems, attributes
    spanning multi-line, etc.). Reports mismatches via the inventory's
    counts vs full_entries.json.
    """
    out: dict[str, str] = {}
    for m in THEOREM_BODY_RE.finditer(source_text):
        name = m.group(1)
        body = m.group(2)
        out[name] = body
    return out


def inventory(etp_root: str, registry_path: str) -> dict:
    print(f'reading registry: {registry_path}')
    with open(registry_path) as f:
        registry = json.load(f)
    print(f'  {len(registry):,} total entries')

    # --- Aggregate by variant -------------------------------------------
    by_variant = Counter()
    by_proven = Counter()
    impl_by_file = Counter()
    facts_by_file = Counter()
    n_false_covered = 0
    magma_size_dist: Counter[int | str] = Counter()

    impl_entries: list[dict] = []  # for source-file inspection
    for e in registry:
        v = list(e['variant'].keys())[0]
        by_variant[v] += 1
        by_proven[e.get('proven', None)] += 1
        if v == 'implication':
            impl_entries.append(e)
            impl_by_file[e['filename']] += 1
        elif v == 'facts':
            facts_by_file[e['filename']] += 1
            n_false_covered += len(e['variant']['facts']['refuted'])
            # Magma size — derive from the source if we can; for now just
            # log presence. Many facts entries don't directly state the
            # magma size in JSON. We'll derive it during proof extraction
            # in Task 2; here we record the field metadata.
            finite_flag = e['variant']['facts'].get('finite', None)
            magma_size_dist[f'finite={finite_flag}'] += 1

    print(f'  by_variant: {dict(by_variant)}')
    print(f'  by_proven : {dict(by_proven)}')
    print(f'  implications: {by_variant["implication"]:,}')
    print(f'  facts entries: {by_variant["facts"]:,} (FALSE pair coverage: {n_false_covered:,})')

    # --- Walk the Lean source to classify mode (term vs tactic) ---------
    print(f'walking Lean source tree: {etp_root}')
    lean_files = []
    total_lines = 0
    for root, _dirs, files in os.walk(etp_root):
        for f in files:
            if f.endswith('.lean'):
                path = os.path.join(root, f)
                lean_files.append(path)
                with open(path, errors='ignore') as fh:
                    total_lines += sum(1 for _ in fh)
    print(f'  {len(lean_files):,} .lean files, {total_lines:,} lines total')

    # Parse each Lean file once for proof bodies of named theorems
    print('parsing proof bodies ...')
    name_to_body: dict[str, str] = {}
    for path in lean_files:
        try:
            text = open(path, errors='ignore').read()
        except OSError:
            continue
        bodies = parse_proof_bodies(text)
        name_to_body.update(bodies)
    print(f'  recovered bodies for {len(name_to_body):,} named theorems')

    # Classify implication proofs by syntactic mode
    mode_counts = Counter()
    unrecovered = 0
    mode_by_file: dict[str, Counter] = {}
    for e in impl_entries:
        bare_name = e['name'].split('.')[-1]   # e.g. 'Equation2_implies_Equation3' from 'Subgraph.Equation2_implies_Equation3'
        body = name_to_body.get(bare_name)
        if body is None:
            unrecovered += 1
            continue
        m = detect_mode(body)
        mode_counts[m] += 1
        mode_by_file.setdefault(e['filename'], Counter())[m] += 1
    print(f'  mode for implication proofs: {dict(mode_counts)}, unrecovered={unrecovered}')

    # --- Lean toolchain version -----------------------------------------
    toolchain_path = os.path.join(etp_root, 'lean-toolchain')
    if not os.path.exists(toolchain_path):
        # ETP layout: lean-toolchain at repo root, not in equational_theories/
        toolchain_path = os.path.join(os.path.dirname(etp_root.rstrip('/')), 'lean-toolchain')
    lean_version = None
    if os.path.exists(toolchain_path):
        lean_version = open(toolchain_path).read().strip()
    print(f'  lean toolchain: {lean_version}')

    # --- Assemble the inventory JSON ------------------------------------
    inv = {
        'etp_root': etp_root,
        'registry_path': registry_path,
        'lean_toolchain': lean_version,
        'totals': {
            'registry_entries': len(registry),
            'implications': by_variant['implication'],
            'facts_entries': by_variant['facts'],
            'unconditional': by_variant['unconditional'],
            'proven': by_proven.get(True, 0),
            'unproven': by_proven.get(False, 0),
            'false_pair_coverage_via_facts': n_false_covered,
            'lean_files': len(lean_files),
            'lean_total_lines': total_lines,
        },
        'implication_proofs_by_file': dict(
            sorted(impl_by_file.items(), key=lambda kv: -kv[1])
        ),
        'facts_entries_by_file': dict(
            sorted(facts_by_file.items(), key=lambda kv: -kv[1])
        ),
        'implication_proof_mode': {
            'term_mode': mode_counts['term'],
            'tactic_mode': mode_counts['tactic'],
            'unrecovered': unrecovered,
        },
        'implication_proof_mode_by_file_top20': [
            {
                'file': fn,
                'term': mode_by_file[fn].get('term', 0),
                'tactic': mode_by_file[fn].get('tactic', 0),
                'total': mode_by_file[fn].get('term', 0) + mode_by_file[fn].get('tactic', 0),
            }
            for fn in sorted(
                mode_by_file.keys(),
                key=lambda f: -(mode_by_file[f].get('term', 0) + mode_by_file[f].get('tactic', 0)),
            )[:20]
        ],
        'facts_finite_flag_dist': dict(magma_size_dist),
    }
    return inv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--etp-root', default='/home/gavin/git/equational_theories/equational_theories',
                        help='path to the ETP equational_theories/ Lean source directory')
    parser.add_argument('--registry', default='/home/gavin/git/equational_theories/full_entries.json',
                        help='path to full_entries.json')
    parser.add_argument('--output', default='et_studies/proof_survey/outputs/inventory.json')
    parser.add_argument('--force', action='store_true', help='overwrite existing inventory')
    args = parser.parse_args(argv)

    out_path = _abs(args.output)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force.', file=sys.stderr)
        return 2

    inv = inventory(args.etp_root, args.registry)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(inv, f, indent=2, sort_keys=False)
    print(f'\nwrote {out_path}')

    print()
    print('=== Summary ===')
    t = inv['totals']
    print(f'  Lean toolchain        : {inv["lean_toolchain"]}')
    print(f'  Registry entries      : {t["registry_entries"]:,}')
    print(f'    implications        : {t["implications"]:,}')
    print(f'    facts entries       : {t["facts_entries"]:,}')
    print(f'    unconditional       : {t["unconditional"]:,}')
    print(f'  Proven / unproven     : {t["proven"]:,} / {t["unproven"]:,}')
    print(f'  FALSE pairs (via facts): {t["false_pair_coverage_via_facts"]:,}')
    print(f'  Lean files            : {t["lean_files"]:,}')
    print(f'  Lean total lines      : {t["lean_total_lines"]:,}')
    print(f'  Implication proof mode:')
    m = inv['implication_proof_mode']
    total = m['term_mode'] + m['tactic_mode']
    if total > 0:
        print(f'    term mode           : {m["term_mode"]:>6,}  ({100*m["term_mode"]/total:5.1f}%)')
        print(f'    tactic mode         : {m["tactic_mode"]:>6,}  ({100*m["tactic_mode"]/total:5.1f}%)')
        print(f'    unrecovered (regex miss): {m["unrecovered"]:>6,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
