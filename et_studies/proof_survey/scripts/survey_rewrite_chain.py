"""Phases 3 + 4 of `rewrite_chain_study_plan.md` — tactic-structure and
rule-set analysis on the 1,636 rewrite-chain bodies.

Reads `et_studies/rewrite_chain_study/data/rewrite_chain_dataset.parquet`. Both
analyses parse the same raw bodies, so they share a single pass.

Task 1 (tactic-structure) reports:
  - chain-length distribution (total rewrite-step args across all
    rewrite tactics in a proof)
  - tactic-verb mix (rw / rwa / simp_rw / nth_rw / nth_rewrite /
    calc / conv counts and per-proof prevalence)
  - structural vs straight-line classification (uses calc / have /
    case-splits → structured; otherwise straight-line)
  - proof length in tokens (Lean-shaped lexer over the raw body)

Task 2 (rule-set) reports:
  - per-proof rule-set classification: {h-only-forward, h-only-reverse,
    h-bidirectional, has-other-name}
  - per-arg classification: {h, ← h, other_name}
  - the fraction of proofs that stay purely within {h, ← h} — the
    key tractability predictor (one of the two decision-driving
    numbers per Methodology Resolution 8)

Outputs (committed):
  et_studies/rewrite_chain_study/outputs/tactic_structure.json
  et_studies/rewrite_chain_study/outputs/ruleset_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


# Rewrite-style tactics that take a `[...]` arg list.
REWRITE_TACTICS = ('rw', 'rwa', 'simp_rw', 'nth_rw', 'nth_rewrite')

# Other tactics worth tracking (presence / count).
OTHER_TACTICS = ('calc', 'conv', 'have', 'show', 'simp', 'rfl',
                 'apply', 'exact', 'assumption', 'intro', 'intros',
                 'repeat', 'first', 'try', 'all_goals', 'any_goals')

# Pattern: <tactic_name> [ <args> ]
# Use a non-greedy match for the bracket contents; can be defeated by
# nested brackets, but those are rare in this corpus.
TACTIC_ARGS_RE = re.compile(
    r'\b(rw|rwa|simp_rw|nth_rw|nth_rewrite)\s*\[([^\[\]]*)\]'
)

# Standalone tactic-keyword presence detector (ignores word-internal matches).
TACTIC_KEYWORD_RE = re.compile(
    r'\b(' + '|'.join(REWRITE_TACTICS + OTHER_TACTICS) + r')\b'
)

# Lean-shaped token lexer for proof-length-in-tokens (matches the survey's
# tokenizer to the extent the regex is the same — Unicode identifiers
# as single tokens).
OP_TOKENS = sorted([
    '<;>', ':=', '=>', '↦', '▸', '..', '←', '→', '∀', '∃', '∧', '∨', '¬',
    '≠', '≤', '≥', '∈', '∉', '⊆', '∪', '∩', '⟨', '⟩', '◇',
], key=len, reverse=True)
PUNCT = list('()[]{}.,;|:')


def _build_lexer() -> re.Pattern:
    parts: list[str] = [re.escape(op) for op in OP_TOKENS]
    parts += [re.escape(p) for p in PUNCT]
    parts.append(r'[A-Za-z_α-ωΑ-Ω][A-Za-z0-9_α-ωΑ-Ωʹ]*')
    parts.append(r'\d+')
    parts.append(r'\S')
    return re.compile('|'.join(parts))


LEXER = _build_lexer()


def classify_arg(arg: str) -> str:
    """Classify a single rewrite-arg into {h_forward, h_reverse, other_name}.

    Lean accepts both `←` (Unicode U+2190) and `<-` (ASCII) for the
    reverse-direction marker; both forms occur in the ETP corpus. Strip
    either before identifier extraction.

    Examples:
      "h"           -> h_forward
      "← h"         -> h_reverse
      "<-h"         -> h_reverse
      "h x y"       -> h_forward (partial application; the rule is still h)
      "← h x y"     -> h_reverse
      "Foo.bar"     -> other_name
      "← Foo.bar"   -> other_name
    """
    s = arg.strip()
    reverse = False
    if s.startswith('←'):
        reverse = True
        s = s[1:].strip()
    elif s.startswith('<-'):
        reverse = True
        s = s[2:].strip()
    # Take the first identifier token as the rule name.
    m = re.match(r'([A-Za-z_α-ωΑ-Ω][A-Za-z0-9_α-ωΑ-Ωʹ.]*)', s)
    if not m:
        return 'other_name'
    name = m.group(1)
    if name == 'h':
        return 'h_reverse' if reverse else 'h_forward'
    return 'other_name'


def parse_proof(body: str) -> dict:
    """Parse one rewrite-chain proof body. Returns a dict with:
      - tactic_counts: how many times each tactic verb appears
      - rewrite_step_count: total rewrite-arg count (chain length)
      - arg_classes: list[str] (h_forward / h_reverse / other_name)
      - has_structural: bool (calc / have / conv / show etc.)
      - tokens: total Lean tokens in the body
    """
    rewrite_args: list[str] = []
    for m in TACTIC_ARGS_RE.finditer(body):
        inside = m.group(2)
        # Split on top-level commas. Args here don't typically contain
        # commas inside parens, but the regex above already excludes
        # nested brackets, so a simple split is safe.
        parts = [p.strip() for p in inside.split(',') if p.strip()]
        rewrite_args.extend(parts)

    arg_classes = [classify_arg(a) for a in rewrite_args]
    tactic_counts: Counter[str] = Counter()
    for m in TACTIC_KEYWORD_RE.finditer(body):
        tactic_counts[m.group(1)] += 1
    has_structural = any(k in tactic_counts
                         for k in ('calc', 'conv', 'have', 'show'))

    return {
        'tactic_counts': dict(tactic_counts),
        'rewrite_step_count': len(rewrite_args),
        'arg_classes': arg_classes,
        'has_structural': has_structural,
        'tokens': len(LEXER.findall(body)),
    }


def _percentiles(xs: list[int], ps=(50, 75, 90, 95, 99)) -> dict:
    import numpy as np
    arr = np.array(xs)
    out = {'min': int(arr.min()), 'max': int(arr.max()),
           'mean': float(arr.mean()), 'median': float(np.median(arr))}
    for p in ps:
        out[f'p{p}'] = float(np.percentile(arr, p))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset',
                        default='et_studies/rewrite_chain_study/data/rewrite_chain_dataset.parquet')
    parser.add_argument('--tactic-output',
                        default='et_studies/rewrite_chain_study/outputs/tactic_structure.json')
    parser.add_argument('--ruleset-output',
                        default='et_studies/rewrite_chain_study/outputs/ruleset_analysis.json')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    for p in [args.tactic_output, args.ruleset_output]:
        if os.path.exists(_abs(p)) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    print(f'loading {args.dataset}')
    df = pq.read_table(_abs(args.dataset)).to_pandas()
    print(f'  {len(df):,} proofs')

    print('parsing proofs ...')
    parsed = [parse_proof(b) for b in df['body_raw'].tolist()]

    # --- Task 1: tactic structure ---
    chain_lens = [p['rewrite_step_count'] for p in parsed]
    proof_tokens = [p['tokens'] for p in parsed]
    n_structural = sum(p['has_structural'] for p in parsed)

    # Tactic-verb mix: per-proof presence (does the proof use this verb at all)
    # and total invocations (how often it appears across the corpus).
    presence: Counter[str] = Counter()
    total_count: Counter[str] = Counter()
    for p in parsed:
        for verb, cnt in p['tactic_counts'].items():
            presence[verb] += 1   # one per proof using it
            total_count[verb] += cnt

    tactic_summary = {
        'n_proofs': int(len(df)),
        'chain_length_distribution': _percentiles(chain_lens),
        'proof_token_length_distribution': _percentiles(proof_tokens),
        'structural_proof_count': int(n_structural),
        'structural_proof_fraction': float(n_structural / len(df)),
        'tactic_presence_per_proof': dict(presence.most_common()),
        'tactic_total_invocations': dict(total_count.most_common()),
    }

    os.makedirs(os.path.dirname(_abs(args.tactic_output)), exist_ok=True)
    with open(_abs(args.tactic_output), 'w') as f:
        json.dump(tactic_summary, f, indent=2, sort_keys=False)
    print(f'wrote {args.tactic_output}')

    # --- Task 2: rule-set analysis ---
    # Per-arg classification, aggregated across the corpus.
    all_args: Counter[str] = Counter()
    for p in parsed:
        for cls in p['arg_classes']:
            all_args[cls] += 1

    # Per-proof classification: assign each proof to one of:
    #   no_rewrite_tactic, h_only_forward, h_only_reverse,
    #   h_bidirectional, has_other_name
    per_proof: Counter[str] = Counter()
    proofs_within_h_set = 0
    for p in parsed:
        classes = set(p['arg_classes'])
        if not classes:
            per_proof['no_rewrite_args'] += 1
            # A proof with no rewrite args might still be a "rewrite-chain"
            # in name (e.g., uses calc + rfl). Treat as within-{h, ← h}
            # for the tractability fraction since no outside rule is named.
            proofs_within_h_set += 1
        elif 'other_name' in classes:
            per_proof['has_other_name'] += 1
        elif classes == {'h_forward'}:
            per_proof['h_only_forward'] += 1
            proofs_within_h_set += 1
        elif classes == {'h_reverse'}:
            per_proof['h_only_reverse'] += 1
            proofs_within_h_set += 1
        elif classes == {'h_forward', 'h_reverse'}:
            per_proof['h_bidirectional'] += 1
            proofs_within_h_set += 1
        else:
            per_proof['other'] += 1

    # Collect outside-lemma names for the has_other_name proofs.
    outside_names: Counter[str] = Counter()
    for p, body in zip(parsed, df['body_raw'].tolist()):
        if any(c == 'other_name' for c in p['arg_classes']):
            for m in TACTIC_ARGS_RE.finditer(body):
                inside = m.group(2)
                for arg in inside.split(','):
                    s = arg.strip()
                    if s.startswith('←'):
                        s = s[1:].strip()
                    elif s.startswith('<-'):
                        s = s[2:].strip()
                    name_m = re.match(r'([A-Za-z_α-ωΑ-Ω][A-Za-z0-9_α-ωΑ-Ωʹ.]*)', s)
                    if name_m and name_m.group(1) != 'h':
                        outside_names[name_m.group(1)] += 1

    ruleset_summary = {
        'n_proofs': int(len(df)),
        'per_arg_classification': dict(all_args.most_common()),
        'per_proof_classification': dict(per_proof.most_common()),
        'within_h_set_count': int(proofs_within_h_set),
        'within_h_set_fraction': float(proofs_within_h_set / len(df)),
        'outside_names_distribution': dict(outside_names.most_common(50)),
    }

    with open(_abs(args.ruleset_output), 'w') as f:
        json.dump(ruleset_summary, f, indent=2, sort_keys=False)
    print(f'wrote {args.ruleset_output}')

    # --- Headline output ---
    print()
    print('=== Task 1: Tactic structure ===')
    cl = tactic_summary['chain_length_distribution']
    print(f"  Chain length: median {cl['median']:.0f}, mean {cl['mean']:.1f}, "
          f"p95 {cl['p95']:.0f}, max {cl['max']}")
    tl = tactic_summary['proof_token_length_distribution']
    print(f"  Proof tokens: median {tl['median']:.0f}, mean {tl['mean']:.1f}, "
          f"p95 {tl['p95']:.0f}, max {tl['max']}")
    print(f"  Structural proofs: {tactic_summary['structural_proof_count']:,} / {len(df):,} "
          f"= {100*tactic_summary['structural_proof_fraction']:.1f}%")
    print(f"  Top tactic verbs by presence:")
    for verb, count in list(tactic_summary['tactic_presence_per_proof'].items())[:8]:
        print(f'    {verb:<12} in {count:>5,} / {len(df):,} proofs '
              f'({100*count/len(df):.1f}%)')
    print()
    print('=== Task 2: Rule-set ===')
    print(f"  Per-arg distribution:")
    for cls, n in ruleset_summary['per_arg_classification'].items():
        print(f'    {cls:<14} {n:>6,}')
    print(f"  Per-proof distribution:")
    for cls, n in ruleset_summary['per_proof_classification'].items():
        print(f'    {cls:<22} {n:>5,} / {len(df):,} = {100*n/len(df):.1f}%')
    print(f"  Within-{{h, ← h}} fraction: {ruleset_summary['within_h_set_count']:,} / {len(df):,} = "
          f"{100*ruleset_summary['within_h_set_fraction']:.1f}%")
    if outside_names:
        print(f"  Top 10 outside lemma names:")
        for name, n in list(ruleset_summary['outside_names_distribution'].items())[:10]:
            print(f'    {name:<32} {n:>5,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
