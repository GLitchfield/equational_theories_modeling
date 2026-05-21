"""Phase 1.1 of `proof_decoder_plan.md` — phase-2 vocab pre-tokenization.

Per Methodology Resolution 5: tokenize matcher-solved proofs + the
rewrite-chain subset of the ETP corpus (excluding Vampire/Egg), bucket
the tokens into instruction / structural / content, and report counts.
Pins the phase-2 vocab number now so it doesn't slip.

This script doesn't change anything else — it's a one-off counting
pass that produces `outputs/analyses/proof_decoder/phase2_vocab.json`.

The rewrite-chain bodies come from the survey's `normalized_proofs.parquet`
(N4 normalization: variables → v1..v6, equation refs → E1..EN,
hypothesis names → h1..hN). That's the form the phase-2 decoder
would train on.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


# Lean-shaped lexer (multi-char Unicode identifiers as single tokens,
# per Methodology Resolution 9 in the proof-corpus survey plan).
OP_TOKENS = sorted([
    '<;>', ':=', '=>', '↦', '▸', '..', '←', '→', '∀', '∃', '∧', '∨', '¬',
    '≠', '≤', '≥', '∈', '∉', '⊆', '∪', '∩', '⟨', '⟩', '◇',
], key=len, reverse=True)
PUNCT = list('()[]{}.,;|:')

def _build_tokenizer() -> re.Pattern:
    parts: list[str] = [re.escape(op) for op in OP_TOKENS]
    parts += [re.escape(p) for p in PUNCT]
    parts.append(r'[A-Za-z_α-ωΑ-Ω][A-Za-z0-9_α-ωΑ-Ωʹ]*')
    parts.append(r'\d+')
    parts.append(r'\S')
    return re.compile('|'.join(parts))


TOKEN_RE = _build_tokenizer()


# Token-bucket classification.
INSTRUCTION_TOKENS = {
    # Tactic verbs the rewrite-chain category typically uses (per
    # survey Tasks 3+4): the closed set is small. Anything in this
    # set is a tactic verb the decoder must learn to emit.
    'rw', 'rewrite', 'rwa', 'simp_rw', 'nth_rw', 'nth_rewrite',
    'calc', 'conv',
    'intro', 'intros', 'exact', 'apply', 'refine',
    'have', 'show', 'suffices', 'let',
    'simp', 'decide', 'aesop', 'grind', 'rfl', 'trivial',
    'by', 'fun',
}

STRUCTURAL_TOKENS = {
    '[', ']', '(', ')', '{', '}', ',', ':', ';', '|',
    ':=', '=>', '↦', '←', '→', '..', '▸', '◇',
    '_', '.', '=',
    '∀', '∃', '∧', '∨', '¬', '⟨', '⟩',
    '<;>',
}


def classify_token(tok: str) -> str:
    if tok in INSTRUCTION_TOKENS:
        return 'instruction'
    if tok in STRUCTURAL_TOKENS:
        return 'structural'
    # Everything else is content (variables, hypotheses, equation refs,
    # numeric literals, qualified identifiers like `h.symm` token-pieces).
    return 'content'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--matcher-proofs',
                        default='et_studies/matcher_deliverable/data/matcher_proofs.parquet')
    parser.add_argument('--rewrite-chain-source',
                        default='et_studies/proof_survey/data/normalized_proofs.parquet')
    parser.add_argument('--rewrite-chain-categories',
                        default='et_studies/proof_survey/data/true_corpus.parquet',
                        help='used to filter normalized_proofs to category==c_rewrite_chain')
    parser.add_argument('--output',
                        default='outputs/analyses/proof_decoder/phase2_vocab.json')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    out_path = _abs(args.output)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force.', file=sys.stderr)
        return 2

    # --- 1. matcher subset tokens (in the same v1..v6 normalized space
    # the rewrite-chain bodies use, for a fair vocab estimate) ---
    matcher_dataset_path = 'et_studies/proof_decoder/data/decoder_phase1_dataset.parquet'
    print(f'tokenizing matcher proofs (normalized, from {matcher_dataset_path}) ...')
    mp = pq.read_table(_abs(matcher_dataset_path), columns=['normalized_proof']).to_pandas()
    matcher_counter: Counter = Counter()
    for body in mp['normalized_proof']:
        matcher_counter.update(TOKEN_RE.findall(body))
    print(f'  {len(mp):,} proofs, vocab so far: {len(matcher_counter):,}')

    # --- 2. rewrite-chain subset tokens ---
    print(f'finding rewrite-chain pairs in {args.rewrite_chain_categories}')
    cat_tbl = pq.read_table(_abs(args.rewrite_chain_categories),
                            columns=['e1', 'e2', 'category']).to_pandas()
    rc_keys = set(zip(
        cat_tbl[cat_tbl['category'] == 'c_rewrite_chain']['e1'].tolist(),
        cat_tbl[cat_tbl['category'] == 'c_rewrite_chain']['e2'].tolist(),
    ))
    print(f'  {len(rc_keys):,} rewrite-chain pairs')

    print(f'loading bodies from {args.rewrite_chain_source}')
    src = pq.read_table(_abs(args.rewrite_chain_source),
                        columns=['e1', 'e2', 'n4_plus_hyps']).to_pandas()
    rc = src[[(e1, e2) in rc_keys for e1, e2 in zip(src['e1'], src['e2'])]]
    print(f'  joined: {len(rc):,} bodies')

    rc_counter: Counter = Counter()
    for body in rc['n4_plus_hyps']:
        rc_counter.update(TOKEN_RE.findall(body))
    print(f'  vocab so far: {len(rc_counter):,}')

    # --- 3. union and bucket ---
    union = matcher_counter + rc_counter
    print(f'\nunion vocab size: {len(union):,}')

    buckets: dict[str, dict[str, int]] = {
        'instruction': {},
        'structural': {},
        'content': {},
    }
    for tok, cnt in union.items():
        buckets[classify_token(tok)][tok] = cnt

    # Sort each bucket by count desc.
    for b in buckets.values():
        items = sorted(b.items(), key=lambda kv: -kv[1])
        b.clear()
        b.update(dict(items))

    counts_per_bucket = {b: len(d) for b, d in buckets.items()}
    total = sum(counts_per_bucket.values())

    summary = {
        'matcher_subset': {
            'n_proofs': int(len(mp)),
            'vocab_size': len(matcher_counter),
            'top_15': dict(matcher_counter.most_common(15)),
        },
        'rewrite_chain_subset': {
            'n_proofs': int(len(rc)),
            'vocab_size': len(rc_counter),
            'top_30': dict(rc_counter.most_common(30)),
        },
        'union_vocab': {
            'total_size': total,
            'instruction_size': counts_per_bucket['instruction'],
            'structural_size': counts_per_bucket['structural'],
            'content_size': counts_per_bucket['content'],
        },
        'buckets': buckets,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'\nwrote {out_path}')

    print()
    print('=== Phase 2 vocab estimate ===')
    print(f"  matcher subset       : {summary['matcher_subset']['vocab_size']:>4} tokens "
          f"({summary['matcher_subset']['n_proofs']:,} proofs)")
    print(f"  rewrite-chain subset : {summary['rewrite_chain_subset']['vocab_size']:>4} tokens "
          f"({summary['rewrite_chain_subset']['n_proofs']:,} proofs)")
    print(f"  union                : {total:>4} tokens")
    print(f"    instruction        : {counts_per_bucket['instruction']:>4}")
    print(f"    structural         : {counts_per_bucket['structural']:>4}")
    print(f"    content            : {counts_per_bucket['content']:>4}")
    print()
    print('  Top 10 instruction tokens:')
    for tok, cnt in list(buckets['instruction'].items())[:10]:
        print(f'    {tok:<20} {cnt:>6,}')
    print('  Top 10 structural tokens:')
    for tok, cnt in list(buckets['structural'].items())[:10]:
        print(f'    {tok:<20} {cnt:>6,}')
    print('  Top 10 content tokens:')
    for tok, cnt in list(buckets['content'].items())[:10]:
        print(f'    {tok:<20} {cnt:>6,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
