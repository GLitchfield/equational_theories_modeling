"""Task 5 of `proof_corpus_survey_plan.md` — tokenization + vocabulary.

Tokenizes every TRUE proof body (from
et_studies/proof_survey/data/normalized_proofs.parquet,
level n1_whitespace) into Lean-shaped
tokens, then reports:

  - Total vocabulary size before normalization
  - Token frequency distribution: how many tokens cover X% of usage
  - Estimated vocab size after the N4 normalization (variables /
    equation refs / hypotheses → placeholders)

Methodology Resolution 9: multi-character Unicode identifiers
(`α`, `β`, `≃`, etc.) are SINGLE tokens, matching Lean's parser.

Output: outputs/analyses/proof_survey/tokenization_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


# Lean tokenization:
#  - whitespace is a separator (drop)
#  - multi-char operators come first: <;>, :=, =>, ↦, ▸, .., …
#  - structural brackets / punctuation: ( ) [ ] { } ⟨ ⟩ , ; .
#  - identifier characters: letters (incl. unicode), digits, underscore,
#    primes, '_', and Unicode greek letters
#
# Tokenize via a single regex with alternatives, longest-first.
#
# Note: this is a heuristic Lean lexer. It's adequate for the regular
# ETP corpus (mostly auto-generated, very uniform syntax).

OP_TOKENS = sorted([
    '<;>', ':=', '=>', '↦', '▸', '..', '←', '→', '∀', '∃', '∧', '∨', '¬',
    '≠', '≤', '≥', '∈', '∉', '⊆', '∪', '∩', '⟨', '⟩', '◇',
], key=len, reverse=True)
# Escape regex specials in single-char ops (most are safe but `.`, `(`, etc. need it)

# Bracket / punctuation single chars
PUNCT = list('()[]{}.,;|:')

# Build the master tokenizer regex.
def _build_tokenizer() -> re.Pattern:
    parts: list[str] = []
    for op in OP_TOKENS:
        parts.append(re.escape(op))
    for p in PUNCT:
        parts.append(re.escape(p))
    # Identifier: at least one letter / underscore / unicode letter,
    # followed by any of letters/digits/underscores/primes
    parts.append(r'[A-Za-z_α-ωΑ-Ω][A-Za-z0-9_α-ωΑ-Ωʹʹ]*')
    # Integer literal
    parts.append(r'\d+')
    # Catch-all for any other non-whitespace char (single token, surfaces unknown chars)
    parts.append(r'\S')
    return re.compile('|'.join(parts))


TOKEN_RE = _build_tokenizer()


def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s)


def cumcoverage(counts: Counter, thresholds=(0.50, 0.80, 0.90, 0.95, 0.99)) -> dict:
    total = sum(counts.values())
    sorted_freqs = sorted(counts.values(), reverse=True)
    cum = 0
    out: dict = {}
    for t in thresholds:
        for i, c in enumerate(sorted_freqs, start=1):
            cum += c
            if cum / total >= t:
                out[f'tokens_for_{int(t*100)}pct'] = i
                cum = sum(sorted_freqs[:i])
                break
        # reset cum for next threshold
        cum = 0
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input',
                        default='et_studies/proof_survey/data/normalized_proofs.parquet')
    parser.add_argument('--output',
                        default='outputs/analyses/proof_survey/tokenization_results.json')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    out_path = _abs(args.output)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force.', file=sys.stderr)
        return 2

    print(f'reading {args.input}')
    df = pd.read_parquet(_abs(args.input))
    print(f'  {len(df):,} proofs')

    # Tokenize both raw (N1) and fully normalized (N4) bodies for comparison
    print('tokenizing raw (N1) bodies ...')
    raw_counter: Counter = Counter()
    proof_lengths: list[int] = []
    for s in df['n1_whitespace']:
        toks = tokenize(s)
        raw_counter.update(toks)
        proof_lengths.append(len(toks))

    print('tokenizing fully normalized (N4) bodies ...')
    norm_counter: Counter = Counter()
    norm_lengths: list[int] = []
    for s in df['n4_plus_hyps']:
        toks = tokenize(s)
        norm_counter.update(toks)
        norm_lengths.append(len(toks))

    # Distributions
    raw_total = sum(raw_counter.values())
    norm_total = sum(norm_counter.values())

    pls = pd.Series(proof_lengths)
    nls = pd.Series(norm_lengths)

    summary = {
        'n_proofs': int(len(df)),
        'tokens_total_raw': raw_total,
        'tokens_total_normalized': norm_total,
        'vocab_size_raw': len(raw_counter),
        'vocab_size_normalized_N4': len(norm_counter),
        'proof_length_tokens_raw': {
            'min': int(pls.min()), 'max': int(pls.max()),
            'mean': float(pls.mean()), 'median': float(pls.median()),
            'p25': float(pls.quantile(0.25)), 'p75': float(pls.quantile(0.75)),
            'p90': float(pls.quantile(0.90)), 'p95': float(pls.quantile(0.95)),
            'p99': float(pls.quantile(0.99)),
        },
        'proof_length_tokens_normalized': {
            'min': int(nls.min()), 'max': int(nls.max()),
            'mean': float(nls.mean()), 'median': float(nls.median()),
            'p25': float(nls.quantile(0.25)), 'p75': float(nls.quantile(0.75)),
            'p90': float(nls.quantile(0.90)), 'p95': float(nls.quantile(0.95)),
            'p99': float(nls.quantile(0.99)),
        },
        'tokens_for_coverage_raw': cumcoverage(raw_counter),
        'tokens_for_coverage_normalized': cumcoverage(norm_counter),
        'top_50_tokens_raw': dict(raw_counter.most_common(50)),
        'top_50_tokens_normalized': dict(norm_counter.most_common(50)),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'\nwrote {out_path}')

    print()
    print('=== Tokenization summary ===')
    print(f"  Proofs                          : {summary['n_proofs']:,}")
    print(f"  Total tokens (raw)              : {summary['tokens_total_raw']:,}")
    print(f"  Vocab size (raw)                : {summary['vocab_size_raw']:,}")
    print(f"  Vocab size (after N4 normalize) : {summary['vocab_size_normalized_N4']:,}")
    pls = summary['proof_length_tokens_raw']
    print(f"  Proof length (raw, tokens)      : median {pls['median']:.0f}, mean {pls['mean']:.1f}, p95 {pls['p95']:.0f}, p99 {pls['p99']:.0f}, max {pls['max']:,}")
    cov = summary['tokens_for_coverage_raw']
    print(f"  Vocab needed (raw) for 95%/99%  : {cov['tokens_for_95pct']:>4} / {cov['tokens_for_99pct']:>4}")
    cov2 = summary['tokens_for_coverage_normalized']
    print(f"  Vocab needed (norm) for 95%/99% : {cov2['tokens_for_95pct']:>4} / {cov2['tokens_for_99pct']:>4}")
    print()
    print('  Top 15 tokens (raw):')
    for tok, n in list(summary['top_50_tokens_raw'].items())[:15]:
        print(f'    {tok:<20} {n:>8,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
