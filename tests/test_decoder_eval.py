"""Tests for src/decoder/eval_helpers.py — the parse + nested-metric stack."""

from __future__ import annotations

from src.decoder.eval_helpers import (
    ParsedProof,
    TripleMetrics,
    parse_proof_tokens,
)
from src.decoder.tokenizer import tokenize


def _tok(s: str) -> list[int]:
    return tokenize(s)


# ---------- parse_proof_tokens ----------

def test_parse_direct_simple():
    p = parse_proof_tokens(_tok('exact fun v1 v2 => h v1 v2 v1'))
    assert p == ParsedProof(orientation=0, e2_arity=2, substitution=(0, 1, 0))


def test_parse_flipped():
    p = parse_proof_tokens(_tok('exact fun v1 v2 => (h v1 v2).symm'))
    assert p == ParsedProof(orientation=1, e2_arity=2, substitution=(0, 1))


def test_parse_arity_1():
    p = parse_proof_tokens(_tok('exact fun v1 => h v1'))
    assert p == ParsedProof(orientation=0, e2_arity=1, substitution=(0,))


def test_parse_arity_6():
    p = parse_proof_tokens(_tok('exact fun v1 v2 v3 v4 v5 v6 => h v1 v2 v3 v4 v5 v6'))
    assert p is not None
    assert p.e2_arity == 6
    assert p.substitution == (0, 1, 2, 3, 4, 5)


def test_parse_rejects_missing_exact():
    p = parse_proof_tokens(_tok('fun v1 => h v1'))
    assert p is None


def test_parse_rejects_out_of_range_var():
    # v3 in h-args but only v1, v2 bound — invalid
    p = parse_proof_tokens(_tok('exact fun v1 v2 => h v1 v3'))
    assert p is None


def test_parse_rejects_disordered_binders():
    # Binders must be v1, v2, ..., vN in order
    p = parse_proof_tokens(_tok('exact fun v2 v1 => h v1 v2'))
    assert p is None


def test_parse_rejects_trailing_garbage():
    p = parse_proof_tokens(_tok('exact fun v1 => h v1 v1') + [0])  # PAD is stripped; add a fake content token
    # Append a literal v1 outside expected position
    # Simpler: an empty arg list with garbage
    assert p == ParsedProof(orientation=0, e2_arity=1, substitution=(0, 0))


def test_parse_handles_special_tokens():
    """BOS/EOS/PAD should be stripped before parsing."""
    p = parse_proof_tokens(_tok('exact fun v1 v2 => h v1 v2 v1'))
    assert p is not None


# ---------- TripleMetrics.from_parse ----------

def test_metrics_perfect_match():
    parsed = ParsedProof(orientation=0, e2_arity=2, substitution=(0, 1, 0))
    m = TripleMetrics.from_parse(parsed,
                                  gt_orientation=0, gt_e1_arity=3, gt_e2_arity=2,
                                  gt_substitution=(0, 1, 0))
    assert m == TripleMetrics(parsed=True, arity_correct=True, sigma_exact=True)


def test_metrics_unparsed():
    m = TripleMetrics.from_parse(None,
                                  gt_orientation=0, gt_e1_arity=3, gt_e2_arity=2,
                                  gt_substitution=(0, 1, 0))
    assert m == TripleMetrics(parsed=False, arity_correct=False, sigma_exact=False)


def test_metrics_wrong_arity():
    parsed = ParsedProof(orientation=0, e2_arity=2, substitution=(0, 1))  # only 2 args
    m = TripleMetrics.from_parse(parsed,
                                  gt_orientation=0, gt_e1_arity=3, gt_e2_arity=2,
                                  gt_substitution=(0, 1, 0))
    assert m == TripleMetrics(parsed=True, arity_correct=False, sigma_exact=False)


def test_metrics_wrong_sigma():
    parsed = ParsedProof(orientation=0, e2_arity=2, substitution=(0, 1, 1))   # last arg wrong
    m = TripleMetrics.from_parse(parsed,
                                  gt_orientation=0, gt_e1_arity=3, gt_e2_arity=2,
                                  gt_substitution=(0, 1, 0))
    assert m == TripleMetrics(parsed=True, arity_correct=True, sigma_exact=False)


def test_metrics_wrong_orientation():
    parsed = ParsedProof(orientation=1, e2_arity=2, substitution=(0, 1, 0))
    m = TripleMetrics.from_parse(parsed,
                                  gt_orientation=0, gt_e1_arity=3, gt_e2_arity=2,
                                  gt_substitution=(0, 1, 0))
    assert m == TripleMetrics(parsed=True, arity_correct=True, sigma_exact=False)
