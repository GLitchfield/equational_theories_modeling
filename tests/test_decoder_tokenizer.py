"""Unit tests for src/decoder/tokenizer.py."""

from __future__ import annotations

import pytest

from et_studies.proof_decoder.code.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    TOKEN_TO_ID,
    VOCAB,
    VOCAB_SIZE,
    build_normalized_proof,
    detokenize,
    tokenize,
)


def test_vocab_size_is_16():
    assert VOCAB_SIZE == 16


def test_special_ids():
    assert PAD_ID == 0
    assert BOS_ID == 1
    assert EOS_ID == 2


def test_no_preamble_tokens_in_vocab():
    """Methodology Resolution 1: drop intro / G / _ from vocab."""
    assert 'intro' not in TOKEN_TO_ID
    assert 'G' not in TOKEN_TO_ID
    assert '_' not in TOKEN_TO_ID


def test_tokenize_direct_proof():
    """`exact fun v1 v2 => h v1 v2 v1` (e2_arity=2, e1_arity=3, σ=(0,1,0))."""
    ids = tokenize('exact fun v1 v2 => h v1 v2 v1')
    expected = [
        BOS_ID,
        TOKEN_TO_ID['exact'],
        TOKEN_TO_ID['fun'],
        TOKEN_TO_ID['v1'],
        TOKEN_TO_ID['v2'],
        TOKEN_TO_ID['=>'],
        TOKEN_TO_ID['h'],
        TOKEN_TO_ID['v1'],
        TOKEN_TO_ID['v2'],
        TOKEN_TO_ID['v1'],
        EOS_ID,
    ]
    assert ids == expected


def test_tokenize_flipped_proof():
    """`exact fun v1 v2 => (h v1 v2).symm`."""
    ids = tokenize('exact fun v1 v2 => (h v1 v2).symm')
    # Just confirm `.symm` is recognized as a single token, not as `.` + `symm`.
    assert TOKEN_TO_ID['.symm'] in ids
    # And ( and ) are present, exactly once each.
    assert ids.count(TOKEN_TO_ID['(']) == 1
    assert ids.count(TOKEN_TO_ID[')']) == 1


def test_tokenize_rejects_unknown_token():
    """Verify the defensive consumed-vs-non-ws check fires."""
    with pytest.raises(ValueError, match='unconsumed'):
        tokenize('intro v1 v2')   # `intro` not in vocab


def test_detokenize_round_trip():
    proof = 'exact fun v1 v2 => h v1 v2 v1'
    ids = tokenize(proof)
    assert detokenize(ids) == proof


def test_detokenize_with_specials():
    ids = [BOS_ID, TOKEN_TO_ID['exact'], EOS_ID, PAD_ID]
    assert detokenize(ids, strip_special=True) == 'exact'
    assert detokenize(ids, strip_special=False) == '<BOS> exact <EOS> <PAD>'


# ---------- build_normalized_proof ----------

def test_build_direct_proof():
    """e2_arity=2, e1_arity=3, σ=(0,1,0) → `exact fun v1 v2 => h v1 v2 v1`."""
    out = build_normalized_proof(
        e2_arity=2, substitution=[0, 1, 0], e1_arity=3, orientation=0,
    )
    assert out == 'exact fun v1 v2 => h v1 v2 v1'


def test_build_flipped_proof():
    out = build_normalized_proof(
        e2_arity=2, substitution=[0, 1], e1_arity=2, orientation=1,
    )
    assert out == 'exact fun v1 v2 => (h v1 v2).symm'


def test_build_arity_1():
    """Edge case: E2 has 1 var, E1 has 1 var, both fill to v1."""
    out = build_normalized_proof(
        e2_arity=1, substitution=[0], e1_arity=1, orientation=0,
    )
    assert out == 'exact fun v1 => h v1'


def test_build_max_arity():
    out = build_normalized_proof(
        e2_arity=6, substitution=[0, 1, 2, 3, 4, 5], e1_arity=6, orientation=0,
    )
    assert out == 'exact fun v1 v2 v3 v4 v5 v6 => h v1 v2 v3 v4 v5 v6'


def test_build_rejects_invalid_arity():
    with pytest.raises(ValueError):
        build_normalized_proof(e2_arity=0, substitution=[], e1_arity=0, orientation=0)
    with pytest.raises(ValueError):
        build_normalized_proof(e2_arity=7, substitution=[0, 1, 2, 3, 4, 5, 6], e1_arity=7, orientation=0)


def test_build_rejects_length_mismatch():
    with pytest.raises(ValueError, match='length'):
        build_normalized_proof(e2_arity=2, substitution=[0, 1, 2], e1_arity=2, orientation=0)


def test_build_then_tokenize_round_trip():
    """The constructed proof string tokenizes without error and emits a
    sequence with BOS at start, EOS at end."""
    out = build_normalized_proof(
        e2_arity=3, substitution=[0, 1, 2, 0], e1_arity=4, orientation=0,
    )
    ids = tokenize(out)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    # detokenize should round-trip
    assert detokenize(ids) == out
