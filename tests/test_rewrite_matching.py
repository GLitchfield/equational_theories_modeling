"""Tests for src/rewrite/matching.py.

Cover the cases that distinguish first-order matching from the
matcher's structural matching, plus the supporting functions
(substitute, rewrite_at, all_rewrites, term_size).
"""

from __future__ import annotations

import pytest

from src.data.parser import Op, Var, parse_equation
from src.rewrite.matching import (
    all_rewrites,
    first_order_match,
    rewrite_at,
    substitute,
    term_size,
)


# ---------- first_order_match: cases the matcher would also handle ----------

def test_var_to_var():
    sigma: dict = {}
    assert first_order_match(Var('a'), Var('x'), sigma) is True
    assert sigma == {'a': Var('x')}


def test_op_to_op():
    p = parse_equation('a ◇ b = c').lhs    # Op(a, b)
    s = parse_equation('x ◇ y = z').lhs    # Op(x, y)
    sigma: dict = {}
    assert first_order_match(p, s, sigma) is True
    assert sigma == {'a': Var('x'), 'b': Var('y')}


def test_consistency_check():
    """Rule variable `a` appears twice; second occurrence must match
    the first binding."""
    p = parse_equation('a ◇ a = b').lhs    # Op(a, a)
    s_ok = parse_equation('x ◇ x = y').lhs
    s_bad = parse_equation('x ◇ y = z').lhs
    sigma_ok: dict = {}
    assert first_order_match(p, s_ok, sigma_ok) is True
    sigma_bad: dict = {}
    assert first_order_match(p, s_bad, sigma_bad) is False


def test_op_pattern_vs_var_subject_fails():
    """If the pattern requires an operator but the subject is a bare
    variable, fail."""
    p = parse_equation('a ◇ b = c').lhs    # Op
    s = parse_equation('x = y').lhs        # Var
    sigma: dict = {}
    assert first_order_match(p, s, sigma) is False


# ---------- first_order_match: cases the matcher REJECTS but we accept ----------

def test_var_pattern_binds_to_compound_term():
    """THE distinguishing test: a rule variable can bind to a
    compound subject (an Op). The matcher's structural_match rejects
    this; first-order match accepts."""
    p = Var('a')
    s = parse_equation('x ◇ y = z').lhs    # Op(x, y)
    sigma: dict = {}
    assert first_order_match(p, s, sigma) is True
    assert sigma == {'a': s}


def test_multiple_rule_vars_binding_to_compounds():
    """Pattern `a ◇ b`, subject `(x ◇ y) ◇ z`. Should bind a → (x ◇ y),
    b → z."""
    p = parse_equation('a ◇ b = c').lhs                # Op(a, b)
    s = parse_equation('(x ◇ y) ◇ z = w').lhs          # Op(Op(x,y), z)
    sigma: dict = {}
    assert first_order_match(p, s, sigma) is True
    assert sigma['a'] == parse_equation('x ◇ y = z').lhs
    assert sigma['b'] == Var('z')


def test_repeated_rule_var_to_same_compound():
    """Pattern `a ◇ a`; subject `(x ◇ y) ◇ (x ◇ y)` — same compound
    on both sides; should succeed with a → x ◇ y."""
    p = parse_equation('a ◇ a = b').lhs
    s = parse_equation('(x ◇ y) ◇ (x ◇ y) = z').lhs
    sigma: dict = {}
    assert first_order_match(p, s, sigma) is True
    assert sigma['a'] == parse_equation('x ◇ y = z').lhs


def test_repeated_rule_var_to_different_compounds_fails():
    """Pattern `a ◇ a`; subject `(x ◇ y) ◇ z` — different RHS; should fail."""
    p = parse_equation('a ◇ a = b').lhs
    s = parse_equation('(x ◇ y) ◇ z = w').lhs
    sigma: dict = {}
    assert first_order_match(p, s, sigma) is False


# ---------- substitute ----------

def test_substitute_simple():
    """σ = {a: x}; apply to Op(a, a) → Op(x, x)."""
    t = parse_equation('a ◇ a = b').lhs
    sigma = {'a': Var('x')}
    out = substitute(t, sigma)
    assert out == parse_equation('x ◇ x = y').lhs


def test_substitute_with_compound_value():
    """σ = {a: y ◇ z}; apply to Op(a, a) → Op(y◇z, y◇z)."""
    t = parse_equation('a ◇ a = b').lhs
    sigma = {'a': parse_equation('y ◇ z = w').lhs}
    out = substitute(t, sigma)
    assert out == parse_equation('(y ◇ z) ◇ (y ◇ z) = w').lhs


def test_substitute_leaves_unmapped_vars():
    """Variables not in σ are left as-is."""
    t = parse_equation('a ◇ b = c').lhs
    sigma = {'a': Var('x')}
    out = substitute(t, sigma)
    assert out == parse_equation('x ◇ b = c').lhs


# ---------- rewrite_at (root only) ----------

def test_rewrite_at_root_succeeds():
    """Rule `a → a ◇ a`, apply at root of `x`: → `x ◇ x`."""
    p = Var('a')
    r = parse_equation('a ◇ a = b').lhs   # Op(a, a)
    term = Var('x')
    out = rewrite_at(term, p, r)
    assert out == parse_equation('x ◇ x = y').lhs


def test_rewrite_at_root_fails():
    """Rule with operator pattern on a bare variable subject fails."""
    p = parse_equation('a ◇ b = c').lhs
    r = Var('a')
    term = Var('x')
    out = rewrite_at(term, p, r)
    assert out is None


# ---------- all_rewrites (every subterm position) ----------

def test_all_rewrites_root_and_children():
    """Rule `a → a ◇ a` applied to `x ◇ y`. Root rewrite: (x ◇ y) → (x ◇ y) ◇ (x ◇ y).
    Plus each child: x → x ◇ x, y → y ◇ y."""
    p = Var('a')
    r = parse_equation('a ◇ a = b').lhs
    term = parse_equation('x ◇ y = z').lhs
    outs = all_rewrites(term, p, r)
    # Three rewrite positions: root, left child, right child.
    assert len(outs) == 3
    # Root rewrite: (x ◇ y) ◇ (x ◇ y)
    root_rw = parse_equation('(x ◇ y) ◇ (x ◇ y) = z').lhs
    # Left rewrite: (x ◇ x) ◇ y
    left_rw = parse_equation('(x ◇ x) ◇ y = z').lhs
    # Right rewrite: x ◇ (y ◇ y)
    right_rw = parse_equation('x ◇ (y ◇ y) = z').lhs
    assert root_rw in outs
    assert left_rw in outs
    assert right_rw in outs


def test_all_rewrites_no_matches():
    """Rule pattern fits nowhere → empty list."""
    p = parse_equation('a ◇ b = c').lhs   # operator pattern
    r = Var('a')
    term = Var('x')                       # bare variable, no operator
    outs = all_rewrites(term, p, r)
    assert outs == []


def test_all_rewrites_is_deterministic():
    """Same inputs → same output list, in same order."""
    p = parse_equation('a ◇ b = c').lhs
    r = parse_equation('b ◇ a = c').lhs   # swap
    term = parse_equation('x ◇ (y ◇ z) = w').lhs
    a = all_rewrites(term, p, r)
    b = all_rewrites(term, p, r)
    assert a == b


# ---------- term_size ----------

def test_term_size():
    assert term_size(Var('a')) == 1
    assert term_size(parse_equation('a ◇ b = c').lhs) == 3      # Op + 2 vars
    assert term_size(parse_equation('(a ◇ b) ◇ c = d').lhs) == 5  # Op + Op + 2 vars + var
