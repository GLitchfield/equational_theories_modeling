"""Tests for src/rewrite/lpo.py — Lexicographic Path Order on magma
terms."""

from __future__ import annotations

from lib.data.parser import Op, Var, parse_equation
from lib.rewrite.lpo import lpo_compare, lpo_gt, orient


def _t(text: str):
    return parse_equation(f'a = {text}').rhs


def _eq(text: str):
    return parse_equation(text).lhs, parse_equation(text).rhs


# ---------- reflexivity / equality ----------

def test_lpo_gt_reflexive_returns_false():
    t = _t('x ◇ y')
    assert not lpo_gt(t, t)


# ---------- Op > Var ----------

def test_op_greater_than_var_unconditionally():
    """An Op is greater than any Var, whether or not the var occurs."""
    assert lpo_gt(_t('x ◇ y'), Var('x'))   # x occurs in lhs
    assert lpo_gt(_t('x ◇ y'), Var('z'))   # z does not occur


def test_var_never_greater_than_op():
    assert not lpo_gt(Var('x'), _t('x ◇ y'))
    assert not lpo_gt(Var('z'), _t('x ◇ y'))


# ---------- Var vs Var: incomparable under strict LPO ----------

def test_distinct_vars_are_incomparable():
    """Under strict LPO (stable under substitution), variables are
    incomparable to each other regardless of any precedence."""
    assert not lpo_gt(Var('z'), Var('x'))
    assert not lpo_gt(Var('x'), Var('z'))
    assert not lpo_gt(Var('y'), Var('x'))


def test_precedence_param_is_unused_for_strict_lpo():
    """The `precedence` parameter is kept for downstream use by
    ordered completion (Phase 4). For strict LPO, it has no effect."""
    assert not lpo_gt(Var('y'), Var('x'), precedence=('x', 'y'))
    assert not lpo_gt(Var('x'), Var('y'), precedence=('y', 'x'))


# ---------- LPO subterm rule (case 3) ----------

def test_op_greater_than_its_subterm_op():
    """(x ◇ y) ◇ z >_lpo x ◇ y — left subterm equality case."""
    assert lpo_gt(_t('(x ◇ y) ◇ z'), _t('x ◇ y'))
    assert lpo_gt(_t('x ◇ (y ◇ z)'), _t('y ◇ z'))


def test_op_greater_than_deeper_var():
    """((x ◇ y) ◇ z) >_lpo x  via subterm-of-subterm + Op>Var."""
    assert lpo_gt(_t('(x ◇ y) ◇ z'), Var('x'))


# ---------- LPO same-head lex (case 4) ----------

def test_lpo_lex_compares_left_first():
    """f(x◇y, z) vs f(x, z) → compare s1 vs t1 first.
    (x◇y) ◇ z vs x ◇ z: left s1=x◇y, t1=x; x◇y > x (case 1).
    Need s > t1, t2: yes. So s > t."""
    assert lpo_gt(_t('(x ◇ y) ◇ z'), _t('x ◇ z'))


def test_lpo_lex_compares_right_when_left_equal():
    """f(x, y◇z) vs f(x, y) → left equal, compare s2=y◇z vs t2=y;
    y◇z > y. Also need s > t1=x, t2=y. Yes."""
    assert lpo_gt(_t('x ◇ (y ◇ z)'), _t('x ◇ y'))


# ---------- associativity orientation ----------

def test_associativity_orients_left_associative_greater():
    """(x◇y)◇z > x◇(y◇z) under LPO with default precedence."""
    s = _t('(x ◇ y) ◇ z')
    t = _t('x ◇ (y ◇ z)')
    assert lpo_gt(s, t)
    assert not lpo_gt(t, s)


# ---------- variable-disjoint terms ----------

def test_lpo_two_vars_incomparable():
    """Strict LPO returns 0 (incomparable) for any pair of distinct
    variables — no implicit precedence among them."""
    assert lpo_compare(Var('x'), Var('y')) == 0
    assert lpo_compare(Var('y'), Var('x')) == 0
    # Same variable: still 0 (equal, not strictly greater).
    assert lpo_compare(Var('x'), Var('x')) == 0


def test_lpo_compare_variable_disjoint_ops_incomparable():
    """Two variable-disjoint Ops with identical structure are
    incomparable under strict LPO: the lex on subterms reduces to
    var-vs-var, which is always incomparable."""
    s = _t('x ◇ y')
    t = _t('u ◇ v')
    assert not lpo_gt(s, t)
    assert not lpo_gt(t, s)
    assert lpo_compare(s, t) == 0


# ---------- orient() integration ----------

def test_orient_simple_one_direction():
    """`x ◇ y = x` orients to (x ◇ y) → x: rhs's vars are a subset of
    lhs's, and lhs > rhs under LPO."""
    lhs, rhs = _eq('x ◇ y = x')
    result = orient(lhs, rhs)
    assert result == (_t('x ◇ y'), Var('x'))


def test_orient_picks_shrinking_direction():
    """`x = x ◇ y` — only valid orientation is `x ◇ y → x` (Vars(x) ⊆
    Vars(x ◇ y) but not vice versa)."""
    lhs, rhs = _eq('x = x ◇ y')
    result = orient(lhs, rhs)
    assert result == (_t('x ◇ y'), Var('x'))


def test_orient_variable_disjoint_returns_none():
    """`x ◇ y = z ◇ w` — neither direction has variable inclusion.
    Phase 0 hard core; orient() returns None."""
    lhs, rhs = _eq('x ◇ y = z ◇ w')
    result = orient(lhs, rhs)
    assert result is None


def test_orient_commutativity_returns_none():
    """`x ◇ y = y ◇ x` — Vars(lhs) == Vars(rhs) (both directions
    variable-valid), but strict LPO leaves x ◇ y and y ◇ x
    INCOMPARABLE (case 4 lex-compare fails on the var-vs-var step).
    orient() returns None. This is the symmetric/order-unorientable
    case that ordered completion (Phase 4) must handle."""
    lhs, rhs = _eq('x ◇ y = y ◇ x')
    assert orient(lhs, rhs) is None


def test_orient_idempotence_succeeds():
    """`x ◇ x = x` orients via subterm rule: x ◇ x > x."""
    lhs, rhs = _eq('x ◇ x = x')
    result = orient(lhs, rhs)
    assert result == (_t('x ◇ x'), Var('x'))


def test_orient_associativity_left_to_right():
    """`(x ◇ y) ◇ z = x ◇ (y ◇ z)` orients left-to-right under strict
    LPO — case 4 lex with s1 = x ◇ y > x = t1."""
    lhs, rhs = _eq('(x ◇ y) ◇ z = x ◇ (y ◇ z)')
    result = orient(lhs, rhs)
    assert result == (lhs, rhs)
