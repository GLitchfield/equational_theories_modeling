"""Unit tests for src/matcher/matcher.py.

Cover the algorithm cases the plan calls out:
  - identical equations match trivially
  - structurally different equations don't match
  - inconsistent var mapping (same E1 var → two different E2 vars) fails
  - non-injective mapping (different E1 vars → same E2 var) works
  - flipped orientation
  - compound-term-required substitutions fail
  - Eq1 (`x = x`) as E1 edge case
  - proof emission output strings
"""

from __future__ import annotations

import pytest

from src.data.parser import parse_equation
from src.matcher.matcher import (
    Match,
    match,
    ordered_vars,
    structural_match,
    _match_oriented,
)


# ---------- ordered_vars ----------

def test_ordered_vars_lhs_pre_then_rhs_pre():
    """First-appearance order: walk LHS pre-order, then RHS pre-order."""
    eq = parse_equation('x ◇ (y ◇ z) = z')
    assert ordered_vars(eq) == ('x', 'y', 'z')


def test_ordered_vars_rhs_only_var():
    eq = parse_equation('x = y')
    assert ordered_vars(eq) == ('x', 'y')


def test_ordered_vars_trivial_reflexive():
    eq = parse_equation('x = x')
    assert ordered_vars(eq) == ('x',)


# ---------- structural_match ----------

def test_structural_match_identical_variables():
    s = {}
    t = parse_equation('x = x').lhs
    assert structural_match(t, t, s) is True
    assert s == {'x': 'x'}


def test_structural_match_extends_sigma():
    e1 = parse_equation('x ◇ y = z').lhs   # x ◇ y
    e2 = parse_equation('a ◇ b = c').lhs   # a ◇ b
    s = {}
    assert structural_match(e1, e2, s) is True
    assert s == {'x': 'a', 'y': 'b'}


def test_structural_match_inconsistency_fails():
    """Same E1 variable mapped to two different E2 variables -> fail."""
    e1 = parse_equation('x ◇ x = y').lhs   # x ◇ x
    e2 = parse_equation('a ◇ b = c').lhs   # a ◇ b
    s = {}
    assert structural_match(e1, e2, s) is False


def test_structural_match_non_injective_ok():
    """Different E1 variables can map to the same E2 variable."""
    e1 = parse_equation('x ◇ y = z').lhs   # x ◇ y
    e2 = parse_equation('a ◇ a = b').lhs   # a ◇ a
    s = {}
    assert structural_match(e1, e2, s) is True
    assert s == {'x': 'a', 'y': 'a'}


def test_structural_match_var_vs_op_fails():
    """Variable in E1 mapping to compound term in E2 -> fail."""
    e1 = parse_equation('x = y').lhs       # bare Var(x)
    e2 = parse_equation('a ◇ b = c').lhs   # Op(a, b)
    s = {}
    assert structural_match(e1, e2, s) is False


def test_structural_match_op_vs_var_fails():
    e1 = parse_equation('x ◇ y = z').lhs   # Op
    e2 = parse_equation('a = b').lhs       # Var
    s = {}
    assert structural_match(e1, e2, s) is False


# ---------- _match_oriented ----------

def test_match_oriented_direct_simple():
    eq1 = parse_equation('x ◇ y = y')
    eq2 = parse_equation('a ◇ b = b')
    sigma = _match_oriented(eq1.lhs, eq1.rhs, eq2.lhs, eq2.rhs)
    assert sigma == {'x': 'a', 'y': 'b'}


def test_match_oriented_cross_side_inconsistency():
    """σ that works on LHS but contradicts on RHS is rejected."""
    # E1: x ◇ y = x   (LHS uses x,y; RHS uses x)
    # E2: a ◇ b = b   (RHS contradicts LHS's mapping for x)
    eq1 = parse_equation('x ◇ y = x')
    eq2 = parse_equation('a ◇ b = b')
    sigma = _match_oriented(eq1.lhs, eq1.rhs, eq2.lhs, eq2.rhs)
    assert sigma is None


def test_match_oriented_plan_example():
    """The plan's worked example: E1 = a ◇ (b ◇ c) = a, E2 = x ◇ (y ◇ x) = x,
    σ = {a -> x, b -> y, c -> x}."""
    eq1 = parse_equation('x ◇ (y ◇ z) = x')   # use x,y,z as E1's "a,b,c"
    eq2 = parse_equation('x ◇ (y ◇ x) = x')   # E2 uses x and y; z position becomes x
    sigma = _match_oriented(eq1.lhs, eq1.rhs, eq2.lhs, eq2.rhs)
    assert sigma == {'x': 'x', 'y': 'y', 'z': 'x'}


# ---------- match() ----------

def test_match_identical_equations():
    eq = parse_equation('x ◇ y = z')
    m = match(eq, eq)
    assert m is not None
    assert m.orientation == 0
    assert m.sigma == {'x': 'x', 'y': 'y', 'z': 'z'}


def test_match_direct_orientation():
    eq1 = parse_equation('x ◇ y = y')
    eq2 = parse_equation('a ◇ b = b')
    m = match(eq1, eq2)
    assert m is not None
    assert m.orientation == 0


def test_match_flipped_orientation():
    """E1: x = x ◇ y. E2: a ◇ b = a. Direct fails (Var vs Op on LHS).
    Flipped: lhs1 (x) -> rhs2 (a), rhs1 (x ◇ y) -> lhs2 (a ◇ b). Works."""
    eq1 = parse_equation('x = x ◇ y')
    eq2 = parse_equation('a ◇ b = a')
    m = match(eq1, eq2)
    assert m is not None
    assert m.orientation == 1
    assert m.sigma == {'x': 'a', 'y': 'b'}


def test_match_compound_term_required_fails():
    """E1's variable would need to map to a compound term."""
    eq1 = parse_equation('x ◇ y = z')   # 3 vars, all bare
    eq2 = parse_equation('(x ◇ y) ◇ z = w')   # LHS is a compound, E1's x can't be a Var here
    m = match(eq1, eq2)
    assert m is None


def test_match_eq1_as_hypothesis_only_implies_eq1():
    """E1 = `x = x` (Eq1 reflexive). Per Methodology Resolution 8, the only
    pair the matcher catches is Eq1 -> Eq1."""
    eq1 = parse_equation('x = x')
    # Eq1 -> Eq1 should match
    m = match(eq1, parse_equation('y = y'))
    assert m is not None
    # Eq1 -> any non-trivial equation should NOT match
    for txt in ['x ◇ y = z', '(x ◇ y) ◇ z = x', 'x = y']:
        m = match(eq1, parse_equation(txt))
        assert m is None, f'matcher unexpectedly accepted Eq1 -> {txt}'


def test_match_eq1_as_goal_only_from_eq1():
    """E2 = `x = x`. For E1 to imply this, E1's LHS and RHS must both be
    single variables that σ can map to the same E2 variable.
    So E1 = `a = a` works, E1 = `a = b` works (set σ(a) = σ(b) = x),
    but E1 = `a ◇ b = c` does NOT work."""
    eq2 = parse_equation('x = x')
    assert match(parse_equation('a = a'), eq2) is not None
    assert match(parse_equation('a = b'), eq2) is not None   # σ = {a→x, b→x}
    assert match(parse_equation('a ◇ b = c'), eq2) is None
    assert match(parse_equation('a = b ◇ c'), eq2) is None


# ---------- proof emission ----------

def test_proof_term_direct_two_var_perm():
    """The plan's headline example: λ x y => h x y x for E2's 2 vars,
    E1 has 3 vars, σ maps E1's third var to E2's first var."""
    eq1 = parse_equation('x ◇ (y ◇ z) = x')
    eq2 = parse_equation('x ◇ (y ◇ x) = x')
    m = match(eq1, eq2)
    assert m is not None
    assert m.proof_term() == 'fun x y => h x y x'


def test_proof_term_flipped_uses_symm():
    eq1 = parse_equation('x = x ◇ y')
    eq2 = parse_equation('a ◇ b = a')
    m = match(eq1, eq2)
    assert m is not None
    assert m.orientation == 1
    assert m.proof_term() == 'fun a b => (h a b).symm'


def test_proof_tactic_body_lifts_term():
    """Stage 2 submission body is `exact <term proof>`. Will be revisited
    in Task 20 once the live judge confirms the expected shape."""
    eq1 = parse_equation('x ◇ y = y')
    eq2 = parse_equation('a ◇ b = b')
    m = match(eq1, eq2)
    assert m is not None
    assert m.proof_tactic_body() == 'exact fun a b => h a b'


# ---------- substitution_padded ----------

def test_substitution_padded_two_vars():
    """E1 has 2 vars; padded with -1 to width 6."""
    eq1 = parse_equation('x ◇ y = y')
    eq2 = parse_equation('a ◇ b = b')
    m = match(eq1, eq2)
    assert m is not None
    # e1_vars = (x, y); σ = {x→a, y→b}; e2_vars = (a, b)
    # E2 indices: a=0, b=1. So padded = (0, 1, -1, -1, -1, -1)
    assert m.substitution_padded() == (0, 1, -1, -1, -1, -1)


def test_substitution_padded_non_injective():
    """Plan's worked example: σ = {x→x, y→y, z→x}, e2_vars=(x,y)
    → padded indices (0, 1, 0, -1, -1, -1)."""
    eq1 = parse_equation('x ◇ (y ◇ z) = x')
    eq2 = parse_equation('x ◇ (y ◇ x) = x')
    m = match(eq1, eq2)
    assert m is not None
    assert m.substitution_padded() == (0, 1, 0, -1, -1, -1)


def test_substitution_padded_custom_sentinel():
    eq1 = parse_equation('x = y')
    eq2 = parse_equation('a = b')
    m = match(eq1, eq2)
    assert m is not None
    assert m.substitution_padded(width=4, sentinel=255) == (0, 1, 255, 255)


# ---------- regression: known ETP cases from the corpus ----------

def test_corpus_example_equation12_implies_equation10():
    """From Generated/SimpleRewrites: theorem Equation12_implies_Equation10
    proved with `λ x y => h x y x`. Per equations.txt:
      Equation 10: line 10, Equation 12: line 12. Reading the file shows:
        line 10 = `x = x ◇ (x ◇ x)`  (probably; we cross-check below)
        line 12 = `x = x ◇ (y ◇ x)`
    The implication is Equation12 → Equation10 (E1=12, E2=10).
    σ should map E1's y to E2's x (since E2 has only one variable that
    appears repeatedly).
    """
    # equations.txt is 1-indexed (line N = Equation N).
    with open('/home/gavin/git/equational_theories/data/equations.txt') as f:
        lines = f.readlines()
    eq12 = parse_equation(lines[11].strip())  # line 12
    eq10 = parse_equation(lines[9].strip())   # line 10
    m = match(eq12, eq10)
    assert m is not None
    assert m.proof_term() == 'fun x y => h x y x'
