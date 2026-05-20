"""Tests for src/rewrite/unification.py — first-order Robinson
unification with occurs-check."""

from __future__ import annotations

from lib.data.parser import Op, Var, parse_equation
from lib.rewrite.unification import (
    apply_subst,
    collect_vars,
    occurs,
    rename_vars,
    unify,
)


def _t(text: str):
    """Parse a single term — wrap in 'x = TEXT' and return the rhs."""
    return parse_equation(f'x = {text}').rhs


# ---------- apply_subst ----------

def test_apply_subst_identity_when_no_bindings():
    t = _t('x ◇ y')
    assert apply_subst(t, {}) == t


def test_apply_subst_replaces_var():
    t = Var('x')
    sigma = {Var('x'): Var('y')}
    assert apply_subst(t, sigma) == Var('y')


def test_apply_subst_walks_recursively():
    """σ = {x → y, y → z}. apply_subst(x, σ) should chase to z."""
    sigma = {Var('x'): Var('y'), Var('y'): Var('z')}
    assert apply_subst(Var('x'), sigma) == Var('z')


def test_apply_subst_descends_into_op():
    t = _t('x ◇ y')   # Op(x, y)
    sigma = {Var('x'): _t('u ◇ v'), Var('y'): Var('w')}
    expected = Op(_t('u ◇ v'), Var('w'))
    assert apply_subst(t, sigma) == expected


# ---------- occurs-check ----------

def test_occurs_var_in_self():
    assert occurs(Var('x'), Var('x'))


def test_occurs_var_in_op():
    assert occurs(Var('x'), _t('x ◇ y'))
    assert occurs(Var('y'), _t('x ◇ y'))


def test_occurs_var_not_in_unrelated_term():
    assert not occurs(Var('z'), _t('x ◇ y'))


# ---------- unify: simple cases ----------

def test_unify_identical_terms_returns_empty():
    s = _t('x ◇ y')
    sigma = unify(s, s)
    assert sigma == {}


def test_unify_var_with_var():
    """unify(x, y) → {x → y} or {y → x} (either is valid mgu)."""
    sigma = unify(Var('x'), Var('y'))
    assert sigma is not None
    assert apply_subst(Var('x'), sigma) == apply_subst(Var('y'), sigma)


def test_unify_var_with_op():
    """unify(x, a ◇ b) → {x → a ◇ b}."""
    sigma = unify(Var('x'), _t('a ◇ b'))
    assert sigma is not None
    assert apply_subst(Var('x'), sigma) == _t('a ◇ b')


def test_unify_op_with_var_symmetric():
    """unify(a ◇ b, x) → {x → a ◇ b}."""
    sigma = unify(_t('a ◇ b'), Var('x'))
    assert sigma is not None
    assert apply_subst(Var('x'), sigma) == _t('a ◇ b')


# ---------- occurs-check failure ----------

def test_unify_occurs_check_blocks_self_referential():
    """unify(x, x ◇ y) should fail — x can't be x ◇ y (occurs-check)."""
    sigma = unify(Var('x'), _t('x ◇ y'))
    assert sigma is None


def test_unify_occurs_check_blocks_indirect_cycle():
    """unify(x, y ◇ z) succeeds; then unifying y with x must fail."""
    sigma = unify(Var('x'), _t('y ◇ z'))
    assert sigma is not None
    # Now try to unify y with x given the existing σ — this would
    # introduce y → x → y ◇ z (cycle).
    sigma2 = unify(Var('y'), Var('x'), sigma)
    assert sigma2 is None


# ---------- unify: compound cases ----------

def test_unify_two_ops_aligning_subterms():
    """unify(x ◇ b, a ◇ y) — must produce SOME mgu under which the
    two sides become equal. Multiple valid mgus exist (binding b→y
    vs y→b), so only check the equality property."""
    s = _t('x ◇ b')
    t = _t('a ◇ y')
    sigma = unify(s, t)
    assert sigma is not None
    assert apply_subst(s, sigma) == apply_subst(t, sigma)


def test_unify_propagates_through_existing_bindings():
    """unify((x ◇ x), (a ◇ y)) — both pairs of leaves must end up
    equal under σ. The specific form of σ depends on _compose's
    direction choice; assert equality, not a specific form."""
    s = _t('x ◇ x')
    t = _t('a ◇ y')
    sigma = unify(s, t)
    assert sigma is not None
    # All four leaves collapse to the same value.
    assert apply_subst(Var('x'), sigma) == apply_subst(Var('a'), sigma)
    assert apply_subst(Var('x'), sigma) == apply_subst(Var('y'), sigma)
    # And the two compound terms become equal.
    assert apply_subst(s, sigma) == apply_subst(t, sigma)


def test_unify_clash_returns_none():
    """unify((a ◇ b), (a ◇ b ◇ c)) → mismatch on second subterm shape."""
    s = _t('a ◇ b')
    t = _t('a ◇ (b ◇ c)')
    sigma = unify(s, t)
    # s.right is Var(b); t.right is Op(b, c). Should unify b → b ◇ c via
    # occurs-check... b doesn't occur in b ◇ c — wait, b occurs in b ◇ c!
    # So occurs-check should block.
    assert sigma is None


def test_unify_var_with_compound_containing_other_var():
    """unify(x, a ◇ b) succeeds with x → a ◇ b. Verify."""
    sigma = unify(Var('x'), _t('a ◇ b'))
    assert sigma is not None
    assert apply_subst(Var('x'), sigma) == _t('a ◇ b')


# ---------- rename_vars helper ----------

def test_rename_vars_appends_suffix():
    t = _t('x ◇ y')
    renamed = rename_vars(t, '_1')
    assert renamed == Op(Var('x_1'), Var('y_1'))


def test_rename_vars_descends_into_nested():
    t = _t('(x ◇ y) ◇ z')
    renamed = rename_vars(t, '_2')
    assert renamed == Op(Op(Var('x_2'), Var('y_2')), Var('z_2'))


def test_collect_vars_returns_all_vars():
    t = _t('(x ◇ y) ◇ x')
    vs = collect_vars(t)
    assert vs == frozenset({Var('x'), Var('y')})


# ---------- the role of variable renaming for rule unification ----------

def test_unification_after_renaming_to_disjoint_var_sets():
    """If two rules share variable names but mean different things,
    rename them apart before unifying. unify(x ◇ y, y ◇ x) without
    renaming binds both vars to each other — meaningful for one
    rule but accidental conflation between two rules.

    With renaming: unify((x_1 ◇ y_1), (y_2 ◇ x_2)) treats all 4 vars
    as independent and finds {x_1 → y_2, y_1 → x_2}."""
    lhs1 = _t('x ◇ y')
    lhs2 = _t('y ◇ x')
    r1 = rename_vars(lhs1, '_1')
    r2 = rename_vars(lhs2, '_2')
    sigma = unify(r1, r2)
    assert sigma is not None
    # apply σ to both sides; they should be equal.
    assert apply_subst(r1, sigma) == apply_subst(r2, sigma)
