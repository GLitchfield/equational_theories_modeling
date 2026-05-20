"""Tests for src/completion/ordered.py — unfailing KB completion."""

from __future__ import annotations

from lib.data.parser import Op, Var, parse_equation
from src.completion.kb import normalize as plain_normalize
from src.completion.ordered import (
    OrderedStatus,
    equation_to_rules,
    ordered_complete,
    ordered_normalize,
)


def _t(text: str):
    return parse_equation(f'a = {text}').rhs


def _e(text: str):
    return parse_equation(text).lhs, parse_equation(text).rhs


# ============================================================
# equation_to_rules
# ============================================================

def test_equation_to_rules_symmetric():
    """Vars equal → two rules."""
    s, t = _e('x ◇ y = y ◇ x')
    rules = equation_to_rules(s, t)
    assert len(rules) == 2
    assert (s, t) in rules
    assert (t, s) in rules


def test_equation_to_rules_directed():
    """Vars(rhs) ⊊ Vars(lhs) → one rule lhs → rhs."""
    s, t = _e('x ◇ y = x')
    rules = equation_to_rules(s, t)
    assert len(rules) == 1
    assert rules[0] == (s, t)


def test_equation_to_rules_variable_disjoint():
    """Vars disjoint → zero rules."""
    s, t = _e('x ◇ y = z ◇ w')
    rules = equation_to_rules(s, t)
    assert rules == []


# ============================================================
# ordered_normalize
# ============================================================

def test_ordered_normalize_per_instance_check_blocks_unsound_step():
    """Rule (y ◇ x → x ◇ x): variable inclusion holds (y is dropped),
    but LPO is abstractly incomparable — applies only per-instance
    when σ(y) is structurally greater than σ(x). On subject (a ◇ b)
    with both vars (incomparable), the rule should NOT apply."""
    # Construct rule manually so the test isn't coupled to orient()'s
    # variable-inclusion gating.
    lhs = _t('y ◇ x')
    rhs = _t('x ◇ x')
    # Subject (a ◇ b) — both vars distinct. LPO incomparable.
    subject = _t('a ◇ b')
    # ordered_normalize should not rewrite.
    result = ordered_normalize(subject, [(lhs, rhs)])
    assert result == subject


def test_ordered_normalize_per_instance_check_allows_sound_step():
    """Same rule. Subject ((c ◇ d) ◇ e) — σ(y)=c◇d (compound),
    σ(x)=e (var). LPO: lpo_gt((c◇d) ◇ e, e ◇ e)? Op>Var via case 2;
    lex (c◇d, e) vs (e, e); c◇d > e (Op>Var); lex yes. Per-instance
    valid; rule should apply."""
    lhs = _t('y ◇ x')
    rhs = _t('x ◇ x')
    subject = _t('(c ◇ d) ◇ e')
    result = ordered_normalize(subject, [(lhs, rhs)])
    assert result == _t('e ◇ e')


def test_ordered_normalize_idempotence_works_like_plain_kb():
    """Rule (x ◇ x → x): always sound, always strictly decreases,
    per-instance check always passes. Same result as plain normalize."""
    rule = (_t('x ◇ x'), Var('x'))
    t = _t('(a ◇ a) ◇ b')
    expected = plain_normalize(t, [rule])
    actual = ordered_normalize(t, [rule])
    assert actual == expected


# ============================================================
# ordered_complete
# ============================================================

def test_ordered_complete_identity_returns_empty_or_only_input():
    """x = x — already joinable, no rules needed."""
    result = ordered_complete([_e('x = x')])
    assert result.status == OrderedStatus.NO_RULES_FROM_INPUT or result.rules == []


def test_ordered_complete_idempotence_yields_one_rule():
    """Same as plain KB: (x ◇ x → x), 0 CPs, COMPLETED."""
    result = ordered_complete([_e('x = x ◇ x')])
    assert result.status == OrderedStatus.COMPLETED
    assert (_t('x ◇ x'), Var('x')) in result.rules


def test_ordered_complete_variable_disjoint_input_classified_no_rules():
    """`x ◇ y = z ◇ w` — Phase 0 hard core. Variable inclusion fails
    both ways; equation_to_rules returns []. Status: NO_RULES_FROM_INPUT."""
    result = ordered_complete([_e('x ◇ y = z ◇ w')])
    assert result.status == OrderedStatus.NO_RULES_FROM_INPUT
    assert result.rules == []


def test_ordered_complete_commutativity_completes():
    """`x ◇ y = y ◇ x` — plain KB returns FAILED_ORIENT. Ordered KB
    keeps both directions and completes. The 'rules' contain at least
    both directions of the input equation."""
    s, t = _e('x ◇ y = y ◇ x')
    result = ordered_complete([(s, t)], max_critical_pairs=200,
                              max_rules=20, wall_seconds=2.0)
    assert result.status == OrderedStatus.COMPLETED
    assert (s, t) in result.rules
    assert (t, s) in result.rules


def test_ordered_complete_decides_a_consequence():
    """After completing idempotence x = x ◇ x, the system should
    decide (a ◇ a) ◇ a = a as TRUE."""
    result = ordered_complete([_e('x = x ◇ x')])
    assert result.status == OrderedStatus.COMPLETED
    lhs, rhs = _e('(a ◇ a) ◇ a = a')
    assert ordered_normalize(lhs, result.rules) == ordered_normalize(rhs, result.rules)


def test_ordered_complete_commutativity_reduces_pair():
    """After completing commutativity, the term (b ◇ a) should
    normalize to its 'canonical' form. Whichever direction LPO
    chooses, the normalize on (b ◇ a) and (a ◇ b) should agree."""
    result = ordered_complete([_e('x ◇ y = y ◇ x')], max_critical_pairs=200)
    assert result.status == OrderedStatus.COMPLETED
    nf1 = ordered_normalize(_t('b ◇ a'), result.rules)
    nf2 = ordered_normalize(_t('a ◇ b'), result.rules)
    # Both should reach the same normal form — that's the whole point
    # of completing the commutative theory.
    assert nf1 == nf2
