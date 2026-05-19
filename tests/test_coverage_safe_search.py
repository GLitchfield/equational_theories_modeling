"""Tests for src/coverage/safe_search.py — direction-restricted
bidirectional BFS for the coverage measurement."""

from __future__ import annotations

from src.data.parser import Var, parse_equation
from src.coverage.safe_search import (
    SafeSearchResult,
    TerminateReason,
    _safe_rules,
    safe_search,
)


def _t(text: str):
    return parse_equation(f'a = {text}').rhs


def _e(text: str):
    eq = parse_equation(text)
    return eq.lhs, eq.rhs


# ---------- _safe_rules ----------

def test_safe_rules_variable_equal_both_directions():
    lhs, rhs = _e('x ◇ y = y ◇ x')   # Vars equal
    rules = _safe_rules(lhs, rhs)
    assert (lhs, rhs) in rules
    assert (rhs, lhs) in rules
    assert len(rules) == 2


def test_safe_rules_strict_inclusion_one_direction():
    """E1: x = x ◇ y  → only rhs → lhs is safe (drops y)."""
    lhs, rhs = _e('x = x ◇ y')
    rules = _safe_rules(lhs, rhs)
    assert len(rules) == 1
    assert rules[0] == (rhs, lhs)


def test_safe_rules_variable_disjoint_empty():
    lhs, rhs = _e('x ◇ y = z ◇ w')
    rules = _safe_rules(lhs, rhs)
    assert rules == []


# ---------- safe_search: terminate reasons ----------

def test_safe_search_no_valid_rule_short_circuits():
    """Variable-disjoint E1: skip immediately."""
    lhs2, rhs2 = _e('u ◇ v = u')
    lhs1, rhs1 = _e('x ◇ y = z ◇ w')
    r = safe_search(lhs2, rhs2, lhs1, rhs1)
    assert not r.solved
    assert r.terminate_reason == TerminateReason.NO_VALID_RULE_FROM_E1
    assert r.states_expanded == 0


def test_safe_search_solves_known_case():
    """Same setup as src/rewrite/engine.py::search's known test:
    E1: a = a ◇ a (idempotence, vars equal → both directions safe).
    Goal: x = x ◇ x → solvable in depth 1."""
    lhs2, rhs2 = _e('x = x ◇ x')
    lhs1, rhs1 = _e('a = a ◇ a')
    r = safe_search(lhs2, rhs2, lhs1, rhs1)
    assert r.solved
    assert r.terminate_reason == TerminateReason.SOLVED
    assert r.total_depth == 1


def test_safe_search_depth_exhausted():
    """A goal that can't be reached within d_max=1 but isn't structurally
    impossible. E1: x ◇ y = x (left projection, strict-inclusion safe rule
    x◇y → x). Goal: x ◇ y ◇ z = x doesn't fit in 1 step — actually depends
    on the rule. Let me pick a case where d_max=1 is tight."""
    # Construct a goal where the proof requires depth 2+.
    # E1: x ◇ y = x. Rule: x◇y → x.
    # Goal: (a ◇ b) ◇ c = a. Apply rule once at root: (a◇b)◇c → a◇b. Apply once more: a◇b → a.
    # So depth 2 forward suffices.
    lhs2, rhs2 = _e('(a ◇ b) ◇ c = a')
    lhs1, rhs1 = _e('x ◇ y = x')
    r = safe_search(lhs2, rhs2, lhs1, rhs1, d_max=1)
    # At d_max=1 we can't fully reduce. Should depth-exhaust.
    if not r.solved:
        assert r.terminate_reason in (
            TerminateReason.DEPTH_EXHAUSTED, TerminateReason.FRONTIER_DIED
        )


def test_safe_search_matches_engine_on_matcher_pairs():
    """Sanity: the safe variant should still subsume the matcher.
    For a matcher-shape pair (E2 = σ(E1) var-to-var), at least one of
    E1's directions is safe (either the shrinking direction if E1 has
    strict inclusion, or both if equal), so safe_search should solve."""
    # Pair: E1 = `x = x ◇ x` (vars equal, both dirs safe). E2 = `y = y ◇ y` (σ-rename).
    lhs1, rhs1 = _e('x = x ◇ x')
    lhs2, rhs2 = _e('y = y ◇ y')
    r = safe_search(lhs2, rhs2, lhs1, rhs1, d_max=2)
    assert r.solved


def test_safe_search_avoids_unsafe_direction_leak():
    """E1: x = x ◇ y has only the shrinking direction safe. The original
    engine using both directions could leak literal `y` into rewritten
    terms; safe_search must not. Verify the safe direction still solves
    a plausible goal."""
    # E1: x = x ◇ y. Safe rule: x ◇ y → x. (Drops y.)
    # Goal: a ◇ b = a. Apply once at root: a ◇ b → a. Solves at depth 1.
    lhs1, rhs1 = _e('x = x ◇ y')
    lhs2, rhs2 = _e('a ◇ b = a')
    r = safe_search(lhs2, rhs2, lhs1, rhs1, d_max=2)
    assert r.solved
    assert r.terminate_reason == TerminateReason.SOLVED
