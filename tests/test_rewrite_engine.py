"""Tests for src/rewrite/engine.py — bidirectional rewrite search.

Cover the cases the engine must handle to be useful:
  - identical sides (depth 0 trivial close)
  - one-rewrite close (depth 1)
  - depth-2 close
  - failure when E1 doesn't apply
  - failure beyond d_max
  - term-size-cap rejection
"""

from __future__ import annotations

from src.data.parser import parse_equation
from src.rewrite.engine import SearchResult, default_size_cap, search


def _e(text: str):
    e = parse_equation(text)
    return e.lhs, e.rhs


# ---------- depth 0: identical sides ----------

def test_search_identical_sides():
    """lhs2 = rhs2; depth 0 immediate close."""
    lhs2 = rhs2 = parse_equation('x = x').lhs   # both are Var(x)
    lhs1, rhs1 = _e('a = a')
    r = search(lhs2, rhs2, lhs1, rhs1)
    assert r.closed
    assert r.total_depth == 0


# ---------- depth 1: E1 = E2 directly ----------

def test_search_e1_equals_e2_direct():
    """E1: a = b. E2: x = y. σ rewrites a→x, hence x=y is reachable in 1 step
    from the LHS by rule a→b (which becomes x→y under σ). Actually no — for E1=E2
    we just need lhs2 -> rhs2 in one step or vice versa."""
    lhs2, rhs2 = _e('x = x ◇ x')
    lhs1, rhs1 = _e('a = a ◇ a')   # E1 = E2 (after renaming a→x)
    r = search(lhs2, rhs2, lhs1, rhs1)
    assert r.closed
    # forward: x -> x ◇ x via a → a ◇ a applied at x with σ(a) = x. Depth 1.
    assert r.total_depth == 1


# ---------- depth 2: chain of length 2 ----------

def test_search_chain_of_two():
    """E1: a = a ◇ a. Proving x = (x ◇ x) ◇ (x ◇ x).
    Apply a → a ◇ a at x: get x ◇ x. Then again on each side, get (x ◇ x) ◇ (x ◇ x).
    Forward depth 2."""
    lhs2 = parse_equation('x = x').lhs   # x
    rhs2 = parse_equation('x = (x ◇ x) ◇ (x ◇ x)').rhs
    lhs1, rhs1 = _e('a = a ◇ a')
    r = search(lhs2, rhs2, lhs1, rhs1)
    # Should close; the exact total_depth depends on which side reaches first.
    assert r.closed


# ---------- failure: no rewrite reaches the goal ----------

def test_search_unreachable_returns_failure():
    """Pick E1 that can't possibly prove an unrelated E2."""
    # E1: a ◇ a = a (idempotence — but lhs has an op, so can rewrite an op
    # into a var)
    lhs1, rhs1 = _e('a ◇ a = a')
    # E2: x ◇ y = z ◇ w (totally different shape; idempotence can't derive this)
    lhs2, rhs2 = _e('x ◇ y = z ◇ w')
    r = search(lhs2, rhs2, lhs1, rhs1, d_max=3)
    assert not r.closed


# ---------- term-size cap ----------

def test_search_size_cap_blocks_both_directions():
    """E1: a = a ◇ a expands forward (a → a ◇ a) and shrinks backward
    (a ◇ a → a). Engine starts forward from `x` (size 1) and backward
    from `(x ◇ x) ◇ (x ◇ x)` (size 7). With a tight size_cap=4, neither
    direction can move — forward growth would produce size 3 (within),
    then size 5 (rejected); backward shrinkage produces size 3 (within),
    can't continue from there because nothing else is reachable within
    the cap.

    Actually forward CAN grow x → x ◇ x (size 3 ≤ cap=4). Backward
    can shrink (x◇x)◇(x◇x) → x ◇ x (size 3 ≤ cap=4). So they meet at
    size 3.

    Use a tighter setup: rhs2 has size 7, forward starts from size 1,
    and the only way to bridge requires going through size > 4 OR
    starting backward (which begins above the cap and must shrink
    immediately).
    """
    # Pick lhs2 = x ◇ x ◇ x ◇ x ◇ ... requiring bigger states to reach.
    # Simpler: use a different rule where neither direction shrinks
    # below the cap usefully.
    lhs2 = parse_equation('x = x').lhs                          # size 1
    rhs2 = parse_equation('x = (x ◇ x) ◇ (x ◇ x)').rhs          # size 7
    lhs1, rhs1 = _e('a = a ◇ a')

    # cap=2: any forward step from x produces x ◇ x (size 3 > 2). Blocked.
    # Backward starting from size 7: try to shrink, a ◇ a → a at root gives
    # x ◇ x (size 3 > 2). Blocked. Neither side can move.
    r = search(lhs2, rhs2, lhs1, rhs1, d_max=4, size_cap=2)
    assert not r.closed


def test_default_size_cap_uses_max():
    """default_size_cap = max(50, 5 × max(|lhs|, |rhs|))."""
    from src.rewrite.matching import term_size
    lhs = parse_equation('x = x').lhs
    assert default_size_cap(lhs, lhs) == 50   # small → 50 floor
    # Build a term big enough that 5× exceeds 50 (need |t| ≥ 11).
    big = parse_equation(
        '(((x ◇ y) ◇ (z ◇ w)) ◇ u) = (((x ◇ y) ◇ (z ◇ w)) ◇ u)'
    ).lhs
    assert default_size_cap(lhs, big) == max(50, 5 * term_size(big))


# ---------- bidirectional meet-in-the-middle ----------

def test_bidirectional_meets_in_middle():
    """E1: a = a ◇ a. Goal: x ◇ x = x ◇ x ◇ x ◇ x.
    Forward: x ◇ x → (x◇x)◇(x◇x). Backward: x◇x◇x◇x → ... well, both
    routes ought to converge. Mostly just checks that both directions
    are exercised — verify by states_expanded > 0 on both sides."""
    lhs2 = parse_equation('x = x ◇ x').rhs
    rhs2 = parse_equation('x = ((x ◇ x) ◇ x) ◇ x').rhs
    lhs1, rhs1 = _e('a = a ◇ a')
    r = search(lhs2, rhs2, lhs1, rhs1, d_max=4)
    # Just confirm we got a result and recorded both-side state counts.
    assert r.forward_states_seen >= 1
    assert r.backward_states_seen >= 1


# ---------- failure beyond d_max ----------

def test_d_max_caps_search():
    """If d_max is 0, only the immediate-meet case can close."""
    lhs2, rhs2 = _e('x = y')
    lhs1, rhs1 = _e('a ◇ a = a')
    r = search(lhs2, rhs2, lhs1, rhs1, d_max=0)
    assert not r.closed
    assert r.forward_depth is None and r.backward_depth is None
