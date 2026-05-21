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

from lib.data.parser import parse_equation
from et_studies.rewrite_chain_study.code.engine import (
    EnumerationResult,
    ProofPath,
    SearchResult,
    default_size_cap,
    enumerate_chains,
    find_path,
    search,
)
from lib.rewrite.matching import term_size


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
    from lib.rewrite.matching import term_size
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


# ---------- find_path: path reconstruction ----------

def test_find_path_identical_sides_returns_trivial_path():
    """Depth-0 close: forward + backward both contain just the term."""
    lhs2 = rhs2 = parse_equation('x = x').lhs
    lhs1, rhs1 = _e('a = a')
    p = find_path(lhs2, rhs2, lhs1, rhs1)
    assert p.closed
    assert p.total_depth == 0
    assert p.forward_path == (lhs2,)
    assert p.backward_path == (rhs2,)
    assert p.meeting_term == lhs2


def test_find_path_forward_only():
    """E1: a = a ◇ a. Goal: x = x ◇ x. Forward step: x → x ◇ x.
    Backward depth 0 (rhs2 IS the meeting term)."""
    lhs2 = parse_equation('x = x').lhs
    rhs2 = parse_equation('x = x ◇ x').rhs
    lhs1, rhs1 = _e('a = a ◇ a')
    p = find_path(lhs2, rhs2, lhs1, rhs1)
    assert p.closed
    assert p.meeting_term == rhs2
    assert p.forward_path[0] == lhs2
    assert p.forward_path[-1] == rhs2
    assert p.backward_depth == 0
    assert p.forward_depth >= 1
    # Each successive term differs from the previous by one rewrite step.
    assert len(p.forward_path) == p.forward_depth + 1
    assert len(p.backward_path) == 1


def test_find_path_meets_in_middle():
    """Build a goal where the engine must move BOTH sides.
    Goal: x ◇ x = (x ◇ x) ◇ (x ◇ x). Rule: a = a ◇ a.
    Forward: x ◇ x → (x◇x) ◇ x → (x◇x) ◇ (x◇x).  Or some such.
    Backward: same target. We mainly check that both paths are nontrivial
    and meet at a common term."""
    lhs2 = parse_equation('x = x ◇ x').rhs
    rhs2 = parse_equation('x = (x ◇ x) ◇ (x ◇ x)').rhs
    lhs1, rhs1 = _e('a = a ◇ a')
    p = find_path(lhs2, rhs2, lhs1, rhs1)
    assert p.closed
    # The meeting term must appear at the end of BOTH paths.
    assert p.forward_path[-1] == p.meeting_term
    assert p.backward_path[-1] == p.meeting_term
    # And each path starts from its respective endpoint.
    assert p.forward_path[0] == lhs2
    assert p.backward_path[0] == rhs2


def test_find_path_unreachable_returns_failure():
    """Same setup as test_search_unreachable: should return closed=False
    with an empty proof path."""
    lhs1, rhs1 = _e('a ◇ a = a')
    lhs2, rhs2 = _e('x ◇ y = z ◇ w')
    p = find_path(lhs2, rhs2, lhs1, rhs1, d_max=3)
    assert not p.closed
    assert p.forward_path == ()
    assert p.backward_path == ()


def test_find_path_total_depth_matches_search():
    """find_path's reported total_depth should match search's, on a case
    where search closes (so we can compare)."""
    lhs2 = parse_equation('x = x').lhs
    rhs2 = parse_equation('x = (x ◇ x) ◇ (x ◇ x)').rhs
    lhs1, rhs1 = _e('a = a ◇ a')
    s = search(lhs2, rhs2, lhs1, rhs1)
    p = find_path(lhs2, rhs2, lhs1, rhs1)
    assert s.closed and p.closed
    assert s.total_depth == p.total_depth


def test_find_path_intermediate_terms_have_bounded_size():
    """All intermediate terms must respect the size cap."""
    lhs2 = parse_equation('x = x').lhs
    rhs2 = parse_equation('x = (x ◇ x) ◇ (x ◇ x)').rhs
    lhs1, rhs1 = _e('a = a ◇ a')
    cap = default_size_cap(lhs2, rhs2)
    p = find_path(lhs2, rhs2, lhs1, rhs1, size_cap=cap)
    assert p.closed
    for t in list(p.forward_path) + list(p.backward_path):
        assert term_size(t) <= cap


# ---------- enumerate_chains: multimodality ----------

def test_enumerate_chains_zero_depth_only_when_sides_equal():
    """target_depth=0 yields one chain iff lhs2 == rhs2."""
    lhs1, rhs1 = _e('a = a ◇ a')
    eq = parse_equation('x = x ◇ x')
    # lhs2 != rhs2 → 0 chains at depth 0
    r = enumerate_chains(eq.lhs, eq.rhs, lhs1, rhs1, target_depth=0)
    assert r.n_distinct_chains == 0
    # lhs2 == rhs2 → 1 chain (the singleton path).
    r = enumerate_chains(eq.lhs, eq.lhs, lhs1, rhs1, target_depth=0)
    assert r.n_distinct_chains == 1
    assert r.chains == ((eq.lhs,),)


def test_enumerate_chains_returns_chains_of_exact_length():
    """All chains returned have target_depth + 1 terms."""
    lhs2 = parse_equation('x = x').lhs
    rhs2 = parse_equation('x = x ◇ x').rhs
    lhs1, rhs1 = _e('a = a ◇ a')
    r = enumerate_chains(lhs2, rhs2, lhs1, rhs1, target_depth=1)
    assert r.n_distinct_chains >= 1
    for chain in r.chains:
        assert len(chain) == 2
        assert chain[0] == lhs2 and chain[-1] == rhs2


def test_enumerate_chains_dedupes_by_term_sequence():
    """If two distinct rewrite-rule applications produce the same
    intermediate-term sequence, they count as ONE chain."""
    lhs2 = parse_equation('x = x').lhs
    rhs2 = parse_equation('x = x ◇ x').rhs
    lhs1, rhs1 = _e('a = a ◇ a')
    r = enumerate_chains(lhs2, rhs2, lhs1, rhs1, target_depth=1)
    # Forward (a→a◇a applied at root) and backward (a◇a→a applied… NO,
    # backward needs an op, but lhs2 is a var). So only forward works.
    # One distinct chain.
    assert r.n_distinct_chains == 1


def test_enumerate_chains_can_find_multiple():
    """Goal: x ◇ x = (x ◇ x) ◇ (x ◇ x) with rule a = a ◇ a.
    Forward chains of length 2: at depth 1, x◇x → (x◇x)◇x or x◇(x◇x);
    at depth 2, each leads to (x◇x)◇(x◇x). So 2 distinct chains."""
    lhs2 = parse_equation('x = x ◇ x').rhs
    rhs2 = parse_equation('x = (x ◇ x) ◇ (x ◇ x)').rhs
    lhs1, rhs1 = _e('a = a ◇ a')
    r = enumerate_chains(lhs2, rhs2, lhs1, rhs1, target_depth=2)
    assert r.n_distinct_chains >= 2


def test_enumerate_chains_states_budget():
    """A tight budget aborts DFS without finishing — chains is a lower
    bound on the true count."""
    lhs2 = parse_equation('x = x ◇ x').rhs
    rhs2 = parse_equation('x = (x ◇ x) ◇ (x ◇ x)').rhs
    lhs1, rhs1 = _e('a = a ◇ a')
    r = enumerate_chains(lhs2, rhs2, lhs1, rhs1, target_depth=4, states_budget=2)
    assert r.hit_budget
    assert r.states_expanded >= 2
