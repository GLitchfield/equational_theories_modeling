"""Bidirectional rewrite-search engine for equational implication.

Per Methodology Resolutions 3, 4, 5 in `rewrite_chain_study_plan.md`:

  - Bidirectional (meet-in-the-middle) iterative-deepening BFS.
    Expand reachable sets from both `lhs2` (the goal's LHS) and
    `rhs2` (the goal's RHS) using E1's rewrite rule (forward and
    reverse). Succeed when the two sets intersect.
  - Per-side depth bound `d_max=6` by default (so total proof
    chain length up to 12).
  - Term-size cap (default `max(50, 5 * max(|lhs2|, |rhs2|))`)
    rejects any rewrite that produces a term larger than the cap.
  - Returns the smallest total depth at which the goal is closed,
    or None on failure within the bounds.

Single rule: E1 = `lhs1 = rhs1` is treated as `{lhs1 → rhs1,
rhs1 → lhs1}` (both directions). Per the survey, 99.7% of
rewrite-chain proofs stay purely within this rule set.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.parser import Term
from src.rewrite.matching import all_rewrites, term_size


@dataclass(frozen=True)
class SearchResult:
    closed: bool                 # did we find a meeting point?
    total_depth: int | None      # forward_depth + backward_depth at intersection (None if !closed)
    forward_depth: int | None
    backward_depth: int | None
    states_expanded: int         # how many new terms generated total
    forward_states_seen: int     # how many unique terms in the forward set at the end
    backward_states_seen: int    # how many unique terms in the backward set at the end


def default_size_cap(lhs2: Term, rhs2: Term) -> int:
    """`max(50, 5 * max(|lhs2|, |rhs2|))` per Methodology Resolution 3."""
    return max(50, 5 * max(term_size(lhs2), term_size(rhs2)))


def search(
    lhs2: Term,
    rhs2: Term,
    lhs1: Term,
    rhs1: Term,
    d_max: int = 6,
    size_cap: int | None = None,
    states_budget: int | None = None,
) -> SearchResult:
    """Bidirectional iterative-deepening BFS for proving `lhs2 = rhs2`
    using E1 (`lhs1 = rhs1`) as a rewrite rule (forward and reverse).

    Args:
        lhs2, rhs2: the goal's two sides.
        lhs1, rhs1: E1's two sides.
        d_max: per-side depth bound. Total proof chain up to 2 * d_max.
        size_cap: maximum allowed |term| at any intermediate state.
            Defaults to `default_size_cap(lhs2, rhs2)`.

    Returns:
        SearchResult with `closed=True` and `total_depth = fd + bd`
        if a meeting point is found at forward depth `fd` and
        backward depth `bd`. Otherwise `closed=False`.

        If `states_budget` is set and the search expands that many
        states without finding a meet, abort early with closed=False
        and `total_depth=None`. Useful for bounding worst-case time
        per pair when running on a corpus.
    """
    if size_cap is None:
        size_cap = default_size_cap(lhs2, rhs2)

    # Each side's "seen" map: term -> minimum depth at which it was reached.
    fwd: dict[Term, int] = {lhs2: 0}
    bwd: dict[Term, int] = {rhs2: 0}

    # If sides start equal, depth-0 meet.
    if lhs2 == rhs2:
        return SearchResult(
            closed=True, total_depth=0, forward_depth=0, backward_depth=0,
            states_expanded=0, forward_states_seen=1, backward_states_seen=1,
        )
    # Or the goal IS the rule (E1 = E2). Depth-1 meet.
    if (lhs2 == lhs1 and rhs2 == rhs1) or (lhs2 == rhs1 and rhs2 == lhs1):
        # Single rewrite either direction yields the other side.
        return SearchResult(
            closed=True, total_depth=1, forward_depth=1, backward_depth=0,
            states_expanded=1, forward_states_seen=2, backward_states_seen=1,
        )

    # Current frontier per side.
    fwd_frontier: list[Term] = [lhs2]
    bwd_frontier: list[Term] = [rhs2]

    states_expanded = 0

    for d in range(1, d_max + 1):
        # --- Expand forward frontier one step ---
        new_fwd: list[Term] = []
        for t in fwd_frontier:
            # Forward rewrite: lhs1 -> rhs1
            for nt in all_rewrites(t, lhs1, rhs1):
                if term_size(nt) > size_cap:
                    continue
                if nt not in fwd:
                    fwd[nt] = d
                    new_fwd.append(nt)
                    states_expanded += 1
                    if nt in bwd:
                        return SearchResult(
                            closed=True, total_depth=d + bwd[nt],
                            forward_depth=d, backward_depth=bwd[nt],
                            states_expanded=states_expanded,
                            forward_states_seen=len(fwd),
                            backward_states_seen=len(bwd),
                        )
                    if states_budget is not None and states_expanded >= states_budget:
                        return SearchResult(
                            closed=False, total_depth=None,
                            forward_depth=None, backward_depth=None,
                            states_expanded=states_expanded,
                            forward_states_seen=len(fwd),
                            backward_states_seen=len(bwd),
                        )
            # Reverse rewrite: rhs1 -> lhs1
            for nt in all_rewrites(t, rhs1, lhs1):
                if term_size(nt) > size_cap:
                    continue
                if nt not in fwd:
                    fwd[nt] = d
                    new_fwd.append(nt)
                    states_expanded += 1
                    if nt in bwd:
                        return SearchResult(
                            closed=True, total_depth=d + bwd[nt],
                            forward_depth=d, backward_depth=bwd[nt],
                            states_expanded=states_expanded,
                            forward_states_seen=len(fwd),
                            backward_states_seen=len(bwd),
                        )
                    if states_budget is not None and states_expanded >= states_budget:
                        return SearchResult(
                            closed=False, total_depth=None,
                            forward_depth=None, backward_depth=None,
                            states_expanded=states_expanded,
                            forward_states_seen=len(fwd),
                            backward_states_seen=len(bwd),
                        )
        fwd_frontier = new_fwd

        # --- Expand backward frontier one step ---
        new_bwd: list[Term] = []
        for t in bwd_frontier:
            for nt in all_rewrites(t, lhs1, rhs1):
                if term_size(nt) > size_cap:
                    continue
                if nt not in bwd:
                    bwd[nt] = d
                    new_bwd.append(nt)
                    states_expanded += 1
                    if nt in fwd:
                        return SearchResult(
                            closed=True, total_depth=fwd[nt] + d,
                            forward_depth=fwd[nt], backward_depth=d,
                            states_expanded=states_expanded,
                            forward_states_seen=len(fwd),
                            backward_states_seen=len(bwd),
                        )
                    if states_budget is not None and states_expanded >= states_budget:
                        return SearchResult(
                            closed=False, total_depth=None,
                            forward_depth=None, backward_depth=None,
                            states_expanded=states_expanded,
                            forward_states_seen=len(fwd),
                            backward_states_seen=len(bwd),
                        )
            for nt in all_rewrites(t, rhs1, lhs1):
                if term_size(nt) > size_cap:
                    continue
                if nt not in bwd:
                    bwd[nt] = d
                    new_bwd.append(nt)
                    states_expanded += 1
                    if nt in fwd:
                        return SearchResult(
                            closed=True, total_depth=fwd[nt] + d,
                            forward_depth=fwd[nt], backward_depth=d,
                            states_expanded=states_expanded,
                            forward_states_seen=len(fwd),
                            backward_states_seen=len(bwd),
                        )
                    if states_budget is not None and states_expanded >= states_budget:
                        return SearchResult(
                            closed=False, total_depth=None,
                            forward_depth=None, backward_depth=None,
                            states_expanded=states_expanded,
                            forward_states_seen=len(fwd),
                            backward_states_seen=len(bwd),
                        )
        bwd_frontier = new_bwd

        # If both frontiers exhausted, no path exists.
        if not fwd_frontier and not bwd_frontier:
            return SearchResult(
                closed=False, total_depth=None,
                forward_depth=None, backward_depth=None,
                states_expanded=states_expanded,
                forward_states_seen=len(fwd),
                backward_states_seen=len(bwd),
            )

    return SearchResult(
        closed=False, total_depth=None,
        forward_depth=None, backward_depth=None,
        states_expanded=states_expanded,
        forward_states_seen=len(fwd),
        backward_states_seen=len(bwd),
    )

