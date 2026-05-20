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

from lib.data.parser import Term
from lib.rewrite.matching import all_rewrites, term_size


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


@dataclass(frozen=True)
class ProofPath:
    closed: bool
    meeting_term: Term | None
    forward_path: tuple[Term, ...]    # lhs2 = forward_path[0], ..., meeting_term = forward_path[-1]
    backward_path: tuple[Term, ...]   # rhs2 = backward_path[0], ..., meeting_term = backward_path[-1]
    forward_depth: int                # len(forward_path) - 1
    backward_depth: int               # len(backward_path) - 1
    total_depth: int                  # forward_depth + backward_depth
    states_expanded: int


def find_path(
    lhs2: Term,
    rhs2: Term,
    lhs1: Term,
    rhs1: Term,
    d_max: int = 6,
    size_cap: int | None = None,
    states_budget: int | None = None,
) -> ProofPath:
    """Like `search`, but tracks parents during BFS and reconstructs one
    shortest proof path on success. Used for Task 3 intermediate-state
    characterization. Slower than `search` (allocates parent dicts), so
    intended for sample-level analysis, not full-corpus runs."""
    if size_cap is None:
        size_cap = default_size_cap(lhs2, rhs2)

    # term -> (depth_reached, parent_term_or_None_for_root)
    fwd: dict[Term, tuple[int, Term | None]] = {lhs2: (0, None)}
    bwd: dict[Term, tuple[int, Term | None]] = {rhs2: (0, None)}

    def reconstruct(meet: Term) -> ProofPath:
        # Forward path: meet -> ... -> lhs2, then reverse.
        rev_fwd = [meet]
        cur = meet
        while True:
            _, prev = fwd[cur]
            if prev is None:
                break
            rev_fwd.append(prev)
            cur = prev
        fwd_path = tuple(reversed(rev_fwd))
        # Backward path: meet -> ... -> rhs2, then reverse.
        rev_bwd = [meet]
        cur = meet
        while True:
            _, prev = bwd[cur]
            if prev is None:
                break
            rev_bwd.append(prev)
            cur = prev
        bwd_path = tuple(reversed(rev_bwd))
        fd = len(fwd_path) - 1
        bd = len(bwd_path) - 1
        return ProofPath(
            closed=True, meeting_term=meet,
            forward_path=fwd_path, backward_path=bwd_path,
            forward_depth=fd, backward_depth=bd, total_depth=fd + bd,
            states_expanded=states_expanded,
        )

    if lhs2 == rhs2:
        return ProofPath(
            closed=True, meeting_term=lhs2,
            forward_path=(lhs2,), backward_path=(rhs2,),
            forward_depth=0, backward_depth=0, total_depth=0,
            states_expanded=0,
        )

    fwd_frontier: list[Term] = [lhs2]
    bwd_frontier: list[Term] = [rhs2]
    states_expanded = 0

    def fail() -> ProofPath:
        return ProofPath(
            closed=False, meeting_term=None,
            forward_path=(), backward_path=(),
            forward_depth=-1, backward_depth=-1, total_depth=-1,
            states_expanded=states_expanded,
        )

    for d in range(1, d_max + 1):
        new_fwd: list[Term] = []
        for t in fwd_frontier:
            for pattern, replacement in ((lhs1, rhs1), (rhs1, lhs1)):
                for nt in all_rewrites(t, pattern, replacement):
                    if term_size(nt) > size_cap:
                        continue
                    if nt not in fwd:
                        fwd[nt] = (d, t)
                        new_fwd.append(nt)
                        states_expanded += 1
                        if nt in bwd:
                            return reconstruct(nt)
                        if states_budget is not None and states_expanded >= states_budget:
                            return fail()
        fwd_frontier = new_fwd

        new_bwd: list[Term] = []
        for t in bwd_frontier:
            for pattern, replacement in ((lhs1, rhs1), (rhs1, lhs1)):
                for nt in all_rewrites(t, pattern, replacement):
                    if term_size(nt) > size_cap:
                        continue
                    if nt not in bwd:
                        bwd[nt] = (d, t)
                        new_bwd.append(nt)
                        states_expanded += 1
                        if nt in fwd:
                            return reconstruct(nt)
                        if states_budget is not None and states_expanded >= states_budget:
                            return fail()
        bwd_frontier = new_bwd

        if not fwd_frontier and not bwd_frontier:
            return fail()

    return fail()


@dataclass(frozen=True)
class EnumerationResult:
    chains: tuple[tuple[Term, ...], ...]   # each chain: (lhs2, t_1, ..., rhs2)
    n_distinct_chains: int
    states_expanded: int
    hit_budget: bool


def enumerate_chains(
    lhs2: Term,
    rhs2: Term,
    lhs1: Term,
    rhs1: Term,
    target_depth: int,
    size_cap: int | None = None,
    states_budget: int | None = 100_000,
) -> EnumerationResult:
    """Enumerate every distinct chain
        lhs2 -> t_1 -> ... -> t_{D-1} -> rhs2
    of length exactly `target_depth` rewrite steps. Used for Task 5
    multimodality.

    Implementation is bidirectional all-parents BFS:
      1) Forward BFS from lhs2, recording every (term, depth, parent)
         tuple so we can backtrack all shortest paths.
      2) Backward BFS from rhs2, same.
      3) For each meeting term `m` with min-fwd-depth + min-bwd-depth ==
         target_depth, enumerate every fwd-path-to-`m` × every
         bwd-path-from-`m`. Concatenate and deduplicate by chain.

    Forward-only DFS does NOT work for this corpus: rules like
    `x = x ◇ y` introduce fresh variables on the RHS, so forward
    rewriting from `x` produces `x ◇ y_rule` (with the literal
    rule-variable name) rather than `x ◇ z_subj`. Bidirectional
    search binds the fresh variable naturally on the backward step.

    Counting semantics: chains are deduplicated by their term sequence.
    Different meeting-point splits of the same chain count once.

    `states_budget` bounds total BFS-expansion + chain-enumeration work."""
    if size_cap is None:
        size_cap = default_size_cap(lhs2, rhs2)

    # Forward BFS — record minimum depth + every parent at that minimum depth.
    fwd_depth: dict[Term, int] = {lhs2: 0}
    fwd_parents: dict[Term, list[Term]] = {lhs2: []}   # root has empty parent list

    bwd_depth: dict[Term, int] = {rhs2: 0}
    bwd_parents: dict[Term, list[Term]] = {rhs2: []}

    states_expanded = 0

    def expand(
        frontier: list[Term],
        d: int,
        depth_map: dict[Term, int],
        parents_map: dict[Term, list[Term]],
    ) -> tuple[list[Term], int]:
        new_frontier: list[Term] = []
        budget_hit = 0
        for t in frontier:
            for pattern, replacement in ((lhs1, rhs1), (rhs1, lhs1)):
                for nt in all_rewrites(t, pattern, replacement):
                    if term_size(nt) > size_cap:
                        continue
                    nonlocal states_expanded
                    states_expanded += 1
                    if states_budget is not None and states_expanded >= states_budget:
                        return new_frontier, 1
                    if nt not in depth_map:
                        depth_map[nt] = d
                        parents_map[nt] = [t]
                        new_frontier.append(nt)
                    elif depth_map[nt] == d:
                        # another shortest-path parent at this same depth
                        if t not in parents_map[nt]:
                            parents_map[nt].append(t)
                    # else: deeper-route arrival — skip (we only keep shortest)
        return new_frontier, budget_hit

    fwd_frontier = [lhs2]
    bwd_frontier = [rhs2]
    hit_budget = False
    for d in range(1, target_depth + 1):
        fwd_frontier, fb = expand(fwd_frontier, d, fwd_depth, fwd_parents)
        if fb:
            hit_budget = True
            break
        bwd_frontier, bb = expand(bwd_frontier, d, bwd_depth, bwd_parents)
        if bb:
            hit_budget = True
            break

    if target_depth == 0:
        if lhs2 == rhs2:
            return EnumerationResult(
                chains=((lhs2,),), n_distinct_chains=1,
                states_expanded=0, hit_budget=False,
            )
        return EnumerationResult(
            chains=(), n_distinct_chains=0,
            states_expanded=0, hit_budget=False,
        )

    # Enumerate all shortest paths to each meeting term, then combine.
    def paths_to(target: Term, depth: int, parents_map: dict[Term, list[Term]],
                 root: Term) -> list[tuple[Term, ...]]:
        """All paths root -> ... -> target of length exactly `depth`."""
        if depth == 0:
            return [(target,)] if target == root else []
        results: list[tuple[Term, ...]] = []
        for prev in parents_map.get(target, []):
            for sub in paths_to(prev, depth - 1, parents_map, root):
                results.append(sub + (target,))
        return results

    chains: set[tuple[Term, ...]] = set()
    meeting_terms = set(fwd_depth) & set(bwd_depth)
    for m in meeting_terms:
        fd = fwd_depth[m]
        bd = bwd_depth[m]
        if fd + bd != target_depth:
            continue
        fwd_paths = paths_to(m, fd, fwd_parents, lhs2)
        bwd_paths = paths_to(m, bd, bwd_parents, rhs2)
        for fp in fwd_paths:
            for bp in bwd_paths:
                # fp ends at m; bp ends at m. Full chain: fp + reverse(bp without final m).
                full = fp + tuple(reversed(bp[:-1]))
                chains.add(full)
                states_expanded += 1
                if states_budget is not None and states_expanded >= states_budget:
                    hit_budget = True
                    return EnumerationResult(
                        chains=tuple(chains),
                        n_distinct_chains=len(chains),
                        states_expanded=states_expanded,
                        hit_budget=True,
                    )

    return EnumerationResult(
        chains=tuple(chains),
        n_distinct_chains=len(chains),
        states_expanded=states_expanded,
        hit_budget=hit_budget,
    )
