"""Soundness-restricted variant of `src/rewrite/engine.py::search` for
the coverage measurement.

The base engine uses both directions of E1 unconditionally. With
first-order matching and σ taken at match-time only, applying a rule
`s → t` where `Vars(t) ⊋ Vars(s)` substitutes literal rule-variable
names into the rewritten term — these names can collide with E2's
free variables and produce unsound "meets" downstream.

This module restricts the engine to *safe* directions per Phase 0's
variable-inclusion classification:

  Vars(t) ⊆ Vars(s)   → rule s → t is safe.
  otherwise           → unsafe direction (would leak fresh rule vars).

For each E1 = `lhs1 = rhs1`, we admit at most two safe rules:
  (lhs1 → rhs1) iff Vars(rhs1) ⊆ Vars(lhs1)
  (rhs1 → lhs1) iff Vars(lhs1) ⊆ Vars(rhs1)

If both fail (variable-disjoint E1), the function returns immediately
with `terminate_reason='no_valid_rule_from_e1'`. Otherwise it runs a
bidirectional iterative-deepening BFS using only the safe rules and
reports `terminate_reason` for sound interpretation of the negative
results.

Termination reasons:
  solved                 — meeting term found within d_max
  no_valid_rule_from_e1  — neither direction safe (variable-disjoint)
  depth_exhausted        — d_max iterations completed, frontier nonempty,
                            no meet found
  frontier_died          — both frontiers became empty inside d_max
  wall_timeout           — externally enforced (SIGALRM); the function
                            itself doesn't install signal handlers
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.data.parser import Term
from src.rewrite.matching import all_rewrites, term_size
from src.rewrite.unification import collect_vars


class TerminateReason(str, Enum):
    SOLVED = 'solved'
    NO_VALID_RULE_FROM_E1 = 'no_valid_rule_from_e1'
    DEPTH_EXHAUSTED = 'depth_exhausted'
    FRONTIER_DIED = 'frontier_died'
    WALL_TIMEOUT = 'wall_timeout'   # set by caller on SIGALRM
    BUDGET_STATES = 'budget_states'  # if a states budget is in play


@dataclass(frozen=True)
class SafeSearchResult:
    solved: bool
    terminate_reason: TerminateReason
    total_depth: int | None
    forward_depth: int | None
    backward_depth: int | None
    states_expanded: int
    forward_states_seen: int
    backward_states_seen: int


def _safe_rules(lhs1: Term, rhs1: Term) -> list[tuple[Term, Term]]:
    """Return the variable-inclusion-safe rules from E1."""
    vs = {v.name for v in collect_vars(lhs1)}
    vt = {v.name for v in collect_vars(rhs1)}
    rules: list[tuple[Term, Term]] = []
    if vt <= vs:
        rules.append((lhs1, rhs1))   # forward direction safe
    if vs <= vt:
        rules.append((rhs1, lhs1))   # reverse direction safe
    return rules


def default_size_cap(lhs2: Term, rhs2: Term) -> int:
    return max(50, 5 * max(term_size(lhs2), term_size(rhs2)))


def safe_search(
    lhs2: Term,
    rhs2: Term,
    lhs1: Term,
    rhs1: Term,
    d_max: int = 6,
    size_cap: int | None = None,
    states_budget: int | None = None,
) -> SafeSearchResult:
    """Bidirectional meet-in-the-middle BFS using only variable-inclusion-safe
    directions of E1. See module docstring for rationale."""
    if size_cap is None:
        size_cap = default_size_cap(lhs2, rhs2)

    rules = _safe_rules(lhs1, rhs1)
    if not rules:
        return SafeSearchResult(
            solved=False,
            terminate_reason=TerminateReason.NO_VALID_RULE_FROM_E1,
            total_depth=None, forward_depth=None, backward_depth=None,
            states_expanded=0, forward_states_seen=0, backward_states_seen=0,
        )

    fwd: dict[Term, int] = {lhs2: 0}
    bwd: dict[Term, int] = {rhs2: 0}

    if lhs2 == rhs2:
        return SafeSearchResult(
            solved=True, terminate_reason=TerminateReason.SOLVED,
            total_depth=0, forward_depth=0, backward_depth=0,
            states_expanded=0, forward_states_seen=1, backward_states_seen=1,
        )

    fwd_frontier: list[Term] = [lhs2]
    bwd_frontier: list[Term] = [rhs2]
    states_expanded = 0

    def _step(frontier: list[Term], seen: dict[Term, int], d: int,
              other: dict[Term, int]) -> tuple[list[Term], Term | None]:
        """Expand `frontier` one step; return (new_frontier, meet_term_or_None).
        Updates `seen` and increments states_expanded by reference via closure."""
        nonlocal states_expanded
        new_frontier: list[Term] = []
        for t in frontier:
            for pattern, replacement in rules:
                for nt in all_rewrites(t, pattern, replacement):
                    if term_size(nt) > size_cap:
                        continue
                    if nt not in seen:
                        seen[nt] = d
                        new_frontier.append(nt)
                        states_expanded += 1
                        if nt in other:
                            return new_frontier, nt
                        if states_budget is not None and states_expanded >= states_budget:
                            return new_frontier, None  # caller checks budget
        return new_frontier, None

    for d in range(1, d_max + 1):
        # Forward step
        fwd_frontier, meet = _step(fwd_frontier, fwd, d, bwd)
        if meet is not None:
            return SafeSearchResult(
                solved=True, terminate_reason=TerminateReason.SOLVED,
                total_depth=d + bwd[meet],
                forward_depth=d, backward_depth=bwd[meet],
                states_expanded=states_expanded,
                forward_states_seen=len(fwd), backward_states_seen=len(bwd),
            )
        if states_budget is not None and states_expanded >= states_budget:
            return SafeSearchResult(
                solved=False, terminate_reason=TerminateReason.BUDGET_STATES,
                total_depth=None, forward_depth=None, backward_depth=None,
                states_expanded=states_expanded,
                forward_states_seen=len(fwd), backward_states_seen=len(bwd),
            )

        # Backward step
        bwd_frontier, meet = _step(bwd_frontier, bwd, d, fwd)
        if meet is not None:
            return SafeSearchResult(
                solved=True, terminate_reason=TerminateReason.SOLVED,
                total_depth=fwd[meet] + d,
                forward_depth=fwd[meet], backward_depth=d,
                states_expanded=states_expanded,
                forward_states_seen=len(fwd), backward_states_seen=len(bwd),
            )
        if states_budget is not None and states_expanded >= states_budget:
            return SafeSearchResult(
                solved=False, terminate_reason=TerminateReason.BUDGET_STATES,
                total_depth=None, forward_depth=None, backward_depth=None,
                states_expanded=states_expanded,
                forward_states_seen=len(fwd), backward_states_seen=len(bwd),
            )

        if not fwd_frontier and not bwd_frontier:
            return SafeSearchResult(
                solved=False, terminate_reason=TerminateReason.FRONTIER_DIED,
                total_depth=None, forward_depth=None, backward_depth=None,
                states_expanded=states_expanded,
                forward_states_seen=len(fwd), backward_states_seen=len(bwd),
            )

    return SafeSearchResult(
        solved=False, terminate_reason=TerminateReason.DEPTH_EXHAUSTED,
        total_depth=None, forward_depth=None, backward_depth=None,
        states_expanded=states_expanded,
        forward_states_seen=len(fwd), backward_states_seen=len(bwd),
    )
