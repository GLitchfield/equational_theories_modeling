"""Ordered (unfailing) Knuth-Bendix completion for magma equations.

Distinct from `src/completion/kb.py` (plain KB) in that equations are
never "given up" because abstract LPO can't orient them strictly.
Instead, each equation contributes 0, 1, or 2 *rules*, one per
direction in which variable inclusion holds; rules are applied
**per-instance**, gated by an LPO comparison of the specific subject
against the specific result.

Per-equation outcomes:

  Vars(s) == Vars(t):              two rules, both per-instance.
                                    (Symmetric / commutativity-shaped.)
  Vars(t) ⊊ Vars(s):               one rule s → t. Per-instance.
  Vars(s) ⊊ Vars(t):               one rule t → s. Per-instance.
  Vars(s), Vars(t) incomparable:   ZERO rules. The equation cannot
                                    participate; the input theory is
                                    not captured. Status:
                                    NO_RULES_FROM_INPUT.

Per-instance orientation: a rule l → r applies to subject `u` only
when `lpo_gt(u, u_after)` — i.e., the specific instance is strictly
decreasing under LPO. This recovers the "symmetric" case (where
abstract LPO is incomparable but specific instances are not) that
plain KB rejects.

This is a Phase 4 component of the completion exploration plan.
"""

from __future__ import annotations

import signal as _signal
import time
from dataclasses import dataclass, field
from enum import Enum

from src.completion.kb import (
    _CompletionTimeout,
    _alarm_handler,
    critical_pairs,
    non_var_positions,
)
from src.data.parser import Term, Var
from src.rewrite.lpo import DEFAULT_VAR_PRECEDENCE, lpo_gt_total
from src.rewrite.matching import all_rewrites, term_size
from src.rewrite.unification import collect_vars


Rule = tuple[Term, Term]


class OrderedStatus(str, Enum):
    COMPLETED = 'completed'
    NO_RULES_FROM_INPUT = 'no_rules_from_input'
    BUDGET_RULES = 'budget_rules'
    BUDGET_CPS = 'budget_cps'
    BUDGET_WALL = 'budget_wall'


@dataclass
class OrderedResult:
    status: OrderedStatus
    rules: list[Rule]
    n_critical_pairs_considered: int = 0
    n_critical_pairs_dropped_by_size_cap: int = 0
    n_iterations: int = 0
    n_equations_kept_as_dormant: int = 0   # variable-disjoint CPs that can't become rules
    wall_seconds: float = 0.0

    @property
    def is_fully_completed(self) -> bool:
        """COMPLETED with no CPs dropped by the size cap. Sound for
        both TRUE and FALSE in the downstream decision procedure."""
        return (self.status == OrderedStatus.COMPLETED
                and self.n_critical_pairs_dropped_by_size_cap == 0
                and self.n_equations_kept_as_dormant == 0)


def equation_to_rules(s: Term, t: Term) -> list[Rule]:
    """Return the 0–2 valid directional rules for equation s = t.

    Variable-inclusion is the construction-time gate; the LPO
    per-instance check happens at rewriting.

      Vars(t) ⊆ Vars(s)  → rule s → t valid.
      Vars(s) ⊆ Vars(t)  → rule t → s valid.

    When both hold (Vars(s) == Vars(t)), both rules are returned —
    the rewriter will pick whichever side is order-decreasing on the
    specific instance. Trivial equations (s == t literally) return []."""
    if s == t:
        return []
    vs = {v.name for v in collect_vars(s)}
    vt = {v.name for v in collect_vars(t)}
    rules: list[Rule] = []
    if vt <= vs:
        rules.append((s, t))
    if vs <= vt:
        rules.append((t, s))
    return rules


def ordered_normalize(
    t: Term,
    rules: list[Rule],
    precedence: tuple[str, ...] = DEFAULT_VAR_PRECEDENCE,
) -> Term:
    """Reduce t to a normal form under the rule set, applying each
    rule only at positions where the specific rewrite is strictly
    decreasing under **total** LPO (variables ordered by `precedence`).

    Per-instance check uses `lpo_gt_total` (not strict LPO) because
    we're comparing two specific terms, not asserting an abstract
    invariant — total order is needed to actually decide which of
    (b ◇ a) vs (a ◇ b) is "greater" so that ordered rewriting can
    rewrite one to the other.

    Termination: each successful rewrite is strictly decreasing under
    total LPO, which is well-founded. So normalize terminates."""
    while True:
        applied = False
        for l, r in rules:
            for nt in all_rewrites(t, l, r):
                if lpo_gt_total(t, nt, precedence):
                    t = nt
                    applied = True
                    break
            if applied:
                break
        if not applied:
            return t


def ordered_complete(
    equations: list[tuple[Term, Term]],
    max_rules: int = 100,
    max_critical_pairs: int = 10_000,
    max_term_size: int = 30,
    wall_seconds: float = 30.0,
    precedence: tuple[str, ...] = DEFAULT_VAR_PRECEDENCE,
) -> OrderedResult:
    """Run ordered (unfailing) KB completion.

    Differences from plain `src/completion/kb.py::complete`:
      - Equations are never rejected as unorientable. Each equation
        contributes 0–2 rules per variable inclusion; per-instance
        LPO gates each rewrite.
      - Status NO_RULES_FROM_INPUT fires only if the INPUT equation
        is variable-disjoint (no rules from it). Otherwise we run
        the cascade.
      - Dormant CPs (variable-disjoint critical pairs generated
        mid-cascade) are tracked but don't participate. They do not
        prevent COMPLETED — the rule set is still ground-confluent
        on the part of the theory it represents.
    """
    t_start = time.time()

    # Initial rules from input equations. A non-trivial equation that
    # produces zero rules (variable-disjoint) flags the whole run as
    # NO_RULES_FROM_INPUT. Trivial (s == t) equations are silently
    # skipped — they contribute nothing but don't fail the run.
    input_rules: list[Rule] = []
    for s, t in equations:
        if s == t:
            continue
        rs = equation_to_rules(s, t)
        if not rs:
            return OrderedResult(
                status=OrderedStatus.NO_RULES_FROM_INPUT,
                rules=[], wall_seconds=time.time() - t_start,
            )
        input_rules.extend(rs)

    pending: list[tuple[Term, Term]] = []   # (s, t) pairs waiting to be processed
    rules: list[Rule] = list(input_rules)
    n_iter = 0
    n_cps = 0
    n_cps_dropped = 0
    n_dormant = 0

    # Seed pending with critical pairs from the initial rules.
    def seed_cps(initial_rules: list[Rule]) -> None:
        nonlocal n_cps, n_cps_dropped
        for i, r_i in enumerate(initial_rules):
            for j, r_j in enumerate(initial_rules):
                same = (i == j)
                for cp_l, cp_r in critical_pairs(r_i, r_j, same_rule=same):
                    n_cps += 1
                    if term_size(cp_l) > max_term_size or term_size(cp_r) > max_term_size:
                        n_cps_dropped += 1
                        continue
                    pending.append((cp_l, cp_r))

    use_alarm = wall_seconds is not None and wall_seconds > 0
    old_handler = None
    if use_alarm:
        old_handler = _signal.signal(_signal.SIGALRM, _alarm_handler)
        _signal.setitimer(_signal.ITIMER_REAL, wall_seconds)

    try:
        seed_cps(input_rules)
        while pending:
            n_iter += 1
            elapsed = time.time() - t_start
            if len(rules) > max_rules:
                return OrderedResult(
                    OrderedStatus.BUDGET_RULES, rules,
                    n_critical_pairs_considered=n_cps,
                    n_critical_pairs_dropped_by_size_cap=n_cps_dropped,
                    n_iterations=n_iter, n_equations_kept_as_dormant=n_dormant,
                    wall_seconds=elapsed,
                )
            if n_cps > max_critical_pairs:
                return OrderedResult(
                    OrderedStatus.BUDGET_CPS, rules,
                    n_critical_pairs_considered=n_cps,
                    n_critical_pairs_dropped_by_size_cap=n_cps_dropped,
                    n_iterations=n_iter, n_equations_kept_as_dormant=n_dormant,
                    wall_seconds=elapsed,
                )

            s, t = pending.pop(0)
            s = ordered_normalize(s, rules, precedence)
            t = ordered_normalize(t, rules, precedence)
            if s == t:
                continue
            # Turn the new equation into rules.
            new_rules = equation_to_rules(s, t)
            if not new_rules:
                # Variable-disjoint CP — dormant. Cannot contribute.
                n_dormant += 1
                continue
            # Add the new rules, compute fresh CPs against ALL rules
            # (including the new ones for self-overlap).
            old_count = len(rules)
            rules.extend(new_rules)
            for new_rule in new_rules:
                for r in rules:
                    same = (r is new_rule)
                    for cp_l, cp_r in critical_pairs(new_rule, r, same_rule=same):
                        n_cps += 1
                        if term_size(cp_l) > max_term_size or term_size(cp_r) > max_term_size:
                            n_cps_dropped += 1
                            continue
                        pending.append((cp_l, cp_r))
                    if r is not new_rule:
                        for cp_l, cp_r in critical_pairs(r, new_rule):
                            n_cps += 1
                            if term_size(cp_l) > max_term_size or term_size(cp_r) > max_term_size:
                                n_cps_dropped += 1
                                continue
                            pending.append((cp_l, cp_r))

        return OrderedResult(
            OrderedStatus.COMPLETED, rules,
            n_critical_pairs_considered=n_cps,
            n_critical_pairs_dropped_by_size_cap=n_cps_dropped,
            n_iterations=n_iter, n_equations_kept_as_dormant=n_dormant,
            wall_seconds=time.time() - t_start,
        )
    except _CompletionTimeout:
        return OrderedResult(
            OrderedStatus.BUDGET_WALL, rules,
            n_critical_pairs_considered=n_cps,
            n_critical_pairs_dropped_by_size_cap=n_cps_dropped,
            n_iterations=n_iter, n_equations_kept_as_dormant=n_dormant,
            wall_seconds=time.time() - t_start,
        )
    finally:
        if use_alarm:
            _signal.setitimer(_signal.ITIMER_REAL, 0)
            _signal.signal(_signal.SIGALRM, old_handler)
