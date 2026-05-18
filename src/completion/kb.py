"""Knuth-Bendix completion for magma equations.

Phase 1 of `completion_exploration_plan.md`. Implements the standard
KB completion loop: orient pending equations via LPO + variable
inclusion (`src/rewrite/lpo.py::orient`), compute critical pairs
between rules (via unification, `src/rewrite/unification.py`),
normalize CPs under the current system, and add the unjoinable
ones as new equations.

A run terminates in one of four states (`CompletionStatus`):

  COMPLETED       — all critical pairs are joinable; the rule set
                    is confluent and terminating; it's a decision
                    procedure for the equational theory of the input.
  FAILED_ORIENT   — encountered an equation that orient() can't turn
                    into a rule (variable-disjoint a priori, or
                    symmetric-unorientable under LPO). For ordered
                    completion (Phase 4) this is the recovery point.
  BUDGET_RULES    — exceeded `max_rules`. May or may not have
                    completed with more.
  BUDGET_CPS      — exceeded `max_critical_pairs`.
  BUDGET_WALL     — exceeded `wall_seconds`.

Position semantics in this file:
  A position is a tuple of {0, 1}-valued integers selecting subterms.
  () is the root, (0,) the left subterm, (1,) the right, etc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from src.data.parser import Op, Term, Var
from src.rewrite.lpo import DEFAULT_VAR_PRECEDENCE, orient
from src.rewrite.matching import all_rewrites
from src.rewrite.unification import (
    apply_subst,
    rename_vars,
    unify,
)


Position = tuple[int, ...]
Rule = tuple[Term, Term]


class CompletionStatus(str, Enum):
    COMPLETED = 'completed'
    FAILED_ORIENT = 'failed_orient'
    BUDGET_RULES = 'budget_rules'
    BUDGET_CPS = 'budget_cps'
    BUDGET_WALL = 'budget_wall'


@dataclass
class CompletionResult:
    status: CompletionStatus
    rules: list[Rule]
    n_critical_pairs_considered: int = 0
    n_iterations: int = 0
    wall_seconds: float = 0.0
    failed_equation: tuple[Term, Term] | None = None
    # For Phase 5: ground normal forms of the completed system.
    # Populated only when status == COMPLETED and the caller requests it.
    ground_normal_forms_finite: bool | None = None
    ground_normal_forms: list[Term] = field(default_factory=list)


# ============================================================
# Position-based subterm access
# ============================================================

def non_var_positions(t: Term) -> list[Position]:
    """All positions in t whose subterm is an Op (non-variable).
    Includes the root if t is an Op."""
    if isinstance(t, Var):
        return []
    out: list[Position] = [()]
    if isinstance(t, Op):
        for p in non_var_positions(t.left):
            out.append((0,) + p)
        for p in non_var_positions(t.right):
            out.append((1,) + p)
    return out


def subterm_at(t: Term, p: Position) -> Term:
    """Subterm of t at position p."""
    if not p:
        return t
    if isinstance(t, Op):
        if p[0] == 0:
            return subterm_at(t.left, p[1:])
        if p[0] == 1:
            return subterm_at(t.right, p[1:])
    raise ValueError(f'invalid position {p} in {t}')


def replace_at(t: Term, p: Position, new: Term) -> Term:
    """Term identical to t except the subterm at position p is `new`."""
    if not p:
        return new
    if isinstance(t, Op):
        if p[0] == 0:
            return Op(replace_at(t.left, p[1:], new), t.right)
        if p[0] == 1:
            return Op(t.left, replace_at(t.right, p[1:], new))
    raise ValueError(f'invalid position {p} in {t}')


# ============================================================
# Normalization
# ============================================================

def normalize(t: Term, rules: list[Rule]) -> Term:
    """Reduce t to a normal form under the current rule set.
    Terminating provided every rule l → r in `rules` satisfies LPO
    (l > r) and variable inclusion (Vars(r) ⊆ Vars(l)) — both
    enforced by `orient()` in `src/rewrite/lpo.py`."""
    while True:
        applied = False
        for l, r in rules:
            rewrites = list(all_rewrites(t, l, r))
            if rewrites:
                t = rewrites[0]
                applied = True
                break
        if not applied:
            return t


def is_joinable(s: Term, t: Term, rules: list[Rule]) -> bool:
    """Are s and t reducible to the same normal form under `rules`?
    A confluent and terminating system makes this complete; during
    completion the system may not be confluent yet, so a False answer
    just means "we need to add this equation"."""
    return normalize(s, rules) == normalize(t, rules)


# ============================================================
# Critical pairs
# ============================================================

def critical_pairs(
    rule1: Rule, rule2: Rule, same_rule: bool = False,
) -> list[tuple[Term, Term]]:
    """All critical pairs from overlapping `rule1`'s LHS onto `rule2`'s
    LHS at non-variable positions.

    For each non-var position p in `rule2.lhs` whose subterm unifies
    with `rule1.lhs` (after renaming variables to be disjoint), the
    critical pair is
        ( σ(rule2.rhs), σ(rule2.lhs[p ← rule1.rhs]) )
    where σ = mgu(rule1.lhs, rule2.lhs|p).

    `same_rule=True` skips the root position to avoid the trivial
    self-overlap that produces a pair like (σ(r), σ(r)).
    """
    l1, r1 = rule1
    l2, r2 = rule2
    l1r = rename_vars(l1, '_1')
    r1r = rename_vars(r1, '_1')
    l2r = rename_vars(l2, '_2')
    r2r = rename_vars(r2, '_2')

    pairs: list[tuple[Term, Term]] = []
    for p in non_var_positions(l2r):
        if same_rule and p == ():
            continue
        sub = subterm_at(l2r, p)
        sigma = unify(l1r, sub)
        if sigma is None:
            continue
        replaced = replace_at(l2r, p, r1r)
        cp_left = apply_subst(r2r, sigma)
        cp_right = apply_subst(replaced, sigma)
        if cp_left != cp_right:
            pairs.append((cp_left, cp_right))
    return pairs


# ============================================================
# Completion loop
# ============================================================

def complete(
    equations: list[tuple[Term, Term]],
    max_rules: int = 100,
    max_critical_pairs: int = 10_000,
    wall_seconds: float = 30.0,
    precedence: tuple[str, ...] = DEFAULT_VAR_PRECEDENCE,
) -> CompletionResult:
    """Run KB completion on the given equations.

    Per Methodology Resolution 4, three independent caps: max_rules,
    max_critical_pairs, wall_seconds. Hitting any one classifies as
    "did not complete within budget."

    Algorithm:
      pending ← list of input equations
      rules ← []
      while pending:
        check budgets
        (s, t) ← pending.pop()
        s, t ← normalize under rules
        if s == t: continue (joinable)
        oriented ← orient(s, t)
        if oriented is None: return FAILED_ORIENT
        new_rule ← oriented
        # Inter-reduce: rewrite old rules' RHSs with new rule;
        # if a rule's LHS itself gets rewritten by new_rule, kick
        # the rule back to pending as an equation.
        kept_rules ← []
        for (l, r) in rules:
          new_l ← normalize(l, [new_rule])
          new_r ← normalize(r, [new_rule])
          if new_l != l:
            pending.append((new_l, new_r))
          else:
            kept_rules.append((l, new_r))
        rules ← kept_rules + [new_rule]
        # Compute critical pairs between new_rule and every rule.
        for (l, r) in rules:
          for cp in critical_pairs(new_rule, (l, r), same_rule=(...)):
            pending.append(cp)
          if (l, r) is not new_rule:
            for cp in critical_pairs((l, r), new_rule):
              pending.append(cp)
      return COMPLETED
    """
    t_start = time.time()
    pending: list[tuple[Term, Term]] = list(equations)
    rules: list[Rule] = []
    n_iter = 0
    n_cps = 0

    while pending:
        n_iter += 1
        elapsed = time.time() - t_start
        if elapsed > wall_seconds:
            return CompletionResult(
                CompletionStatus.BUDGET_WALL, rules,
                n_critical_pairs_considered=n_cps, n_iterations=n_iter,
                wall_seconds=elapsed,
            )
        if len(rules) > max_rules:
            return CompletionResult(
                CompletionStatus.BUDGET_RULES, rules,
                n_critical_pairs_considered=n_cps, n_iterations=n_iter,
                wall_seconds=elapsed,
            )
        if n_cps > max_critical_pairs:
            return CompletionResult(
                CompletionStatus.BUDGET_CPS, rules,
                n_critical_pairs_considered=n_cps, n_iterations=n_iter,
                wall_seconds=elapsed,
            )

        s, t = pending.pop(0)
        s = normalize(s, rules)
        t = normalize(t, rules)
        if s == t:
            continue
        oriented = orient(s, t, precedence)
        if oriented is None:
            return CompletionResult(
                CompletionStatus.FAILED_ORIENT, rules,
                n_critical_pairs_considered=n_cps, n_iterations=n_iter,
                wall_seconds=time.time() - t_start,
                failed_equation=(s, t),
            )
        new_rule = oriented

        # Inter-reduction: simplify existing rules with the new rule.
        kept_rules: list[Rule] = []
        for (l, r) in rules:
            # If new_rule rewrites l, the old rule is subsumed; turn back
            # into a pending equation.
            l_after = normalize(l, [new_rule])
            r_after = normalize(r, [new_rule])
            if l_after != l:
                pending.append((l_after, r_after))
            else:
                kept_rules.append((l, r_after))
        rules = kept_rules + [new_rule]

        # Compute critical pairs between new_rule and every rule
        # (including itself for non-root overlaps).
        for r in rules:
            for cp in critical_pairs(new_rule, r, same_rule=(r is new_rule)):
                pending.append(cp)
                n_cps += 1
            if r is not new_rule:
                for cp in critical_pairs(r, new_rule):
                    pending.append(cp)
                    n_cps += 1

    return CompletionResult(
        CompletionStatus.COMPLETED, rules,
        n_critical_pairs_considered=n_cps, n_iterations=n_iter,
        wall_seconds=time.time() - t_start,
    )
