"""First-order matching and substitution for the rewrite engine.

Per Methodology Resolution 1 in `rewrite_chain_study_plan.md`: the
rewrite engine needs first-order matching, not the matcher's
structural-match. Standard rewriting semantics:

  Given a rule LHS containing rule-variables (`a, b, c, ...`) and a
  subject term that's ground (with respect to the rule's variables —
  in the equational implication setting, the subject is built from
  the goal equation's variables `x, y, z, ...`, which the matcher
  treats as constants relative to the rule's pattern variables),
  find σ such that σ(pattern) ≡ subject.

  Crucially, σ can map a rule variable to ANY term — a variable,
  an operator application, anything. The matcher's `structural_match`
  prohibits variable-to-compound mappings because its scope is
  variable-RENAMING (the variable-substitution category in the proof
  survey was defined by var-to-var maps). Rewriting has no such
  restriction.

Same shape as `src/matcher/matcher.py::structural_match`, one rule
changed: when the pattern is a variable, bind to whatever the
subject is (consistency-check across re-occurrences). When the
pattern is an Op, the subject must also be an Op with matching
substructure.
"""

from __future__ import annotations

from lib.data.parser import Op, Term, Var


def first_order_match(
    pattern: Term,
    subject: Term,
    sigma: dict[str, Term],
) -> bool:
    """Match `pattern` against `subject`, extending `sigma`.

    `sigma` maps rule-variable name → subject Term. The same dict is
    threaded across recursive calls so a rule variable appearing
    multiple times must bind to the same subject subterm (the
    "consistency check").

    Returns True on success (with sigma extended in-place), False on
    structural mismatch or consistency violation. On failure sigma
    may be partially populated; callers should not reuse it across
    failed attempts.
    """
    if isinstance(pattern, Var):
        # Pattern variable: bind to subject (any Term).
        existing = sigma.get(pattern.name)
        if existing is not None:
            return existing == subject
        sigma[pattern.name] = subject
        return True
    # Pattern is Op: subject must be Op, with matching substructure.
    if isinstance(subject, Op):
        return (
            first_order_match(pattern.left, subject.left, sigma)
            and first_order_match(pattern.right, subject.right, sigma)
        )
    # Pattern Op vs subject Var: an operator pattern requires the subject
    # to have the same operator at the top, so it fails.
    return False


def substitute(t: Term, sigma: dict[str, Term]) -> Term:
    """Apply a substitution σ to a term.

    Rule variables in `t` whose name is in `sigma` are replaced by
    the corresponding subject term. Names not in `sigma` are left
    as-is (this is what you want for "use σ from a match, apply to
    the rule's RHS to get the rewritten term").
    """
    if isinstance(t, Var):
        return sigma.get(t.name, t)
    return Op(substitute(t.left, sigma), substitute(t.right, sigma))


def rewrite_at(
    term: Term,
    pattern: Term,
    replacement: Term,
) -> Term | None:
    """One-shot rewrite at the ROOT of `term`. Returns the rewritten
    term if `pattern` matches `term`; None if it doesn't.

    `pattern` and `replacement` come from a rewrite rule (e.g.,
    `pattern = E1.lhs`, `replacement = E1.rhs` for forward rewriting
    with E1).
    """
    sigma: dict[str, Term] = {}
    if not first_order_match(pattern, term, sigma):
        return None
    return substitute(replacement, sigma)


def all_rewrites(
    term: Term,
    pattern: Term,
    replacement: Term,
) -> list[Term]:
    """Generate every one-step rewrite of `term` via the rule
    `pattern → replacement`, applied at EVERY subterm position.

    Returns a list of rewritten Terms (one per matching position).
    Order is deterministic (pre-order walk).
    """
    out: list[Term] = []
    _walk_and_rewrite(term, pattern, replacement, out)
    return out


def _walk_and_rewrite(
    here: Term,
    pattern: Term,
    replacement: Term,
    out: list[Term],
) -> None:
    """Helper: at every node, try to rewrite at the root; then recurse
    into children, reconstructing the whole term with the rewritten
    child substituted at that position."""
    # Try the rewrite at this position.
    sigma: dict[str, Term] = {}
    if first_order_match(pattern, here, sigma):
        out.append(substitute(replacement, sigma))
    # Recurse into children (rebuild around the rewritten child).
    if isinstance(here, Op):
        left_rewrites: list[Term] = []
        _walk_and_rewrite(here.left, pattern, replacement, left_rewrites)
        for new_left in left_rewrites:
            out.append(Op(new_left, here.right))
        right_rewrites: list[Term] = []
        _walk_and_rewrite(here.right, pattern, replacement, right_rewrites)
        for new_right in right_rewrites:
            out.append(Op(here.left, new_right))


def term_size(t: Term) -> int:
    """Count of nodes (variables + operator applications) in a term."""
    if isinstance(t, Var):
        return 1
    return 1 + term_size(t.left) + term_size(t.right)
