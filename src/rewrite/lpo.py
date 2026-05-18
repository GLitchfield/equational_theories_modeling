"""Lexicographic Path Order (LPO) for magma terms.

A reduction order is a well-founded partial order on terms, compatible
with operations and substitutions, used to orient equations into
rewrite rules (the greater side becomes the LHS). LPO is the standard
choice for term rewriting when the signature is small and orientation
behavior is predictable.

For our setting (a single binary operator '◇' plus variables), the
order definition simplifies to:

  Precedence: 'Op > Var'. The only function symbol is '◇'; there is
              no precedence among variables (variables are incomparable
              to each other under strict LPO — this is what makes the
              order stable under substitution).

  s >_lpo t  iff  s != t  AND one of:

    (1) s is Op and t is Var. Op > Var by precedence convention. This
        subsumes the textbook "t ∈ Vars(s)" subterm case AND the
        out-of-term case; the combination matters only when combined
        with variable inclusion, checked separately in `orient()`.

    (3) s = (s1 ◇ s2), t any term, and some s_i ≥_lpo t (subterm rule).

    (4) Both s, t are Op; (s1, s2) > (t1, t2) lexicographically
        (s1 >_lpo t1, or s1 = t1 and s2 >_lpo t2); AND s >_lpo t1,
        s >_lpo t2.

Variables are incomparable: `lpo_gt(Var('x'), Var('y'))` is False
for any x ≠ y, even when one alphabetically precedes the other. This
is the standard strict-LPO definition and the only one stable under
substitution — which means a rule oriented by LPO at the abstract
level remains LPO-decreasing on every ground instance. Without
variable incomparability, an "orientation" of commutativity like
`y ◇ x → x ◇ y` is unstable: applying it to `(a ◇ b)` gives `(b ◇ a)`,
and applying it again gives back `(a ◇ b)` — non-terminating.
Symmetric equations (commutativity-like) MUST go to ordered completion
(Phase 4) for proper handling.

The `precedence` parameter is kept for downstream use (ordered
completion in Phase 4 uses per-instance variable comparisons) but
plays no role in the plain-LPO comparison here.

Termination of rewriting under R = {l_i → r_i} is guaranteed if every
l_i >_lpo r_i AND Vars(r_i) ⊆ Vars(l_i) — both checked by `orient()`.

This file is a Phase 1 component of the completion exploration plan.
"""

from __future__ import annotations

from src.data.parser import Op, Term, Var
from src.rewrite.unification import collect_vars

# Default variable precedence — alphabetical over the ETP alphabet.
DEFAULT_VAR_PRECEDENCE: tuple[str, ...] = ('u', 'v', 'w', 'x', 'y', 'z')


def _var_index(v: Var, precedence: tuple[str, ...]) -> int:
    """Position of `v` in the precedence tuple. Variables not in the
    precedence are appended with stable lexicographic order at the
    higher end (so they're always greater than declared variables)."""
    try:
        return precedence.index(v.name)
    except ValueError:
        # Variables outside the alphabet — order by their name lexicographically.
        return len(precedence) + ord(v.name[0])


def lpo_gt(s: Term, t: Term, precedence: tuple[str, ...] = DEFAULT_VAR_PRECEDENCE) -> bool:
    """Return True if s >_lpo t under strict LPO with the 'Op > Var'
    convention. Variables are incomparable to each other — this is
    what makes the order stable under substitution. The `precedence`
    parameter is unused here; it's kept for downstream use by ordered
    completion (Phase 4), which needs per-instance comparisons."""
    if s == t:
        return False

    # Variables are incomparable to each other under strict LPO.
    if isinstance(s, Var) and isinstance(t, Var):
        return False

    # Case Var vs Op: a variable is never greater than an Op under
    # 'Op > Var' precedence.
    if isinstance(s, Var):
        return False

    # s is Op from here.
    assert isinstance(s, Op)

    # Case Op vs Var: by 'Op > Var' precedence, always True.
    if isinstance(t, Var):
        return True

    # Both are Op (single function symbol, same head).
    assert isinstance(t, Op)
    s1, s2 = s.left, s.right
    t1, t2 = t.left, t.right

    # Case (3): some subterm s_i ≥_lpo t.
    if (s1 == t or lpo_gt(s1, t, precedence)) or \
       (s2 == t or lpo_gt(s2, t, precedence)):
        return True

    # Case (4): same head, lexicographic on subterms AND s > each t_i.
    lex_gt = (
        lpo_gt(s1, t1, precedence)
        or (s1 == t1 and lpo_gt(s2, t2, precedence))
    )
    if lex_gt:
        if lpo_gt(s, t1, precedence) and lpo_gt(s, t2, precedence):
            return True

    return False


def lpo_compare(s: Term, t: Term, precedence: tuple[str, ...] = DEFAULT_VAR_PRECEDENCE) -> int:
    """Return +1 if s >_lpo t, -1 if t >_lpo s, 0 if incomparable
    (i.e., neither side strictly greater). For total well-foundedness
    on the rewrite side, callers should also check variable-inclusion
    (`Vars(r) ⊆ Vars(l)`) separately."""
    if s == t:
        return 0
    if lpo_gt(s, t, precedence):
        return 1
    if lpo_gt(t, s, precedence):
        return -1
    return 0


def orient(
    lhs: Term, rhs: Term,
    precedence: tuple[str, ...] = DEFAULT_VAR_PRECEDENCE,
) -> tuple[Term, Term] | None:
    """Orient an equation lhs = rhs into a rewrite rule l → r such that
    l >_lpo r AND Vars(r) ⊆ Vars(l). Returns (l, r) or None if neither
    direction yields a valid rule.

    Variable-inclusion is checked first because it's the cheaper and
    stricter test; LPO comparison only runs on directions that pass it.
    """
    v_lhs = {v.name for v in collect_vars(lhs)}
    v_rhs = {v.name for v in collect_vars(rhs)}
    lhs_to_rhs_ok = v_rhs <= v_lhs
    rhs_to_lhs_ok = v_lhs <= v_rhs

    if not (lhs_to_rhs_ok or rhs_to_lhs_ok):
        # Variable-disjoint or incomparable — Phase 0 hard core.
        return None

    if lhs_to_rhs_ok and lpo_gt(lhs, rhs, precedence):
        return (lhs, rhs)
    if rhs_to_lhs_ok and lpo_gt(rhs, lhs, precedence):
        return (rhs, lhs)
    # Variable inclusion holds in at least one direction, but LPO doesn't
    # strictly orient. This is the "symmetric / order-unorientable" case
    # — ordered completion (Phase 4) handles it.
    return None
