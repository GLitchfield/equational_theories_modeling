"""First-order unification over magma terms.

Distinct from `src/rewrite/matching.py::first_order_match`, which is
one-sided (pattern's variables bind to subject's subterms; subject's
variables are constants). Unification is two-sided: both sides'
variables are subject to binding.

Used in Phase 1 of completion (computing critical pairs between two
rules whose variables have been made disjoint by renaming).

A substitution σ is a dict mapping `Var` → `Term`. `apply_subst(t, σ)`
walks t and replaces each Var v by σ[v] (recursively if σ[v] itself
contains other vars, until fixpoint via apply_subst's recursion).

`unify(s, t)` returns the most-general unifier σ such that
`apply_subst(s, σ) == apply_subst(t, σ)`, or None if s and t are
not unifiable.
"""

from __future__ import annotations

from src.data.parser import Op, Term, Var

Substitution = dict[Var, Term]


def apply_subst(t: Term, sigma: Substitution) -> Term:
    """Apply σ to t, recursively replacing any bound variable until
    fixpoint."""
    if isinstance(t, Var):
        if t in sigma:
            return apply_subst(sigma[t], sigma)
        return t
    if isinstance(t, Op):
        return Op(apply_subst(t.left, sigma), apply_subst(t.right, sigma))
    raise TypeError(t)


def occurs(v: Var, t: Term, sigma: Substitution | None = None) -> bool:
    """Does variable v appear anywhere in t (after applying σ)?"""
    if sigma is not None:
        t = apply_subst(t, sigma)
    if isinstance(t, Var):
        return t == v
    if isinstance(t, Op):
        return occurs(v, t.left) or occurs(v, t.right)
    raise TypeError(t)


def _compose(sigma: Substitution, v: Var, t: Term) -> Substitution:
    """Extend σ with the binding v → t, propagating through existing
    bindings so that the result is in solved form (no variable on the
    left also appears on the right of another binding)."""
    inner = {v: t}
    new = {x: apply_subst(s, inner) for x, s in sigma.items()}
    new[v] = t
    return new


def unify(s: Term, t: Term, sigma: Substitution | None = None) -> Substitution | None:
    """Robinson unification with occurs-check. Returns the most-general
    unifier σ such that `apply_subst(s, σ) == apply_subst(t, σ)`, or
    None if not unifiable.

    If `sigma` is given, extends it; otherwise starts from the empty
    substitution. The caller MUST ensure s and t have disjoint variable
    sets (rename one set if necessary). Otherwise unification may
    produce a "wrong" answer due to accidental variable identity.
    """
    if sigma is None:
        sigma = {}
    s = apply_subst(s, sigma)
    t = apply_subst(t, sigma)
    if s == t:
        return sigma
    if isinstance(s, Var):
        if occurs(s, t):
            return None
        return _compose(sigma, s, t)
    if isinstance(t, Var):
        if occurs(t, s):
            return None
        return _compose(sigma, t, s)
    if isinstance(s, Op) and isinstance(t, Op):
        sigma = unify(s.left, t.left, sigma)
        if sigma is None:
            return None
        return unify(s.right, t.right, sigma)
    return None


def rename_vars(t: Term, suffix: str) -> Term:
    """Append `suffix` to every variable name in t. Used to make two
    rules' variable sets disjoint before unifying them.

    Note: ETP variables are single chars from {u, v, w, x, y, z}, so
    appending a non-conflicting suffix (e.g. '_1') is safe."""
    if isinstance(t, Var):
        return Var(t.name + suffix)
    if isinstance(t, Op):
        return Op(rename_vars(t.left, suffix), rename_vars(t.right, suffix))
    raise TypeError(t)


def collect_vars(t: Term) -> frozenset[Var]:
    if isinstance(t, Var):
        return frozenset({t})
    if isinstance(t, Op):
        return collect_vars(t.left) | collect_vars(t.right)
    raise TypeError(t)
