"""First-order matcher for equational implication.

Given two equations E1 and E2 (each `lhs = rhs` over a binary `◇`
operator and variables `x, y, z, w, u, v`), determines whether E2 is
a **variable-substitution instance** of E1: does there exist a
substitution σ mapping each variable of E1 to a variable of E2 (not
necessarily injective) such that `σ(lhs1) ≡ lhs2` and `σ(rhs1) ≡ rhs2`?

When such a σ exists, E1 implies E2 and the proof is:

    fun <e2_vars> => h <σ(e1_var_1)> <σ(e1_var_2)> ...

This module implements the deterministic algorithm — pure symbolic
computation on parse trees, no learning, no embeddings. It backs
Track 1 of the proof-generator architecture, covering the ~62.5%
of ETP TRUE proofs that are arg-permutation closes (per the
proof-corpus survey).

Key entry points:

  - ``match(eq1, eq2)`` -> Optional[Match]      both orientations
  - ``Match.proof_term()``                      Lean term-mode proof string
  - ``Match.proof_tactic_body()``               Stage 2 tactic-body proof string
  - ``Match.substitution_padded(width=6)``      fixed-length substitution tuple

See ``matcher_plan.md`` for the algorithm and methodology resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.data.parser import Equation, Op, Term, Var, canonical_var_ids


# --- core structural match ---------------------------------------------------

def structural_match(
    t1: Term,
    t2: Term,
    sigma: dict[str, str],
) -> bool:
    """Walk E1-term t1 and E2-term t2 in parallel, extending sigma.

    sigma maps E1-variable-name -> E2-variable-name. The same dict
    is shared across both the LHS and RHS matches so the substitution
    is consistent across both sides of the equation.

    Returns True if t1 can be renamed to t2 consistently with sigma,
    False on any structural mismatch or consistency conflict.

    Mutates sigma on success; on failure sigma may be partially populated
    (callers should not reuse it across attempts).
    """
    if isinstance(t1, Var) and isinstance(t2, Var):
        if t1.name in sigma:
            return sigma[t1.name] == t2.name
        sigma[t1.name] = t2.name
        return True
    if isinstance(t1, Op) and isinstance(t2, Op):
        return (
            structural_match(t1.left, t2.left, sigma)
            and structural_match(t1.right, t2.right, sigma)
        )
    # Var vs Op or Op vs Var -> structural mismatch (a variable would
    # have to map to a compound term, which this matcher prohibits by
    # design — those proofs aren't in the variable-substitution category).
    return False


def _match_oriented(
    e1_lhs: Term, e1_rhs: Term,
    e2_lhs: Term, e2_rhs: Term,
) -> dict[str, str] | None:
    """Try to match (e1_lhs -> e2_lhs, e1_rhs -> e2_rhs) consistently.

    Returns the substitution dict on success, None on failure.
    """
    sigma: dict[str, str] = {}
    if not structural_match(e1_lhs, e2_lhs, sigma):
        return None
    if not structural_match(e1_rhs, e2_rhs, sigma):
        return None
    return sigma


# --- public match API --------------------------------------------------------

@dataclass(frozen=True)
class Match:
    """Result of a successful structural match.

    Attributes:
      orientation: 0 = direct (lhs1 -> lhs2, rhs1 -> rhs2)
                   1 = flipped (lhs1 -> rhs2, rhs1 -> lhs2); proof needs .symm
      sigma:       {e1_var_name -> e2_var_name}, consistent across LHS+RHS
      e1_vars:     E1's variables in canonical (first-appearance) order,
                   matching how Lean's Equation_N binds its universals
      e2_vars:     E2's variables in canonical order, matching how Lean
                   binds the goal equation's universals
    """
    orientation: int                # 0 direct, 1 flipped
    sigma: dict[str, str]
    e1_vars: tuple[str, ...]
    e2_vars: tuple[str, ...]

    @property
    def applied_args(self) -> tuple[str, ...]:
        """The E2-variable names in E1's argument order (for `h <args>`)."""
        return tuple(self.sigma[v] for v in self.e1_vars)

    def proof_term(self) -> str:
        """Lean term-mode proof string.

        Direct:  `fun e2_v1 e2_v2 => h σ(e1_v1) σ(e1_v2) ...`
        Flipped: `fun e2_v1 e2_v2 => (h σ(e1_v1) ...).symm`
        """
        lam_vars = ' '.join(self.e2_vars) if self.e2_vars else '_'
        applied = ' '.join(self.applied_args)
        body = f'h {applied}' if applied else 'h'
        if self.orientation == 1:
            body = f'({body}).symm'
        return f'fun {lam_vars} => {body}'

    def proof_tactic_body(self) -> str:
        """Stage 2 submission body (the judge wraps in `:= by <body>`).

        Format pinned by inspection of
        `tests/fixtures/answers/accepted_true_basic.answer.json` in the
        Stage 2 repo:

            import JudgeProblem

            def submission : Goal := by
              intro G _ h
              <user tactic body>

        After the judge's `intro G _ h`, the user's body inherits a goal
        of `Equation_Y G = ∀ <e2_vars>, lhs2 = rhs2`. The natural body
        is a single `exact fun <e2_vars> => h <σ(e1_vars)>` for the
        direct case (or with `.symm` wrap for the flipped case).

        Live-judge confirmation is pending a built Stage 2 environment
        (Lean 4.30.0-rc2 + mathlib). The format here is structurally
        correct by construction (a value of the goal type) and matches
        the fixture pattern.
        """
        return f'exact {self.proof_term()}'

    def proof_full_code(self) -> str:
        """Stage 2 ``code`` field for the answer JSON.

        Wraps ``proof_tactic_body()`` in the canonical submission shell:

            import JudgeProblem

            def submission : Goal := by
              intro G _ h
              <body>
        """
        body = self.proof_tactic_body()
        # Indent each line of the body by 2 spaces to nest inside the `by`.
        indented = '\n'.join(f'  {line}' for line in body.splitlines())
        return (
            'import JudgeProblem\n'
            '\n'
            'def submission : Goal := by\n'
            '  intro G _ h\n'
            f'{indented}\n'
        )

    def answer_json(self) -> dict:
        """Stage 2 ``{verdict, code}`` answer dict — serialize to JSON
        and pass as ``raw_answer`` to ``judge.verify.verify_answer``."""
        return {'verdict': 'true', 'code': self.proof_full_code()}

    def substitution_padded(self, width: int = 6, sentinel: int = -1) -> tuple[int, ...]:
        """Substitution as a fixed-length-`width` tuple of E2-variable indices.

        Per Methodology Resolution 4: max var count in the ETP is 6, so the
        default width=6 covers all equations with a single sentinel-padded
        layout. Entry i is the index (into e2_vars) of σ(e1_vars[i]), or
        `sentinel` if E1 has fewer than `width` variables.
        """
        e2_idx = {v: i for i, v in enumerate(self.e2_vars)}
        out: list[int] = []
        for i in range(width):
            if i < len(self.e1_vars):
                out.append(e2_idx[self.sigma[self.e1_vars[i]]])
            else:
                out.append(sentinel)
        return tuple(out)


def ordered_vars(eq: Equation) -> tuple[str, ...]:
    """E1/E2's variable names in canonical order (first appearance, LHS pre-order
    then RHS pre-order). Mirrors how the ETP's Equation_N theorems bind their
    universals: ``∀ x y z, ...`` follows the same first-appearance order.
    """
    ids = canonical_var_ids(eq)
    return tuple(sorted(ids, key=ids.get))


def match(eq1: Equation, eq2: Equation) -> Match | None:
    """Try to match E2 as a variable-substitution instance of E1.

    Returns a ``Match`` if such a substitution exists, ``None`` otherwise.

    Tries both orientations:
      0 (direct):  σ(lhs1) = lhs2 and σ(rhs1) = rhs2
      1 (flipped): σ(lhs1) = rhs2 and σ(rhs1) = lhs2 (proof needs .symm)

    Only these two are tried (Methodology Resolution 6) — flipping both
    equations is the same as flipping neither, so they're provably
    sufficient.

    Edge case (Methodology Resolution 8): if E1 is Eq1 (`x = x`) the
    matcher succeeds only when E2 is also of the form `v = v`, since
    Eq1 has a single variable on each side and the structural walk
    forbids a variable mapping to a compound term. So Eq1 -> Eq1 is
    the only Eq1-as-hypothesis pair the matcher catches, which is
    correct (Eq1 implies nothing else in this restricted sense).
    """
    e1_vars = ordered_vars(eq1)
    e2_vars = ordered_vars(eq2)
    # Direct orientation
    sigma = _match_oriented(eq1.lhs, eq1.rhs, eq2.lhs, eq2.rhs)
    if sigma is not None:
        return Match(orientation=0, sigma=sigma, e1_vars=e1_vars, e2_vars=e2_vars)
    # Flipped orientation (proof will need .symm)
    sigma = _match_oriented(eq1.lhs, eq1.rhs, eq2.rhs, eq2.lhs)
    if sigma is not None:
        return Match(orientation=1, sigma=sigma, e1_vars=e1_vars, e2_vars=e2_vars)
    return None
