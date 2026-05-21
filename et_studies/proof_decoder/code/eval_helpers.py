"""Parse generated decoder token sequences and compute the nested metric
stack (parse rate -> arity-correct -> σ-exact-match) per Methodology
Resolution 3 of `proof_decoder_plan.md`.

Generated proofs are expected to be:
  Direct:  exact fun v_{p_1} v_{p_2} ... v_{p_N} => h v_{a_1} v_{a_2} ... v_{a_K}
  Flipped: exact fun v_{p_1} ... v_{p_N} => ( h v_{a_1} ... v_{a_K} ) .symm

Where N = E2's arity and K = E1's arity. The λ-binders are v1, v2, ...,
v_N in order (so {p_1, ..., p_N} = {1, ..., N}). The h-arguments are
the substitution σ in normalized-E2 indices (so each v_{a_i} ∈ {v_1, ..., v_N}).
"""

from __future__ import annotations

from dataclasses import dataclass

from et_studies.proof_decoder.code.tokenizer import (
    BOS_ID, EOS_ID, ID_TO_TOKEN, PAD_ID, TOKEN_TO_ID,
)


@dataclass(frozen=True)
class ParsedProof:
    orientation: int          # 0 = direct, 1 = flipped
    e2_arity: int             # number of λ-binders
    substitution: tuple[int, ...]  # 0-indexed E2-var indices, one per E1 arg


def _strip_special(tokens: list[int]) -> list[int]:
    """Drop BOS/EOS/PAD from a token list."""
    return [t for t in tokens if t not in (BOS_ID, EOS_ID, PAD_ID)]


_V_TOKENS = {TOKEN_TO_ID[f'v{i}']: i for i in range(1, 7)}   # token_id -> arity index (1..6)


def parse_proof_tokens(tokens: list[int]) -> ParsedProof | None:
    """Parse a generated token sequence and extract (orientation, e2_arity, σ).

    Returns None on any structural failure. This is the "parse rate"
    gate of the nested metric stack — generated outputs that don't fit
    the expected `exact fun ... => h ...` shape return None.
    """
    toks = _strip_special(tokens)
    if not toks:
        return None
    expect = lambda i, t: i < len(toks) and toks[i] == TOKEN_TO_ID[t]   # noqa: E731

    if not expect(0, 'exact'):
        return None
    if not expect(1, 'fun'):
        return None

    # Read λ-binders until `=>`.
    i = 2
    binders: list[int] = []
    while i < len(toks) and toks[i] != TOKEN_TO_ID['=>']:
        if toks[i] not in _V_TOKENS:
            return None
        binders.append(_V_TOKENS[toks[i]])
        i += 1
    if i == len(toks):                  # no `=>` found
        return None
    if not binders:                     # zero-arity not supported by this grammar
        return None
    # Binders must be exactly v_1, v_2, ..., v_N in order.
    e2_arity = len(binders)
    if binders != list(range(1, e2_arity + 1)):
        return None
    i += 1                              # past `=>`

    if i >= len(toks):
        return None

    # Now: either `h v_{a_1} ... v_{a_K}` (direct) or `( h v_{a_1} ... v_{a_K} ) .symm` (flipped).
    orientation = 0
    if toks[i] == TOKEN_TO_ID['(']:
        orientation = 1
        i += 1

    if not expect(i, 'h'):
        return None
    i += 1

    # Read h-arguments (must all be v_i tokens).
    args: list[int] = []
    while i < len(toks) and toks[i] in _V_TOKENS:
        v_idx = _V_TOKENS[toks[i]]
        # Substitution stores 0-indexed E2 var positions; the v_j token
        # has arity index j (1-based), so subtract 1.
        if v_idx > e2_arity:
            return None           # σ-value exceeds bound λ-binders
        args.append(v_idx - 1)
        i += 1

    if orientation == 1:
        # Must close with `)` then `.symm`.
        if not expect(i, ')'):
            return None
        i += 1
        if not expect(i, '.symm'):
            return None
        i += 1
    # After this point, anything else is trailing garbage — reject.
    if i != len(toks):
        return None

    return ParsedProof(
        orientation=orientation,
        e2_arity=e2_arity,
        substitution=tuple(args),
    )


@dataclass(frozen=True)
class TripleMetrics:
    """One row's contribution to the nested metric stack."""
    parsed: bool          # did the output parse at all?
    arity_correct: bool   # did binders + h-arg count match ground truth?
    sigma_exact: bool     # did substitution + orientation match ground truth?

    @classmethod
    def from_parse(
        cls,
        parsed: ParsedProof | None,
        gt_orientation: int,
        gt_e1_arity: int,
        gt_e2_arity: int,
        gt_substitution: tuple[int, ...],
    ) -> 'TripleMetrics':
        if parsed is None:
            return cls(parsed=False, arity_correct=False, sigma_exact=False)
        # Arity-correct: λ-binders count == gt_e2_arity AND h-arg count == gt_e1_arity.
        arity_ok = (
            parsed.e2_arity == gt_e2_arity
            and len(parsed.substitution) == gt_e1_arity
        )
        if not arity_ok:
            return cls(parsed=True, arity_correct=False, sigma_exact=False)
        # σ-exact: orientation matches AND every substitution entry matches.
        sigma_ok = (
            parsed.orientation == gt_orientation
            and parsed.substitution == gt_substitution
        )
        return cls(parsed=True, arity_correct=True, sigma_exact=sigma_ok)
