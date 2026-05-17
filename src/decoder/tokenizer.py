"""Phase 1 tokenizer for the proof decoder.

Vocab (16 tokens, per Methodology Resolution 1 in
`proof_decoder_plan.md`): 13 proof-emitting tokens plus 3 special.

The decoder generates only the proof body (`exact fun <vars> => h
<args>` with optional `.symm` wrap). The judge's preamble
`intro G _ h` is outside the decoder's output, so `intro`, `G`,
and `_` are NOT in the vocab.

Token IDs are deterministic and stable across runs — they form the
softmax labels for the autoregressive decoder.
"""

from __future__ import annotations

import re

# Special tokens get the low IDs by convention so PAD=0 (used by
# torch.nn.CrossEntropyLoss(ignore_index=0)).
PAD = '<PAD>'
BOS = '<BOS>'
EOS = '<EOS>'

# Vocab list, in fixed order. ID = index in this list.
VOCAB: tuple[str, ...] = (
    PAD,        # 0 - padding (also used to mask loss)
    BOS,        # 1 - beginning of sequence
    EOS,        # 2 - end of sequence
    'exact',    # 3 - instruction
    'fun',      # 4 - structural
    '=>',       # 5 - structural
    'h',        # 6 - hypothesis identifier
    '(',        # 7 - structural (only used for .symm wrap)
    ')',        # 8 - structural
    '.symm',    # 9 - structural (flipped orientation)
    'v1',       # 10 - variable 1
    'v2',       # 11 - variable 2
    'v3',       # 12 - variable 3
    'v4',       # 13 - variable 4
    'v5',       # 14 - variable 5
    'v6',       # 15 - variable 6
)

TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN: dict[int, str] = {i: tok for i, tok in enumerate(VOCAB)}

VOCAB_SIZE = len(VOCAB)           # 16
PAD_ID = TOKEN_TO_ID[PAD]         # 0
BOS_ID = TOKEN_TO_ID[BOS]         # 1
EOS_ID = TOKEN_TO_ID[EOS]         # 2

# Matches any of: `exact`, `fun`, `=>`, `h`, `(`, `)`, `.symm`, `v1`-`v6`.
# Longest-match-first: `.symm` before `.`, `v10` would beat `v1` if present
# (but v10 isn't in the vocab; max var index is 6).
_TOKEN_RE = re.compile(
    r'\.symm|=>|exact|fun|v[1-6]|h|\(|\)'
)


def tokenize(proof_body: str) -> list[int]:
    """Tokenize a phase-1 proof body string to a list of token IDs.

    Input is the matcher's normalized proof body (e.g.,
    `exact fun v1 v2 => h v1 v2 v1` or
    `exact fun v1 v2 => (h v1 v2).symm`). The output sequence has
    BOS prepended and EOS appended.

    Raises ValueError if any out-of-vocab character is encountered
    (defensive — caller's normalization should produce only
    in-vocab tokens).
    """
    # Strip leading/trailing whitespace; split by the token regex.
    s = proof_body.strip()
    matches = list(_TOKEN_RE.finditer(s))
    # Sanity: every non-whitespace character must be inside some match.
    consumed = sum(m.end() - m.start() for m in matches)
    non_ws_len = len(re.sub(r'\s+', '', s))
    if consumed != non_ws_len:
        raise ValueError(
            f'tokenization left {non_ws_len - consumed} non-whitespace chars '
            f'unconsumed in {s!r}'
        )
    ids: list[int] = [BOS_ID]
    for m in matches:
        tok = m.group(0)
        if tok not in TOKEN_TO_ID:
            raise ValueError(f'unknown token {tok!r} in {s!r}')
        ids.append(TOKEN_TO_ID[tok])
    ids.append(EOS_ID)
    return ids


def detokenize(ids: list[int], strip_special: bool = True) -> str:
    """Inverse of tokenize: convert token IDs back to a proof body string.

    Spaces are inserted between tokens except for the parenthesis
    pair around `.symm`-wrapped terms (which is purely cosmetic — the
    decoder's downstream consumer parses either form).

    When `strip_special` is True, BOS/EOS/PAD are dropped.
    """
    toks = [ID_TO_TOKEN[i] for i in ids]
    if strip_special:
        toks = [t for t in toks if t not in (PAD, BOS, EOS)]
    # Simple join with spaces; readers can collapse if needed.
    return ' '.join(toks)


def build_normalized_proof(
    e2_arity: int,
    substitution: list[int],
    e1_arity: int,
    orientation: int,
) -> str:
    """Construct the normalized proof body from (substitution, orientation,
    arities), in v1..v6 space.

    Args:
        e2_arity: number of variables in E2 (determines λ-binders).
        substitution: list of length e1_arity, each entry is the index
            (0-based) of σ(e1_var) into E2's variable list.
        e1_arity: number of variables in E1 (length of substitution).
        orientation: 0 = direct, 1 = flipped (wrap in `(...).symm`).

    Returns the normalized proof body string, e.g.
    `exact fun v1 v2 => h v1 v2 v1` or `exact fun v1 v2 => (h v1 v2).symm`.
    """
    if e2_arity < 1 or e2_arity > 6:
        raise ValueError(f'e2_arity must be 1..6; got {e2_arity}')
    if len(substitution) != e1_arity:
        raise ValueError(
            f'substitution length {len(substitution)} != e1_arity {e1_arity}'
        )
    binders = ' '.join(f'v{i + 1}' for i in range(e2_arity))
    if e1_arity == 0:
        # Pathological case (Eq1 → Eq1?); h applies to nothing.
        applied = 'h'
    else:
        args = ' '.join(f'v{substitution[i] + 1}' for i in range(e1_arity))
        applied = f'h {args}'
    if orientation == 1:
        applied = f'({applied}).symm'
    return f'exact fun {binders} => {applied}'
