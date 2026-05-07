"""Parser for equational laws.

Equation syntax (verified against the ETP equations.txt corpus, 4,694 lines):
  - Operator: '◇' (U+25C7 WHITE DIAMOND), always binary, infix
  - Variables: single lowercase letters from {u, v, w, x, y, z}
  - Sub-products are always parenthesized; top-level products never parenthesized
  - 'LHS = RHS' with whitespace-separated tokens; max depth 4

Encoding (deviations from the engineering plan, recorded for reviewability):
  - 6 variable slots (var_id_0..5) instead of the plan's 4 — actual data uses 6.
  - Family-C root-to-root edge gets edge_attr = [0, 0] (neither left nor right);
    keeps the plan's 2-dim edge feature without inventing a third class.
  - is_lhs_var = 1 only when LHS is a single variable AND the RHS variable node
    matches it. For Family-C (LHS is a product), is_lhs_var = 0 everywhere; the
    LHS subtree's presence and the directional root-link encode the distinction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

OPERATOR = '◇'  # ◇
NUM_VAR_SLOTS = 6
NODE_FEATURE_DIM = 2 + NUM_VAR_SLOTS + 1  # is_op, is_var, var_id_0..5, is_lhs_var = 9
EDGE_FEATURE_DIM = 2  # [is_left, is_right]; root-link in Family-C uses [0, 0]


@dataclass(frozen=True)
class Var:
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Op:
    left: 'Term'
    right: 'Term'

    def __repr__(self) -> str:
        return f'({self.left} {OPERATOR} {self.right})'


Term = Union[Var, Op]


@dataclass(frozen=True)
class Equation:
    lhs: Term
    rhs: Term

    def __repr__(self) -> str:
        def inner(t: Term) -> str:
            return t.name if isinstance(t, Var) else f'({inner(t.left)} {OPERATOR} {inner(t.right)})'

        def top(t: Term) -> str:
            return t.name if isinstance(t, Var) else f'{inner(t.left)} {OPERATOR} {inner(t.right)}'

        return f'{top(self.lhs)} = {top(self.rhs)}'


# --- tokenizer -----------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for ch in text:
        if ch.isspace():
            continue
        if ch in '()=' or ch == OPERATOR:
            tokens.append(ch)
        elif 'a' <= ch <= 'z':
            tokens.append(ch)
        else:
            raise ValueError(f'unexpected character {ch!r} in {text!r}')
    return tokens


# --- recursive-descent parser --------------------------------------------------

class _Parser:
    def __init__(self, tokens: list[str], text: str):
        self.tokens = tokens
        self.text = text
        self.pos = 0

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, want: str) -> None:
        tok = self._peek()
        if tok != want:
            raise ValueError(
                f'expected {want!r} at token index {self.pos}, got {tok!r} in {self.text!r}'
            )
        self._advance()

    def _atom(self) -> Term:
        tok = self._peek()
        if tok == '(':
            self._advance()
            node = self._product()
            self._expect(')')
            return node
        if tok is None or not (len(tok) == 1 and 'a' <= tok <= 'z'):
            raise ValueError(
                f'expected variable or "(" at token index {self.pos}, got {tok!r} in {self.text!r}'
            )
        self._advance()
        return Var(tok)

    def _product(self) -> Op:
        # Inside parens: must be `atom ◇ atom`.
        left = self._atom()
        self._expect(OPERATOR)
        right = self._atom()
        return Op(left, right)

    def top_term(self) -> Term:
        # At top level: atom, optionally `◇ atom` to form an unparenthesized product.
        left = self._atom()
        if self._peek() == OPERATOR:
            self._advance()
            right = self._atom()
            return Op(left, right)
        return left


def parse_equation(text: str) -> Equation:
    """Parse equation text into an AST."""
    tokens = _tokenize(text)
    p = _Parser(tokens, text)
    lhs = p.top_term()
    p._expect('=')
    rhs = p.top_term()
    if p.pos != len(tokens):
        raise ValueError(f'trailing tokens after parse: {tokens[p.pos:]} in {text!r}')
    return Equation(lhs=lhs, rhs=rhs)


# --- canonical variable IDs ----------------------------------------------------

def canonical_var_ids(eq: Equation) -> dict[str, int]:
    """DeBruijn-style IDs assigned by first appearance, walking LHS pre-order
    then RHS pre-order. Returns {variable_name -> id in 0..NUM_VAR_SLOTS-1}."""
    ids: dict[str, int] = {}

    def walk(t: Term) -> None:
        if isinstance(t, Var):
            if t.name not in ids:
                ids[t.name] = len(ids)
        else:
            walk(t.left)
            walk(t.right)

    walk(eq.lhs)
    walk(eq.rhs)
    if len(ids) > NUM_VAR_SLOTS:
        raise ValueError(
            f'equation uses {len(ids)} distinct variables, exceeds NUM_VAR_SLOTS={NUM_VAR_SLOTS}'
        )
    return ids


# --- node/edge array construction (pure Python; no torch dep) -----------------

def _var_features(canon_id: int, is_lhs_var: bool) -> list[float]:
    feat = [0.0] * NODE_FEATURE_DIM
    feat[1] = 1.0  # is_var
    feat[2 + canon_id] = 1.0  # var_id one-hot
    if is_lhs_var:
        feat[2 + NUM_VAR_SLOTS] = 1.0
    return feat


def _op_features() -> list[float]:
    feat = [0.0] * NODE_FEATURE_DIM
    feat[0] = 1.0
    return feat


def equation_to_arrays(
    text: str,
) -> tuple[list[list[float]], list[tuple[int, int]], list[list[float]]]:
    """Parse equation text and emit (node_features, edges, edge_attrs) as lists.

    Encoding rules:
      - bare-LHS (LHS is a single variable): emit only the RHS subtree; var nodes
        in the RHS whose name matches the LHS variable get is_lhs_var=1.
      - Family-C (LHS is a product): emit the LHS subtree, then the RHS subtree,
        then a root-to-root edge from LHS_root to RHS_root with edge_attr=[0,0].
        is_lhs_var=0 throughout.

    Edges are parent→child as the plan specifies; edge_attr is one-hot
    [is_left, is_right] for tree edges, and [0, 0] for the Family-C root-link.
    """
    eq = parse_equation(text)
    var_ids = canonical_var_ids(eq)
    bare_lhs_var = eq.lhs.name if isinstance(eq.lhs, Var) else None

    node_feats: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    edge_attrs: list[list[float]] = []

    def emit(t: Term) -> int:
        if isinstance(t, Var):
            is_lhs_v = bare_lhs_var is not None and t.name == bare_lhs_var
            idx = len(node_feats)
            node_feats.append(_var_features(var_ids[t.name], is_lhs_v))
            return idx
        idx = len(node_feats)
        node_feats.append(_op_features())
        l = emit(t.left)
        r = emit(t.right)
        edges.append((idx, l))
        edge_attrs.append([1.0, 0.0])
        edges.append((idx, r))
        edge_attrs.append([0.0, 1.0])
        return idx

    if bare_lhs_var is not None:
        emit(eq.rhs)
    else:
        lhs_root = emit(eq.lhs)
        rhs_root = emit(eq.rhs)
        edges.append((lhs_root, rhs_root))
        edge_attrs.append([0.0, 0.0])

    return node_feats, edges, edge_attrs


# --- PyG wrapper (lazy torch import) ------------------------------------------

def equation_to_pyg(text: str):
    """Parse equation text and return a PyG ``Data`` object.

    Imports torch and torch_geometric on call so the rest of the module stays
    usable in environments without those installed (e.g. unit tests).
    """
    import torch
    from torch_geometric.data import Data

    node_feats, edges, edge_attrs = equation_to_arrays(text)
    x = torch.tensor(node_feats, dtype=torch.float)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
