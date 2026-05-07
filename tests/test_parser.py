"""Tests for src/data/parser.py.

Pure-Python tests — no torch dependency. Covers tokenize, parse, canonical
variable IDs, and the array emission logic. The PyG-tensor wrapper
(equation_to_pyg) is exercised optionally via importorskip when torch is
present.

Runnable two ways:
  - ``pytest tests/test_parser.py``
  - ``python tests/test_parser.py`` (uses the __main__ driver below)
"""

from __future__ import annotations

import os
import sys

# Make src/ importable when running the file directly (not via pytest).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.parser import (  # noqa: E402
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NUM_VAR_SLOTS,
    OPERATOR,
    Equation,
    Op,
    Var,
    canonical_var_ids,
    equation_to_arrays,
    parse_equation,
)


# --- parser ------------------------------------------------------------------

def test_parse_trivial_reflexive():
    assert parse_equation('x = x') == Equation(lhs=Var('x'), rhs=Var('x'))


def test_parse_trivial_distinct():
    assert parse_equation('x = y') == Equation(lhs=Var('x'), rhs=Var('y'))


def test_parse_top_level_product():
    eq = parse_equation('x = x ◇ y')
    assert eq == Equation(lhs=Var('x'), rhs=Op(Var('x'), Var('y')))


def test_parse_nested_right():
    eq = parse_equation('x = x ◇ (y ◇ z)')
    assert eq == Equation(
        lhs=Var('x'),
        rhs=Op(Var('x'), Op(Var('y'), Var('z'))),
    )


def test_parse_nested_left():
    eq = parse_equation('x = (x ◇ y) ◇ z')
    assert eq == Equation(
        lhs=Var('x'),
        rhs=Op(Op(Var('x'), Var('y')), Var('z')),
    )


def test_parse_family_c():
    # Line 38 of the corpus.
    eq = parse_equation('x ◇ x = x ◇ y')
    assert eq == Equation(
        lhs=Op(Var('x'), Var('x')),
        rhs=Op(Var('x'), Var('y')),
    )


def test_parse_max_depth():
    # Line 411: deepest right-nest.
    inner = Op(Var('x'), Var('x'))
    inner = Op(Var('x'), inner)
    inner = Op(Var('x'), inner)
    expected = Equation(lhs=Var('x'), rhs=Op(Var('x'), inner))
    assert parse_equation('x = x ◇ (x ◇ (x ◇ (x ◇ x)))') == expected


def test_parse_invalid_char():
    try:
        parse_equation('x = x + y')
    except ValueError:
        return
    raise AssertionError('expected ValueError for invalid character')


def test_parse_missing_eq():
    try:
        parse_equation('x x')
    except ValueError:
        return
    raise AssertionError('expected ValueError for missing =')


def test_parse_unmatched_paren():
    try:
        parse_equation('x = (x ◇ y')
    except ValueError:
        return
    raise AssertionError('expected ValueError for unmatched paren')


def test_parse_trailing_garbage():
    try:
        parse_equation('x = x ◇ y z')
    except ValueError:
        return
    raise AssertionError('expected ValueError for trailing garbage')


# --- canonical variable IDs --------------------------------------------------

def test_canonical_simple():
    assert canonical_var_ids(parse_equation('x = y')) == {'x': 0, 'y': 1}


def test_canonical_lhs_first():
    assert canonical_var_ids(parse_equation('x = y ◇ z')) == {'x': 0, 'y': 1, 'z': 2}


def test_canonical_six_distinct():
    eq = parse_equation('x = y ◇ (z ◇ (w ◇ (u ◇ v)))')
    assert canonical_var_ids(eq) == {'x': 0, 'y': 1, 'z': 2, 'w': 3, 'u': 4, 'v': 5}


def test_canonical_invariance_to_rename():
    """`x = y ◇ z` and `u = v ◇ w` differ only by variable names; their
    canonical encodings must match exactly."""
    a1, e1, ea1 = equation_to_arrays('x = y ◇ z')
    a2, e2, ea2 = equation_to_arrays('u = v ◇ w')
    assert a1 == a2
    assert e1 == e2
    assert ea1 == ea2


def test_canonical_swap_lhs_rhs_var_renaming():
    """`x = y` and `y = x` are the same canonically (both: LHS-var-equals-other-var)."""
    a1, e1, ea1 = equation_to_arrays('x = y')
    a2, e2, ea2 = equation_to_arrays('y = x')
    assert a1 == a2
    assert e1 == e2
    assert ea1 == ea2


# --- array encoding ----------------------------------------------------------

def _var_id_of(node: list[float]) -> int | None:
    for i in range(NUM_VAR_SLOTS):
        if node[2 + i]:
            return i
    return None


def test_arrays_x_eq_x():
    nodes, edges, eattr = equation_to_arrays('x = x')
    assert len(nodes) == 1
    assert edges == []
    assert eattr == []
    # var_id_0, is_lhs_var=1
    assert nodes[0] == [0, 1, 1, 0, 0, 0, 0, 0, 1]


def test_arrays_x_eq_y():
    nodes, edges, eattr = equation_to_arrays('x = y')
    assert len(nodes) == 1
    # y is canonical id 1 (x already took 0); is_lhs_var=0 since y != x.
    assert nodes[0] == [0, 1, 0, 1, 0, 0, 0, 0, 0]


def test_arrays_simple_op():
    nodes, edges, eattr = equation_to_arrays('x = x ◇ y')
    assert len(nodes) == 3
    assert edges == [(0, 1), (0, 2)]
    assert eattr == [[1.0, 0.0], [0.0, 1.0]]
    assert nodes[0] == [1, 0, 0, 0, 0, 0, 0, 0, 0]                # Op
    assert nodes[1] == [0, 1, 1, 0, 0, 0, 0, 0, 1]                # Var x, is_lhs_var
    assert nodes[2] == [0, 1, 0, 1, 0, 0, 0, 0, 0]                # Var y


def test_arrays_lhs_var_does_not_appear_in_rhs():
    """If LHS var doesn't occur in RHS, no node is flagged is_lhs_var."""
    nodes, _, _ = equation_to_arrays('x = y ◇ z')
    is_lhs_idx = 2 + NUM_VAR_SLOTS
    assert all(n[is_lhs_idx] == 0 for n in nodes)


def test_arrays_family_c_root_link():
    nodes, edges, eattr = equation_to_arrays('x ◇ x = x ◇ y')
    # 6 nodes: LHS Op(0), Var(x)(1), Var(x)(2); RHS Op(3), Var(x)(4), Var(y)(5).
    assert len(nodes) == 6
    assert edges == [(0, 1), (0, 2), (3, 4), (3, 5), (0, 3)]
    assert eattr == [[1, 0], [0, 1], [1, 0], [0, 1], [0, 0]]
    is_lhs_idx = 2 + NUM_VAR_SLOTS
    assert all(n[is_lhs_idx] == 0 for n in nodes)


def test_arrays_family_c_canonical_ids():
    """`x ◇ y = y ◇ x`: both vars used on both sides; canonical x=0, y=1."""
    nodes, _, _ = equation_to_arrays('x ◇ y = y ◇ x')
    assert _var_id_of(nodes[1]) == 0  # LHS left = x
    assert _var_id_of(nodes[2]) == 1  # LHS right = y
    assert _var_id_of(nodes[4]) == 1  # RHS left = y
    assert _var_id_of(nodes[5]) == 0  # RHS right = x


def test_arrays_node_feature_shape():
    """Every node feature must have exactly NODE_FEATURE_DIM components."""
    for eq_text in ['x = x', 'x = y ◇ z', 'x ◇ x = x ◇ y',
                    '(x ◇ y) ◇ z = (w ◇ u) ◇ v']:
        nodes, _, eattrs = equation_to_arrays(eq_text)
        for n in nodes:
            assert len(n) == NODE_FEATURE_DIM
        for e in eattrs:
            assert len(e) == EDGE_FEATURE_DIM


def test_arrays_one_hot_variable_id():
    """Variable nodes have exactly one var_id slot set."""
    nodes, _, _ = equation_to_arrays('(x ◇ y) ◇ z = (w ◇ u) ◇ v')
    for n in nodes:
        if n[1] == 1.0:  # is_var
            slots = sum(n[2:2 + NUM_VAR_SLOTS])
            assert slots == 1.0


def test_arrays_op_nodes_have_no_var_id():
    """Operator nodes have all var_id slots zero."""
    nodes, _, _ = equation_to_arrays('(x ◇ y) ◇ z = (w ◇ u) ◇ v')
    for n in nodes:
        if n[0] == 1.0:  # is_op
            assert n[2:2 + NUM_VAR_SLOTS] == [0.0] * NUM_VAR_SLOTS
            assert n[1] == 0.0


# --- corpus sanity check -----------------------------------------------------

def test_parse_full_corpus():
    """Parse every line of the corpus without errors. Skipped if data
    symlink is not configured."""
    path = os.path.join(_REPO_ROOT, 'data', 'raw', 'etp', 'equations.txt')
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) == 4694
    family_c = 0
    for line in lines:
        eq = parse_equation(line)
        nodes, edges, eattrs = equation_to_arrays(line)
        assert len(nodes) >= 1
        assert len(edges) == len(eattrs)
        ids = canonical_var_ids(eq)
        assert len(ids) <= NUM_VAR_SLOTS
        # Edge count = 2 * num_internal_op_nodes + (1 if family-C else 0)
        ops = sum(1 for n in nodes if n[0] == 1.0)
        family_c_link = 0 if isinstance(eq.lhs, Var) else 1
        assert len(edges) == 2 * ops + family_c_link
        if family_c_link:
            family_c += 1
    assert family_c == 1555


# --- optional torch wrapper smoke test ---------------------------------------

def test_pyg_data_smoke():
    try:
        from src.data.parser import equation_to_pyg
    except ImportError:
        return
    try:
        data = equation_to_pyg('x = x ◇ y')
    except ImportError:
        return  # torch_geometric not installed
    assert tuple(data.x.shape) == (3, NODE_FEATURE_DIM)
    assert tuple(data.edge_index.shape) == (2, 2)
    assert tuple(data.edge_attr.shape) == (2, EDGE_FEATURE_DIM)


# --- driver for running directly (no pytest) --------------------------------

if __name__ == '__main__':
    import inspect
    import traceback

    test_funcs = [
        v for k, v in sorted(globals().items())
        if k.startswith('test_') and inspect.isfunction(v)
    ]
    fails = []
    for fn in test_funcs:
        try:
            fn()
            print(f'PASS  {fn.__name__}')
        except Exception as e:
            fails.append(fn.__name__)
            print(f'FAIL  {fn.__name__}: {type(e).__name__}: {e}')
            traceback.print_exc()
    print()
    print(f'{len(test_funcs) - len(fails)}/{len(test_funcs)} passed')
    sys.exit(1 if fails else 0)
