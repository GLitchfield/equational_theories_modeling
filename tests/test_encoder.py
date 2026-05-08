"""Tests for src/model/encoder.py."""

from __future__ import annotations

import os
import sys

import torch
from torch_geometric.data import Batch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.parser import (  # noqa: E402
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    equation_to_pyg,
)
from src.model.encoder import EquationEncoder  # noqa: E402


def _example_batch() -> Batch:
    return Batch.from_data_list([
        equation_to_pyg('x = x'),                          # 1 node, 0 edges
        equation_to_pyg('x = x ◇ y'),                      # 3 nodes, 2 edges
        equation_to_pyg('(x ◇ y) ◇ z = (w ◇ u) ◇ v'),     # family-C, multi-subtree
    ])


def test_encoder_default_constants_match_parser():
    enc = EquationEncoder()
    assert enc.node_features == NODE_FEATURE_DIM
    assert enc.edge_features == EDGE_FEATURE_DIM


def test_encoder_forward_shape():
    enc = EquationEncoder()
    batch = _example_batch()
    out = enc(batch)
    assert out.shape == (3, 64)
    assert torch.isfinite(out).all()


def test_encoder_handles_single_node_graph():
    """Trivial reflexive 'x = x' is a 1-node graph with 0 edges; with the
    GATv2Conv default ``add_self_loops=True`` it should still produce a
    finite embedding."""
    enc = EquationEncoder()
    enc.eval()
    g = equation_to_pyg('x = x')
    assert g.x.shape[0] == 1
    assert g.edge_index.shape[1] == 0
    batch = Batch.from_data_list([g])
    with torch.no_grad():
        out = enc(batch)
    assert out.shape == (1, 64)
    assert torch.isfinite(out).all()


def test_encoder_distinguishes_equations():
    """Two structurally different equations should produce different
    embeddings even at random init (the encoder isn't trivially constant)."""
    torch.manual_seed(0)
    enc = EquationEncoder()
    enc.eval()
    batch = Batch.from_data_list([
        equation_to_pyg('x = x'),
        equation_to_pyg('x = x ◇ y'),
    ])
    with torch.no_grad():
        out = enc(batch)
    assert not torch.allclose(out[0], out[1], atol=1e-4)


def test_encoder_eval_determinism():
    enc = EquationEncoder()
    enc.eval()
    batch = _example_batch()
    with torch.no_grad():
        out1 = enc(batch)
        out2 = enc(batch)
    assert torch.allclose(out1, out2)


def test_encoder_backward_pass_yields_gradients():
    enc = EquationEncoder()
    batch = _example_batch()
    loss = enc(batch).pow(2).sum()
    loss.backward()
    has_nonzero_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in enc.parameters()
    )
    assert has_nonzero_grad


def test_encoder_param_count_in_range():
    """The plan estimates ~1-2M parameters; the literal 128/64/4/4 config
    comes in lower in practice. Just sanity-check the order of magnitude."""
    enc = EquationEncoder()
    n = enc.num_parameters()
    assert 50_000 < n < 10_000_000, f'unexpected param count: {n}'


def test_encoder_invalid_hidden_dim_raises():
    """hidden_dim must be divisible by num_heads."""
    try:
        EquationEncoder(hidden_dim=130, num_heads=4)
    except ValueError:
        return
    raise AssertionError('expected ValueError for non-divisible hidden_dim')


def test_encoder_cuda_forward():
    if not torch.cuda.is_available():
        return
    enc = EquationEncoder().cuda()
    batch = _example_batch().to('cuda')
    out = enc(batch)
    assert out.shape == (3, 64)
    assert out.device.type == 'cuda'
    assert torch.isfinite(out).all()


def test_bidirectional_edges_propagate_root_from_leaf_in_one_layer():
    """With bidirectional edges, a single message-passing step is enough for
    the root node to receive information from a leaf. This is the entire
    reason for the 4-dim bidirectional edge representation: with the prior
    parent→child-only spec, the root would only see itself in 1 layer.
    """
    from torch_geometric.nn import GATv2Conv

    from src.data.parser import EDGE_FEATURE_DIM, equation_to_pyg

    # Three nodes: op (0), var x (1), var y (2). Bidirectional edges mean
    # the op node has incoming edges from both leaves.
    g = equation_to_pyg('x = x ◇ y')
    assert g.x.shape[0] == 3

    torch.manual_seed(0)
    conv = GATv2Conv(g.x.shape[1], 16, edge_dim=EDGE_FEATURE_DIM)
    conv.eval()

    with torch.no_grad():
        out_orig = conv(g.x, g.edge_index, g.edge_attr)

    # Perturb only the leaf at index 1; root output (index 0) must change.
    x_pert = g.x.clone()
    x_pert[1, 0] += 1.0
    with torch.no_grad():
        out_pert = conv(x_pert, g.edge_index, g.edge_attr)

    assert not torch.allclose(out_orig[0], out_pert[0], atol=1e-6), (
        "Root output did not change when a leaf's input was perturbed — "
        "upward edges may not be wired up correctly."
    )


# --- driver -----------------------------------------------------------------

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
