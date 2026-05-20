"""Tests for src/model/encoder.py."""

from __future__ import annotations

import os
import sys

import torch
from torch_geometric.data import Batch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.parser import (  # noqa: E402
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


def test_encoder_return_pooled_returns_pooled_alongside_embedding():
    """``return_pooled=True`` exposes the pre-projection
    (mean_pool || max_pool) representation for the auxiliary heads.
    Pool dim = 2 * hidden_dim."""
    enc = EquationEncoder(hidden_dim=128, embed_dim=64)
    batch = _example_batch()
    out, pooled = enc(batch, return_pooled=True)
    assert out.shape == (3, 64)
    assert pooled.shape == (3, 256)  # 2 * hidden_dim
    assert torch.isfinite(pooled).all()
    # The L2-normalized output should still be unit-norm — return_pooled
    # is purely additive, doesn't change the embedding path.
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_encoder_output_is_unit_norm():
    """Encoder output is L2-normalized: every embedding lives on the unit
    sphere. Bounds the contrastive loss's distance space."""
    torch.manual_seed(0)
    enc = EquationEncoder()
    enc.eval()
    batch = _example_batch()
    with torch.no_grad():
        out = enc(batch)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_encoder_normalize_false_returns_raw_projection():
    """``normalize=False`` skips the L2 step so VICReg can operate on the
    pre-normalization space. The raw projection generally has non-unit
    norm; otherwise we'd be on the unit sphere by accident."""
    torch.manual_seed(0)
    enc = EquationEncoder()
    enc.eval()
    batch = _example_batch()
    with torch.no_grad():
        normed = enc(batch, normalize=True)
        raw = enc(batch, normalize=False)

    # Same direction — raw / ||raw|| equals normed.
    raw_dir = torch.nn.functional.normalize(raw, dim=-1)
    assert torch.allclose(raw_dir, normed, atol=1e-5)

    # ... but raw is generally not unit-norm.
    raw_norms = raw.norm(dim=-1)
    assert not torch.allclose(raw_norms, torch.ones_like(raw_norms), atol=1e-3)


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


# --- skip-connection tests ------------------------------------------------

def test_skip_pooled_init_dims():
    """skip_pooled=True derives proj_dim from embed_dim - 2*hidden_dim
    when proj_dim is not given."""
    enc = EquationEncoder(hidden_dim=128, embed_dim=768, skip_pooled=True)
    assert enc.skip_pooled is True
    assert enc.embed_dim == 768
    assert enc.proj_dim == 512  # 768 - 2*128
    assert enc.pool_dim == 256
    assert enc.proj.out_features == 512
    assert enc.proj_norm is not None
    assert enc.skip_norm is not None
    assert enc.proj_norm.normalized_shape == (512,)
    assert enc.skip_norm.normalized_shape == (256,)


def test_skip_pooled_explicit_proj_dim():
    """proj_dim can be specified explicitly; embed_dim must equal proj_dim + pool_dim."""
    enc = EquationEncoder(hidden_dim=128, embed_dim=768, proj_dim=512, skip_pooled=True)
    assert enc.proj_dim == 512
    assert enc.embed_dim == 768


def test_skip_pooled_invalid_dim_combo_raises():
    """proj_dim + 2*hidden_dim must equal embed_dim under skip_pooled."""
    try:
        EquationEncoder(hidden_dim=128, embed_dim=600, proj_dim=512, skip_pooled=True)
    except ValueError as e:
        assert "proj_dim + 2*hidden_dim == embed_dim" in str(e)
        return
    raise AssertionError("expected ValueError on mismatched skip dims")


def test_skip_pooled_forward_unit_norm():
    """The full 768-dim output should be L2-normalized."""
    enc = EquationEncoder(hidden_dim=128, embed_dim=768, skip_pooled=True).eval()
    batch = _example_batch()
    out = enc(batch)
    assert out.shape == (3, 768)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_skip_pooled_unnormalized_concat_layout():
    """In skip mode, normalize=False returns the un-normalized concat;
    the first proj_dim slots are the LayerNorm'd projection, the
    remaining pool_dim slots are the LayerNorm'd pooled rep."""
    enc = EquationEncoder(hidden_dim=128, embed_dim=768, skip_pooled=True).eval()
    batch = _example_batch()
    raw = enc(batch, normalize=False)
    assert raw.shape == (3, 768)
    z_proj = raw[:, :512]
    z_skip = raw[:, 512:]
    # LayerNorm'd outputs have ~zero mean and ~unit variance per row.
    assert z_proj.mean(dim=-1).abs().max() < 1e-4
    assert z_skip.mean(dim=-1).abs().max() < 1e-4


def test_skip_pooled_init_balances_channels():
    """The point of the LayerNorms: at init, neither half should
    dominate the pre-normalize embedding by more than ~2x in mean
    per-dim magnitude. Without LayerNorm, the skip channel (LayerNorm'd
    GNN output post-pool) would dwarf the random Linear's output."""
    torch.manual_seed(0)
    enc = EquationEncoder(hidden_dim=128, embed_dim=768, skip_pooled=True).eval()
    batch = _example_batch()
    raw = enc(batch, normalize=False)
    z_proj_mean_abs = raw[:, :512].abs().mean()
    z_skip_mean_abs = raw[:, 512:].abs().mean()
    ratio = max(z_proj_mean_abs, z_skip_mean_abs) / min(z_proj_mean_abs, z_skip_mean_abs)
    assert ratio < 3.0, (
        f"channel magnitudes diverge ({z_proj_mean_abs:.4f} vs "
        f"{z_skip_mean_abs:.4f}, ratio {ratio:.2f}); LayerNorm should "
        f"keep them comparable at init"
    )


def test_skip_pooled_return_pooled():
    """return_pooled still works in skip mode and exposes the raw pooled
    rep (not the LayerNorm'd one) for the aux heads."""
    enc = EquationEncoder(hidden_dim=128, embed_dim=768, skip_pooled=True).eval()
    batch = _example_batch()
    out, pooled = enc(batch, return_pooled=True)
    assert out.shape == (3, 768)
    assert pooled.shape == (3, 256)


def test_skip_pooled_param_count_increase():
    """Adding LayerNorm to both halves should add ~1.5K params at the
    canonical 512+256 split (LayerNorm has 2*dim params)."""
    base = EquationEncoder(hidden_dim=128, embed_dim=512)
    skip = EquationEncoder(hidden_dim=128, embed_dim=768, skip_pooled=True)
    base_n = base.num_parameters()
    skip_n = skip.num_parameters()
    # Skip adds: nothing to GNN backbone; proj is now 512 instead of 512
    # (same), proj_norm = 2*512 = 1024 params, skip_norm = 2*256 = 512
    # params. Total +1536 over the same proj_dim baseline.
    delta = skip_n - base_n
    assert 1500 <= delta <= 1600, f"unexpected param delta: {delta}"


def test_skip_pooled_cuda_forward():
    if not torch.cuda.is_available():
        return
    enc = EquationEncoder(hidden_dim=128, embed_dim=768, skip_pooled=True).cuda()
    batch = _example_batch().to('cuda')
    out = enc(batch)
    assert out.shape == (3, 768)
    assert out.device.type == 'cuda'
    assert torch.isfinite(out).all()


def test_bidirectional_edges_propagate_root_from_leaf_in_one_layer():
    """With bidirectional edges, a single message-passing step is enough for
    the root node to receive information from a leaf. This is the entire
    reason for the 4-dim bidirectional edge representation: with the prior
    parent→child-only spec, the root would only see itself in 1 layer.
    """
    from torch_geometric.nn import GATv2Conv

    from lib.data.parser import EDGE_FEATURE_DIM, equation_to_pyg

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
