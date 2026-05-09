"""Tests for src/model/jepa.py."""

from __future__ import annotations

import os
import sys

import torch
from torch_geometric.data import Batch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.parser import equation_to_pyg  # noqa: E402
from src.model.encoder import EquationEncoder  # noqa: E402
from src.model.jepa import JEPA  # noqa: E402
from src.model.predictor import ImplicationPredictor  # noqa: E402


def _make_jepa() -> JEPA:
    return JEPA(EquationEncoder(), ImplicationPredictor())


def _example_pair_batch() -> tuple[Batch, Batch]:
    eq1 = Batch.from_data_list([
        equation_to_pyg('x = x ◇ y'),
        equation_to_pyg('x = x'),
        equation_to_pyg('(x ◇ y) ◇ z = (w ◇ u) ◇ v'),
    ])
    eq2 = Batch.from_data_list([
        equation_to_pyg('x = y ◇ x'),
        equation_to_pyg('x ◇ x = x'),
        equation_to_pyg('x = x ◇ (y ◇ z)'),
    ])
    return eq1, eq2


# --- construction ----------------------------------------------------------

def test_target_initialized_from_context_at_construction():
    jepa = _make_jepa()
    for p_t, p_c in zip(
        jepa.target_encoder.parameters(),
        jepa.context_encoder.parameters(),
    ):
        assert torch.equal(p_t, p_c)


def test_target_parameters_have_no_grad():
    jepa = _make_jepa()
    for p in jepa.target_encoder.parameters():
        assert not p.requires_grad


def test_context_and_predictor_parameters_keep_grad():
    """The freeze on the target must not leak to the context or predictor."""
    jepa = _make_jepa()
    assert all(p.requires_grad for p in jepa.context_encoder.parameters())
    assert all(p.requires_grad for p in jepa.predictor.parameters())


# --- forward --------------------------------------------------------------

def test_forward_returns_two_tensors_with_correct_shape():
    jepa = _make_jepa()
    eq1, eq2 = _example_pair_batch()
    z_pred, z_tgt = jepa(eq1, eq2)
    assert z_pred.shape == (3, 64)
    assert z_tgt.shape == (3, 64)
    assert torch.isfinite(z_pred).all()
    assert torch.isfinite(z_tgt).all()


def test_forward_target_has_no_grad():
    jepa = _make_jepa()
    eq1, eq2 = _example_pair_batch()
    _, z_tgt = jepa(eq1, eq2)
    assert not z_tgt.requires_grad


def test_forward_return_encoder_raw_returns_four_tensors():
    """``return_encoder_raw=True`` returns the *encoder* pre-normalization
    outputs alongside the normalized predictor + target. VICReg consumes
    the raws to push the encoder representation off rank-9 — applying it
    further downstream (on the predictor) leaves the encoder->predictor
    L2-normalization bottleneck in place."""
    jepa = _make_jepa()
    eq1, eq2 = _example_pair_batch()
    z_pred, z_tgt, z_context_raw, z_tgt_raw = jepa(eq1, eq2, return_encoder_raw=True)
    assert z_pred.shape == (3, 64)
    assert z_tgt.shape == (3, 64)
    assert z_context_raw.shape == (3, 64)
    assert z_tgt_raw.shape == (3, 64)


def test_forward_encoder_raw_directions_match_post_norm():
    """``encoder_raw / ||encoder_raw|| == post_norm_for_predictor``. The
    raw output of the context encoder, divided by its norm, equals the
    z_context that gets fed into the predictor and ultimately compared
    against z_target via the contrastive loss."""
    import torch.nn.functional as F
    jepa = _make_jepa()
    jepa.eval()
    eq1, eq2 = _example_pair_batch()
    with torch.no_grad():
        # Use return_encoder_raw to get raw encoder outputs.
        _, z_tgt, z_context_raw, z_tgt_raw = jepa(eq1, eq2, return_encoder_raw=True)
        # Compare against z_context produced via the encoder directly.
        z_context_via_encoder = jepa.context_encoder(eq1)
    assert torch.allclose(F.normalize(z_context_raw, dim=-1), z_context_via_encoder, atol=1e-5)
    assert torch.allclose(F.normalize(z_tgt_raw, dim=-1), z_tgt, atol=1e-5)


def test_forward_encoder_raw_grad_flow():
    """``z_context_raw`` carries gradient (this is the whole point — it
    gives VICReg a direct path into the context encoder, bypassing the
    L2 bottleneck). ``z_target_raw`` is detached (target frozen +
    no_grad)."""
    jepa = _make_jepa()
    eq1, eq2 = _example_pair_batch()
    _, _, z_context_raw, z_tgt_raw = jepa(eq1, eq2, return_encoder_raw=True)
    assert z_context_raw.requires_grad
    assert not z_tgt_raw.requires_grad


def test_forward_outputs_are_unit_norm_and_distance_is_bounded():
    """With both predictor and target encoder L2-normalizing their outputs,
    pairwise distances must lie in [0, 2]. This is the property that lets
    the contrastive loss's margin (1.0) sit inside a meaningful range."""
    import torch.nn.functional as F
    jepa = _make_jepa()
    jepa.eval()
    eq1, eq2 = _example_pair_batch()
    with torch.no_grad():
        z_pred, z_tgt = jepa(eq1, eq2)
    pred_norms = z_pred.norm(dim=-1)
    tgt_norms = z_tgt.norm(dim=-1)
    assert torch.allclose(pred_norms, torch.ones_like(pred_norms), atol=1e-5)
    assert torch.allclose(tgt_norms, torch.ones_like(tgt_norms), atol=1e-5)
    distances = F.pairwise_distance(z_pred, z_tgt)
    assert (distances >= 0).all()
    assert (distances <= 2.0 + 1e-5).all()


def test_backward_only_updates_context_and_predictor():
    """Loss gradient must flow through context_encoder + predictor only."""
    jepa = _make_jepa()
    eq1, eq2 = _example_pair_batch()
    z_pred, z_tgt = jepa(eq1, eq2)
    loss = (z_pred - z_tgt).pow(2).sum()
    loss.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in jepa.context_encoder.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in jepa.predictor.parameters()
    )
    # Frozen target params have grad=None (autograd never created tensors).
    for p in jepa.target_encoder.parameters():
        assert p.grad is None


# --- EMA update -----------------------------------------------------------

def test_update_target_tau_zero_copies_context():
    jepa = _make_jepa()
    # Perturb context away from initial target state.
    with torch.no_grad():
        for p in jepa.context_encoder.parameters():
            p.add_(torch.randn_like(p))

    jepa.update_target(tau=0.0)

    for p_t, p_c in zip(
        jepa.target_encoder.parameters(),
        jepa.context_encoder.parameters(),
    ):
        assert torch.equal(p_t, p_c)


def test_update_target_tau_one_keeps_target():
    jepa = _make_jepa()
    initial_target = [p.clone() for p in jepa.target_encoder.parameters()]

    with torch.no_grad():
        for p in jepa.context_encoder.parameters():
            p.add_(torch.randn_like(p))

    jepa.update_target(tau=1.0)

    for p_t, init in zip(jepa.target_encoder.parameters(), initial_target):
        assert torch.equal(p_t, init)


def test_update_target_tau_half_is_midpoint():
    jepa = _make_jepa()
    initial_target = [p.clone() for p in jepa.target_encoder.parameters()]
    with torch.no_grad():
        for p in jepa.context_encoder.parameters():
            p.add_(torch.randn_like(p))
    new_context = [p.clone() for p in jepa.context_encoder.parameters()]

    jepa.update_target(tau=0.5)

    for p_t, init, new in zip(
        jepa.target_encoder.parameters(), initial_target, new_context
    ):
        expected = 0.5 * init + 0.5 * new
        assert torch.allclose(p_t, expected, atol=1e-6)


def test_update_target_invalid_tau_raises():
    jepa = _make_jepa()
    for bad in [-0.1, 1.1, 2.0]:
        try:
            jepa.update_target(tau=bad)
        except ValueError:
            continue
        raise AssertionError(f'expected ValueError for tau={bad}')


# --- CUDA ----------------------------------------------------------------

def test_jepa_cuda_forward_and_update():
    if not torch.cuda.is_available():
        return
    jepa = _make_jepa().cuda()
    eq1, eq2 = _example_pair_batch()
    eq1, eq2 = eq1.to('cuda'), eq2.to('cuda')

    z_pred, z_tgt = jepa(eq1, eq2)
    assert z_pred.shape == (3, 64)
    assert z_pred.device.type == 'cuda'
    assert z_tgt.device.type == 'cuda'

    # Train step + EMA should also work end-to-end.
    loss = (z_pred - z_tgt).pow(2).sum()
    loss.backward()
    jepa.update_target(tau=0.99)


# --- driver --------------------------------------------------------------

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
