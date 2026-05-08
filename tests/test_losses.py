"""Tests for src/training/losses.py."""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.training.losses import contrastive_loss  # noqa: E402


def test_loss_returns_scalar():
    z1 = torch.randn(8, 64)
    z2 = torch.randn(8, 64)
    labels = torch.randint(0, 2, (8,)).float()
    loss = contrastive_loss(z1, z2, labels)
    assert loss.shape == ()


def test_loss_near_zero_for_perfect_TRUE_at_zero_distance():
    """All-TRUE batch with z_pred == z_target: loss should be ≈ 0.

    Not exactly zero because ``F.pairwise_distance`` has a default
    ``eps=1e-6`` numerical floor, but several orders of magnitude below
    any meaningful contribution.
    """
    z = torch.randn(4, 64)
    labels = torch.ones(4)
    loss = contrastive_loss(z, z, labels)
    assert loss.item() < 1e-8


def test_loss_zero_for_perfect_FALSE_beyond_margin():
    """All-FALSE batch with d ≥ margin: relu(margin-d)=0 → loss=0."""
    z1 = torch.zeros(4, 64)
    z2 = torch.zeros(4, 64)
    z2[:, 0] = 2.0  # d=2.0
    labels = torch.zeros(4)
    loss = contrastive_loss(z1, z2, labels, margin=1.0)
    assert loss.item() == 0.0


def test_loss_value_for_TRUE_at_unit_distance():
    z1 = torch.zeros(4, 64)
    z2 = torch.zeros(4, 64)
    z2[:, 0] = 1.0  # d=1.0
    labels = torch.ones(4)
    loss = contrastive_loss(z1, z2, labels)
    # Per pair: 1 * 1.0² = 1.0; mean over 4 = 1.0
    assert torch.isclose(loss, torch.tensor(1.0))


def test_loss_value_for_FALSE_within_margin():
    z1 = torch.zeros(4, 64)
    z2 = torch.zeros(4, 64)
    z2[:, 0] = 0.5  # d=0.5
    labels = torch.zeros(4)
    loss = contrastive_loss(z1, z2, labels, margin=1.0)
    # Per pair: relu(1-0.5)² = 0.25; mean over 4 = 0.25
    assert torch.isclose(loss, torch.tensor(0.25))


def test_loss_true_weight_scales_TRUE_term():
    z1 = torch.zeros(2, 64)
    z2 = torch.zeros(2, 64)
    z2[:, 0] = 1.0
    labels = torch.ones(2)
    loss_1 = contrastive_loss(z1, z2, labels, true_weight=1.0)
    loss_2 = contrastive_loss(z1, z2, labels, true_weight=2.0)
    assert torch.isclose(loss_2, 2.0 * loss_1)


def test_loss_true_weight_does_not_affect_FALSE_term():
    """true_weight should multiply only the TRUE-term contribution."""
    z1 = torch.zeros(2, 64)
    z2 = torch.zeros(2, 64)
    z2[:, 0] = 0.5
    labels = torch.zeros(2)  # all FALSE
    loss_1 = contrastive_loss(z1, z2, labels, true_weight=1.0)
    loss_2 = contrastive_loss(z1, z2, labels, true_weight=5.0)
    assert torch.isclose(loss_1, loss_2)


def test_loss_mixed_batch():
    """2 TRUE pairs at d=1, 2 FALSE pairs at d=0.5, margin=1."""
    z1 = torch.zeros(4, 64)
    z2 = torch.zeros(4, 64)
    z2[0, 0] = 1.0
    z2[1, 0] = 1.0
    z2[2, 0] = 0.5
    z2[3, 0] = 0.5
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss = contrastive_loss(z1, z2, labels, margin=1.0, true_weight=1.0)
    # TRUE contribution: 2 * 1.0² = 2.0
    # FALSE contribution: 2 * (0.5)² = 0.5
    # Mean over 4: 2.5/4 = 0.625
    assert torch.isclose(loss, torch.tensor(0.625))


def test_loss_backward_yields_gradients_on_z_predicted():
    z1 = torch.randn(4, 64, requires_grad=True)
    z2 = torch.randn(4, 64)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = contrastive_loss(z1, z2, labels)
    loss.backward()
    assert z1.grad is not None
    assert z1.grad.abs().sum().item() > 0


def test_loss_shape_mismatch_raises():
    z1 = torch.randn(4, 64)
    z2 = torch.randn(4, 32)
    labels = torch.zeros(4)
    try:
        contrastive_loss(z1, z2, labels)
    except ValueError:
        return
    raise AssertionError('expected ValueError for embedding-shape mismatch')


def test_loss_label_batch_mismatch_raises():
    z1 = torch.randn(4, 64)
    z2 = torch.randn(4, 64)
    labels = torch.zeros(3)
    try:
        contrastive_loss(z1, z2, labels)
    except ValueError:
        return
    raise AssertionError('expected ValueError for label batch mismatch')


def test_loss_cuda():
    if not torch.cuda.is_available():
        return
    z1 = torch.randn(4, 64, device='cuda', requires_grad=True)
    z2 = torch.randn(4, 64, device='cuda')
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0], device='cuda')
    loss = contrastive_loss(z1, z2, labels)
    loss.backward()
    assert loss.device.type == 'cuda'
    assert z1.grad is not None


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
