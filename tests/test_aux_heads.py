"""Tests for the AuxiliaryHeads module.

Verifies shape, gradient flow, loss assembly, and accuracy reporting.
"""

from __future__ import annotations

import pytest
import torch

from src.model.aux_heads import (
    HEAD_SPECS,
    AuxiliaryHeads,
    auxiliary_accuracy,
    auxiliary_loss,
)


def _make_random_labels(batch: int) -> dict[str, torch.Tensor]:
    return {
        name: torch.randint(0, n, (batch,), dtype=torch.long)
        for name, n in HEAD_SPECS.items()
    }


def test_head_specs_match_plan():
    """The 10 heads from aux_heads_implementation_plan.md."""
    assert HEAD_SPECS == {
        "spine_type": 6,
        "lhs_structure": 2,
        "num_variables": 6,
        "magma_bit_0": 2,
        "magma_bit_1": 2,
        "magma_bit_2": 2,
        "magma_bit_3": 2,
        "magma_bit_4": 2,
        "magma_bit_6": 2,
        "magma_bit_7": 2,
    }


def test_forward_returns_per_head_logits_with_correct_shapes():
    heads = AuxiliaryHeads(pool_dim=64)
    pooled = torch.randn(8, 64)
    out = heads(pooled)
    assert set(out.keys()) == set(HEAD_SPECS.keys())
    for name, num_classes in HEAD_SPECS.items():
        assert out[name].shape == (8, num_classes), name


def test_param_count_is_minimal():
    """Plan expectation: ~2K params for h=256 (pool_dim=512). Should
    be 'training signal, not model capacity'."""
    heads = AuxiliaryHeads(pool_dim=512)
    n_params = sum(p.numel() for p in heads.parameters())
    # Per head: pool_dim * num_classes + num_classes (weight + bias)
    # 6 + 2 + 6 + 2*7 = 28 total output units, weights = 28*512 = 14336
    # plus 28 bias = 14364
    assert n_params == 14364, f"unexpected aux head param count: {n_params}"


def test_gradient_flows_through_pooled_input():
    heads = AuxiliaryHeads(pool_dim=32)
    pooled = torch.randn(4, 32, requires_grad=True)
    out = heads(pooled)
    labels = _make_random_labels(batch=4)
    loss, _ = auxiliary_loss(out, labels)
    loss.backward()
    assert pooled.grad is not None
    assert torch.isfinite(pooled.grad).all()
    assert (pooled.grad.abs() > 0).any()


def test_loss_returns_scalar_and_per_head_dict():
    heads = AuxiliaryHeads(pool_dim=32)
    out = heads(torch.randn(4, 32))
    labels = _make_random_labels(batch=4)
    total, per_head = auxiliary_loss(out, labels)
    assert total.shape == ()
    assert torch.isfinite(total)
    assert set(per_head.keys()) == set(HEAD_SPECS.keys())
    for v in per_head.values():
        assert isinstance(v, float)
        # ln(num_classes) is the random-baseline cross-entropy value;
        # untrained heads should land near it but the test just checks
        # the value is sane (positive, finite).
        assert v > 0


def test_loss_at_init_is_near_random_baseline():
    """Untrained heads on random pooled input should give cross-entropy
    near ln(num_classes) per head (the entropy of uniform predictions).
    """
    torch.manual_seed(0)
    heads = AuxiliaryHeads(pool_dim=32)
    out = heads(torch.randn(256, 32))
    labels = _make_random_labels(batch=256)
    _, per_head = auxiliary_loss(out, labels)
    import math
    for name, value in per_head.items():
        target = math.log(HEAD_SPECS[name])
        # Allow generous slack for finite-batch noise; this is a sanity
        # bound, not a tight assertion.
        assert 0.3 * target < value < 3.0 * target, (
            f"head {name} loss {value:.3f} far from random baseline "
            f"ln({HEAD_SPECS[name]}) = {target:.3f}"
        )


def test_loss_decreases_when_logits_match_labels():
    """Sanity check: if logits put all mass on the true class, loss → 0."""
    batch = 8
    out = {
        name: torch.zeros(batch, n)
        for name, n in HEAD_SPECS.items()
    }
    labels = _make_random_labels(batch=batch)
    # Cheat: set the true-class logit very high so cross-entropy is tiny.
    for name in HEAD_SPECS:
        out[name][torch.arange(batch), labels[name]] = 100.0
    total, _ = auxiliary_loss(out, labels)
    assert total.item() < 1e-3


def test_accuracy_returns_per_head_floats():
    heads = AuxiliaryHeads(pool_dim=32)
    out = heads(torch.randn(64, 32))
    labels = _make_random_labels(batch=64)
    acc = auxiliary_accuracy(out, labels)
    assert set(acc.keys()) == set(HEAD_SPECS.keys())
    for v in acc.values():
        assert 0.0 <= v <= 1.0


def test_loss_raises_on_mismatched_label_keys():
    out = {name: torch.randn(4, n) for name, n in HEAD_SPECS.items()}
    bad_labels = {"spine_type": torch.zeros(4, dtype=torch.long)}
    with pytest.raises(ValueError, match="missing aux labels"):
        auxiliary_loss(out, bad_labels)


def test_works_on_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    heads = AuxiliaryHeads(pool_dim=32).cuda()
    pooled = torch.randn(8, 32, device="cuda")
    out = heads(pooled)
    labels = {
        name: torch.randint(0, n, (8,), device="cuda", dtype=torch.long)
        for name, n in HEAD_SPECS.items()
    }
    loss, _ = auxiliary_loss(out, labels)
    assert loss.device.type == "cuda"
    assert torch.isfinite(loss)
