"""Tests for the AuxiliaryHeads module.

Verifies shape, gradient flow, loss assembly, and accuracy reporting.
"""

from __future__ import annotations

import pytest
import torch

from embeddings.src.model.aux_heads import (
    HEAD_SPECS,
    HEAD_SUBSETS,
    AuxiliaryHeads,
    auxiliary_accuracy,
    auxiliary_loss,
    resolve_head_names,
)


def _make_random_labels(batch: int) -> dict[str, torch.Tensor]:
    return {
        name: torch.randint(0, n, (batch,), dtype=torch.long)
        for name, n in HEAD_SPECS.items()
    }


def test_head_specs_match_plan():
    """The 10 structural heads from aux_heads_implementation_plan.md
    plus the 8 cohomology summary heads added in the cohom-aux follow-up."""
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
        "cohom_count_2": 5,
        "cohom_count_3": 5,
        "cohom_count_5": 6,
        "cohom_count_7": 6,
        "cohom_mean_h2_2": 5,
        "cohom_mean_h2_3": 6,
        "cohom_mean_h2_5": 6,
        "cohom_mean_h2_7": 6,
    }


def test_forward_returns_per_head_logits_with_correct_shapes():
    heads = AuxiliaryHeads(pool_dim=64)
    pooled = torch.randn(8, 64)
    out = heads(pooled)
    assert set(out.keys()) == set(HEAD_SPECS.keys())
    for name, num_classes in HEAD_SPECS.items():
        assert out[name].shape == (8, num_classes), name


def test_param_count_is_minimal():
    """Heads should remain small relative to the encoder backbone
    (training signal, not model capacity)."""
    heads = AuxiliaryHeads(pool_dim=512)
    n_params = sum(p.numel() for p in heads.parameters())
    # Per head: pool_dim * num_classes + num_classes (weight + bias).
    # Structural heads: 6 + 2 + 6 + 2*7 = 28 output units.
    # Cohom heads: 5 + 5 + 6 + 6 + 5 + 6 + 6 + 6 = 45 output units.
    # Total = 73 output units, params = 73 * 512 + 73 = 37449.
    assert n_params == 37449, f"unexpected aux head param count: {n_params}"


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


# --- head subset behavior -------------------------------------------------

def test_resolve_head_names_default_returns_all():
    assert resolve_head_names(None) == list(HEAD_SPECS.keys())


def test_resolve_head_names_named_subsets():
    structural = resolve_head_names("structural")
    assert structural == HEAD_SUBSETS["structural"]
    assert "spine_type" in structural
    assert "cohom_count_2" not in structural

    cohom = resolve_head_names("cohom")
    assert cohom == HEAD_SUBSETS["cohom"]
    assert "cohom_count_2" in cohom
    assert "spine_type" not in cohom

    all_heads = resolve_head_names("all")
    assert all_heads == list(HEAD_SPECS.keys())


def test_resolve_head_names_explicit_list():
    spec = ["spine_type", "magma_bit_3"]
    assert resolve_head_names(spec) == spec


def test_resolve_head_names_unknown_subset_raises():
    with pytest.raises(ValueError, match="unknown head subset"):
        resolve_head_names("not_a_subset")


def test_resolve_head_names_unknown_head_in_list_raises():
    with pytest.raises(ValueError, match="unknown head names"):
        resolve_head_names(["spine_type", "not_a_head"])


def test_resolve_head_names_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        resolve_head_names([])


def test_aux_heads_subset_only_creates_active_heads():
    heads = AuxiliaryHeads(pool_dim=32, head_names=HEAD_SUBSETS["structural"])
    assert set(heads.heads.keys()) == set(HEAD_SUBSETS["structural"])
    assert heads.head_names == HEAD_SUBSETS["structural"]
    pooled = torch.randn(4, 32)
    out = heads(pooled)
    assert set(out.keys()) == set(HEAD_SUBSETS["structural"])
    # Param count: only structural heads, not all 18.
    n_params = sum(p.numel() for p in heads.parameters())
    # Structural: 6 + 2 + 6 + 2*7 = 28 output units * (32+1) = 924
    assert n_params == 924


def test_auxiliary_loss_with_subset_heads():
    """Loss should only use the heads present in logits, ignoring extra
    labels (this is how the train loop calls it: dataset always loads
    all 18 head labels, but logits only contains active heads)."""
    structural_names = HEAD_SUBSETS["structural"]
    heads = AuxiliaryHeads(pool_dim=32, head_names=structural_names)
    out = heads(torch.randn(8, 32))
    # Labels dict contains ALL 18 heads (extra ones are ignored).
    labels = {
        name: torch.randint(0, n, (8,), dtype=torch.long)
        for name, n in HEAD_SPECS.items()
    }
    total, per_head = auxiliary_loss(out, labels)
    assert torch.isfinite(total)
    # Per-head dict has only the active subset.
    assert set(per_head.keys()) == set(structural_names)


def test_auxiliary_loss_raises_when_label_missing_for_active_head():
    heads = AuxiliaryHeads(pool_dim=32, head_names=["spine_type", "magma_bit_0"])
    out = heads(torch.randn(4, 32))
    labels = {"spine_type": torch.zeros(4, dtype=torch.long)}  # missing magma_bit_0
    with pytest.raises(ValueError, match="missing aux labels"):
        auxiliary_loss(out, labels)


def test_aux_heads_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown head names"):
        AuxiliaryHeads(pool_dim=32, head_names=["spine_type", "not_a_head"])
