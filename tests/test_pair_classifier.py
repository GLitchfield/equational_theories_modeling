"""Unit tests for ``src/model/pair_classifier.py``.

Cover shape contracts, parameter counts (so accidental architecture
changes show up), and the "deterministic on frozen inputs" property
that downstream evaluation depends on.
"""

from __future__ import annotations

import pytest
import torch

from src.model.pair_classifier import (
    MODELS,
    PairLinear,
    PairMLPConcat,
    PairMLPFeatures,
    build_model,
)


# ---------- shape contracts ----------

def _z(batch: int = 4, dim: int = 768) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(batch, dim), torch.randn(batch, dim)


def test_pair_linear_returns_one_logit_per_pair():
    m = PairLinear(embed_dim=768)
    z1, z2 = _z()
    out = m(z1, z2)
    assert out.shape == (4, 1)


def test_pair_mlp_concat_returns_one_logit_per_pair():
    m = PairMLPConcat(embed_dim=768)
    z1, z2 = _z()
    out = m(z1, z2)
    assert out.shape == (4, 1)


def test_pair_mlp_features_returns_one_logit_per_pair():
    m = PairMLPFeatures(embed_dim=768)
    z1, z2 = _z()
    out = m(z1, z2)
    assert out.shape == (4, 1)


def test_models_accept_smaller_embedding_dim():
    z1, z2 = _z(batch=2, dim=256)
    for name in MODELS:
        m = build_model(name, embed_dim=256)
        out = m(z1, z2)
        assert out.shape == (2, 1), f'{name} returned shape {tuple(out.shape)}'


# ---------- parameter counts ----------
# These are the headline architecture numbers in the plan. A surprising
# change here means the model spec drifted from what the plan promised.

def test_pair_linear_param_count():
    # Linear(2*768=1536, 1) -> 1536 + 1 = 1537
    assert PairLinear(embed_dim=768).num_parameters() == 1537


def test_pair_mlp_concat_param_count():
    # Linear(1536, 512) + Linear(512, 256) + Linear(256, 1)
    # = (1536*512 + 512) + (512*256 + 256) + (256 + 1)
    # = 786,944         + 131,328          + 257
    # = 918,529
    assert PairMLPConcat(embed_dim=768).num_parameters() == 918_529


def test_pair_mlp_features_param_count():
    # Linear(3072, 512) + Linear(512, 256) + Linear(256, 1)
    # = (3072*512 + 512) + 131,328         + 257
    # = 1,573,376         + 131,328         + 257
    # = 1,704,961
    assert PairMLPFeatures(embed_dim=768).num_parameters() == 1_704_961


# ---------- determinism & gradient flow ----------

def test_eval_mode_is_deterministic():
    """Dropout is off in eval(); same input twice must give same output."""
    m = PairMLPConcat(embed_dim=768).eval()
    z1, z2 = _z()
    out_a = m(z1, z2)
    out_b = m(z1, z2)
    assert torch.allclose(out_a, out_b)


def test_train_mode_uses_dropout():
    """In train(), two passes on the same input should differ due to dropout."""
    torch.manual_seed(1)
    m = PairMLPConcat(embed_dim=768).train()
    z1, z2 = _z()
    out_a = m(z1, z2)
    out_b = m(z1, z2)
    assert not torch.allclose(out_a, out_b), 'dropout should make outputs differ across passes in train mode'


def test_gradient_flows_to_classifier_not_to_embeddings():
    """Frozen-embedding training: grad on the head, no grad on inputs.

    The training script never sets requires_grad on the embedding tensors,
    so this test pins that contract.
    """
    m = PairMLPConcat(embed_dim=768)
    z1, z2 = _z()
    # Inputs are leaves with requires_grad=False (the typical training case).
    assert not z1.requires_grad and not z2.requires_grad
    logit = m(z1, z2).sum()
    logit.backward()
    # Every classifier parameter received a gradient.
    for name, p in m.named_parameters():
        assert p.grad is not None, f'no grad on {name}'
        assert p.grad.abs().sum() > 0, f'zero grad on {name}'


# ---------- factory ----------

def test_build_model_dispatches_by_name():
    assert isinstance(build_model('linear'), PairLinear)
    assert isinstance(build_model('mlp_concat'), PairMLPConcat)
    assert isinstance(build_model('mlp_features'), PairMLPFeatures)


def test_build_model_rejects_unknown():
    with pytest.raises(ValueError, match='unknown pair-classifier name'):
        build_model('nope')


# ---------- input asymmetry (z_e1 vs z_e2) ----------

def test_mlp_concat_is_order_sensitive():
    """concat(z1, z2) != concat(z2, z1) — the MLP can distinguish them.

    This matters: implication is directional (E1 -> E2), so the
    classifier must treat the two sides asymmetrically.
    """
    m = PairMLPConcat(embed_dim=768).eval()
    z1, z2 = _z()
    out_12 = m(z1, z2)
    out_21 = m(z2, z1)
    assert not torch.allclose(out_12, out_21)
