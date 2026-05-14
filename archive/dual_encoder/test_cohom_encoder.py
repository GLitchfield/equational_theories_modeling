"""Tests for the cohomology MLP."""

from __future__ import annotations

import pytest
import torch

from src.model.cohom_encoder import CohomologyEncoder


def test_default_shape_and_param_count():
    enc = CohomologyEncoder()
    assert enc.input_dim == 174
    assert enc.hidden_dim == 128
    assert enc.output_dim == 128
    # Linear(174, 128) + Linear(128, 128) + biases = 174*128 + 128 + 128*128 + 128 = 38912
    assert enc.num_parameters() == 174 * 128 + 128 + 128 * 128 + 128


def test_forward_shape():
    enc = CohomologyEncoder()
    x = torch.randn(8, 174)
    y = enc(x)
    assert y.shape == (8, 128)


def test_custom_dims():
    enc = CohomologyEncoder(input_dim=87, hidden_dim=64, output_dim=32)
    x = torch.randn(4, 87)
    y = enc(x)
    assert y.shape == (4, 32)
    assert enc.num_parameters() == 87 * 64 + 64 + 64 * 32 + 32


def test_rejects_wrong_input_dim():
    enc = CohomologyEncoder(input_dim=174)
    with pytest.raises(ValueError, match="174"):
        enc(torch.randn(4, 87))


def test_rejects_non_2d_input():
    enc = CohomologyEncoder()
    with pytest.raises(ValueError):
        enc(torch.randn(174))


def test_gradient_flow():
    enc = CohomologyEncoder()
    x = torch.randn(4, 174, requires_grad=True)
    y = enc(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad.abs() > 0).any()


def test_zero_input_produces_finite_output():
    """All-null equations get an all-zero profile; the MLP must handle it."""
    enc = CohomologyEncoder()
    x = torch.zeros(4, 174)
    y = enc(x)
    assert torch.isfinite(y).all()
    # Output is the second linear layer's bias for all-zero input
    # (first layer's bias passes through ReLU only if non-negative).
    assert y.shape == (4, 128)


def test_works_on_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    enc = CohomologyEncoder().cuda()
    x = torch.randn(8, 174, device="cuda")
    y = enc(x)
    assert y.device.type == "cuda"
    assert y.shape == (8, 128)
