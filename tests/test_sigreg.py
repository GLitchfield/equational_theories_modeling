"""Tests for SIGReg (LeJEPA Algorithm 1).

Sanity tests verify:
  - Shape and finiteness of the per-slice statistic
  - Loss is small for samples drawn from the target distribution
    (standard Gaussian) and large for non-Gaussian samples
  - Gradients flow back to the input
  - Direction sampling is reproducible with the same global_step
  - Gradients are bounded (LeJEPA Theorem 4)
"""

from __future__ import annotations

import math

import pytest
import torch

from embeddings.src.training.sigreg import sigreg


def test_returns_tensor_of_expected_shape():
    x = torch.randn(64, 32)
    out = sigreg(x, global_step=0, num_slices=8)
    assert out.shape == (8,)
    assert torch.isfinite(out).all()


def test_loss_is_small_for_standard_gaussian_input():
    """Standard Gaussian input matches the target N(0,1); loss should be in
    the finite-N noise floor.

    Note: the paper's bias is O(1/N) (Theorem 6), but Algorithm 1 multiplies
    the per-slice statistic by N at the end, so the floor for a true
    Gaussian is O(1), not O(0). The point of this test is that a true
    Gaussian sits in that floor while the anisotropic/collapsed tests
    below produce values 5-10x larger.
    """
    torch.manual_seed(0)
    x = torch.randn(2048, 64)  # N(0, I)
    out = sigreg(x, global_step=0, num_slices=256)
    assert out.mean().item() < 3.0, (
        f"Expected loss in finite-N floor for Gaussian input, "
        f"got {out.mean().item():.4f}"
    )


def test_loss_is_large_for_anisotropic_input():
    """Heavily anisotropic input should produce much larger loss than Gaussian."""
    torch.manual_seed(0)
    # Same N and K, but the first two dimensions are scaled out
    # (anisotropic — fails the variance test along those directions).
    gauss = torch.randn(2048, 64)
    aniso = gauss.clone()
    aniso[:, 0] *= 10.0
    aniso[:, 1] *= 10.0
    g_loss = sigreg(gauss, global_step=0, num_slices=256).mean().item()
    a_loss = sigreg(aniso, global_step=0, num_slices=256).mean().item()
    assert a_loss > g_loss * 5, (
        f"Expected anisotropic loss to dominate Gaussian; "
        f"got gauss={g_loss:.4f}, aniso={a_loss:.4f}"
    )


def test_loss_is_large_for_collapsed_input():
    """All-zeros input is a degenerate (collapsed) distribution."""
    torch.manual_seed(0)
    x = torch.zeros(2048, 64)
    out = sigreg(x, global_step=0, num_slices=256)
    # Collapsed point mass is far from Gaussian.
    assert out.mean().item() > 5.0, f"Expected large loss for collapsed input, got {out.mean().item():.4f}"


def test_gradient_flows_back_to_input():
    torch.manual_seed(0)
    x = torch.randn(256, 32, requires_grad=True)
    loss = sigreg(x, global_step=0, num_slices=16).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad.abs() > 0).any()  # at least some non-zero gradient


def test_directions_are_reproducible_for_same_global_step():
    """Same global_step → same projection directions → same loss."""
    x = torch.randn(128, 32)
    a = sigreg(x, global_step=42, num_slices=16)
    b = sigreg(x, global_step=42, num_slices=16)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_directions_change_across_steps():
    """Different global_step → different directions → (almost surely) different loss."""
    x = torch.randn(128, 32)
    a = sigreg(x, global_step=0, num_slices=16)
    b = sigreg(x, global_step=1, num_slices=16)
    assert not torch.allclose(a, b), "Expected different directions to produce different statistics"


def test_gradient_norm_is_bounded():
    """LeJEPA Theorem 4: |dEP/dz_i| ≤ 4σ²/N. With σ=1 (Gaussian window),
    per-sample gradient should be ≤ 4/N. Aggregated over N samples, the
    total gradient norm is bounded by a constant — verified empirically
    here by checking that a reasonable input doesn't produce wild grads.
    """
    torch.manual_seed(0)
    x = torch.randn(512, 32, requires_grad=True)
    loss = sigreg(x, global_step=0, num_slices=64).mean()
    loss.backward()
    grad_norm = x.grad.norm().item()
    # Per-sample bound 4/N = 4/512 ≈ 0.008. Total grad norm over 512×32
    # entries should be O(sqrt(512)) at most = ~23 in the worst case;
    # in practice much smaller. Very loose check: not exploding.
    assert grad_norm < 10.0, f"Gradient norm {grad_norm:.4f} exceeds expected bound"


def test_works_on_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.randn(128, 32, device="cuda")
    out = sigreg(x, global_step=0, num_slices=64)
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
