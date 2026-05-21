"""Sketched Isotropic Gaussian Regularization (SIGReg).

From LeJEPA (Balestriero & LeCun, 2025), Algorithm 1 + Section 4.

SIGReg pushes the encoder's pre-norm embedding distribution toward an
isotropic standard Gaussian by:

  1. Sampling `num_slices` random unit-norm directions in R^embed_dim
  2. Projecting each embedding onto each direction (univariate)
  3. Comparing the empirical characteristic function (ECF) of each
     projection against the theoretical CF of N(0, 1) using the
     Epps-Pulley test
  4. Returning per-slice statistics (caller takes .mean() for scalar loss)

Differences from VICReg:
  - VICReg matches only moments 1-2 (mean, var, cov); SIGReg matches
    the full distribution via the characteristic function
  - SIGReg has provably bounded gradients (LeJEPA Theorem 4)
  - SIGReg has a single hyperparameter (lambda) that the paper validates
    as width- and architecture-independent

Single-GPU implementation. For DDP, the ECF averaging step would need
an all_reduce; not relevant for our single-RTX-4080 setup.
"""

from __future__ import annotations

import torch


def sigreg(
    x: torch.Tensor,
    global_step: int,
    num_slices: int = 1024,
    integration_points: int = 17,
    integration_range: float = 5.0,
) -> torch.Tensor:
    """Compute per-slice Epps-Pulley statistics for SIGReg.

    Args:
        x: (N, K) tensor of encoder pre-norm outputs. N = batch size,
           K = embedding dimension.
        global_step: integer training step. Used to seed direction
           sampling so directions are reproducible and (in DDP)
           synchronized across ranks. Must increment monotonically.
        num_slices: number of random projection directions (|A| in paper).
           Paper recommends 1024 as default.
        integration_points: number of trapezoidal-rule knots over the
           integration range. Paper says 17 is sufficient (Table 1a).
        integration_range: half-width of the integration domain
           [-range, range]. Paper recommends 5.0.

    Returns:
        Tensor of shape (num_slices,) — per-slice Epps-Pulley statistics.
        Caller should .mean() to get the scalar loss.

    Notes:
        - Apply to the encoder's pre-normalization output, NOT the
          L2-normalized one. SIGReg shapes the unconstrained distribution.
        - Bounded gradients per LeJEPA Theorem 4: |dEP/dz_i| ≤ 4σ²/N.
        - Memory footprint dominated by the (N, num_slices,
          integration_points) complex tensor of exp(i*t*<a,x>).
    """
    dev = {"device": x.device}

    # Sample random unit-norm projection directions, seeded by step.
    # Resampling every step is critical (LeJEPA Figure 7 — cumulative
    # coverage from resampling beats fixed-direction sets by orders of
    # magnitude).
    g = torch.Generator(**dev)
    g.manual_seed(int(global_step))
    A = torch.randn((x.size(1), num_slices), generator=g, **dev)
    A = A / A.norm(p=2, dim=0)  # unit columns

    # Integration points for the trapezoidal-rule estimate of the
    # weighted L2 distance between empirical and theoretical CFs.
    t = torch.linspace(-integration_range, integration_range,
                       integration_points, **dev)

    # Theoretical characteristic function of N(0, 1) at t is exp(-t²/2).
    # The Epps-Pulley weight w(t) is also exp(-t²/2) (Gaussian window
    # with sigma=1; paper's recommended choice). The two coincide in
    # this implementation: the same exp_f vector is both the target CF
    # and the weight.
    exp_f = torch.exp(-0.5 * t.pow(2))

    # Empirical characteristic function: (1/N) Σ exp(i * t * <a, x_n>).
    # Project: (N, num_slices), then broadcast against t to get the
    # exponent grid (N, num_slices, integration_points).
    x_t = (x @ A).unsqueeze(2) * t           # (N, M, T)
    ecf = (1j * x_t).exp().mean(dim=0)       # (M, T), complex

    # If using DDP, we'd all_reduce(ecf, op='AVG') here so the test
    # statistic uses the full global batch. Single-GPU: skip.

    # Weighted L2 distance |ECF - target_CF|² · w(t).
    # ecf is complex (M, T); exp_f is real (T,) — broadcasts, abs().square()
    # returns real, then weighted by exp_f.
    err = (ecf - exp_f).abs().square() * exp_f         # (M, T), real

    # Trapezoidal integration over the t axis. The * N scaling matches
    # the paper's Algorithm 1 — makes the test statistic scale with the
    # sample size, standard for Epps-Pulley.
    N = x.size(0)
    return torch.trapezoid(err, t, dim=1) * N          # (M,)
