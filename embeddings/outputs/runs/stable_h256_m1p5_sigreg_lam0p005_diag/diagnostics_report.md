# `stable_h256_m1p5_sigreg_lam0p005_diag` — Diagnostic Report

Second SIGReg run: λ dropped 10× from the paper default (0.05 → 0.005)
to fix the warmup-clip-saturation problem from the previous run. Same
recipe as `stable_h256_m1p5_sigreg_diag` otherwise.

**Result: stability is fixed, but the model still doesn't learn the
task. Best bal_acc 0.6891.** `cl_TRUE` is pinned at 1.97 (random
unit-sphere distance²) the entire run, same failure mode as λ=0.05 —
SIGReg's distributional pressure prevents the contrastive force from
pulling TRUE pairs together. **The problem is architectural, not λ
tuning.**

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `stable_h256_m1p5_sigreg_lam0p005_diag` |
| Wandb id | `uu4dxh64` |
| Wandb dir | `wandb/run-20260511_194804-uu4dxh64/` |
| Config | `configs/experiments/stable_h256_m1p5_sigreg_lam0p005_diag.yaml` |
| Wall clock | 93.7 min (5625.0 s) |
| Outcome | 10 epochs completed, exit code 0 — but model did not learn task |

Single-knob change from `stable_h256_m1p5_sigreg_diag.yaml`:
`sigreg_lambda` 0.05 → 0.005.

---

## Per-epoch monitor

| ep | bal_acc | true_R | false_R | thresh | eff_rank | gap | tr_p50 | fa_p50 | over_TaF | over_FbT | val_loss | tr_loss |
|---:|--------:|-------:|--------:|-------:|---------:|----:|-------:|-------:|---------:|---------:|---------:|--------:|
| 0 | 0.6635 | 0.5704 | 0.7566 | 1.3977 | 45.60 | 0.0452 | 1.3841 | 1.4293 | 0.2694 | 0.1896 | 0.7152 | 0.8340 |
| **1** | **0.6891** | 0.6153 | 0.7629 | 1.4007 | 50.90 | 0.0495 | 1.3797 | 1.4292 | 0.2422 | 0.1576 | 0.7119 | 0.7617 |
| 2 | 0.6766 | 0.5943 | 0.7588 | 1.4009 | 52.77 | 0.0440 | 1.3846 | 1.4286 | 0.2501 | 0.1736 | 0.7137 | 0.7596 |
| 3 | 0.6677 | 0.5695 | 0.7658 | 1.4005 | 52.78 | 0.0386 | 1.3891 | 1.4277 | 0.2682 | 0.1794 | 0.7157 | 0.7589 |
| 4 | 0.6680 | 0.5602 | 0.7759 | 1.3975 | 55.38 | 0.0397 | 1.3866 | 1.4263 | 0.2869 | 0.1795 | 0.7165 | 0.7584 |
| 5 | 0.6667 | 0.5585 | 0.7749 | 1.3967 | 57.85 | 0.0414 | 1.3850 | 1.4264 | 0.2923 | 0.1802 | 0.7139 | 0.7578 |
| 6 | 0.6810 | 0.5689 | 0.7931 | 1.3964 | 59.20 | 0.0448 | 1.3827 | 1.4275 | 0.2680 | 0.1606 | 0.7106 | 0.7571 |
| 7 | 0.6848 | 0.5816 | 0.7880 | 1.3995 | 60.47 | 0.0438 | 1.3835 | 1.4273 | 0.2710 | 0.1578 | 0.7102 | 0.7565 |
| 8 | 0.6847 | 0.5754 | 0.7941 | 1.3983 | 61.83 | 0.0433 | 1.3831 | 1.4263 | 0.2788 | 0.1565 | 0.7090 | 0.7561 |
| 9 | 0.6819 | 0.5730 | 0.7908 | 1.3992 | 62.24 | 0.0425 | 1.3844 | 1.4269 | 0.2836 | 0.1596 | 0.7100 | 0.7558 |

bal_acc bumped from 0.61 (λ=0.05) to 0.69 — modest but still terrible
relative to VICReg's 0.93+. Recall is FALSE-biased like λ=0.05.

---

## Per-step diagnostic aggregates

| ep | total | cont | cl_TRUE | cl_FALSE | grad | g_enc | g_pred | sigreg | lr | tau |
|---:|-----:|-----:|--------:|---------:|-----:|------:|-------:|-------:|----:|----:|
| 0 | 0.8414 | 0.7210 | 1.9230 | 0.0128 | 0.60 | 0.60 | 0.023 | 24.08 | warmup | 0.9960 |
| 1 | 0.7601 | 0.7340 | 1.9677 | 0.0089 | 0.12 | 0.12 | 0.012 | 5.21 | 9.5e-4 | 0.9962 |
| 2 | 0.7581 | 0.7343 | 1.9685 | 0.0088 | 0.11 | 0.10 | 0.012 | 4.76 | 8.6e-4 | 0.9964 |
| 3 | 0.7568 | 0.7337 | 1.9684 | 0.0087 | 0.11 | 0.11 | 0.014 | 4.62 | 7.4e-4 | 0.9968 |
| 4 | 0.7591 | 0.7365 | 1.9686 | 0.0087 | 0.11 | 0.11 | 0.016 | 4.52 | 5.8e-4 | 0.9973 |
| 5 | 0.7575 | 0.7356 | 1.9690 | 0.0086 | 0.12 | 0.11 | 0.019 | 4.38 | 4.4e-4 | 0.9977 |
| 6 | 0.7570 | 0.7357 | 1.9686 | 0.0085 | 0.12 | 0.12 | 0.023 | 4.27 | 2.7e-4 | 0.9982 |
| 7 | 0.7580 | 0.7373 | 1.9683 | 0.0084 | 0.13 | 0.13 | 0.029 | 4.14 | 1.5e-4 | 0.9986 |
| 8 | 0.7540 | 0.7338 | 1.9699 | 0.0084 | 0.14 | 0.13 | 0.033 | 4.03 | 6e-5 | 0.9988 |
| 9 | 0.7553 | 0.7354 | 1.9686 | 0.0083 | 0.14 | 0.13 | 0.036 | 3.99 | 0 | 0.9990 |

### Stability is fixed

| ep | max grad | clip_frac |
|---:|---------:|----------:|
| 0 | 12.81 | 0.158 |
| 1 | 0.17 | 0.000 |
| 2 | 0.13 | 0.000 |
| ... | ... | 0.000 |
| 9 | 0.17 | 0.000 |

Zero clipping ep1–9 (vs 92% for λ=0.05). The training is "stable" in
the sense of well-behaved gradients. But the model isn't learning.

---

## Diagnosis — structural incompatibility, not λ tuning

`cl_TRUE` is **stuck at ~1.97 the entire run** in both SIGReg variants:

| step | λ=0.005 cl_TRUE |
|---:|---:|
| 0 | 2.22 |
| 100 | 1.11 |
| **150** | **0.58** ← contrastive briefly wins |
| 250 | 1.05 |
| ep1 mean | **1.97** |
| ep9 mean | **1.97** |

The contrastive force *can* pull TRUE pairs together initially (cl_TRUE
hits 0.58 at step 150), but SIGReg gradually undoes it within the
first epoch. λ=0.005 isn't lowering the strength enough to stop the
slide; it's just slowing it.

`cl_TRUE = 1.97 ≈ 2.0 = random unit-sphere distance²`. TRUE pairs end
up at the same distance as random pairs. The encoder learned to
produce isotropic Gaussian pre-norm features (good per SIGReg), which
after L2-normalization land uniformly on the unit sphere — and on a
uniform sphere, no TRUE pair is closer than random.

### The architectural conflict

- Our encoder has **L2-normalization on the predictor output** (unit
  sphere; margin=1.5 in [0, 2] range)
- SIGReg shapes the **pre-norm distribution** toward N(0, I)
- Pre-norm N(0, I) → after L2-norm → **uniform on unit sphere**
- Contrastive on the sphere wants TRUE pairs **clustered**
  (anisotropic on sphere) and FALSE pairs **spread**
- These objectives **directly conflict**: SIGReg → uniform on sphere;
  contrastive → clustered on sphere

The LeJEPA paper sidesteps this because it uses **MSE on unnormalized
embeddings** (Section 5.1, Eq. 5–7) — no L2 normalization, no
unit-sphere geometry. Their predictor and prediction targets live in
the same unconstrained space SIGReg operates on.

For us, dropping λ further just makes SIGReg ineffective. λ=0.0005
would land at "SIGReg disabled" performance, not "SIGReg working."

---

## Comparison

| run | regularizer | best ep | bal_acc | true_R | false_R | cl_TRUE end | gap_median | eff_rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| h=128 stable_m1p5_vic0p1 | VICReg 0.1/0.1 | 6 | **0.9492** | 0.95 | 0.95 | 1.43 | 0.31 | 146 |
| h=256 stable_m1p5_vic0p1 | VICReg 0.1/0.1 | 0 | 0.9333 | 0.96 | 0.91 | 1.59 | 0.27 | 206 |
| h=256 stable_m1p5_vic0p05 | VICReg 0.05/0.05 | 0 | 0.9469 | 0.94 | 0.95 | 1.46 | 0.35 | 136 |
| h=256 stable_m1p5_sigreg λ=0.05 | SIGReg 0.05 | 1 | 0.6095 | 0.48 | 0.74 | **1.97** | 0.03 | 52 |
| **h=256 stable_m1p5_sigreg λ=0.005** | SIGReg 0.005 | 1 | 0.6891 | 0.62 | 0.76 | **1.97** | 0.04 | 62 |

Both SIGReg variants land in the same failure mode: TRUE pairs at
random-pair distance, classes overlapping, eff-rank under 65.

---

## Where to go from here

Three paths:

1. **Remove L2-normalization from the predictor output** to match
   LeJEPA's setup. This directly addresses the architectural conflict.
   Substantial recipe rework — `margin=1.5` no longer lives in [0, 2],
   contrastive loss landscape changes.
2. **Apply SIGReg post-norm with a sphere-target distribution** (e.g.
   uniform on S^{K-1}). Requires deriving an Epps-Pulley equivalent
   for the spherical uniform; the paper doesn't cover this case. Real
   custom math.
3. **Accept SIGReg isn't a fit for this architecture**, drop the
   SIGReg track, and address h=256 via other means: early-stop at
   ep0 (h=256/vic=0.05 already gets 0.9469 at ep0), LR schedule
   tweaks, true_weight scheduling, etc.

---

## Headline numbers

- **Best: ep1 bal_acc 0.6891** — modest improvement over λ=0.05's 0.6095,
  but still 26pp below h=128 VICReg baseline.
- **Stability fixed**: zero clipping ep1–9, post-warmup max grad 0.17.
- **`cl_TRUE` stuck at 1.97** the entire run = random unit-sphere
  distance². TRUE pairs not pulled together.
- **gap_median 0.04–0.05** — TRUE/FALSE classes essentially overlap.
- **eff_rank 45–62** — encoder operates in a low-effective-rank regime.
- **Mechanism**: SIGReg's pre-norm Gaussian target conflicts with our
  L2-normalized predictor's unit-sphere geometry. After L2-norm,
  Gaussian pre-norm becomes uniform on sphere, which is the *opposite*
  of what TRUE-pair clustering needs. Lowering λ slows but doesn't
  solve the conflict.
