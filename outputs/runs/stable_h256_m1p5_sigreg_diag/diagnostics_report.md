# `stable_h256_m1p5_sigreg_diag` — Diagnostic Report

First SIGReg run: replaces VICReg with SIGReg (LeJEPA Algorithm 1) at
the paper's recommended λ=0.05. Same recipe as
`stable_h256_m1p5_vic0p1_diag` otherwise (margin 1.5, h=256, no-holdout,
10 epochs).

**Result: catastrophic. Best bal_acc 0.6095 — far worse than any prior
run.** The implementation is correct (9 sanity tests pass; math matches
the paper); the problem is that λ=0.05 is grossly mistuned for our
operating regime.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `stable_h256_m1p5_sigreg_diag` |
| Wandb id | `ryd8rgoc` |
| Wandb dir | `wandb/run-20260511_180134-ryd8rgoc/` |
| Config | `configs/experiments/stable_h256_m1p5_sigreg_diag.yaml` |
| Wall clock | 99.3 min (5960.5 s) |
| Outcome | 10 epochs completed, exit code 0 — but training did not converge |
| Encoder params | 799,744 (same as other h=256 runs) |

---

## Per-epoch monitor

| ep | bal_acc | true_R | false_R | thresh | eff_rank | gap | tr_p50 | fa_p50 | over_TaF | over_FbT | val_loss | tr_loss |
|---:|--------:|-------:|--------:|-------:|---------:|----:|-------:|-------:|---------:|---------:|---------:|--------:|
| 0 | 0.5737 | 0.3995 | 0.7480 | 1.4071 | 50.96 | 0.0189 | 1.4244 | 1.4433 | 0.3737 | 0.3628 | 0.7467 | 1.8269 |
| **1** | **0.6095** | 0.4835 | 0.7355 | 1.4075 | 51.82 | 0.0310 | 1.4103 | 1.4414 | 0.3092 | 0.2819 | 0.7359 | 0.9805 |
| 2 | 0.6062 | 0.4829 | 0.7295 | 1.4050 | 52.72 | 0.0315 | 1.4082 | 1.4397 | 0.3198 | 0.2887 | 0.7358 | 0.9709 |
| 3 | 0.5808 | 0.3854 | 0.7762 | 1.3956 | 52.67 | 0.0186 | 1.4182 | 1.4369 | 0.3912 | 0.3669 | 0.7413 | 0.9655 |
| 4 | 0.5885 | 0.4305 | 0.7464 | 1.3991 | 53.41 | 0.0228 | 1.4114 | 1.4342 | 0.3605 | 0.3325 | 0.7359 | 0.9598 |
| 5 | 0.5763 | 0.3487 | 0.8039 | 1.3878 | 53.61 | 0.0133 | 1.4203 | 1.4337 | 0.4226 | 0.4022 | 0.7405 | 0.9546 |
| 6 | 0.5749 | 0.3387 | 0.8111 | 1.3830 | 54.13 | 0.0129 | 1.4155 | 1.4285 | 0.4197 | 0.4018 | 0.7365 | 0.9498 |
| 7 | 0.5856 | 0.3855 | 0.7858 | 1.3869 | 54.42 | 0.0152 | 1.4098 | 1.4249 | 0.4112 | 0.3771 | 0.7311 | 0.9449 |
| 8 | 0.5871 | 0.3940 | 0.7802 | 1.3885 | 54.71 | 0.0151 | 1.4097 | 1.4248 | 0.4111 | 0.3730 | 0.7307 | 0.9404 |
| 9 | 0.5866 | 0.3850 | 0.7882 | 1.3870 | 54.85 | 0.0150 | 1.4101 | 1.4251 | 0.4123 | 0.3750 | 0.7309 | 0.9377 |

Best epoch: ep1 (0.6095). After that, bal_acc drifts down then
plateaus. Recall is *FALSE-biased* — the opposite direction from
VICReg's TRUE bias. TRUE p50 ≈ 1.41 vs FALSE p50 ≈ 1.43 — *the two
classes are essentially overlapping*.

---

## Per-step diagnostic aggregates (mean per epoch)

| ep | total | cont | cl_T | cl_F | grad | g_enc | g_pred | sigreg | lr | tau |
|---:|-----:|-----:|-----:|-----:|-----:|------:|-------:|-------:|----:|----:|
| 0 | 1.9058 | 0.7222 | 1.9259 | 0.0129 | 5.40 | 5.40 | 0.024 | 23.67 | warmup | 0.9960 |
| 1 | 0.9789 | 0.7356 | 1.9721 | 0.0089 | 0.89 | 0.89 | 0.013 | 4.87 | 9.5e-4 | 0.9962 |
| 2 | 0.9695 | 0.7358 | 1.9725 | 0.0089 | 0.90 | 0.90 | 0.014 | 4.67 | 8.6e-4 | 0.9964 |
| 3 | 0.9626 | 0.7356 | 1.9732 | 0.0088 | 0.93 | 0.93 | 0.015 | 4.54 | 7.4e-4 | 0.9968 |
| 4 | 0.9601 | 0.7384 | 1.9734 | 0.0088 | 0.96 | 0.96 | 0.018 | 4.44 | 5.8e-4 | 0.9973 |
| 5 | 0.9547 | 0.7376 | 1.9743 | 0.0087 | 1.00 | 1.00 | 0.021 | 4.34 | 4.4e-4 | 0.9977 |
| 6 | 0.9492 | 0.7376 | 1.9736 | 0.0086 | 1.03 | 1.03 | 0.024 | 4.23 | 2.7e-4 | 0.9982 |
| 7 | 0.9468 | 0.7395 | 1.9741 | 0.0086 | 1.08 | 1.08 | 0.028 | 4.15 | 1.5e-4 | 0.9986 |
| 8 | 0.9382 | 0.7355 | 1.9742 | 0.0086 | 1.11 | 1.11 | 0.031 | 4.05 | 6e-5 | 0.9988 |
| 9 | 0.9373 | 0.7373 | 1.9735 | 0.0086 | 1.12 | 1.12 | 0.033 | 4.00 | 0 | 0.9990 |

### Clip behavior + SIGReg ranges

| ep | max grad | clip fraction | max sigreg | min sigreg |
|---:|---------:|--------------:|-----------:|-----------:|
| 0 | 127.86 | **0.937** | 1396.46 | 4.98 |
| 1 | 1.13 | 0.070 | 5.56 | 4.23 |
| 2 | 1.12 | 0.069 | 5.29 | 4.16 |
| 3 | 1.11 | 0.171 | 5.01 | 4.01 |
| 4 | 1.16 | 0.289 | 4.88 | 3.90 |
| 5 | 1.31 | 0.459 | 4.91 | 3.86 |
| 6 | 1.32 | **0.654** | 4.77 | 3.76 |
| 7 | 1.37 | **0.824** | 4.59 | 3.69 |
| 8 | 1.36 | **0.901** | 4.46 | 3.65 |
| 9 | 1.41 | **0.922** | 4.45 | 3.63 |

The training is in a perpetually clip-saturated regime by mid-late
training (>90% of steps clipped in ep8–9).

---

## Diagnosis — SIGReg dominates, contrastive starved

**TRUE-pair contrastive loss is stuck at ~1.97 throughout the entire
run.** For comparison:
- h=128 VICReg baseline: cl_TRUE settled at ~1.43
- h=256 VICReg 0.1 baseline: ~1.59
- h=256 VICReg 0.05 baseline: ~1.46

cl_TRUE = 1.97 ≈ 2.0 means TRUE-pair distances ≈ √1.97 ≈ 1.40 — that's
the value you get for *random unit-sphere pairs*. **The encoder never
learned to pull TRUE pairs together at all.**

The mechanism, traced through the per-step diagnostics:

1. **At init, SIGReg loss is ~1400.** The freshly-initialized 800K-param
   GAT produces wildly non-Gaussian pre-norm features, so the
   regularizer has enormous early mass.
2. **With λ=0.05, that's 70 units of loss from SIGReg** vs 0.84 from
   contrastive — an 80× imbalance. Encoder gradients hit 127 at step 0;
   the clip threshold (1.0) fires every step in ep0 (94%).
3. **The encoder is yanked toward Gaussian very fast** — SIGReg drops
   from 1400 → 5 in epoch 0. By ep1 it's settled at ~4.5 steady-state.
4. **But contrastive never gets a chance to shape geometry.** The
   encoder ends up producing roughly-isotropic features in pre-norm
   space, which after L2-normalization land roughly uniformly on the
   unit sphere — the *opposite* of what the contrastive task needs
   (which wants TRUE pairs *close* and FALSE pairs *spread*).
5. **Even in steady state, SIGReg contributes 30% of the loss
   gradient.** With cl_TRUE stuck at 1.97 (the cap for contrastive
   on unit sphere), the contrastive gradient has nowhere to push.

The encoder ended up in a "Gaussian but uninformative" attractor.

### Why this happens here but not in the paper

The LeJEPA paper validates λ=0.05 across ResNet-50 to ViT-g (1.8B
params) on ImageNet. Why does it fail at our smaller 800K-param GAT
on equation graphs?

- **Initial SIGReg mass scales with how non-Gaussian the encoder's
  initial output is.** Larger pre-trained or pre-normalized models
  start much closer to Gaussian. Our random-init small GAT does not.
- **Their batch sizes (≥128, often 2048+) and the `* N` scaling in
  Algorithm 1 mean their absolute SIGReg values are also high — but
  their contrastive (or prediction) losses are correspondingly higher
  too,** so the ratio is balanced.
- **Their warmup schedules and learning rates are tuned for the
  combined loss landscape.** We dropped SIGReg into our existing
  recipe without retuning.
- **Most importantly: the `* N` scaling makes the SIGReg loss scale
  with batch size, so the recommended λ is implicitly tied to a batch
  size norm.** At our batch=2048 with random-init features, the early
  warmup mass overwhelms everything.

---

## Comparison

| run | regularizer | best ep | bal_acc | TRUE recall | FALSE recall | TRUE p50 | gap_median |
|---|---|---:|---:|---:|---:|---:|---:|
| h=128 stable_m1p5_vic0p1 | VICReg 0.1/0.1 | 6 | **0.9492** | 0.95 | 0.95 | 1.18 | 0.31 |
| h=256 stable_m1p5_vic0p1 | VICReg 0.1/0.1 | 0 | 0.9333 | 0.96 | 0.91 | 1.23 | 0.27 |
| h=256 stable_m1p5_vic0p05 | VICReg 0.05/0.05 | 0 | 0.9469 | 0.94 | 0.95 | 1.15 | 0.35 |
| **h=256 stable_m1p5_sigreg** | SIGReg λ=0.05 | 1 | **0.6095** | 0.48 | 0.74 | 1.41 | 0.03 |

---

## What to try next

The implementation is sound (9 sanity tests pass, math matches paper).
The issue is mistuned λ for our regime. Options ranked by impact:

1. **Drop λ 10–50×** — try λ=0.005 or λ=0.001. Direct single-knob fix.
   At λ=0.005: steady-state contribution 0.005 × 4.5 = 0.022 (3% of
   contrastive); warmup contribution 0.005 × 1400 = 7 (still 10×
   contrastive at warmup, but recoverable).
2. **Add a SIGReg warmup schedule** — ramp λ from 0 to target over
   first ~1000 steps. Lets contrastive establish geometry first.
   Requires code change but more principled.
3. **Combine** — λ=0.005 with warmup. Belt and braces.

Recommended next: option 1 with λ=0.005. Single-knob test, directly
addresses the diagnosed problem, takes the same ~100 min.

If λ=0.005 still collapses, there's a deeper interaction between
SIGReg (which shapes pre-norm distribution) and our L2-normalized
predictor / unit-sphere contrastive geometry that needs investigation.
The paper's setup uses a different prediction loss (MSE between views)
and doesn't necessarily L2-normalize the predictor output.

---

## Headline numbers

- **Best: ep1 bal_acc 0.6095** — disastrous; 33pp below VICReg baseline.
- **TRUE-pair contrastive loss stuck at 1.97** the entire run — encoder
  never learned to pull TRUE pairs together.
- **gap_median 0.013–0.031** vs VICReg's 0.27–0.35. Classes overlap.
- **eff_rank 50–55** vs VICReg's 145–350. Encoder is low-effective-rank.
- **Clip fires 92% of steps in ep9** — perpetually saturated.
- **SIGReg loss starts at 1400, settles at ~4.** With λ=0.05, even
  steady-state contributes 30% of total loss.
- **Mechanism**: λ=0.05 (paper default) is mistuned for our small-GAT
  random-init regime. SIGReg's enormous warmup gradient drives the
  encoder into an uninformative Gaussian attractor before contrastive
  can shape the geometry.
