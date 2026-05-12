# `stable_h256_lejepa_3ep_diag` — Diagnostic Report

LeJEPA-style: SIGReg with L2-normalization removed from encoder and
predictor outputs. Tests whether the prior SIGReg failure (TRUE pairs
pinned at random unit-sphere distance) was caused by the L2-norm /
SIGReg structural conflict.

**Result: L2-norm conflict resolved, but a new failure mode emerged.
Best bal_acc 0.5545 (essentially chance). TRUE pairs got *further
apart* during training.** Clip fires 100% of steps in all 3 epochs —
contrastive force is rate-limited and SIGReg's expand-features
pressure dominates.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `stable_h256_lejepa_3ep_diag` |
| Wandb id | `abu9a47l` |
| Wandb dir | `wandb/run-20260511_214000-abu9a47l/` |
| Config | `configs/experiments/stable_h256_lejepa_3ep_diag.yaml` |
| Wall clock | 30.8 min (1848.8 s) |
| Outcome | 3 epochs completed, exit code 0 — but model did not learn |

Architectural change vs prior runs: `model.l2_normalize_outputs: false`
(default true). Encoder and predictor outputs unconstrained R^512.
SIGReg shapes the encoder distribution; contrastive operates in
unconstrained space. Margin rescaled 1.5 → 25.0.

---

## Per-epoch monitor

| ep | bal_acc | true_R | false_R | thresh | eff_rank | gap | tr_p50 | fa_p50 | val_loss | tr_loss |
|---:|--------:|-------:|--------:|-------:|---------:|----:|-------:|-------:|---------:|--------:|
| 0 | 0.5241 | 0.3261 | 0.7221 | 1.346 | 73.98 | 0.008 | 1.389 | 1.397 | 351.30 | 181.93 |
| 1 | 0.5450 | 0.5298 | 0.5603 | 1.413 | 87.59 | 0.018 | 1.407 | 1.425 | 350.89 | 191.13 |
| **2** | **0.5545** | 0.4959 | 0.6130 | 1.421 | 91.20 | 0.023 | 1.422 | 1.445 | 350.69 | 190.90 |

Note: val distances ~1.4 because the eval path still applies L2-norm
(JEPA.forward default `normalize_outputs=True` for evaluation). Eval
measures direction-only similarity on the unit sphere; training
optimizes contrastive on raw distances at scale ~22. The val numbers
show the *direction* geometry (which is what threshold-calibration
sees) is essentially uniform — gap_median 0.008–0.023, classes
overlap.

---

## Per-step diagnostic aggregates

| ep | loss | cont | cl_TRUE | cl_FALSE | grad | g_enc | g_pred | sigreg | lr |
|---:|-----:|-----:|--------:|---------:|-----:|------:|-------:|-------:|----:|
| 0 | 181.83 | 180.15 | **478.91** | 4.16 | 11.88 | 11.21 | 3.68 | 33.68 | warmup |
| 1 | 190.62 | 190.20 | **511.02** | 1.64 | 4.48 | 3.67 | 2.56 | 8.29 | 5.2e-4 |
| 2 | 190.41 | 190.09 | **511.0** | 1.48 | 3.88 | 2.90 | 2.57 | 6.34 | 9e-5 |

### Clip behavior + SIGReg

| ep | max grad | clip_frac | max sigreg | min sigreg |
|---:|---------:|----------:|-----------:|-----------:|
| 0 | 127.53 | **1.000** | 1396.46 | 8.86 |
| 1 | 6.94 | **1.000** | 10.76 | 6.54 |
| 2 | 5.45 | **1.000** | 7.46 | 5.26 |

---

## What happened — SIGReg won, contrastive lost

Three observations to read together:

1. **`cl_TRUE` got worse** (478 → 511). TRUE-pair distance went from
   √478 ≈ 21.9 → √511 ≈ 22.6 — *increasing*. And 22.6² = 511 ≈ K = 512 =
   E[||x||²] for x ~ N(0, I) in R^512. TRUE pairs ended up at "single-
   sample-norm" distance, which is the random-pair distance for
   orthogonal samples in R^K. **The encoder produced features
   uniformly distributed in N(0, I), with no clustering whatsoever.**
   Same flavor of failure as the L2-norm SIGReg runs (TRUE pairs at
   random-pair distance), just expressed in unconstrained space.
2. **Clip fires 100% of steps in all 3 epochs**, not just warmup. With
   clip=1.0 and pre-clip grads at 4–12, every step is rate-limited.
   The contrastive pull-together signal gets normalized in with
   everything else and can't outcompete SIGReg's globally-coherent
   spread-features pressure.
3. **`cl_FALSE` worked fine** (4.16 → 1.48). FALSE pairs successfully
   pushed past margin. SIGReg's "spread to N(0, I)" pressure is
   *aligned* with FALSE-push (both want spread); both fight
   TRUE-pull.

The architectural conflict from prior runs is gone — there's no L2-norm
forcing features onto the unit sphere. But a new pathology emerged:
in unconstrained space, SIGReg's per-step gradient mass is enough to
saturate the clip every step, and within the clip-normalized step,
SIGReg's globally-consistent direction beats contrastive's per-pair
gradient. The model converged to "isotropic Gaussian features without
semantic clustering" — exactly LeJEPA's stated failure mode if
prediction loss isn't strong enough.

---

## Comparison

| run | regularizer | l2_norm | best ep | bal_acc | cl_TRUE end | gap | clip% |
|---|---|:-:|---:|---:|---:|---:|---:|
| h=128 stable_m1p5_vic0p1 | VICReg 0.1/0.1 | yes | 6 | **0.9492** | 1.43 | 0.31 | 0% |
| h=256 stable_m1p5_vic0p05 | VICReg 0.05/0.05 | yes | 0 | 0.9469 | 1.46 | 0.35 | 0% |
| h=256 sigreg λ=0.05 | SIGReg 0.05 | yes | 1 | 0.6095 | 1.97 | 0.03 | 92% |
| h=256 sigreg λ=0.005 | SIGReg 0.005 | yes | 1 | 0.6891 | 1.97 | 0% |
| **h=256 lejepa 3ep** | SIGReg 0.05 | **no** | 2 | **0.5545** | **511** | 0.02 | **100%** |

`cl_TRUE` numbers are not directly comparable across L2-norm settings
(distance scale differs by sqrt(K)), but the pattern is the same: in
all three SIGReg variants, TRUE pairs end up at the regularizer-
preferred typical distance (random unit-sphere or random R^K),
gap_median is near zero, classes overlap.

---

## What to try next

The clip-saturation is the new bottleneck. Options:

1. **Raise clip threshold to ~10–20** (or remove). Direct fix; cheapest
   test. Lets the contrastive force drive geometry rather than
   competing with SIGReg in a clip-normalized step.
2. **Lower λ to 0.005 + raise clip** — combine softer SIGReg with
   clip headroom.
3. **Rescale contrastive loss by 1/K** (divide d² by 512). Brings
   contrastive magnitudes back to O(1) so the existing clip=1.0
   isn't grossly mismatched. Effectively mirrors what L2-norm was
   doing implicitly.
4. **Step back** — the "SIGReg won, contrastive starved" pattern is
   consistent across all three SIGReg variants we've tried. May be
   fundamental for our setup (small encoder, small batch, our
   contrastive task). LeJEPA's prediction loss is MSE between same-
   instance views; ours is contrastive-with-margin between paired
   instances with TRUE/FALSE labels. The task structure may be
   incompatible with SIGReg-as-anti-collapse.

Recommended next: option 1 (raise clip to ~10–20), 3-epoch test. If
clip-out-of-the-way lets `cl_TRUE` actually drop, the rate-limiting
was the bottleneck. If `cl_TRUE` still doesn't drop with clip out of
the way, the conflict is more fundamental and option 4 (drop SIGReg
track) is in scope.

---

## Headline numbers

- **Best: ep2 bal_acc 0.5545** — barely above chance.
- **`cl_TRUE` increased** (478 → 511) — TRUE pairs pulled *apart*, not
  together. Ended at single-sample-norm distance = the orthogonal-
  random-pair distance for R^512 Gaussian features.
- **Clip fires 100% of steps in all 3 epochs.** Contrastive force is
  perpetually rate-limited.
- **gap_median 0.008–0.023** in val (sphere-projection of features).
  Classes essentially overlap.
- **eff_rank 74–91** — better than L2-norm SIGReg runs (45–62) but
  still well below VICReg baselines (135–350).
- **Mechanism**: removing L2-norm dissolved the prior structural
  conflict, but in unconstrained space SIGReg's pressure saturates
  the clip every step. SIGReg's globally-coherent spread direction
  beats contrastive's per-pair pull within the clip-normalized step.
  Model converges to "Gaussian features without clustering" =
  LeJEPA's stated failure mode if the prediction signal is too weak.
