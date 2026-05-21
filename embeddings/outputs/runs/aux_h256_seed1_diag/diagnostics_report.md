# `aux_h256_seed1_diag` — Diagnostic Report

Reproducibility check on `aux_h256_diag`'s late-training phase
transition. Identical config except seed (42 → 1).

**Result: the phase transition mechanism IS reproducible, but its
depth and stability are seed-dependent.** Both seeds show the same
A→B transition signatures at ep7 (cl_TRUE drops, eff_rank drops, gap
widens, train_loss drops, recall flips toward FALSE-biased). Seed=42
fell deep into regime B and stabilized (bal_acc 0.9635, eff_rank 58).
Seed=1 only brushed the basin's edge (bal_acc 0.9364 at ep7) then
slipped back into regime A by ep8.

**Best epoch on seed=1 is ep0 at bal_acc 0.9492** — matching the
h=128 VICReg-only baseline. The aux heads still help (vs h=256
vic-only's 0.9469), just not as dramatically as seed=42 suggested.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `aux_h256_seed1_diag` |
| Wandb id | `mn0fatlb` |
| Config | `configs/experiments/aux_h256_seed1_diag.yaml` |
| Wall clock | 146.9 min (8816.6 s) |
| Outcome | 10 epochs completed, exit code 0 |
| Encoder params | 799,744 (h=256) |
| Aux heads params | 14,364 (10 heads at pool_dim=512) |

Single-knob change vs `aux_h256_diag`: `seed: 42 → 1` (and the
matching `holdout.seed: 42 → 1`, though no-holdout mode doesn't use
this).

---

## Per-epoch monitor

| ep | bal_acc | true_R | false_R | thresh | eff_rk | gap | tr_p50 | fa_p50 | val_l | tr_l |
|---:|--------:|-------:|--------:|-------:|-------:|----:|-------:|-------:|------:|-----:|
| **0** | **0.9492** | 0.9649 | 0.9335 | 1.270 | 143.4 | 0.314 | 1.185 | 1.499 | 0.529 | 0.527 |
| 1 | 0.9359 | 0.9580 | 0.9138 | 1.305 | 224.5 | 0.248 | 1.241 | 1.489 | 0.580 | 0.602 |
| 2 | 0.9305 | 0.9854 | 0.8755 | 1.341 | 272.1 | 0.232 | 1.252 | 1.483 | 0.589 | 0.577 |
| 3 | 0.9324 | 0.9788 | 0.8860 | 1.296 | 272.6 | 0.272 | 1.215 | 1.487 | 0.557 | 0.586 |
| 4 | 0.9339 | 0.9876 | 0.8802 | 1.327 | 295.9 | 0.242 | 1.245 | 1.487 | 0.584 | 0.593 |
| 5 | 0.9312 | 0.9888 | 0.8736 | 1.328 | 301.5 | 0.257 | 1.234 | 1.491 | 0.574 | 0.570 |
| 6 | 0.9319 | 0.9876 | 0.8763 | 1.356 | 315.6 | 0.211 | 1.271 | 1.482 | 0.606 | 0.621 |
| 7 | 0.9364 | 0.9341 | **0.9388** | **1.149** | **200.8** | **0.421** | **1.081** | 1.502 | 0.452 | 0.494 |
| 8 | 0.9304 | 0.9920 | 0.8687 | 1.310 | 269.1 | 0.321 | 1.173 | 1.495 | 0.521 | 0.529 |
| 9 | 0.9304 | 0.9922 | 0.8686 | 1.312 | 272.2 | 0.314 | 1.179 | 1.493 | 0.525 | 0.555 |

**Best epoch: ep0 (0.9492)**, then drift into regime A (0.93 plateau
with TRUE-bias). At ep7 the transition signature kicks in (TRUE p50
1.27→1.08, eff_rank 316→201, gap 0.21→0.42, recall briefly balanced)
but it doesn't deepen — by ep8 the encoder slips back into regime A
(TRUE 0.99 / FALSE 0.87, eff_rank 269) and stays there.

---

## Per-step diagnostic aggregates

| ep | total | cont | cl_TRUE | cl_FALSE | grad | g_enc | g_pred | vc_var | vc_cov | aux | lr | tau |
|---:|-----:|-----:|--------:|---------:|-----:|------:|-------:|-------:|-------:|----:|----:|----:|
| 0 | 0.5277 | 0.4929 | 1.295 | 0.018 | 0.072 | 0.064 | 0.027 | 0.525 | 0.104 | 0.0341 | warmup | 0.9960 |
| 1 | 0.6012 | 0.5817 | 1.554 | 0.009 | 0.024 | 0.018 | 0.016 | 0.239 | 0.149 | 0.0013 | 9.5e-4 | 0.9962 |
| 2 | 0.5753 | 0.5596 | 1.492 | 0.012 | 0.023 | 0.016 | 0.016 | 0.126 | 0.186 | 0.0008 | 8.6e-4 | 0.9964 |
| 3 | 0.5847 | 0.5699 | 1.521 | 0.011 | 0.022 | 0.015 | 0.015 | 0.098 | 0.197 | 0.0005 | 7.4e-4 | 0.9968 |
| 4 | 0.5940 | 0.5796 | 1.542 | 0.010 | 0.022 | 0.013 | 0.017 | 0.088 | 0.200 | 0.0004 | 5.8e-4 | 0.9973 |
| 5 | 0.5709 | 0.5567 | 1.479 | 0.012 | 0.025 | 0.014 | 0.020 | 0.082 | 0.202 | 0.0004 | 4.4e-4 | 0.9977 |
| 6 | 0.6204 | 0.6064 | 1.622 | 0.008 | 0.024 | 0.011 | 0.021 | 0.075 | 0.204 | 0.0003 | 2.7e-4 | 0.9982 |
| **7** | **0.4945** | 0.4806 | **1.263** | 0.018 | 0.037 | 0.018 | 0.032 | 0.071 | 0.205 | 0.0003 | 1.5e-4 | 0.9986 |
| 8 | 0.5292 | 0.5155 | **1.364** | 0.015 | 0.038 | 0.014 | 0.035 | 0.069 | 0.205 | 0.0002 | 6e-5 | 0.9988 |
| 9 | 0.5552 | 0.5415 | **1.436** | 0.013 | 0.038 | 0.013 | 0.036 | 0.069 | 0.205 | 0.0002 | 0 | 0.9990 |

The cl_TRUE trajectory is the smoking gun for the partial transition:
1.62 → 1.26 (ep7, transition starts) → 1.36 (ep8, reverts) → 1.44 (ep9,
back into regime A). seed=42's cl_TRUE went 1.62 → 1.17 → 0.87 → 0.89
(deepened and stabilized).

### Aux head accuracy

All 10 heads at 100% by ep0, stay there every epoch. Same as
`aux_h256_diag`. The structural features are present in the
representation throughout both regimes. Aux loss drops 0.034 → 0.0002
over the run (essentially zero from ep1 onward).

### Stability

| ep | max grad | clip_frac |
|---:|---------:|----------:|
| 0 | 1.01 | 0.0020 |
| 1-9 | 0.04–0.09 | 0.000 |

Identical stability profile to `aux_h256_diag`. Fine throughout.

---

## Comparison: seed=42 vs seed=1

### Trajectory side-by-side

| ep | s42 bal_acc | s1 bal_acc | s42 cl_TRUE | s1 cl_TRUE | s42 eff_rk | s1 eff_rk |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9546 | 0.9492 | 1.265 | 1.295 | 132.6 | 143.4 |
| 1 | 0.9342 | 0.9359 | 1.558 | 1.554 | 202.8 | 224.5 |
| 2 | 0.9320 | 0.9305 | 1.519 | 1.492 | 237.4 | 272.1 |
| 3 | 0.9296 | 0.9324 | 1.578 | 1.521 | 285.8 | 272.6 |
| 4 | 0.9296 | 0.9339 | 1.459 | 1.542 | 289.2 | 295.9 |
| 5 | 0.9315 | 0.9312 | 1.473 | 1.479 | 275.9 | 301.5 |
| 6 | 0.9302 | 0.9319 | **1.649** | **1.622** | 324.0 | 315.6 |
| 7 | **0.9635** | 0.9364 | **1.173** | **1.263** | **58.4** | 200.8 |
| 8 | 0.9612 | 0.9304 | **0.871** | 1.364 | 69.2 | 269.1 |
| 9 | 0.9611 | 0.9304 | 0.890 | 1.436 | 70.7 | 272.2 |

Both trajectories are nearly identical through ep0–6 (the regime-A
plateau). Both transition at ep7 (cl_TRUE drops, eff_rank drops). But:
seed=42 deepens through ep8–9 and stabilizes at bal_acc 0.96+;
seed=1 reverts at ep8–9 back to regime A.

### Best-epoch deltas

| metric | seed=42 ep7 | seed=1 ep0 |
|---|---:|---:|
| bal_acc | 0.9635 | 0.9492 |
| true_recall | 0.9349 | 0.9649 |
| false_recall | 0.9920 | 0.9335 |
| recall gap | 5.7 pp | 3.1 pp |
| TRUE p50 | 0.846 | 1.185 |
| gap_median | 0.654 | 0.314 |
| eff_rank | 58.4 | 143.4 |

---

## Comparison vs all prior baselines

| run | seed | best ep | bal_acc | true_R | false_R | TRUE p50 | eff_rk |
|---|---:|---:|---:|---:|---:|---:|---:|
| h=128 vic-only | 42 | 6 | 0.9492 | 0.95 | 0.95 | 1.18 | 145 |
| h=256 vic-only (vic 0.05) | 42 | 0 | 0.9469 | 0.94 | 0.95 | 1.15 | 136 |
| h=128 + aux (Test 1) | 42 | 3 | 0.9582 | 0.96 | 0.95 | 1.18 | 131 |
| h=256 + aux (Test 2) | 42 | 7 | **0.9635** | 0.93 | 0.99 | 0.85 | 58 |
| **h=256 + aux (this run)** | **1** | **0** | **0.9492** | 0.96 | 0.93 | 1.19 | 143 |

Even on the unlucky seed, the h=256 + aux ep0 (0.9492) **matches the
h=128 vic-only baseline best**. The recall is well-balanced (TRUE 0.96
/ FALSE 0.93). So aux at h=256 has a solid floor — it always reaches at
least the prior best baseline at ep0 — but the +1.4pp gain that
seed=42 delivered is not reliable.

---

## What this tells us

1. **The regime B phase transition is REAL and recipe-driven.** Both
   seeds transition at ep7 with the same signatures (cl_TRUE drops,
   eff_rank drops, gap widens, train_loss drops). The mechanism is
   robust.
2. **The basin depth is seed-dependent.** Seed=42 fell into the deep
   stable basin (eff_rank 58, bal_acc 0.96+); seed=1 only touched the
   basin's edge (eff_rank 201, bal_acc 0.94 briefly) then reverted.
3. **2 seeds is too few to estimate the basin-discovery rate.** Could
   be 50% lucky, could be 10% lucky. Need more seeds (or recipe
   modifications that make regime B more reliably reachable).
4. **Aux at h=256 has a solid floor.** Worst-case ep0 result (0.9492)
   matches h=128 vic-only best. So the aux pivot is at minimum a
   neutral move at h=256, with upside.
5. **Late training is volatile in this regime.** seed=1 went 0.9319
   (ep6) → 0.9364 (ep7) → 0.9304 (ep8) → 0.9304 (ep9). The optimizer
   is bouncing between basin edges in the low-LR cosine tail.

---

## Where to go from here

The seed=42 result was real but not fully reproducible on a second
draw. Options for chasing reliable regime B:

1. **Longer training** (15 epochs at the same recipe). Cheapest test
   of the "seed=1 needed more time" hypothesis — if the partial
   transition at ep7 would have deepened given more low-LR steps, a
   longer run would catch it.
2. **Modified LR schedule** — push the LR lower earlier (cosine over
   5 epochs then constant low LR for 5 more), so the optimizer
   spends more steps in the transition-favoring regime instead of
   only the last 1-2 epochs.
3. **Larger aux_weight** — bump 0.1 → 0.2 or 0.3. A stronger anchor
   might pull the encoder toward regime B from earlier in training,
   not relying on late-training low LR to enable the basin escape.
4. **Multi-seed sweep** — run 3-5 seeds at the current recipe to
   estimate the basin-discovery rate. Tells us how lucky seed=42 was.

Recommendation order depends on goal:
  - "Is regime B real?" → answered: YES.
  - "Can we ship 0.9635?" → option 4 (sweep) tells us how often the
    current recipe finds it; option 1 (longer training) tells us if
    we just need more patience.
  - "Can we make regime B the typical outcome?" → option 2 or 3.

---

## Headline numbers

- **Best: ep0 bal_acc 0.9492** — matches the h=128 vic-only baseline
  best. Recall 0.9649 / 0.9335 (gap 3.1 pp).
- **Phase transition mechanism reproduces** at ep7 (cl_TRUE 1.62 →
  1.26, eff_rank 316 → 201, gap 0.21 → 0.42) — but not as deep as
  seed=42 (cl_TRUE 1.65 → 1.17 → 0.87, eff_rank 324 → 58).
- **Transition reverts** by ep8 on seed=1 (cl_TRUE bounces back to
  1.36, eff_rank to 269, bal_acc to 0.93).
- **Aux floor is solid**: even worst-case ep0 matches prior best
  baseline, with balanced recall. seed=42's +1.4 pp gain isn't
  reproducible on demand.
- **2 seeds is too few** to estimate basin-discovery rate; multi-seed
  sweep would tell.
