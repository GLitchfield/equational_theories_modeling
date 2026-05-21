# `aux_h256_diag` — Diagnostic Report

Test 2 from `aux_heads_implementation_plan.md`: aux heads at h=256
paired with the recipe that produced the best h=256 baseline so far
(margin 1.5, VICReg 0.05/0.05). Single-knob change vs
`stable_h256_m1p5_vic0p05_diag.yaml`: aux heads enabled.

**Headline result: new project record bal_acc 0.9635 at ep7** — beats
Test 1 by +0.5pp and the h=128 VICReg-only baseline by +1.4pp.

But the trajectory is unexpected: training has two distinct regimes
separated by a phase transition between ep6 and ep7. The aux heads
did NOT prevent the early TRUE-bias drift (the plan's hypothesis);
instead, late-training low LR allowed the encoder to escape the drift
plateau into a much tighter, FALSE-biased regime with a record bal_acc.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `aux_h256_diag` |
| Wandb id | `8tvn4ndj` |
| Config | `configs/experiments/aux_h256_diag.yaml` |
| Wall clock | 162 min (9717.8 s) — significantly longer than expected |
| Outcome | 10 epochs completed, exit code 0 |
| Encoder params | 799,744 (h=256) |
| Aux heads params | 10,268 (10 heads at pool_dim=512) |

The 162 min wall clock is +60% over h=256 VICReg-only's ~95 min and
+78% over h=128/aux's 91 min. The eq2 second forward through the
context encoder is more expensive at h=256.

---

## Per-epoch monitor

| ep | bal_acc | true_R | false_R | thresh | eff_rk | gap | tr_p05 | tr_p50 | tr_p95 | fa_p05 | fa_p50 | fa_p95 | val_l | tr_l |
|---:|--------:|-------:|--------:|-------:|-------:|----:|-------:|-------:|-------:|-------:|-------:|-------:|------:|-----:|
| 0 | 0.9546 | 0.9681 | 0.9411 | 1.240 | 132.6 | 0.354 | 1.087 | 1.159 | 1.225 | 1.227 | 1.512 | 1.611 | 0.507 | 0.517 |
| 1 | 0.9342 | 0.9774 | 0.8909 | 1.324 | 202.8 | 0.227 | 1.187 | 1.254 | 1.318 | 1.292 | 1.481 | 1.554 | 0.590 | 0.603 |
| 2 | 0.9320 | 0.9707 | 0.8932 | 1.276 | 237.4 | 0.294 | 1.139 | 1.207 | 1.276 | 1.255 | 1.501 | 1.566 | 0.550 | 0.586 |
| 3 | 0.9296 | 0.9849 | 0.8744 | 1.355 | 285.8 | 0.204 | 1.211 | 1.278 | 1.348 | 1.310 | 1.482 | 1.541 | 0.611 | 0.606 |
| 4 | 0.9296 | 0.9818 | 0.8774 | 1.348 | 289.2 | 0.210 | 1.205 | 1.269 | 1.339 | 1.302 | 1.480 | 1.539 | 0.603 | 0.563 |
| 5 | 0.9315 | 0.9870 | 0.8760 | 1.329 | 275.9 | 0.246 | 1.181 | 1.245 | 1.317 | 1.272 | 1.491 | 1.564 | 0.582 | 0.568 |
| 6 | 0.9302 | 0.9692 | 0.8912 | 1.353 | 324.0 | 0.197 | 1.220 | 1.285 | 1.353 | 1.305 | 1.482 | 1.546 | 0.617 | 0.630 |
| **7** | **0.9635** | 0.9349 | **0.9920** | **1.054** | **58.4** | **0.654** | 0.768 | **0.846** | 0.961 | 1.270 | 1.500 | 1.581 | 0.306 | 0.461 |
| 8 | 0.9612 | 0.9291 | 0.9933 | 1.038 | 69.2 | 0.624 | 0.811 | 0.880 | 0.987 | 1.276 | 1.503 | 1.583 | 0.324 | 0.354 |
| 9 | 0.9611 | 0.9294 | 0.9928 | 1.038 | 70.7 | 0.622 | 0.812 | 0.882 | 0.991 | 1.279 | 1.504 | 1.583 | 0.325 | 0.361 |

### Two distinct regimes

**Regime A (ep0–6) — VICReg-h=256-baseline-like.** Looks essentially
identical to the `stable_h256_m1p5_vic0p05_diag` baseline trajectory:
bal_acc plateaus at 0.93 after a strong ep0, TRUE recall climbs
0.97 → 0.99, FALSE recall falls 0.94 → 0.87 (the simplicity-bias
drift), eff_rank growing 130 → 324, distances ~1.25/1.50.

**Regime B (ep7–9) — record territory.** Phase transition between
ep6 and ep7:

| metric | ep6 | ep7 | Δ |
|---|---:|---:|---:|
| bal_acc | 0.9302 | 0.9635 | +3.3 pp |
| TRUE recall | 0.9692 | 0.9349 | -3.4 pp |
| FALSE recall | 0.8912 | 0.9920 | **+10.1 pp** |
| TRUE p50 | 1.285 | 0.846 | **-34%** |
| eff_rank | 324.0 | 58.4 | **-82%** |
| threshold | 1.353 | 1.054 | -22% |
| gap_median | 0.197 | 0.654 | **+232%** |
| train_loss | 0.6303 | 0.4611 | -27% |

The encoder collapsed onto a much lower-dimensional subspace (eff_rank
58 vs 324) in which TRUE pairs are tightly concentrated (p50 0.85)
while FALSE pairs stay at typical distance (1.50). Recall flipped from
TRUE-biased to FALSE-biased — the model now nearly always catches
FALSE pairs (0.99) at the cost of missing more TRUE (0.93).

---

## Aux head accuracy (val on monitor batch)

| ep | spine | lhs | nvars | mb_0 | mb_1 | mb_2 | mb_3 | mb_4 | mb_6 | mb_7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2-9 | 1.000 (all) | | | | | | | | | |

100% on every head every epoch. The structural features are present in
the representation throughout (including in regime B's compact eff_rank-58
subspace — which makes that subspace meaningfully task-aligned, not
arbitrary).

---

## Per-step diagnostic aggregates

| ep | total | cont | cl_TRUE | cl_FALSE | grad | g_enc | g_pred | vc_var | vc_cov | aux | lr | tau |
|---:|-----:|-----:|--------:|---------:|-----:|------:|-------:|-------:|-------:|----:|----:|----:|
| 0 | 0.5167 | 0.4808 | 1.265 | 0.019 | 0.080 | 0.071 | 0.028 | 0.539 | 0.102 | 0.0384 | warmup | 0.9960 |
| 1 | 0.6038 | 0.5845 | 1.558 | 0.009 | 0.021 | 0.016 | 0.013 | 0.232 | 0.151 | 0.0016 | 9.5e-4 | 0.9962 |
| 2 | 0.5844 | 0.5692 | 1.519 | 0.011 | 0.022 | 0.016 | 0.015 | 0.109 | 0.193 | 0.0009 | 8.6e-4 | 0.9964 |
| 3 | 0.6055 | 0.5910 | 1.578 | 0.009 | 0.022 | 0.014 | 0.017 | 0.089 | 0.200 | 0.0006 | 7.4e-4 | 0.9968 |
| 4 | 0.5630 | 0.5487 | 1.459 | 0.013 | 0.025 | 0.016 | 0.019 | 0.080 | 0.204 | 0.0005 | 5.8e-4 | 0.9973 |
| 5 | 0.5691 | 0.5551 | 1.473 | 0.012 | 0.027 | 0.014 | 0.023 | 0.075 | 0.205 | 0.0004 | 4.4e-4 | 0.9977 |
| 6 | 0.6298 | 0.6159 | 1.649 | 0.007 | 0.025 | 0.011 | 0.022 | 0.071 | 0.206 | 0.0003 | 2.7e-4 | 0.9982 |
| **7** | **0.4626** | 0.4488 | **1.173** | 0.020 | 0.043 | 0.023 | 0.036 | 0.068 | 0.207 | 0.0003 | 1.5e-4 | 0.9986 |
| 8 | **0.3538** | 0.3401 | **0.871** | 0.027 | 0.050 | 0.027 | 0.042 | 0.067 | 0.206 | 0.0003 | 6e-5 | 0.9988 |
| 9 | 0.3611 | 0.3474 | 0.890 | 0.027 | 0.050 | 0.026 | 0.042 | 0.067 | 0.206 | 0.0003 | 0 | 0.9990 |

`cl_TRUE` stuck at 1.45–1.65 through ep1–6 (typical h=256 stuck
plateau), then **drops 1.65 → 0.87 between ep6 and ep8** as the
encoder finds a tighter TRUE basin. Train loss drops 0.63 → 0.35.

VICReg components flat throughout (var 0.07, cov 0.21 from ep4 on),
even through the phase transition — VICReg is satisfied in both regimes.

Aux loss is essentially zero from ep1 onward — the aux heads aren't
the active force during the transition.

---

## Stability

| ep | max grad | clip_frac |
|---:|---------:|----------:|
| 0 | 1.32 | 0.0022 |
| 1-6 | 0.04–0.07 | 0.000 |
| 7-9 | 0.05–0.08 | 0.000 |

Stability is fine throughout. The phase transition isn't driven by a
gradient spike — it's a smooth optimization shift enabled by low LR.

---

## Comparison

| run | best ep | bal_acc | true_R | false_R | TRUE p50 | gap | eff_rk |
|---|---:|---:|---:|---:|---:|---:|---:|
| h=128 vic-only | 6 | 0.9492 | 0.95 | 0.95 | 1.18 | 0.31 | 145 |
| h=256 vic-only (vic 0.05) | 0 | 0.9469 | 0.94 | 0.95 | 1.15 | 0.35 | 136 |
| h=128 + aux (Test 1) | 3 | 0.9582 | 0.96 | 0.95 | 1.18 | 0.33 | 131 |
| **h=256 + aux (Test 2)** | **7** | **0.9635** | 0.93 | **0.99** | **0.85** | **0.65** | **58** |

---

## What this tells us

1. **The plan's hypothesis ("aux anchors FALSE-detection features → FALSE
   recall holds steady") is not borne out for early training.** Ep1–6
   looks essentially identical to VICReg-only h=256 — same plateau,
   same TRUE-bias drift. The aux heads at 100% accuracy from ep0 don't
   prevent the encoder from settling into the regime-A plateau where
   FALSE detection underperforms.

2. **But aux heads + low LR enables a basin escape that VICReg-only
   couldn't reach.** The phase transition at ep7 produces a much
   better representation. Whether the aux heads are causally
   responsible for this transition (vs. just a coincidence with low
   LR) needs separate investigation — see open questions below.

3. **The transition correlates with low LR.** Cosine LR was 2.7e-4 at
   ep6, 1.5e-4 at ep7, 6e-5 at ep8. Late-training low LR allowed the
   optimizer to take the smaller, more careful steps needed to escape
   the regime-A plateau.

4. **Recall reversed direction (not balanced).** TRUE 0.93 / FALSE 0.99
   in regime B vs the typical TRUE 0.96 / FALSE 0.95. The model now
   nearly always catches FALSE pairs (great for false-positive control)
   at the cost of missing some TRUE pairs. Different operating point
   than h=128's balanced result.

5. **eff_rank 58 is very low.** The encoder converged to a compact
   representation. Whether this is a feature (focused on what matters)
   or a bug (overfit to syntactic shortcuts) needs a true holdout
   evaluation to disambiguate.

6. **The no-holdout monitor val IS in-sample.** The 0.9635 is on a 1M
   random subsample of the 22M training set. The h=128/aux 0.9582 is
   also in-sample, so the +0.5pp comparison is apples-to-apples — but
   the absolute number may overestimate generalization.

---

## Open questions

1. **Is the phase transition reproducible?** Could be a single-seed
   lucky basin escape. Different seed run would tell.
2. **Can we trigger regime B earlier?** A modified LR schedule (lower
   terminal LR earlier, or LR warm restart) might find regime B in
   fewer epochs.
3. **Does regime B generalize?** True L1-holdout eval needed — the
   in-sample 0.9635 might overestimate.
4. **Is regime B aux-dependent?** Run h=256 vic 0.05 *without* aux
   for the same 10 epochs and see if it also transitions late. The
   `stable_h256_m1p5_vic0p05_diag` baseline only ran in regime A
   through ep9 — but it had the SAME LR schedule. So the comparison
   suggests aux heads ARE necessary for regime B (probably by anchoring
   the structural features that regime B's compact representation
   relies on).
5. **What's the regime B representation actually doing?** eff_rank 58
   with 100% aux accuracy means the compact subspace still encodes
   spine, lhs, num_vars, magma signature. Is it just those 10
   structural features? Linear-probing the embeddings against other
   invariants (the `06_leaf_variables`, `07_tree_metrics`,
   `02_trivial` parquets we didn't anchor) would show whether
   non-anchored features were dropped.

---

## Headline numbers

- **Best: ep7 bal_acc 0.9635** — new project record. +0.5 pp over Test 1
  (h=128 + aux, ep3, 0.9582), +1.4 pp over the h=128 VICReg-only
  baseline.
- **Recall**: TRUE 0.935, FALSE 0.992 (gap 5.7 pp, reversed direction
  vs typical h=256 TRUE-bias).
- **Phase transition** ep6→ep7: bal_acc +3.3 pp, eff_rank crashes 324→58,
  TRUE p50 drops 1.28→0.85, gap_median triples.
- **All 10 aux heads at 100% accuracy** every epoch.
- **Aux heads did NOT prevent the early TRUE-bias drift** ep1–6 — the
  plan's hypothesis is not supported in that form.
- **Stability fine**, no clipping after ep0 warmup.
- **Wall clock 162 min** — significantly more than expected (h=256 +
  aux is 60-78% slower than VICReg variants; investigate if the cost
  becomes a constraint).
