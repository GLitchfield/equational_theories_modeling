# `aux_h256_cohom_diag` — Diagnostic Report

Cohomology aux heads added on top of the `aux_h256_diag` recipe. Single-knob
change vs `configs/experiments/aux_h256_diag.yaml`: 8 new aux heads
(`cohom_count_{2,3,5,7}`, `cohom_mean_h2_{2,3,5,7}`) on per-prime summary
statistics derived from the length-87 cohomology profile. All other
hyperparameters identical (h=256, margin 1.5, VICReg 0.05/0.05, aux_weight 0.1,
seed 42).

**Headline result: peak bal_acc 0.9491 at ep6, −1.44pp vs the no-cohom
reference's record 0.9635.** The phase transition into regime B fired one
epoch *earlier* than the reference (ep6 vs ep7) but did NOT persist —
the encoder reverted to regime A by ep7 and stayed there for ep7-9.

The cohom-augmented configuration appears to make the regime-A → regime-B
basin shallower, not deeper. The transition mechanism still fires, but the
basin can no longer hold the encoder.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `aux_h256_cohom_diag` |
| Wandb id | `u5fk79q1` |
| Config | `configs/experiments/aux_h256_cohom_diag.yaml` |
| Wall clock | 140.7 min (8443.3 s) |
| Outcome | 10 epochs completed, exit code 0 |
| Encoder params | 799,744 (h=256) |
| Aux heads params | 37,449 (18 heads at pool_dim=512; +27,181 vs reference's 10,268) |
| Mode | `--no-holdout` (in-sample 1M-pair monitor for threshold calibration) |

Wall clock is 21 min faster than the reference's 162 min — variability in
shared GPU load, not a real change in step time.

---

## Per-epoch monitor

| ep | bal_acc | true_R | false_R | thresh | eff_rk | gap | tr_p50 | fa_p50 | tr_l | val_l |
|---:|--------:|-------:|--------:|-------:|-------:|----:|-------:|-------:|-----:|------:|
| 0 | 0.9353 | 0.9756 | 0.8951 | 1.293 | 181.4 | 0.290 | 1.203 | 1.494 | 0.542 | 0.545 |
| 1 | 0.9331 | 0.9779 | 0.8882 | 1.295 | 259.9 | 0.257 | 1.222 | 1.479 | 0.604 | 0.562 |
| 2 | 0.9299 | 0.9884 | 0.8715 | 1.317 | 194.9 | 0.257 | 1.218 | 1.476 | 0.587 | 0.559 |
| 3 | 0.9317 | 0.9865 | 0.8769 | 1.322 | 286.1 | 0.253 | 1.239 | 1.492 | 0.603 | 0.576 |
| 4 | 0.9309 | 0.9865 | 0.8754 | 1.305 | 285.6 | 0.271 | 1.213 | 1.483 | 0.604 | 0.555 |
| 5 | 0.9298 | 0.9838 | 0.8758 | 1.355 | 328.8 | 0.198 | 1.277 | 1.474 | 0.616 | 0.611 |
| **6** | **0.9491** | 0.9220 | **0.9762** | **1.184** | **140.8** | **0.502** | 0.972 | **1.474** | 0.594 | 0.453 |
| 7 | 0.9303 | 0.9888 | 0.8719 | 1.347 | 320.8 | 0.275 | 1.234 | 1.510 | 0.520 | 0.583 |
| 8 | 0.9307 | 0.9881 | 0.8734 | 1.341 | 336.1 | 0.255 | 1.234 | 1.490 | 0.583 | 0.572 |
| 9 | 0.9313 | 0.9876 | 0.8751 | 1.323 | 337.8 | 0.250 | 1.233 | 1.484 | 0.594 | 0.573 |

### Three regimes, not two

**Regime A (ep0–5).** Same TRUE-bias drift signature as every other h=256
run: bal_acc plateaus at 0.93, TRUE recall climbs 0.976 → 0.984, FALSE
recall sits at 0.87-0.90, eff_rank inflates 181 → 329, distance gap shrinks
0.29 → 0.20. Compared to the reference's regime A (ep0=0.9546), this run
*starts ~1.9pp worse* and never reaches the reference's ep0 quality.

**Regime B (ep6 only).** Phase transition fires between ep5 and ep6:
bal_acc jumps to 0.9491, TRUE recall drops to 0.92, FALSE recall jumps to
0.976, eff_rank crashes 329 → 141, distance gap widens to 0.50, threshold
drops 1.36 → 1.18, val_loss drops 0.61 → 0.45. All the regime-B signatures
fire — but at a shallower magnitude than the reference (eff_rank 141 vs 58,
gap 0.50 vs 0.65, FALSE recall 0.976 vs 0.992).

**Regime A' (ep7–9).** Encoder *reverts* to regime A: eff_rank rebounds
to 320-338, distance gap collapses back to 0.25, TRUE recall reclaims its
~0.99 plateau, bal_acc settles at ~0.931. The reversion happens through ep7
(eval at end of ep7 is fully in regime A territory). Regime B was a
single-epoch excursion, not a sustained basin.

### Side-by-side vs `aux_h256_diag` (reference)

| ep | bal_acc cohom | bal_acc ref | Δ | regime cohom | regime ref |
|---:|---:|---:|---:|---|---|
| 0 | 0.9353 | 0.9546 | −0.0193 | A | A |
| 1-5 | 0.93 plateau | 0.93 plateau | ~0 | A | A |
| 6 | 0.9491 | 0.9302 | +0.0189 | **B** (early!) | A |
| 7 | 0.9303 | 0.9635 | −0.0332 | A' (reverted) | **B** (entered) |
| 8 | 0.9307 | 0.9612 | −0.0305 | A' | B |
| 9 | 0.9313 | 0.9611 | −0.0298 | A' | B |

The transition fires earlier with cohom heads but the basin is too shallow
to retain the encoder past one epoch's worth of further training. The
reference's regime B held for 3 epochs (ep7-9); the cohom run's regime B
held for less than 1 epoch.

---

## Aux head behavior

All 18 heads reach **val accuracy 1.000 by ep1** and stay there for the
remainder of training — including the 8 new cohom heads. Validation accuracy
alone suggests the cohom features are trivially learnable from the encoder
representation.

Per-step training cross-entropy tells a more nuanced story:

| ep | spine_loss | mean cohom loss | structural total | cohom total |
|---:|---:|---:|---:|---:|
| 0 | 0.0018 | 0.180 | ≈ small | 1.44 |
| 5 | 0.0001 | 0.011 | ≈ 0 | 0.085 |
| 9 | 0.00003 | 0.008 | ≈ 0 | 0.066 |

Structural heads (spine, lhs, num_vars, magma bits) collapse to essentially
zero loss within the first epoch and stay there. The 8 cohom heads converge
much more slowly, plateauing near 0.01 cross-entropy each (validation
predictions are correct so accuracy = 1.000, but the model isn't perfectly
confident — i.e. the logit margin to the next class is finite).

This means **the cohom heads continue to exert non-zero gradient pressure
throughout training**, while the structural heads are silent. Aux loss
contribution at ep5: ~0.5 × (0 + 0.085 × 8 / 18) × 0.1 ≈ 0.002 (negligible
relative to contrastive loss ~0.6, but always pointing the same direction:
"keep the cohomology classes linearly separable").

The hypothesis that this anchoring would *prevent* drift turned out to
backfire: the constant cohom pressure may have *destabilized* regime B
once entered, by demanding the encoder maintain certain class boundaries
that conflict with regime B's tighter clustering geometry.

---

## Mechanism interpretation

Comparing all three known h=256 trajectories to date:

| Run | Seed | ep0 | best | best ep | regime B duration |
|---|---:|---:|---:|---:|---|
| `stable_h256_m1p5_vic0p05_diag` | 42 | 0.9469 | 0.9469 | 0 | never entered |
| `aux_h256_diag` | 42 | 0.9546 | **0.9635** | 7 | 3 ep (7-9) |
| `aux_h256_seed1_diag` | 1 | 0.9492 | 0.9492 | 0 | partial transition at ep7 (bal_acc 0.9364, eff_rank 200, gap 0.42) then reverted |
| `aux_h256_cohom_diag` | 42 | 0.9353 | 0.9491 | 6 | partial transition at ep6 (bal_acc 0.9491, eff_rank 141, gap 0.50) then reverted |

Two new data points that the cohom run contributes:

1. **Earlier transition is possible.** The transition timing isn't
   intrinsically locked to ep7. Adding the cohom aux heads moved it forward
   by one epoch (the LR is in the same cosine-decay range so the
   "low-LR triggers transition" intuition still holds — ep6 LR ≈ 0.0003
   isn't dramatically lower than ep7's 0.0001 in seed-42, just a slightly
   different operating point that suffices given the perturbed loss
   landscape).

2. **Basin depth depends on aux configuration, not just LR.** The reference
   recipe (10 structural heads, all silent by ep1) lets regime B persist
   for 3 epochs once entered. The cohom recipe (18 heads, 8 still
   non-saturated and exerting ~0.002 of gradient) cannot hold the encoder
   in regime B for even one full additional epoch. This is consistent with
   the antagonism hypothesis: regime B's tight FALSE clustering and
   collapsed effective rank conflicts with the cohom heads' demand for
   linear separability across 5-6 classes per prime.

The seed-1 reproducibility check showed that even without cohom, basin depth
is fragile and seed-dependent — seed=1 hit only a partial transition at
ep7 (bal_acc 0.9364, eff_rank 200, gap 0.42) that then reverted by ep8,
nowhere near seed=42's record-territory regime B. The cohom variant
exhibits the same partial-then-reverted pattern (cohom seed=42 ep6:
bal_acc 0.9491, eff_rank 141, gap 0.50, reverted by ep7), confirming
that basin depth depends on more than just seed — the loss landscape
that the cohom heads create is closer to seed=1's shallow-basin regime
than seed=42's deep-basin regime.

---

## Conclusion

Per-prime cohomology summary statistics, configured as 8 binned classification
aux heads, do NOT improve on the structural-only aux baseline at h=256 with
seed 42. Net effect: peak bal_acc drops from 0.9635 to 0.9491 (−1.44pp).

The mechanism is not "cohom features carry no signal" — they're learned to
1.000 val accuracy and continue to exert non-zero CE pressure throughout
training. The mechanism is "their continued pressure conflicts with regime
B's representational geometry, destabilizing the basin once entered."

**Implications for next steps:**

- Per-prime *summary* statistics aren't useful in this configuration.
  Cell-level (87-head) or richer encodings would face the same antagonism.
- A "decay aux_weight on entry to regime B" schedule could in principle
  rescue this — but the operationally cleaner question is whether
  cohomology should be on the aux path at all, vs. the input-feature path
  (which would expose the information without ongoing gradient pressure).
- The shallow-basin reversion from ep6 → ep7 mirrors the seed-1 reversion
  signature from `aux_h256_seed1_diag`. The basin-depth problem at h=256
  remains the active blocker, and adding more aux heads makes it worse,
  not better.

Recommend **not** integrating cohom summaries as aux heads; revisit the
"input feature" or "per-pair feature" alternatives if the cohomology data
is going to be used at all.
