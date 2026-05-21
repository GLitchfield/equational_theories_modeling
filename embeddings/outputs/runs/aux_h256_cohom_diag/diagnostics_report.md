# `aux_h256_cohom_diag` — Diagnostic Report

Cohomology aux heads added on top of the `aux_h256_diag` recipe. Single-knob
change vs `configs/experiments/aux_h256_diag.yaml`: 8 new aux heads
(`cohom_count_{2,3,5,7}`, `cohom_mean_h2_{2,3,5,7}`) on per-prime summary
statistics derived from the length-87 cohomology profile. All other
hyperparameters identical (h=256, margin 1.5, VICReg 0.05/0.05, aux_weight 0.1,
seed 42).

**Headline result: peak bal_acc 0.9491 at ep6, −1.44pp vs the no-cohom
reference's record 0.9635.** A partial regime-B phase transition fired
one epoch earlier than the reference (ep6 vs ep7) but reverted to regime
A by ep7 and stayed there for ep7-9.

**This run does NOT confirm or refute the simplicity-bias-anchoring
hypothesis** — see the Conclusion. The cohom heads stayed unsaturated
and contributed persistent gradient (the theory's prescription), but at
aux_weight=0.1 their share of total loss was only ~0.08%, too small to
plausibly fight the contrastive simplicity bias. The drift in regime A
was identical to the structural-only run (which had no anti-drift
contribution at all once structural heads saturated by ep1). The
reference's 0.9635 came from a separate, unrelated regime-B phase
transition that happened to fire and stick under seed 42 — not from
aux anchoring suppressing drift.

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

The structural heads' silence means they exerted essentially zero
gradient pressure for 9 of 10 epochs — they could not have prevented
simplicity-bias drift even in principle. The cohom heads stayed
unsaturated but at aux_weight=0.1 their loss contribution was

  aux_weight × 0.5 × per_head_mean × n_heads
   = 0.1 × 0.5 × ((10 × 0.0001) + (8 × 0.011)) / 18
   ≈ 5×10⁻⁴

vs contrastive ≈ 0.6 → **~0.08% of total loss**. That magnitude is
rounding error relative to the simplicity-bias signal it would need to
fight. So the persistent-gradient property was satisfied in form but
not in magnitude.

---

## What this run does (and does not) tell us

**What it tells us:**

- The h=256 simplicity-bias drift in regime A (TRUE recall climbing,
  FALSE recall dropping, eff_rank inflating) is **identical** between
  the structural-only run and the cohom run through ep5. Both plateau
  at ~0.93 with the same TRUE-bias signature. No version of aux-anchoring
  tested so far has changed regime-A drift behavior.
- The phase transition timing isn't intrinsically locked to a specific
  epoch — adding the cohom heads moved it from ep7 to ep6 in seed-42.
  Some perturbation of the loss landscape suffices to shift it. This is
  a useful side observation, not the headline.
- Basin depth at h=256 is shallow without help. Three of four h=256 runs
  to date hit only partial transitions that reverted; only the
  reference seed-42 run produced a sustained basin (3 epochs).

**What it does NOT tell us:**

- Whether persistent aux gradient *can* prevent simplicity-bias drift.
  The cohom heads were configured at aux_weight=0.1, where total cohom
  contribution is ~0.08% of loss. That's too weak for the comparison to
  test the theory. A meaningful test requires aux_weight high enough
  that the cohom contribution is ~1-5% of loss — i.e., aux_weight in the
  1.0-5.0 range.
- Whether basin depth at h=256 has a fix that doesn't require
  fundamentally redesigning the loss. All four h=256 runs to date used
  the same recipe up to aux composition; we haven't varied the
  contrastive recipe itself.

Comparing all four known h=256 trajectories:

| Run | Seed | ep0 | best | best ep | regime B duration |
|---|---:|---:|---:|---:|---|
| `stable_h256_m1p5_vic0p05_diag` | 42 | 0.9469 | 0.9469 | 0 | never entered (no aux) |
| `aux_h256_diag` | 42 | 0.9546 | **0.9635** | 7 | 3 ep (7-9) sustained |
| `aux_h256_seed1_diag` | 1 | 0.9492 | 0.9492 | 0 | partial ep7 (bal_acc 0.9364), reverted |
| `aux_h256_cohom_diag` | 42 | 0.9353 | 0.9491 | 6 | partial ep6 (bal_acc 0.9491), reverted |

The aux_h256_diag basin holding for 3 epochs is the outlier, not the
rule — cohom seed=42 looks like seed=1 (partial then reverted), which
is the more typical h=256 behavior.

---

## Conclusion

Per-prime cohomology summary statistics as aux heads at **aux_weight=0.1**
do not improve on the structural-only aux baseline at h=256. Peak bal_acc
0.9491 vs reference 0.9635 (−1.44pp). But this is **not** evidence that
cohom anchoring is wrong-headed; it's evidence that the test was not
sized correctly to evaluate the anchoring hypothesis.

The right next test is **aux_weight=5.0** (50× the current setting),
which pushes cohom contribution from ~0.08% to ~4% of total loss — the
range where aux signal can plausibly compete with the contrastive
simplicity bias. The structural heads are saturated regardless of weight
multiplier, so this is functionally equivalent to "cohom-specific weight
bump" with no code change. Risk: the same SIGReg-style instability
where aux signal overwhelms contrastive — but cohom magnitudes are an
order smaller than SIGReg's, so headroom exists.

If aux_weight=5.0 prevents the regime-A drift (FALSE recall stays in
the 0.93+ range through ep5), the anchoring theory is confirmed and the
basin question becomes secondary. If it still drifts, then either the
cohom features carry no useful signal beyond what's already in the
contrastive loss, or aux-head anchoring at this scale needs a different
form entirely (e.g., contrastive between same-cohom-class equation
pairs rather than per-equation classification).

Implications for the basin-depth question (h=256 regime-B fragility):
that's a separate phenomenon from the drift question, and this run
didn't move the needle on it. Future work on basin depth should be
labeled as such and not conflated with anti-drift evidence.
