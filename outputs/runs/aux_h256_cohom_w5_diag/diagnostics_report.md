# `aux_h256_cohom_w5_diag` — Diagnostic Report

50× aux_weight bump (0.1 → 5.0) on top of `aux_h256_cohom_diag` to test
whether persistent cohomology-head gradient at a *meaningful* magnitude
prevents simplicity-bias drift at h=256. Single-knob change vs
`configs/experiments/aux_h256_cohom_diag.yaml`: `loss.aux_weight`.

**Headline result: anti-drift anchoring hypothesis refuted at this
configuration.** FALSE recall converged to ~0.87 by ep1 and stayed there
for ep1-9, identical to every prior h=256 run regardless of aux setup.
Best epoch was ep0 at bal_acc 0.9329 — no later epoch improved on it,
and no regime-B transition fired (reference fired at ep7,
`aux_h256_cohom_diag` fired at ep6 — w5 hit neither).

The increased aux gradient successfully made cohom contribution ~3-4%
of total loss (vs 0.08% at w=0.1, the magnitude the literature treats
as "meaningful"), but the simplicity-bias drift was unchanged. So the
hypothesis "persistent aux constraints at meaningful magnitude prevent
simplicity-bias drift" is not supported for these specific cohom
features in this contrastive setup.

A secondary observation: the encoder geometry slowly drifted *toward*
regime B over ep6-9 (eff_rank 286→196, dist_gap_median 0.30→0.42, TRUE
p50 1.197→1.078) without ever crossing the threshold for a bal_acc
gain. This suggests a deeper basin is reachable but the larger aux
gradient is suppressing the rapid transition that the lower-weight
configurations triggered.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `aux_h256_cohom_w5_diag` |
| Wandb id | `968k1wo4` |
| Config | `configs/experiments/aux_h256_cohom_w5_diag.yaml` |
| Wall clock | 161.9 min (9711.3 s) |
| Outcome | 10 epochs completed, exit code 0 |
| Encoder params | 799,744 (h=256) |
| Aux heads params | 37,449 (18 heads at pool_dim=512, same as cohom_diag) |
| Mode | `--no-holdout` |

---

## Per-epoch monitor

| ep | bal_acc | true_R | false_R | thresh | eff_rk | gap | tr_p50 | fa_p50 |
|---:|--------:|-------:|--------:|-------:|-------:|----:|-------:|-------:|
| **0** | **0.9329** | 0.9832 | 0.8825 | 1.304 | 161.1 | 0.291 | 1.197 | 1.487 |
| 1 | 0.9254 | 0.9861 | 0.8647 | 1.338 | 248.7 | 0.239 | 1.241 | 1.479 |
| 2 | 0.9265 | 0.9858 | 0.8673 | 1.345 | 258.5 | 0.248 | 1.244 | 1.492 |
| 3 | 0.9282 | 0.9850 | 0.8713 | 1.340 | 272.8 | 0.257 | 1.237 | 1.493 |
| 4 | 0.9258 | 0.9862 | 0.8654 | 1.337 | 271.3 | 0.269 | 1.222 | 1.491 |
| 5 | 0.9281 | 0.9857 | 0.8705 | 1.335 | 280.2 | 0.268 | 1.224 | 1.492 |
| 6 | 0.9272 | 0.9869 | 0.8675 | 1.324 | 286.1 | 0.296 | 1.197 | 1.494 |
| 7 | 0.9277 | 0.9891 | 0.8663 | 1.320 | 242.2 | 0.348 | 1.148 | 1.496 |
| 8 | 0.9275 | 0.9878 | 0.8673 | 1.308 | 208.9 | 0.401 | 1.099 | 1.499 |
| 9 | 0.9278 | 0.9886 | 0.8667 | 1.310 | 196.2 | 0.422 | 1.078 | 1.500 |

**bal_acc band: 0.9254-0.9329** across all 10 epochs. ep0 is the best.
TRUE recall at 0.98+ throughout, FALSE recall at 0.86-0.88 throughout.
Classic regime-A profile, no excursion to regime B.

### Slow approach to regime-B geometry without crossing

ep6-9 show the encoder gradually adopting regime-B characteristics:

| ep | eff_rank | gap | tr_p50 | fa_p50 |
|---:|---:|---:|---:|---:|
| 6 | 286 | 0.296 | 1.197 | 1.494 |
| 7 | 242 | 0.348 | 1.148 | 1.496 |
| 8 | 209 | 0.401 | 1.099 | 1.499 |
| 9 | 196 | 0.422 | 1.078 | 1.500 |

eff_rank dropping (286 → 196), gap widening (0.30 → 0.42), TRUE p50
collapsing (1.20 → 1.08) — all the regime-B signatures except
threshold/bal_acc adjustment. By ep9 the encoder is partway between
regime-A geometry and regime-B (reference's regime B had eff_rank ~58,
gap ~0.65, TRUE p50 ~0.85). If we ran 2-3 more epochs, the trajectory
suggests it would eventually reach regime B — but the rapid phase
transition that the w=0.1 configurations triggered (in 1-2 epochs of
flat-LR training) did not fire here.

### Side-by-side vs `aux_h256_cohom_diag` and reference

FALSE recall (the anti-drift indicator):

| ep | w5 (5.0) | cohom (0.1) | ref (no cohom) |
|----|---------:|------------:|---------------:|
| 0  | 0.8825   | 0.8951      | 0.9411         |
| 1  | 0.8647   | 0.8882      | 0.8909         |
| 2  | 0.8673   | 0.8715      | 0.8932         |
| 3  | 0.8713   | 0.8769      | 0.8744         |
| 4  | 0.8654   | 0.8754      | 0.8774         |
| 5  | 0.8705   | 0.8758      | 0.8760         |

All three runs converge to the same ~0.87 plateau by ep4-5. The aux
configuration (none / cohom-w0.1 / cohom-w5.0) has **no measurable
effect** on the simplicity-bias drift trajectory in regime A.

bal_acc (regime-B test):

| ep | w5 | cohom (0.1) | ref |
|----|---:|------------:|----:|
| 6 | 0.9272 | **0.9491** (transition!) | 0.9302 |
| 7 | 0.9277 | 0.9303 (reverted) | **0.9635** (transition!) |
| 8 | 0.9275 | 0.9307 | 0.9612 |
| 9 | 0.9278 | 0.9313 | 0.9611 |

w5 fired no transition at all in the 10-epoch budget.

---

## Loss-share confirmation

At ep2 step end (representative mid-training step), loss components were:

| component | magnitude | share of total |
|---|---:|---:|
| contrastive | 0.597 | 94.4% |
| aux (cohom dominates) | 0.022 | 3.4% |
| VICReg var+cov | 0.014 | 2.3% |
| **total** | **0.633** | 100% |

Aux contribution did reach the meaningful 3-4% range, validating that
the test was sized correctly to evaluate the anchoring hypothesis.

Per-head cohom CE at ep2 step end was ~0.006-0.013 — essentially
identical to what `aux_h256_cohom_diag` (w=0.1) had at the same epoch.
The 50× weight produced a 50× larger gradient signal but the encoder
pushed cohom losses to the same residual floor either way; the floor
appears to be set by irreducible class-boundary uncertainty rather than
by encoder capacity. So the aux pressure was applied early (ep0 had
3.7% grad-clip rate vs cohom_diag's 0.35%) and then dissipated as
the encoder reached the residual floor.

---

## What this run tells us

1. **Anti-drift anchoring is not supported by these features in this
   form.** Even at meaningful aux loss share (3-4%), the cohom heads
   produced no detectable change in the FALSE-recall drift trajectory.
   The encoder satisfies aux + drift simultaneously by routing them
   into orthogonal subspaces of the 512-dim pool.

2. **Stronger aux pressure suppresses the regime-B phase transition.**
   Reference and cohom_diag both fired transitions; w5 fired none. The
   larger aux gradient appears to keep the encoder in a deeper regime-A
   well from which the rapid transition mechanism doesn't escape, even
   though the encoder's geometry is slowly drifting toward regime B at
   end of training.

3. **Cohom features are trivially representable.** All 18 aux heads
   reach val accuracy 0.999+ by ep0, and the per-head CE residuals are
   nearly identical at w=0.1 and w=5.0. So whatever anti-drift signal
   the cohom features could in principle carry, the encoder was
   already free to encode them at any aux_weight; the gradient from
   the aux loss is not what determines whether they're represented.

---

## Conclusion

The aux-head paradigm with cohomology summary statistics — at any tested
aux_weight (0.1, 5.0) — does not prevent simplicity-bias drift at h=256.
The hypothesis "persistent aux gradient anchors useful features" is
unsupported for these features in this contrastive setup.

Three remaining possibilities for the cohom data, in order of how cleanly
they fall out of these results:

1. **Per-pair cohom feature, not per-equation aux head.** The simplest
   shape that respects the contrastive task's pair structure: feed
   `(cohom_eq1, cohom_eq2)` similarity (or pair-level features like
   "do they share a satisfying-prime profile") into the predictor or
   contrastive loss directly. This skips the aux-head paradigm entirely.

2. **Concat as input feature.** Encoder gets per-equation cohom
   summaries directly via concatenation to the pooled rep. Doesn't
   change inference (full corpus has cohom data). The encoder doesn't
   have to learn them; they're available to the contrastive loss for
   discrimination if useful.

3. **Drop cohom entirely as an auxiliary signal.** The data suggests
   the per-prime summaries don't carry information whose suppression
   would degrade the contrastive task. The signal that distinguishes
   TRUE from FALSE pairs at h=256 doesn't appear to live in this
   particular cohomological projection.

The basin-depth question (h=256 regime-B reproducibility) is unaffected
by these results and remains the actual blocker on h=256 progress. With
three of four h=256 runs failing to find regime B and the fourth
producing a partial-then-reverted transition, the seed-42 reference run
is increasingly looking like an outlier rather than a recipe.
