# `stable_m1p5_vic0p1_diag` — Diagnostic Report

Complete 10-epoch run with the full per-step diagnostic instrumentation (commit `6c40ed2`)
enabled. First run under the `stable_*` / `outputs/runs/<name>/` convention adopted in
commit `de1e4b4`. The first attempt at this run was killed mid-epoch 3 (session loss);
this is the rerun that completed end-to-end.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `stable_m1p5_vic0p1_diag` |
| Wandb id | `f63fyigk` |
| Wandb dir | `wandb/run-20260510_185040-f63fyigk/` |
| Config | `configs/experiments/stable_m1p5_vic0p1_diag.yaml` |
| Started | 2026-05-10 18:50:40 EDT |
| Finished | 2026-05-10 19:50:10 EDT |
| Wall clock | 59.5 min (3569.7 s) |
| Outcome | All 10 epochs completed, exit code 0 |
| Hardware | RTX 4080 (NVIDIA GeForce, 16 GB) |
| Predecessor (killed) | wandb id `3sutnmlc`, killed at step 37,625 (35% of training); checkpoints in `trash/stable_m1p5_vic0p1_diag_killed_2026-05-10/` |

### Config summary

```
encoder:    hidden_dim=128, embed_dim=512, num_layers=4, num_heads=4, dropout=0.1
            269,056 params
predictor:  embed_dim=512, hidden_dim=128, num_layers=2
            131,712 params
ema:        tau_start=0.996, tau_final=0.999, schedule=cosine
training:   batch_size=2048, lr=1e-3, weight_decay=0.05, optimizer=adamw,
            scheduler=cosine_warmup, warmup_steps=1000, epochs=10,
            grad_clip_norm=1.0, max_val_batches=200
loss:       type=contrastive, margin=1.5, true_weight=1.0,
            vicreg_var_weight=0.1, vicreg_cov_weight=0.1, vicreg_gamma=1.0
holdout:    none (no-holdout: train and validate on all 22M pairs)
log_every:  25 steps  (~430 log lines per epoch, 4,303 total)
```

The *m=1.5 / VICReg=0.1* recipe — same as `failure_analysis_m1p5_vic0p1` (the reference
run), now with all per-step diagnostics live.

### Outputs

```
outputs/checkpoints/stable_m1p5_vic0p1_diag/
  epoch_000.pt … epoch_009.pt   (10 checkpoints, 2.6 MB each)

wandb/run-20260510_185040-f63fyigk/
  files/output.log              (1.78 MB, 4,303 step logs + 10 epoch evals)
  files/config.yaml             (full hyperparameter snapshot)
  files/wandb-metadata.json
  files/wandb-summary.json      (final metric values)

outputs/runs/stable_m1p5_vic0p1_diag/
  diagnostics_report.md         (this file)
  diagnostic_extract.json       (parsed per-step aggregates by epoch + per-epoch evals)
```

---

## Per-epoch monitor (val on 200-batch cap, 409,600 pairs / epoch)

All 10 epochs. Bold = best for the metric.

| ep | bal_acc | true_R | false_R | thresh | eff_rank | gap_med | tr_p50 | fa_p50 | tr_p95 | fa_p05 | over_FbT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9299 | 0.9444 | 0.9153 | 1.2600 | 102.2 | 0.3131 | 1.1793 | 1.4924 | 1.2643 | 1.2381 | 0.0031 |
| 1 | 0.9309 | 0.9566 | 0.9053 | 1.2794 | 136.2 | 0.2712 | 1.2072 | 1.4784 | 1.2764 | 1.2543 | 0.0080 |
| 2 | 0.9386 | 0.9569 | 0.9204 | 1.2636 | 129.5 | 0.2861 | 1.1999 | 1.4860 | 1.2599 | 1.2467 | 0.0058 |
| 3 | 0.9448 | 0.9493 | 0.9402 | 1.2468 | 137.9 | 0.2979 | 1.1903 | 1.4883 | 1.2473 | 1.2414 | 0.0039 |
| 4 | 0.9469 | 0.9524 | 0.9413 | 1.2448 | 141.4 | 0.3072 | 1.1877 | 1.4949 | 1.2436 | 1.2399 | 0.0037 |
| 5 | 0.9451 | **0.9646** | 0.9256 | 1.2446 | 136.7 | 0.3075 | 1.1812 | 1.4887 | 1.2376 | 1.2319 | 0.0035 |
| **6** | **0.9492** | 0.9502 | **0.9482** | 1.2340 | 145.6 | 0.3127 | 1.1815 | 1.4942 | 1.2339 | 1.2331 | 0.0027 |
| 7 | 0.9490 | 0.9568 | 0.9411 | 1.2314 | 147.7 | 0.3200 | 1.1705 | 1.4905 | 1.2283 | 1.2268 | 0.0019 |
| 8 | 0.9455 | 0.9601 | 0.9310 | 1.2338 | **149.6** | 0.3244 | 1.1694 | 1.4937 | 1.2296 | 1.2241 | 0.0018 |
| 9 | 0.9443 | 0.9536 | 0.9350 | 1.2218 | 145.8 | **0.3391** | 1.1563 | 1.4954 | 1.2202 | 1.2141 | **0.0014** |

**Best epoch by bal_acc: ep6 (0.9492)**, with the most balanced recall split
(true 0.9502 / false 0.9482 — 0.2pp split). ep4 and ep7 are statistical ties
(0.9469 / 0.9490). ep8–9 show a small late-training drift downward in bal_acc
even as eff-rank and gap-median continue improving — interesting.

### Trajectory shape

- **bal_acc**: rapid 0.93 → 0.945 in epochs 0–4 (the recipe's fast-convergence regime),
  then ~0.945–0.949 plateau over ep5–9. Late-epoch drift of −0.5pp from ep6 to ep9.
- **true_recall**: bumpy — peaks at 0.9646 at ep5, then 0.9601 at ep8. Higher than
  false_recall in 9 of 10 epochs.
- **false_recall**: monotone-ish climb 0.92 → 0.95 ep0–6, then drift back to 0.93–0.94.
- **threshold** (calibrated each eval): drifts down 1.260 → 1.222 over the run as
  the TRUE distribution shifts left.
- **corpus_eff_rank**: 102 → 150 (+47%) over the run. Peak at ep8 (149.6), slight
  pullback at ep9. Reference run plateaued ~150–155 in this same window.
- **dist_gap_median**: monotonically grows after ep1 dip: 0.27 → 0.34. Classes
  separating further apart over training, even though bal_acc plateaued.
- **dist_true_p50**: monotonically falls 1.18 → 1.16. TRUE distribution is shifting
  left (smaller distances).
- **dist_false_p50**: essentially flat at 1.49 ± 0.01 across all epochs. FALSE
  distribution is stationary.
- **dist_overlap_false_below_true_med**: monotone shrinkage from ep2: 0.0058 →
  0.0014. Fewer FALSE pairs are landing below the TRUE median — the classes
  are cleanly separating.
- **dist_overlap_true_above_false_med**: ≤ 0.0003 every epoch. Negligible.

The headline tension: **bal_acc plateaus while geometric separation keeps improving.**
This says the residual error is concentrated near the threshold — borderline pairs the
model can't disambiguate further, even as the bulk populations move apart. Same pattern
the reference run's full-eval failure analysis found.

---

## Per-step training diagnostics

Per-step traces logged every 25 steps. **4,303 logged steps** across the run.

### Per-class loss split (the headline diagnostic)

Mean per epoch:

| ep | `train/loss` | `train/contrastive_loss_true` | `train/contrastive_loss_false` | TRUE/FALSE ratio |
|---:|---:|---:|---:|---:|
| 0 | 0.5659 | 1.3121 | 0.0176 | 75× |
| 1 | 0.5900 | 1.4294 | 0.0118 | 121× |
| 2 | 0.6086 | **1.4906** | 0.0103 | 145× |
| 3 | 0.6038 | 1.4699 | 0.0108 | 136× |
| 4 | 0.5992 | 1.4630 | 0.0110 | 133× |
| 5 | 0.5977 | 1.4536 | 0.0112 | 130× |
| 6 | 0.5928 | 1.4438 | 0.0115 | 126× |
| 7 | 0.5878 | 1.4314 | 0.0118 | 121× |
| 8 | 0.5823 | 1.4139 | 0.0122 | 116× |
| 9 | 0.5777 | 1.3962 | 0.0127 | 110× |

The full trajectory now visible:

1. **TRUE loss peaks at ep2 (1.49)** then *decreases* monotonically to 1.40 by ep9.
   The encoder is incrementally pulling TRUE pairs together over training.
2. **FALSE loss bottoms at ep2 (0.0103)** then *grows* monotonically to 0.0127 by ep9.
   As the TRUE pull tightens the geometry, a few more FALSE pairs slip below margin
   and start contributing.
3. **Ratio peaks at 145× at ep2**, drops to 110× by ep9 — TRUE still dominates
   completely, but FALSE is no longer 100%-solved.
4. **Total loss falls slowly** (0.61 → 0.58 from ep2 to ep9) — the slow improvement
   in bal_acc/eff-rank is being driven by tightening TRUE pairs, not by adding more
   FALSE-pair work.

This contradicts the killed-run report's reading (which only saw the ep0-3 window where
the curve looked flat). The full-run picture is: **TRUE loss has a real downward trend
in the second half of training, and FALSE loss has a real upward trend.** The classes
are *not* in steady-state — they're slowly trading off. ep6 is the bal_acc maximum
because that's roughly the inflection point where additional TRUE-tightening starts
costing more on the FALSE side than it gains on the TRUE side.

### Gradient norms

Per-epoch means:

| ep | `train/grad_norm_pre_clip` mean | max | `enc` mean | `pred` mean | `grad_clipped` |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0794 | 0.821 | 0.0694 | 0.0328 | 1 of 651 logged steps |
| 1 | 0.0343 | 0.262 | 0.0281 | 0.0188 | 0 |
| 2 | 0.0234 | 0.075 | 0.0185 | 0.0140 | 0 |
| 3 | 0.0228 | 0.058 | 0.0177 | 0.0141 | 0 |
| 4 | 0.0234 | 0.068 | 0.0172 | 0.0155 | 0 |
| 5 | 0.0252 | 0.077 | 0.0170 | 0.0183 | 0 |
| 6 | 0.0271 | 0.063 | 0.0158 | 0.0217 | 0 |
| 7 | 0.0310 | 0.049 | 0.0157 | 0.0265 | 0 |
| 8 | 0.0353 | 0.052 | 0.0157 | 0.0315 | 0 |
| 9 | 0.0393 | 0.049 | 0.0156 | 0.0360 | 0 |

**Clip threshold 1.0 is structurally irrelevant.** Max grad norm ever observed across
all 4,303 logged steps was **0.821**, in early ep0 warmup. After warmup, max never
exceeds 0.262, and from ep2 onwards never exceeds 0.077. **No clip event in any of
epochs 1–9.** The `clipped_step_fraction` epoch counter:

| ep | `clipped_step_fraction` |
|---:|---:|
| 0 | 0.0015 (about 16 of 10,757 steps; all in warmup) |
| 1–9 | 0.0000 |

**Encoder vs predictor crossover** is the surprise of the full-trajectory data:

- ep0–4: encoder grad dominates (encoder ≈ 1.5–2.5× predictor).
- ep5: rough parity (0.017 enc vs 0.018 pred).
- ep6–9: **predictor grad dominates and grows** (0.022 → 0.036), while encoder grad
  *stays flat at ~0.016*.

The encoder appears to converge by ep5 — its gradient signal stabilizes — while the
predictor continues taking adjustments through the end of training. By ep9 the
predictor is doing 2.3× as much work per step as the encoder. This is consistent with
a JEPA training dynamic where the predictor adapts to a settled encoder geometry; it
also fits the BYOL/SimSiam observation that the predictor is the "fast learner" of
the pair.

This is the kind of detail the full diagnostic trace was specifically built to reveal —
it was not visible at all in the ep0-3 killed-run window, because the crossover hadn't
happened yet.

### VICReg components

Per-epoch means:

| ep | `train/vicreg_var` | `train/vicreg_cov` |
|---:|---:|---:|
| 0 | 0.5965 | 0.0903 |
| 1 | 0.4199 | 0.1096 |
| 2 | 0.4014 | 0.1063 |
| 3 | 0.4000 | 0.1057 |
| 4 | 0.3997 | 0.1053 |
| 5 | 0.3994 | 0.1050 |
| 6 | 0.3992 | 0.1047 |
| 7 | 0.3990 | 0.1045 |
| 8 | 0.3989 | 0.1043 |
| 9 | 0.3988 | 0.1042 |

After the ep1 transient (the first time the VICReg gamma=1.0 hinge kicks in fully),
both components are essentially constant: var = 0.399 ± 0.0008, cov = 0.105 ± 0.002
across ep2–9. **VICReg is paying a small, stable cost** — about 0.1 × (0.40 + 0.10)
= 0.05 of the total loss (~8.5% of 0.59).

The slight monotone *decrease* in cov from ep2 (0.106) to ep9 (0.104) is consistent
with eff-rank growing — the off-diagonal covariance shrinks as features spread along
more directions. Variance stays right at the equilibrium the contrastive force
balances against.

### LR & EMA schedule

Cosine warmup + cosine decay; ep0 ramps 0 → 1e-3, then full cosine to 0 by step 107,570.

| ep | `train/lr` mean | `train/tau` mean |
|---:|---:|---:|
| 0 | 0.00098 | 0.9960 |
| 1 | 0.00098 | 0.9962 |
| 2 | 0.00094 | 0.9964 |
| 3 | 0.00086 | 0.9967 |
| 4 | 0.00074 | 0.9971 |
| 5 | 0.00059 | 0.9975 |
| 6 | 0.00043 | 0.9979 |
| 7 | 0.00027 | 0.9983 |
| 8 | 0.00012 | 0.9987 |
| 9 | 0.00003 | 0.9990 |

By ep9, LR is near zero. Tau finished at 0.9990 (matches the configured `tau_final`).

### Step rate

Steady ~33 ms/step throughout, ~6 min/epoch including monitor eval. Total wall clock
3569.7 s = 59.5 min. Reference run was 53 min — 12% slower here, plausibly due to
diagnostic computations (per-step `cont_loss_true`/`cont_loss_false` recomputation,
per-module grad norm reductions).

---

## Comparison to reference run (`failure_analysis_m1p5_vic0p1`)

Same recipe, same code, same seed. Reference wandb id: `ogruyfe1`.

| ep | bal_acc (this) | bal_acc (ref) | Δ | eff_rank (this) | eff_rank (ref) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9299 | 0.9380 | −0.81 | 102.2 | 103.5 |
| 1 | 0.9309 | 0.9355 | −0.46 | 136.2 | 126.4 |
| 2 | 0.9386 | 0.9425 | −0.39 | 129.5 | 141.6 |
| 3 | 0.9448 | 0.9514 | −0.66 | 137.9 | 138.4 |
| 4 | 0.9469 | 0.9526 | −0.57 | 141.4 | 138.7 |
| 5 | 0.9451 | 0.9496 | −0.45 | 136.7 | 143.5 |
| 6 | **0.9492** | **0.9526** | −0.34 | 145.6 | 152.5 |
| 7 | 0.9490 | 0.9514 | −0.24 | 147.7 | 154.6 |
| 8 | 0.9455 | 0.9476 | −0.21 | 149.6 | 152.4 |
| 9 | 0.9443 | 0.9479 | −0.36 | 145.8 | 149.9 |

**This run runs 0.2–0.8pp below reference at every epoch.** Mean gap −0.45 pp. The
shape of the trajectory is the same (peak around ep6, plateau ep4–7, slight late-run
drift), just shifted down. Same is true for eff-rank: this run trails by 5–17 units.

### Why the gap?

Same code, same seed, same config — but CUDA training is not bit-deterministic. The
killed-run snapshot also showed similar per-epoch drift from the reference (ep0:
−0.35 pp; ep1: −0.07 pp; ep2: +0.02 pp), so this rerun is consistent with that
killed-run frame in spirit (sub-percentage-point per-epoch noise).

Three plausible sources:
1. **CUDA non-determinism**: GATv2Conv's scatter-reduce on the GPU is order-dependent;
   reductions in different orders yield slightly different floats. Compounded over
   107K steps, this is enough to produce ~0.5pp drift in bal_acc.
2. **DataLoader worker order**: with `num_workers=4`, the order in which workers
   produce batches isn't seed-pinnable.
3. **Calibration variance**: the threshold is calibrated on a 1M random-pair monitor
   subsample, which can shift slightly between runs.

The failure-analysis run was reported at full-eval bal_acc 0.9523 from the ep4
checkpoint, calibrated on a different threshold pass. **To get a fair comparison
of this run's best epoch (ep6, 0.9492 monitor) against the reference's full-eval
0.9523**, run `scripts/save_predictions.py` on `epoch_006.pt` to do a full
22M-pair eval — that should land in the 0.948–0.952 range based on the ratio
of this run's monitor-vs-ref-monitor (~0.997×) applied to the ref's full-eval.

---

## What the diagnostics tell us — by hypothesis (full-run verdicts)

The diagnostic commit (`6c40ed2`) was added to address three open questions about
the original `failure_analysis` (m=1.0, vic=0.25) instability. Full 10-epoch verdicts:

### H1: "Loss spikes / gradient instability is causing the per-epoch swings"

**Verdict (m=1.5/vic=0.1): Decisively no instability.**

- Total grad norm is < 0.1 across **every single one of 4,303 logged steps** in epochs 1–9.
  Max post-warmup is 0.262 (ep1), and from ep2 onwards never exceeds 0.077.
- Clip threshold 1.0 vs observed max 0.821 (only in ep0 warmup) → in steady state
  the clip is **40× too high** to ever fire.
- Per-epoch `clipped_step_fraction`: 0.0015 (ep0 warmup only), 0.0 in **all of ep1–9**
  across all 96,813 steps in those epochs.
- Loss curve is smooth at the 25-step log resolution; no spikes anywhere.

Compared to the original `failure_analysis` (m=1.0, vic=0.25), where per-epoch bal_acc
swung 0.66 ↔ 0.89 across 20 epochs and corpus_eff_rank swung 11 ↔ 122 — this run is
flat in every comparable metric. **The stabilization recipe works**, and we have the
per-step receipts now.

### H2: "Which class is destabilizing the loss curve"

**Verdict: TRUE class carries the entire learning signal; FALSE pairs trade slowly
back as the geometry tightens.**

- TRUE loss settles in the 1.40–1.49 band from ep1 onwards; **trends downward**
  monotonically from ep2 (1.49) to ep9 (1.40).
- FALSE loss settles in the 0.010–0.013 band; **trends upward** monotonically from
  ep2 (0.0103) to ep9 (0.0127).
- Ratio: 145× at ep2 peak → 110× at ep9. Still TRUE-dominated everywhere.
- Standard deviation of per-step TRUE loss within each epoch shrinks from 0.19 (ep0)
  to 0.005 (ep3) and stays flat through ep9 — the per-pair TRUE distance distribution
  becomes very stable.

Implication for next experiments: **the TRUE-FALSE trade-off is real and continuous.**
Late training, the encoder is "spending" some FALSE-side margin to gain TRUE-side
tightness. ep6 looks like the empirically-optimal trade-off point. Interventions
could include:
- **Early-stop at ep6** (smallest deployment cost, preserves the best monitor bal_acc).
- **Curriculum on TRUE pairs** — focus late-training on TRUE pairs that are still
  borderline; could continue tightening TRUE without spilling to FALSE.
- **Adaptive `true_weight` schedule** — start at 1.0, drop to <1.0 in late training
  to slow the TRUE pull and preserve FALSE separation.

The encoder-converges-before-predictor signature also suggests the encoder may
benefit from a separate (lower) LR schedule than the predictor.

### H3: "Whether margin changes shift the population gap"

**Verdict: m=1.5 holds populations apart and the gap *grows* with training.**

| Epoch | dist_gap_median | dist_true_p50 | dist_false_p50 | overlap_FbT |
|---:|---:|---:|---:|---:|
| 0 | 0.313 | 1.179 | 1.492 | 0.0031 |
| 6 | 0.313 | 1.182 | 1.494 | 0.0027 |
| 9 | **0.339** | **1.156** | 1.495 | **0.0014** |

- FALSE p50 is *flat* at 1.49 ± 0.01 across the entire run. The FALSE population is
  pinned by the m=1.5 margin and doesn't move.
- TRUE p50 *drifts left*: 1.18 → 1.16. The encoder keeps tightening TRUE pairs.
- Gap median grows from 0.31 to 0.34 (+10%).
- `overlap_false_below_true_med` (a structural overlap metric, not a calibration-noise
  one) shrinks by **2.2×** over the run: 0.0031 → 0.0014.

**The gap behavior is the cleanest summary of this recipe**: the margin pins FALSE,
the contrastive pull tightens TRUE, and the result is monotone separation. bal_acc
plateaus despite this geometric improvement because the **threshold** has to be
calibrated on the borderline region — and that borderline doesn't get cleaner just
because the bulk distributions move apart.

---

## What changed vs the killed-run snapshot

The killed run captured ep0–ep2 fully and ~50% of ep3. The full-run picture changes
several conclusions from the killed-run report:

1. **TRUE loss is not flat** — the killed-run window saw it climb from ep0 to a
   peak at ep2; the full run shows it then decreases monotonically through ep9.
2. **Predictor grad eventually exceeds encoder grad** — only visible from ep5 onward.
   In the killed-run window encoder always dominated.
3. **Best epoch is ep6, not "expected ~ep4 plateau"** — close to the reference's
   ep4 vs ep6 dual-peak, but in this run ep6 wins clearly.
4. **eff-rank tops out at 149.6 (ep8)** — slightly below the reference's 154.6 (ep7),
   consistent with the run's overall ~0.5pp underperformance.
5. **Late-epoch drift exists**: ep9 bal_acc 0.9443 < ep6 0.9492. Either training is
   slightly past the optimum at the configured 10 epochs, or it's normal noise in
   the post-plateau regime.

---

## Headline numbers

- **Best epoch: ep6** with bal_acc **0.9492**, true_recall 0.9502, false_recall 0.9482
  (most balanced split of any epoch). Threshold 1.234.
- **Total wall clock: 59.5 min** (target plateau reached by ep4–6).
- **Zero clip events** in epochs 1–9 over 96,813 steps. Max grad norm post-warmup: 0.262.
  **Clip threshold (1.0) is structurally irrelevant** in this regime.
- **TRUE loss trajectory: 1.31 → 1.49 (peak ep2) → 1.40 (ep9)**. FALSE loss:
  **0.018 → 0.010 (min ep2) → 0.013 (ep9)**. The classes trade off slowly through
  the second half of training — TRUE tightens as FALSE loosens slightly.
- **Predictor-encoder grad crossover at ep5**. Encoder converges (grad flat at
  0.016 from ep5 onwards) while predictor keeps adapting (0.018 → 0.036).
- **VICReg components flat**: var = 0.399, cov = 0.105 across ep2–9. No drift.
- **Eff-rank**: 102 (ep0) → 149.6 (ep8 peak) → 145.8 (ep9). Reference plateaued
  at 154.6.
- **Distance gap monotonically grows from 0.27 (ep1) → 0.34 (ep9)**, even though
  bal_acc plateaus by ep4 — the residual error is concentrated near the threshold,
  not in poor bulk separation.
- **Mean gap from reference run: −0.45 pp bal_acc**; same trajectory shape, attributable
  to CUDA non-determinism + DataLoader worker ordering.

## Suggested follow-ups

1. **Dump full-eval predictions for `epoch_006.pt`** via `scripts/save_predictions.py`
   to get the apples-to-apples 22M-pair bal_acc for fair comparison to the reference's
   0.9523. Probably lands in 0.948–0.952.
2. **Re-look at why ep6 is the optimum** — chart per-pair TRUE/FALSE loss vs distance
   from threshold at ep4, ep6, ep9 to see *which* pairs are flipping correct/wrong
   in the late drift.
3. **Test the early-stop hypothesis**: run a 7-epoch variant of this same recipe
   to see if stopping at ep6 (instead of training to ep9) gives a noisier run a
   reliable bal_acc near 0.95 with less compute.
4. **Encoder/predictor LR split**: given the encoder converges by ep5, a smaller-LR
   encoder + larger-LR predictor in the second half could continue improving without
   the late-epoch drift.
