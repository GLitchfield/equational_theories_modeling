# `stable_h256_m1p5_vic0p1_diag` — Diagnostic Report

Wider encoder (hidden_dim 128 → 256) under the stabilization recipe.
Tests whether the prior `hidden_dim_256_*` failures were instability artifacts
or a real capacity-vs-recipe mismatch.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `stable_h256_m1p5_vic0p1_diag` |
| Wandb id | `5t3fskb0` |
| Wandb dir | `wandb/run-20260510_212220-5t3fskb0/` |
| Config | `configs/experiments/stable_h256_m1p5_vic0p1_diag.yaml` |
| Wall clock | 103.8 min (6229.7 s) — **1.75× h=128** |
| Outcome | 10 epochs completed, exit code 0 |
| Encoder params | 799,744 (vs 269,056 at h=128; +197%) |
| Predictor params | 131,712 (unchanged) |

Recipe identical to `stable_m1p5_vic0p1_diag` except `encoder.hidden_dim`:
margin 1.5, VICReg 0.1/0.1, no-holdout 22M-pair train, batch 2048, lr 1e-3,
warmup 1000, 10 epochs, EMA cosine 0.996→0.999, seed 42.

---

## Per-epoch monitor (200-batch val cap, 409,600 pairs)

| ep | bal_acc | true_R | false_R | thresh | eff_rank | gap | tr_p05 | tr_p50 | tr_p95 | fa_p05 | fa_p50 | fa_p95 | over_TaF | over_FbT | val_loss | tr_loss |
|---:|--------:|-------:|--------:|-------:|---------:|----:|-------:|-------:|-------:|-------:|-------:|-------:|---------:|---------:|---------:|--------:|
| **0** | **0.9333** | 0.9612 | 0.9054 | 1.2893 | 206.04 | 0.2648 | 1.1844 | 1.2295 | 1.2843 | 1.2671 | 1.4943 | 1.5657 | 0.0008 | 0.0050 | 0.5682 | 0.5753 |
| 1 | 0.9242 | 0.9796 | 0.8689 | 1.3418 | 272.68 | 0.2112 | 1.2239 | 1.2695 | 1.3249 | 1.3004 | 1.4806 | 1.5426 | 0.0002 | 0.0046 | 0.6038 | 0.6331 |
| 2 | 0.9262 | 0.9737 | 0.8786 | 1.3397 | 305.11 | 0.2034 | 1.2261 | 1.2731 | 1.3276 | 1.3060 | 1.4765 | 1.5394 | 0.0003 | 0.0046 | 0.6064 | 0.5951 |
| 3 | 0.9262 | 0.9632 | 0.8893 | 1.3386 | 320.81 | 0.2061 | 1.2205 | 1.2733 | 1.3327 | 1.3072 | 1.4794 | 1.5399 | 0.0003 | 0.0052 | 0.6065 | 0.6199 |
| 4 | 0.9294 | 0.9819 | 0.8769 | 1.3317 | 304.28 | 0.2500 | 1.1887 | 1.2421 | 1.3080 | 1.2757 | 1.4921 | 1.5616 | 0.0004 | 0.0074 | 0.5794 | 0.6282 |
| 5 | 0.9290 | 0.9804 | 0.8776 | 1.2887 | 280.29 | **0.2908** | 1.1509 | 1.1988 | 1.2644 | 1.2361 | 1.4896 | 1.5745 | 0.0003 | **0.0031** | 0.5431 | 0.5965 |
| 6 | 0.9255 | 0.9746 | 0.8765 | 1.3461 | 330.65 | 0.2126 | 1.2139 | 1.2679 | 1.3333 | 1.3003 | 1.4805 | 1.5526 | 0.0003 | 0.0056 | 0.6022 | 0.6155 |
| 7 | 0.9250 | 0.9810 | 0.8691 | 1.3603 | 345.88 | 0.2017 | 1.2214 | 1.2764 | 1.3423 | 1.3090 | 1.4781 | 1.5498 | 0.0003 | 0.0070 | 0.6098 | 0.6372 |
| 8 | 0.9256 | 0.9722 | 0.8789 | 1.3503 | 346.80 | 0.2029 | 1.2197 | 1.2750 | 1.3388 | 1.3068 | 1.4779 | 1.5511 | 0.0002 | 0.0083 | 0.6079 | 0.6367 |
| 9 | 0.9273 | 0.9801 | 0.8745 | 1.3548 | **349.46** | 0.2039 | 1.2190 | 1.2752 | 1.3387 | 1.3076 | 1.4791 | 1.5522 | 0.0002 | 0.0084 | 0.6076 | 0.6354 |

**Best epoch = ep0 (0.9333).** Trajectory is *flat-to-degrading* rather than
climb-and-plateau. bal_acc drops 0.9pp ep0→ep1, partial recovery ep4–5, then
plateau around 0.925–0.928 through ep9.

---

## Per-step diagnostic aggregates (mean per epoch over ~430 logged steps)

| ep | loss | cont | cl_TRUE | cl_FALSE | grad | g_enc | g_pred | vic_var | vic_cov | lr | tau |
|---:|-----:|-----:|--------:|---------:|-----:|------:|-------:|--------:|--------:|----:|----:|
| 0 | 0.5751 | 0.5214 | 1.3718 | 0.0205 | 0.0865 | 0.0779 | 0.0289 | 0.4230 | 0.1138 | warmup→1e-3 | 0.9960 |
| 1 | 0.6316 | 0.6017 | 1.6102 | 0.0089 | 0.0305 | 0.0273 | 0.0130 | **0.1064** | 0.1931 | 9.5e-4 | 0.9962 |
| 2 | 0.5938 | 0.5663 | 1.5095 | 0.0118 | 0.0306 | 0.0257 | 0.0158 | 0.0630 | 0.2122 | 8.6e-4 | 0.9964 |
| 3 | 0.6183 | 0.5914 | 1.5818 | 0.0097 | 0.0251 | 0.0179 | 0.0171 | 0.0526 | 0.2162 | 7.4e-4 | 0.9968 |
| 4 | 0.6287 | 0.6022 | 1.6062 | 0.0092 | 0.0263 | 0.0161 | 0.0205 | 0.0464 | 0.2185 | 5.8e-4 | 0.9973 |
| 5 | 0.5962 | 0.5699 | 1.5174 | 0.0114 | 0.0313 | 0.0174 | 0.0257 | 0.0437 | 0.2192 | 4.4e-4 | 0.9977 |
| 6 | 0.6154 | 0.5893 | 1.5718 | 0.0098 | 0.0336 | 0.0154 | 0.0296 | 0.0420 | 0.2193 | 2.7e-4 | 0.9982 |
| 7 | 0.6385 | 0.6125 | 1.6333 | 0.0082 | 0.0355 | 0.0135 | 0.0327 | 0.0406 | 0.2195 | 1.5e-4 | 0.9986 |
| 8 | 0.6348 | 0.6088 | 1.6322 | 0.0082 | 0.0370 | 0.0130 | 0.0346 | 0.0399 | 0.2194 | 6e-5 | 0.9988 |
| 9 | 0.6352 | 0.6093 | 1.6290 | 0.0082 | 0.0380 | 0.0128 | 0.0357 | 0.0396 | **0.2193** | 0 | 0.9990 |

### Max grad_norm + clip events per epoch

| ep | max grad_pre | clip_frac | epoch wall (s) |
|---:|-------------:|----------:|---------------:|
| 0 | **1.8216** | 0.0045 (~48 events) | 614 |
| 1 | 0.1480 | 0.0000 | 639 |
| 2 | 0.1868 | 0.0000 | 680 |
| 3 | 0.0661 | 0.0000 | 696 |
| 4 | 0.0548 | 0.0000 | 695 |
| 5 | 0.0521 | 0.0000 | 673 |
| 6 | 0.0626 | 0.0000 | 612 |
| 7 | 0.0451 | 0.0000 | 568 |
| 8 | 0.0453 | 0.0000 | 540 |
| 9 | 0.0469 | 0.0000 | 512 |

Clip fires ~48 times in ep0 warmup (vs ~16 at h=128) — the wider encoder
spikes higher during warmup. Post-warmup grads are actually *smaller* than
h=128 (max 0.19 vs 0.26). Stability is not the failure mode.

---

## Side-by-side vs h=128 (`f63fyigk`)

| ep | bal_acc h128 | bal_acc h256 | Δ | eff_rank h128 | eff_rank h256 | true_R h128 | true_R h256 | false_R h128 | false_R h256 |
|---:|-------------:|-------------:|------:|--------------:|--------------:|------------:|------------:|-------------:|-------------:|
| 0 | 0.9299 | **0.9333** | +0.34 | 102.2 | 206.0 | 0.9444 | 0.9612 | 0.9153 | 0.9054 |
| 1 | 0.9309 | 0.9242 | -0.67 | 136.2 | 272.7 | 0.9566 | 0.9796 | 0.9053 | 0.8689 |
| 2 | 0.9386 | 0.9262 | -1.24 | 129.5 | 305.1 | 0.9569 | 0.9737 | 0.9204 | 0.8786 |
| 3 | 0.9448 | 0.9262 | -1.86 | 137.9 | 320.8 | 0.9493 | 0.9632 | 0.9402 | 0.8893 |
| 4 | 0.9469 | 0.9294 | -1.75 | 141.4 | 304.3 | 0.9524 | 0.9819 | 0.9413 | 0.8769 |
| 5 | 0.9451 | 0.9290 | -1.61 | 136.7 | 280.3 | 0.9646 | 0.9804 | 0.9256 | 0.8776 |
| 6 | **0.9492** | 0.9255 | -2.37 | 145.6 | 330.6 | 0.9502 | 0.9746 | 0.9482 | 0.8765 |
| 7 | 0.9490 | 0.9250 | -2.40 | 147.7 | 345.9 | 0.9568 | 0.9810 | 0.9411 | 0.8691 |
| 8 | 0.9455 | 0.9256 | -1.99 | 149.6 | 346.8 | 0.9601 | 0.9722 | 0.9374 | 0.8789 |
| 9 | 0.9443 | 0.9273 | -1.70 | 145.8 | 349.5 | 0.9536 | 0.9801 | 0.9350 | 0.8745 |

Wider encoder underperforms by 1.5–2.4 pp from ep1 onwards. Stability is
fine, but the model is heavily TRUE-biased: true_R 0.97–0.98 vs false_R
0.87–0.90 — a 9–11 pp gap, where h=128 was within 0.5–4 pp.

---

## Comparison to prior `hidden_dim_256_vicreg` (Phase 1)

Prior result under the *pre-stabilization* recipe (VICReg 0.25/0.25):

| metric | hidden_dim_256_vicreg | stable_h256_m1p5_vic0p1_diag | Δ |
|---|---:|---:|---:|
| best bal_acc | 0.7571 | 0.9333 | +0.176 |
| best true_recall | 0.8826 | 0.9612 | +0.079 |
| best false_recall | 0.6317 | 0.9054 | +0.274 |
| corpus_eff_rank (peak) | 64 | 350 | +5.5× |
| trajectory | bal_acc collapses 0.76 → 0.59 by ep3 | flat 0.93 from ep0 onwards |

The stable recipe **fixes the collapse** — wider encoder is no longer broken,
just suboptimal. h=256 now plateaus at a stable but worse level than h=128;
previously it actively degraded. Stabilization works at h=256, the recipe
just isn't capacity-matched.

---

## Mechanism: why does wider hurt?

The VICReg components landed in a qualitatively different regime:

| | h=128 (ep5–9) | h=256 (ep5–9) |
|---|---:|---:|
| vic_var | 0.399 | **0.041** (10×) |
| vic_cov | 0.105 | **0.219** (2.1×) |
| TRUE contrastive loss | 1.43 | 1.59 (+11%) |
| FALSE contrastive loss | 0.012 | 0.009 (-25%) |
| dist_true p50 | 1.18 | 1.27 (+8%) |
| dist_false p50 | 1.49 | 1.48 (flat) |
| eff_rank | 145 | 345 |

The wider encoder's pre-norm features:
- have **far higher per-feature variance** → vic_var saturates the γ=1 hinge
  (var_loss = ReLU(γ−std) hits the floor)
- but also have **2× the off-diagonal covariance** the cov term is penalizing
- net effect: 0.1 × (0.04 + 0.22) = 0.026 VICReg loss, lower than h=128's
  0.05 — VICReg is paying *less* in total but is structurally different

Consequence: the wider encoder spreads features broadly (high eff-rank), but
can't *concentrate* TRUE pairs as tightly. TRUE p50 lands at 1.27 vs 1.18 —
this matters because the threshold sits near TRUE p95, and a left-shifted
TRUE distribution would give cleaner classification. The wider model
struggles to pull TRUE pairs in toward each other in this VICReg regime.

The 10× higher per-feature variance under the same γ=1 hinge means the
variance term is *weaker* per feature (more features sit at the floor).
With ~2× as many features, the model has more dimensions to disperse into
but is fighting against the variance pressure less effectively per dim.

---

## Suggested follow-ups

1. **Lower VICReg at h=256**: try `vicreg_var_weight=0.05, vicreg_cov_weight=0.05`
   (or 0.02/0.02) to release the variance pressure and let the contrastive
   force tighten TRUE pairs. The h=128 recipe was tuned for 128-dim
   pre-norm features; the per-feature pressure should likely scale with
   width.
2. **Or rebalance with stronger contrastive**: a higher `true_weight`
   (1.5–2.0) might fight back against the TRUE-biased recall imbalance.
3. **Sanity probe**: extract embeddings from `epoch_009.pt` and check
   whether the 9pp FALSE-recall gap is uniformly distributed or
   concentrated in a specific equation cluster.

---

## Headline numbers

- **Best: ep0 bal_acc 0.9333** (true_R 0.9612, false_R 0.9054, threshold 1.289).
  Best epoch is the *first* — wider model never improves over its warmup state.
- **Total wall clock 103.8 min** (1.75× h=128).
- **Eff-rank: 206 → 349** (peak ep9). 2.4× higher than h=128 peak.
- **VICReg regime shift**: var 0.40 → 0.04 (10×), cov 0.11 → 0.22 (2×).
- **Recall imbalance**: 9–11 pp gap (TRUE >> FALSE) the entire run.
- **Stability holds**: 1 clip-event-cluster in ep0 warmup (~48 steps), zero
  clipping in ep1–9 (96,813 steps). Max post-warmup grad 0.19.
- **Mean bal_acc gap from h=128 ep1–9: −1.62 pp.** Wider is consistently
  worse under this recipe.
