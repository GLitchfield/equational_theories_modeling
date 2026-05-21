# `stable_h256_m1p5_vic0p05_diag` — Diagnostic Report

Variant A from the h=256 follow-up. Same recipe as
`stable_h256_m1p5_vic0p1_diag` but VICReg weights halved (0.1 → 0.05 on
both var and cov). Tests whether the wider encoder's degradation under
the stable recipe is driven by VICReg over-regularization.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `stable_h256_m1p5_vic0p05_diag` |
| Wandb id | `nsmxa1r5` |
| Wandb dir | `wandb/run-20260511_085856-nsmxa1r5/` |
| Config | `configs/experiments/stable_h256_m1p5_vic0p05_diag.yaml` |
| Wall clock | 80.5 min (4830.3 s) — faster than h=256/vic=0.1 baseline (103.8 min) |
| Outcome | 10 epochs completed, exit code 0 |
| Encoder params | 799,744 (same as h=256/vic=0.1) |

Single-knob change: `vicreg_var_weight 0.1 → 0.05`, `vicreg_cov_weight 0.1 → 0.05`.

---

## Per-epoch monitor (200-batch val cap, 409,600 pairs)

| ep | bal_acc | true_R | false_R | thresh | eff_rank | gap | tr_p05 | tr_p50 | tr_p95 | fa_p05 | fa_p50 | fa_p95 | over_TaF | over_FbT | val_loss | tr_loss |
|---:|--------:|-------:|--------:|-------:|---------:|----:|-------:|-------:|-------:|-------:|-------:|-------:|---------:|---------:|---------:|--------:|
| **0** | **0.9469** | 0.9441 | **0.9497** | 1.2200 | 135.47 | 0.3494 | 1.0770 | 1.1495 | 1.2238 | 1.2198 | 1.4988 | 1.5973 | 0.0001 | **0.0004** | 0.4995 | 0.5076 |
| 1 | 0.9283 | 0.9778 | 0.8788 | 1.3234 | 228.74 | 0.2418 | 1.1952 | 1.2459 | 1.3083 | 1.2855 | 1.4877 | 1.5572 | 0.0001 | 0.0051 | 0.5828 | 0.5945 |
| 2 | 0.9299 | 0.9787 | 0.8811 | 1.3211 | 289.62 | 0.2491 | 1.1917 | 1.2410 | 1.3029 | 1.2788 | 1.4900 | 1.5591 | 0.0002 | 0.0052 | 0.5784 | 0.5849 |
| 3 | 0.9343 | 0.9756 | 0.8929 | 1.2941 | 248.84 | 0.2751 | 1.1649 | 1.2144 | 1.2750 | 1.2527 | 1.4895 | 1.5569 | 0.0002 | 0.0057 | 0.5556 | 0.5947 |
| 4 | 0.9280 | 0.9826 | 0.8734 | 1.3477 | **299.15** | 0.2212 | 1.2184 | 1.2638 | 1.3276 | 1.2983 | 1.4850 | 1.5489 | 0.0002 | 0.0039 | 0.5998 | 0.6137 |
| 5 | 0.9278 | 0.9848 | 0.8708 | 1.2953 | 274.31 | 0.3144 | 1.1381 | 1.1836 | 1.2511 | 1.2232 | 1.4979 | 1.5752 | 0.0001 | 0.0039 | 0.5314 | 0.5477 |
| 6 | 0.9275 | 0.9881 | 0.8669 | 1.3058 | 268.52 | 0.3382 | 1.1137 | 1.1587 | 1.2271 | 1.1962 | 1.4969 | 1.5735 | 0.0001 | 0.0027 | 0.5114 | 0.5289 |
| 7 | 0.9277 | 0.9875 | 0.8679 | 1.3029 | 257.82 | 0.3449 | 1.1096 | 1.1552 | 1.2261 | 1.2002 | 1.5001 | 1.5780 | 0.0001 | 0.0029 | 0.5084 | 0.5473 |
| 8 | 0.9283 | 0.9884 | 0.8682 | 1.3063 | 281.52 | 0.3142 | 1.1360 | 1.1818 | 1.2487 | 1.2216 | 1.4960 | 1.5736 | 0.0001 | 0.0043 | 0.5298 | 0.5567 |
| 9 | 0.9284 | 0.9886 | 0.8682 | 1.3093 | 288.46 | 0.3094 | 1.1401 | 1.1864 | 1.2525 | 1.2262 | 1.4958 | 1.5728 | 0.0001 | 0.0044 | 0.5335 | 0.5641 |

**Best epoch: ep0 (0.9469)** — strongest h=256 result so far. Same
flat-after-warmup trajectory as h=256/vic=0.1; the best-at-ep0 pattern
persists. Late-training recall drift: true_R climbs 0.94→0.99, false_R
falls 0.95→0.87 by ep9. TRUE-bias regime re-emerges by ep1.

---

## Per-step diagnostic aggregates (mean per epoch)

| ep | loss | cont | cl_TRUE | cl_FALSE | grad | g_enc | g_pred | vic_var | vic_cov | lr | tau |
|---:|-----:|-----:|--------:|---------:|-----:|------:|-------:|--------:|--------:|----:|----:|
| 0 | 0.5074 | 0.4751 | 1.2434 | 0.0221 | 0.0722 | 0.0634 | 0.0267 | 0.5456 | 0.1013 | warmup→1e-3 | 0.9960 |
| 1 | 0.5932 | 0.5725 | 1.5299 | 0.0099 | 0.0236 | 0.0171 | 0.0159 | 0.2737 | 0.1395 | 9.5e-4 | 0.9962 |
| 2 | 0.5836 | 0.5679 | 1.5150 | 0.0111 | 0.0237 | 0.0168 | 0.0164 | 0.1309 | 0.1840 | 8.6e-4 | 0.9964 |
| 3 | 0.5931 | 0.5783 | 1.5458 | 0.0102 | 0.0218 | 0.0138 | 0.0166 | 0.0987 | 0.1970 | 7.4e-4 | 0.9968 |
| 4 | 0.6143 | 0.5999 | 1.6005 | 0.0088 | 0.0204 | 0.0117 | 0.0166 | 0.0865 | 0.2008 | 5.8e-4 | 0.9973 |
| 5 | 0.5473 | 0.5332 | 1.4141 | 0.0140 | 0.0318 | 0.0208 | 0.0236 | 0.0786 | 0.2039 | 4.4e-4 | 0.9977 |
| 6 | 0.5288 | 0.5149 | 1.3616 | 0.0155 | 0.0318 | 0.0170 | 0.0266 | 0.0743 | 0.2052 | 2.7e-4 | 0.9982 |
| 7 | 0.5484 | 0.5346 | 1.4143 | 0.0138 | 0.0344 | 0.0154 | 0.0306 | 0.0720 | 0.2051 | 1.5e-4 | 0.9986 |
| 8 | 0.5550 | 0.5412 | 1.4413 | 0.0130 | 0.0334 | 0.0140 | 0.0301 | 0.0709 | 0.2048 | 6e-5 | 0.9988 |
| 9 | 0.5640 | 0.5502 | 1.4626 | 0.0123 | 0.0336 | 0.0131 | 0.0308 | 0.0705 | 0.2046 | 0 | 0.9990 |

### Max grad_norm + clip events per epoch

| ep | max grad | clip_frac | epoch wall (s) |
|---:|---------:|----------:|---------------:|
| 0 | 1.0232 | 0.0033 (~36 events) | 487 |
| 1 | 0.1408 | 0.0000 | 488 |
| 2 | 0.1130 | 0.0000 | 492 |
| 3 | 0.0461 | 0.0000 | 494 |
| 4 | 0.0345 | 0.0000 | 495 |
| 5 | 0.1063 | 0.0000 | 493 |
| 6 | 0.0560 | 0.0000 | 475 |
| 7 | 0.0570 | 0.0000 | 468 |
| 8 | 0.0436 | 0.0000 | 468 |
| 9 | 0.0431 | 0.0000 | 468 |

Single clip-event-cluster in ep0 warmup (~36 events, fewer than vic=0.1's 48).
Zero clipping ep1–9. Post-warmup max grad 0.14.

---

## Three-way side-by-side (best epoch)

| metric | h=128 vic=0.1 ep6 | h=256 vic=0.1 ep0 | **h=256 vic=0.05 ep0** |
|---|---:|---:|---:|
| bal_acc | **0.9492** | 0.9333 | **0.9469** |
| true_recall | 0.9502 | 0.9612 | 0.9441 |
| false_recall | **0.9482** | 0.9054 | **0.9497** |
| recall gap | 0.20 pp | 5.58 pp | 0.56 pp |
| threshold | 1.234 | 1.289 | 1.220 |
| TRUE p05/p50/p95 | 1.13/1.18/1.23 | 1.18/1.23/1.28 | **1.08/1.15/1.22** |
| FALSE p05/p50/p95 | 1.23/1.49/1.58 | 1.27/1.49/1.57 | 1.22/1.50/1.60 |
| gap_median | 0.313 | 0.265 | **0.349** |
| eff_rank | 145.6 | 206.0 | 135.5 |

The ep0 of vic=0.05 produces **tighter TRUE clusters than h=128** (TRUE p50
1.15 vs 1.18) and a **wider gap** (0.349 vs 0.313). The mechanism is
confirmed: halving VICReg let the contrastive force concentrate TRUE pairs.

---

## What changed vs h=256 vic=0.1

Both variants degrade after ep0 — vic=0.05 just starts from a much better
ep0. The shape of the descent into the TRUE-biased plateau is similar:

| trajectory | vic=0.1 | vic=0.05 |
|---|---:|---:|
| best epoch | 0 | 0 |
| ep0 → ep9 bal_acc drift | 0.9333 → 0.9273 (-0.6 pp) | 0.9469 → 0.9284 (-1.9 pp) |
| ep0 true_R → ep9 true_R | 0.9612 → 0.9801 | 0.9441 → 0.9886 |
| ep0 false_R → ep9 false_R | 0.9054 → 0.8745 | 0.9497 → 0.8682 |
| vic_var trajectory | 0.42 → 0.04 (10×) | 0.55 → 0.07 (8×) |
| vic_cov trajectory | 0.11 → 0.22 (2×) | 0.10 → 0.20 (2×) |

Halving VICReg slowed but did not stop the variance saturation. The cov
term climbs to nearly the same equilibrium. The TRUE-bias drift
(true_R climbing toward 0.99, false_R falling to 0.87) is *identical
in shape* to vic=0.1 — just starting from a much better recall split.

**Implication:** The TRUE-bias is not a VICReg artifact. The wider
encoder has an intrinsic drift toward over-fitting TRUE pairs at the
cost of FALSE separation. VICReg controls *where the drift starts from*
(ep0 quality) but not whether it happens.

---

## Mechanism — ep0 vs ep9 within this run

VICReg trajectory shows two phases:

| | ep0 | ep1 | ep4 | ep9 |
|---|---:|---:|---:|---:|
| vic_var | 0.55 | 0.27 | 0.087 | 0.071 |
| vic_cov | 0.10 | 0.14 | 0.20 | 0.20 |
| TRUE loss | 1.24 | 1.53 | 1.60 | 1.46 |
| TRUE p50 (val) | 1.15 | 1.25 | 1.26 | 1.19 |
| false_recall | 0.95 | 0.88 | 0.87 | 0.87 |

After ep0, var collapses while cov climbs — the encoder finds a
regime where features are spread enough to satisfy the variance hinge
but become correlated. By ep5–6, vic_var settles around 0.07 and the
TRUE loss starts dropping again (1.41–1.36) — but TRUE pairs are now
*overfit* (TRUE recall 0.99) at the expense of FALSE recall (0.87).

The encoder's late-training optimization finds a TRUE-biased
representation and never recovers the balanced ep0 split.

---

## Suggested next experiments

1. **Variant B: `vic_var=0.025, vic_cov=0.025`** — further halve. If ep0
   still climbs and the descent looks the same, VICReg level is
   confirmed not the cause.
2. **Variant C: `vic=0.05` + `true_weight=0.7`** — directly counter the
   TRUE-bias by softening the TRUE pull. Hypothesis: TRUE recall stops
   climbing past 0.95 and FALSE recall stays in the 0.93–0.95 range.
   This is the variant most likely to change the trajectory shape.
3. **Variant D: `vic=0.05` + `epochs=3`** — if ep0 is the answer, just
   don't train past it. Cheapest experiment. Cuts wall clock to ~25 min.
4. **Variant E: `vic=0.05` + `lr=5e-4`** — slower LR may prevent the
   late-epoch slide into TRUE-bias.

Recommended: variant C. The recall-balance data points squarely at
TRUE-bias drift as the late-training failure mechanism; soften the
TRUE pull and re-test.

---

## Headline numbers

- **Best: ep0 bal_acc 0.9469** — strongest h=256 result. Within 0.23 pp
  of h=128's overall best (0.9492). Balanced recall (0.94 / 0.95).
- **TRUE clusters at ep0 are *tighter* than h=128** (TRUE p50 1.15 vs 1.18).
- **Distance gap at ep0 is wider than h=128** (0.349 vs 0.313).
- **But late-training degradation is similar to h=256/vic=0.1** — same
  TRUE-bias drift; ep9 bal_acc 0.9284 (TRUE 0.99, FALSE 0.87).
- **Wall clock 80.5 min** (1.35× h=128, 0.78× h=256/vic=0.1 — variance
  in step rate, not config-driven; both h=256 runs use same compute per step).
- **Stability holds**: 1 clip-event-cluster in ep0 warmup, zero clipping ep1–9.
- **Mechanism confirmed for ep0**, mechanism rejected for ep1–9:
  VICReg drives ep0 quality, *something else* drives the late drift.
