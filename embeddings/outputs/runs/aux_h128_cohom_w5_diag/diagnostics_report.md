# `aux_h128_cohom_w5_diag` — Diagnostic Report

Same test as `aux_h256_cohom_w5_diag` but at h=128: cohom heads at
aux_weight=5.0 to test whether the cohom anchoring effect (or lack of
it) transfers down to the smaller, more stable h=128 regime.

**Headline result: peak bal_acc 0.9362 at ep0, then optimization
pathology — gradient clipping rate climbs from 23% at ep0 to 100% at
ep4 and stays there for ep4-9.** The avg pre-clip gradient norm grows
across training (0.88 → 2.86), meaning the clip threshold of 1.0 is
permanently active and the optimizer is no longer following the
configured LR schedule — it's running at a fixed gradient norm of 1.0
per step regardless of natural-gradient direction or magnitude.

The result is **not a clean test of the anti-drift anchoring hypothesis
at h=128**. Both possible failure modes (anchoring doesn't work; clipping
breaks optimization) would produce the same outcome of "stuck in regime
A near 0.93." A clean test would require aux_weight ≤ 1.0 to keep
clip-rate sane at this configuration.

That said, the indirect evidence we do have:

- ep0 (when clipping was only 23%, optimization still partly intact)
  bal_acc was 0.9362 — better than `aux_h256_cohom_w5_diag`'s 0.9329
  but worse than `aux_h128_diag`'s 0.9492 baseline.
- FALSE recall plateau (~0.87 across all epochs after ep0) is identical
  to every other h=256 cohom variant. Drift wasn't prevented even at
  ep0 where optimization was still functioning.

So the data hints at "h=128 + cohom + heavy aux weight is no better at
anti-drift anchoring than h=256," but the heavy clipping makes it hard
to assign confidence to that hint.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `aux_h128_cohom_w5_diag` |
| Wandb id | `vkxfuo01` |
| Config | `configs/experiments/aux_h128_cohom_w5_diag.yaml` |
| Wall clock | 112.5 min (6750.2 s) |
| Outcome | 10 epochs completed, exit code 0 (no NaN, no crash) |
| Encoder params | ~262K (h=128) |
| Aux heads params | 18,761 (18 heads at pool_dim=256) |
| Mode | `--no-holdout` |

---

## Per-epoch monitor

| ep | bal_acc | true_R | false_R | eff_rk | gap | clip_frac | grad_pre |
|---:|--------:|-------:|--------:|-------:|----:|----------:|---------:|
| **0** | **0.9362** | 0.9802 | 0.8922 | 81 | 0.375 | 0.234 | 0.879 |
| 1 | 0.9288 | 0.9816 | 0.8759 | 100 | 0.372 | 0.883 | 1.298 |
| 2 | 0.9277 | 0.9865 | 0.8689 | 86 | 0.396 | 0.993 | 1.673 |
| 3 | 0.9273 | 0.9842 | 0.8703 | 92 | 0.392 | 0.999 | 1.900 |
| 4 | 0.9288 | 0.9886 | 0.8691 | 106 | 0.397 | 1.000 | 2.128 |
| 5 | 0.9290 | 0.9891 | 0.8689 | 107 | 0.389 | 1.000 | 2.372 |
| 6 | 0.9290 | 0.9904 | 0.8676 | 112 | 0.401 | 1.000 | 2.628 |
| 7 | 0.9288 | 0.9900 | 0.8675 | 113 | 0.411 | 1.000 | 2.853 |
| 8 | 0.9287 | 0.9908 | 0.8667 | 103 | 0.435 | 1.000 | 2.940 |
| 9 | 0.9285 | 0.9905 | 0.8665 | 94 | 0.466 | 1.000 | 2.860 |

Two distinct phases: **ep0** (clip 23%, bal_acc 0.9362) where the
optimizer is partly working, and **ep1-9** (clip 88-100%, bal_acc
0.927-0.929) where it is not.

### Gradient clip pathology

Pre-clip gradient norm grew across training, never converged:

```
ep0: 0.88     ep1: 1.30     ep2: 1.67     ep3: 1.90     ep4: 2.13
ep5: 2.37     ep6: 2.63     ep7: 2.85     ep8: 2.94     ep9: 2.86
```

With clip_norm=1.0, this means every step's effective gradient was
rescaled to norm 1.0 with the configured LR multiplied through, so the
*effective* step size was determined by clipping, not the LR schedule.
The slow eff_rank dynamics (81 → 113 → 94) and gap widening (0.375 →
0.466) suggest some directional preference was preserved through the
rescaling, but it never stabilized into a regime-B basin.

### Side-by-side vs analogous runs

| Run | Peak bal_acc | Peak ep | Sustained clip rate |
|-----|-------------:|--------:|--------------------:|
| `aux_h128_diag` (h=128, structural-only, w=0.1) | **0.9582** | 3 | <1% |
| `aux_h256_cohom_w5_diag` (h=256, cohom, w=5.0) | 0.9329 | 0 | 0-3% |
| `aux_h128_cohom_w5_diag` (h=128, cohom, w=5.0) | 0.9362 | 0 | 23% → 100% |

FALSE recall trajectory comparison:

| ep | h128_cohom_w5 | h256_cohom_w5 | h128_diag (no cohom) |
|----|---:|---:|---:|
| 0 | 0.892 | 0.883 | 0.934 |
| 1 | 0.876 | 0.865 | 0.911 |
| 2 | 0.869 | 0.867 | 0.902 |
| 3 | 0.870 | 0.871 | 0.879 |
| 5 | 0.869 | 0.871 | 0.879 |

h128_cohom_w5 starts ~0.04pp better in FALSE recall than h256_cohom_w5
at ep0 but converges to the same ~0.87 plateau. The structural-only
h128_diag baseline maintains FALSE recall ~0.88-0.93 throughout — this
is the only configuration that doesn't drop into the deep ~0.87 basin.

---

## What this run tells us

1. **aux_weight=5.0 is too aggressive for h=128.** Sustained 100% clip
   rate from ep4 onward means the test was contaminated by optimizer
   pathology. The result cannot be cleanly attributed to "anchoring
   failed" vs "optimization failed." A re-test with aux_weight in the
   1.0-2.0 range would give a cleaner data point.

2. **Indirect evidence still points the same way.** Even at ep0 where
   optimization was partly functional (clip 23%), FALSE recall was
   already 0.892 (started drifting) and bal_acc 0.9362 (worse than
   h=128 baseline 0.9492 ep0). The drift signature is the same direction
   as every other h=256 cohom variant.

3. **The smaller capacity didn't help anchoring.** The hypothesis was
   that h=128's smaller pool_dim might force the encoder to actually
   compromise contrastive performance to satisfy aux. The data we have
   (despite optimizer issues) shows the same TRUE-bias direction with
   no anti-drift effect. The encoder still finds enough capacity at
   h=128 to satisfy aux + drift simultaneously.

4. **The aux gradient is much larger relative to encoder size at h=128.**
   This is a mechanical consequence: the cohom heads have similar param
   counts at both widths (~18K vs ~37K), but the encoder is ~3× smaller
   at h=128 (262K vs 800K). The same aux loss with the same weight
   produces a relatively larger gradient signal at h=128, which is why
   clipping becomes pathological here but not at h=256. This is
   important for any future cohom + h=128 experiment design.

---

## Conclusion

This run mostly tests "what happens when 100% of gradients are clipped"
rather than "do cohom aux heads at meaningful weight prevent drift at
h=128." The indirect evidence is consistent with the h=256 cohom_w5
finding (no anti-drift effect, regime A throughout), but a cleaner
test would use aux_weight ≤ 1.0 to keep clipping under control at this
encoder scale.

For the project as a whole: the cohom-aux-head paradigm is now
unsupported at both widths and at multiple aux_weights (0.1 in
`aux_h256_cohom_diag`, 5.0 here and in `aux_h256_cohom_w5_diag`). The
remaining unexplored cohom configurations are (a) per-pair cohom
features fed to predictor/contrastive directly, (b) input concatenation,
or (c) dropping cohom for the model and using it for analysis only.
This run reinforces the position that the per-equation-classification
aux-head shape is wrong for cohom information, regardless of width or
weight.

The structural-only `aux_h128_diag` at 0.9582 remains the project's
actually-stable best.
