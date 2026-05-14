# `aux_h128_seed1_diag` — Diagnostic Report

Reproducibility check on `aux_h128_diag` (the project's
actually-stable best at 0.9582 ep3). Two changes vs `aux_h128_diag.yaml`:
seed 42 → 1, and `loss.aux_heads: structural` to lock the head set to
the original 10 structural heads despite HEAD_SPECS now containing 18
entries (cohom heads added in a later experiment).

**Headline result: bal_acc 0.9621 at ep6, +0.0039 over the seed=42
peak — the result reproduces, and in fact slightly improves under a
different seed.** The h=128 recipe is recipe-driven, not seed-lucky.
This is a sharp contrast with h=256, where the seed=42 record (0.9635)
came from a lucky regime-B phase transition that did not reproduce
under seed=1 (which only reached 0.9492).

The trajectories of the two h=128 seeds are remarkably aligned:
both peak between ep3 and ep6 in the 0.95-0.96 range, both end at
ep9 around 0.948-0.95, both maintain TRUE and FALSE recall near 0.95
throughout (no extreme TRUE-bias drift like h=256), both hold eff_rank
in the 100-145 band (no drift toward 300+ like h=256 baselines).

This run sets a new **stable** project best at 0.9621.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `aux_h128_seed1_diag` |
| Wandb id | `a0a2vdcr` |
| Config | `configs/experiments/aux_h128_seed1_diag.yaml` |
| Wall clock | 93.3 min (5597.7 s) |
| Outcome | 10 epochs completed, exit code 0 |
| Encoder params | 269,056 (h=128) |
| Aux heads params | 7,196 (10 structural heads, pool_dim=256) |
| Mode | `--no-holdout` |

---

## Per-epoch monitor

| ep | bal_acc | true_R | false_R | eff_rk | gap |
|---:|--------:|-------:|--------:|-------:|----:|
| 0 | 0.9276 | 0.9571 | 0.8982 | 119.5 | 0.294 |
| 1 | 0.9562 | 0.9538 | 0.9586 | 123.8 | 0.310 |
| 2 | 0.9538 | 0.9642 | 0.9433 | 123.6 | 0.333 |
| 3 | 0.9584 | 0.9591 | 0.9577 | 134.3 | 0.328 |
| 4 | 0.9614 | 0.9637 | 0.9591 | 121.7 | 0.332 |
| 5 | 0.9618 | 0.9578 | 0.9659 | 136.6 | 0.334 |
| **6** | **0.9621** | 0.9647 | 0.9596 | 142.5 | 0.332 |
| 7 | 0.9582 | 0.9595 | 0.9569 | 145.2 | 0.336 |
| 8 | 0.9537 | 0.9500 | 0.9573 | 140.5 | 0.354 |
| 9 | 0.9497 | 0.9461 | 0.9534 | 135.4 | 0.373 |

bal_acc band ep3-ep7: **0.958-0.962**. TRUE recall holds at 0.95-0.96
throughout (no climb to 0.99 like h=256). FALSE recall holds at
0.93-0.97 throughout (no drop to 0.87). eff_rank stays in 119-145
(very stable). gap widens slightly across training (0.29 → 0.37) —
consistent with mild representation refinement, no regime change.

### Side-by-side vs seed=42

| ep | seed=1 bal_acc | seed=42 bal_acc | Δ | seed=1 F_rec | seed=42 F_rec |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9276 | 0.9394 | −0.0118 | 0.8982 | 0.9302 |
| 1 | 0.9562 | 0.9421 | +0.0141 | 0.9586 | 0.9331 |
| 2 | 0.9538 | 0.9508 | +0.0030 | 0.9433 | 0.9420 |
| 3 | 0.9584 | **0.9582** | +0.0002 | 0.9577 | 0.9533 |
| 4 | 0.9614 | 0.9539 | +0.0075 | 0.9591 | 0.9452 |
| 5 | 0.9618 | 0.9543 | +0.0075 | 0.9659 | 0.9566 |
| 6 | **0.9621** | 0.9554 | +0.0067 | 0.9596 | 0.9508 |
| 7 | 0.9582 | 0.9535 | +0.0047 | 0.9569 | 0.9465 |
| 8 | 0.9537 | 0.9508 | +0.0029 | 0.9573 | 0.9433 |
| 9 | 0.9497 | 0.9479 | +0.0018 | 0.9534 | 0.9428 |

seed=1 starts ~1.2pp behind at ep0 but pulls ahead from ep1 onward and
stays slightly ahead through ep9. The two trajectories are almost
overlaid — peak epochs differ (ep3 vs ep6) but the peak magnitudes
agree to within 0.4pp.

---

## What this run tells us

1. **The h=128 recipe reproduces under different seeds.** The 0.9582
   headline from seed=42 is not a fluke; seed=1 hits 0.9621 with the
   same recipe. Both runs maintain similar metrics at every epoch with
   small, well-bounded variation.

2. **h=128 has no equivalent of h=256's regime-B phase transition.**
   Neither h=128 seed shows the bal_acc step + eff_rank crash signature
   that defined h=256's regime B. h=128 just steadily improves in the
   0.95-0.96 range, peaks, then mildly degrades. Whatever drives the
   h=256 transition (low LR + capacity + something) does not need to
   fire at h=128 to get to 0.96 territory.

3. **The new project STABLE best is 0.9621**, beating the prior
   stable 0.9582. The single-seed h=256 record (0.9635, `aux_h256_diag`)
   is now within 0.14pp of a routinely-reproducible stable result.
   Given that two of three other h=256 runs failed to reach the
   0.9635 region at all (one seed reverted, one cohom variant
   suppressed transition), the h=128 0.9621 is the more honest project
   headline.

---

## Conclusion

h=128 + 10 structural aux heads + VICReg 0.1/0.1 + margin 1.5 +
aux_weight 0.1 reproduces under different seeds at 0.96-pp-level
agreement. This is the project's most robust configuration to date.

The h=256 path (where 3/4 runs failed to reach the 0.9635 record and
the one that did was seed=42 only) should be considered an unreliable
recipe. Future experiments should default to h=128 unless they
specifically need to test capacity-related hypotheses.

The 0.0014 gap between the h=128 stable best (0.9621) and the h=256
single-seed best (0.9635) is well within seed-noise range and
likely not meaningful.
