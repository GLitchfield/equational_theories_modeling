# `skip_h128_seed42_20ep_diag` — Diagnostic Report

Extension of `skip_h128_seed42_diag` (peak 0.9840 ep9 with monotonic
climb suggesting more headroom) to 20 epochs. Single-knob change vs
`skip_h128_seed42_diag.yaml`: `training.epochs` 10 → 20.

**Headline result: peak bal_acc 0.9774 at ep15 — 0.66pp WORSE than
the 10-epoch run's 0.9840 ep9.** The 20-ep extension underperformed
the original at every directly-comparable epoch (ep0-9: −0.66pp to
−1.43pp). The "monotonic climb suggests headroom" reading was
misleading — that climb was specifically the cosine-LR-decay
trajectory of a 10-epoch budget, not extra information being learned.

The mechanism is the cosine LR scheduler. With `epochs=20`, LR is
normalized over twice as many steps, so absolute LR stays higher
longer. The model trains in a more exploratory regime for longer
and settles into a different basin once LR fully decays. The
aggressive 10-epoch LR decay (steeper schedule) found a deeper
local minimum than the slow 20-epoch decay.

**Project best remains `skip_h128_seed42_diag` at 0.9840.**

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `skip_h128_seed42_20ep_diag` |
| Wandb id | `tact8wh0` |
| Config | `configs/experiments/skip_h128_seed42_20ep_diag.yaml` |
| Wall clock | 214.8 min (≈ 3.6 hours) |
| Outcome | 20 epochs completed, exit code 0 |
| Encoder params | 270,592 (same arch as 10-ep) |
| Predictor params | 197,504 |
| Aux head params | 7,196 |

---

## Per-epoch trajectory

| ep | bal_acc | TRUE_R | FALSE_R |
|---:|--------:|-------:|--------:|
| 0 | 0.9683 | 0.9586 | 0.9780 |
| 1 | 0.9627 | 0.9695 | 0.9559 |
| 2 | 0.9664 | 0.9677 | 0.9652 |
| 3 | 0.9658 | 0.9678 | 0.9639 |
| 4 | 0.9690 | 0.9750 | 0.9631 |
| 5 | 0.9661 | 0.9694 | 0.9628 |
| 6 | 0.9705 | 0.9741 | 0.9669 |
| 7 | 0.9732 | 0.9691 | 0.9773 |
| 8 | 0.9719 | 0.9667 | 0.9771 |
| 9 | 0.9707 | 0.9728 | 0.9686 |
| 10 | 0.9735 | 0.9759 | 0.9711 |
| 11 | 0.9757 | 0.9770 | 0.9744 |
| 12 | 0.9751 | 0.9749 | 0.9752 |
| 13 | 0.9743 | 0.9712 | 0.9774 |
| 14 | 0.9758 | 0.9725 | 0.9791 |
| **15** | **0.9774** | 0.9742 | 0.9806 |
| 16 | 0.9758 | 0.9699 | 0.9818 |
| 17 | 0.9757 | 0.9667 | 0.9847 |
| 18 | 0.9756 | 0.9658 | 0.9854 |
| 19 | 0.9758 | 0.9650 | 0.9866 |

Best at ep15 (0.9774). bal_acc plateaus at ~0.975-0.977 from ep11
onward. Notably, FALSE recall keeps rising (0.969 → 0.987) through
the run while TRUE recall drops (0.970 → 0.965), suggesting the
late-training low-LR regime is biasing toward FALSE-confidence —
similar shape to what we saw in some no-skip baselines but milder
and well-bounded.

### Side-by-side: 20-ep extension vs original 10-ep

| ep | 20ep | 10ep | Δ |
|---:|-----:|-----:|--:|
| 0 | 0.9683 | 0.9693 | −0.001 |
| 1 | 0.9627 | 0.9721 | −0.009 |
| 2 | 0.9664 | 0.9754 | −0.009 |
| 3 | 0.9658 | 0.9757 | −0.010 |
| 4 | 0.9690 | 0.9799 | −0.011 |
| 5 | 0.9661 | 0.9804 | −0.014 |
| 6 | 0.9705 | 0.9818 | −0.011 |
| 7 | 0.9732 | 0.9838 | −0.011 |
| 8 | 0.9719 | 0.9839 | −0.012 |
| 9 | 0.9707 | 0.9840 | −0.013 |

The 20-ep run is **uniformly behind** the 10-ep run by 0.6-1.4pp at
every epoch ep0-9. This isn't seed noise (same seed); it's the
slower LR schedule keeping the model in a higher-LR exploratory
regime longer. The extra 10 epochs only get the model to 0.9774,
still 0.66pp below where the 10-ep run finished.

---

## Mechanism

The cosine-warmup-then-decay LR schedule is parameterized by total
training steps (= `epochs × steps_per_epoch`). With epochs=20:
- 1000 step warmup is the same in absolute terms
- Cosine decay phase lasts 7,608 steps (vs 3,304 with epochs=10)
- LR at any given step is HIGHER under epochs=20 in the early-to-mid
  training window

So when comparing ep5 of the two runs, both have done the same
~2,150 update steps with the same data, but the 20-ep run has been
seeing higher LR throughout. Higher LR → larger steps → noisier
updates → less convergence at any given step count.

By ep15 of the 20-ep run, LR has reached the same low values that
the 10-ep run reached at ep8-9. But the 20-ep run is now optimizing
from a different starting point (the parameters at ep15 ≠ params at
the equivalent step in the 10-ep run, even though the data and
seed are the same — different LR trajectories yield different
parameter trajectories). It settles into a different (worse) local
minimum.

**Counterintuitive but well-documented finding in the literature:**
when LR schedule is normalized over total epochs, "more training"
isn't strictly better; the LR schedule co-determines which optimum
you find. Aggressive LR decay (10-epoch schedule) found a deeper
optimum than gentle LR decay (20-epoch schedule) for this
architecture.

---

## Limitations — the schedule hypothesis was not directly tested

The cosine-LR-scaling story above is a plausible mechanism and the
trajectory shape (uniformly behind by 0.6-1.4pp at every comparable
epoch ep0-9; FALSE recall continues rising past ep10 while bal_acc
plateaus) is consistent with it. But the experiment itself doesn't
isolate the schedule from other things that change with more epochs
(more EMA target updates, different cumulative VICReg pressure, more
opportunities for the aux heads to saturate, etc.). The mechanism
is a hypothesis, not a finding.

A clean isolation experiment would run 10 epochs while the cosine
scheduler plans for 20 — i.e., truncate the 20-ep recipe at ep10.
If the model lands at the 20-ep ep10 number (≈0.9735), schedule is
confirmed as the explanation. If it lands at ~0.98, schedule isn't
the explanation and one of the other "more-epochs" effects is
responsible.

Not run yet — the project best remains `skip_h128_seed42_diag`
either way, so isolating the mechanism doesn't change the
recommendation to use the 10-epoch budget. Worth revisiting if a
future training run needs to motivate a different schedule shape
(constant LR + late decay, multi-cycle cosine restart, etc.) or if
the dual-encoder / ensemble work surfaces it as relevant.

---

## Conclusion

The "extra headroom past ep9" hypothesis is refuted. The 10-ep run
wasn't artificially stopped before convergence; it had found the
best minimum the cosine schedule could deliver under this recipe
budget, and extending the budget actively moved the optimum to a
worse basin.

**Implications:**

1. **Project best stays at `skip_h128_seed42_diag` (0.9840 ep9).**
   The 20-ep variant doesn't replace it.
2. **Don't conclude "model has headroom" from a monotonic climb
   under a fixed LR schedule.** The climb is the schedule, not
   leftover capacity. Need a fundamentally different intervention
   (architecture, recipe, schedule shape) to push past 0.9840.
3. **Two interventions that might still help:**
    - Different LR schedule entirely (e.g., constant LR + late-
      training decay, or multi-cycle cosine restart). Tests whether
      a different optimization trajectory finds a better optimum.
    - The dual-encoder hypothesis (still untested on the skip
      baseline). Adds information rather than just changing the
      optimization.
4. **For multi-seed averaging,** stick with the 10-epoch budget —
   it's faster and finds better optima at this seed at least.
