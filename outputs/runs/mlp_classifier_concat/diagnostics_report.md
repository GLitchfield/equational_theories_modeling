# `mlp_classifier_concat` — Diagnostic Report

MLP pair-classifier (`PairMLPConcat`, 2-hidden-layer MLP on
`concat(z_e1, z_e2)`) trained on frozen embeddings from
`skip_h128_seed42_diag` epoch 9 (the current stable JEPA, 0.9840
full-corpus bal_acc). Backs Phase 3 of `mlp_classifier_plan.md`.

**Headline result: holdout bal_acc 0.9996 — +1.55pp over the JEPA
distance baseline on the same 10% holdout (0.9841). The embeddings
contain substantially more information than the JEPA's geometric
distance metric exploits; the contrastive bottleneck is the
decision rule, not the embedding capacity.**

Joint vacuity criterion (Methodology Resolution 4) is met:
1. Overall holdout bal_acc (0.9996) does not regress vs the matched
   JEPA baseline (0.9841) — clears it by +1.55pp.
2. Accuracy on always-wrong-trained ∩ holdout: **91.37%**
   (2,932 / 3,209) vs JEPA's 0.00% (0 / 3,209) on the same subset.

So the MLP captured the TRUE-by-vacuity pattern that the JEPA
distance metric structurally cannot represent.

---

## Run metadata

| Field | Value |
|---|---|
| Run name | `mlp_classifier_concat` |
| Model | `PairMLPConcat` (Linear(1536,512) → ReLU → Drop(0.1) → Linear(512,256) → ReLU → Drop(0.1) → Linear(256,1)) |
| Parameters | 918,529 |
| Embedding source | `outputs/runs/skip_h128_seed42_diag/equation_embeddings.npy` (768-dim, ep9, context encoder, L2-normalized) |
| Train pairs | 19,826,048 (90% of 22M corpus) |
| Holdout pairs | 2,202,894 (10%, `data/splits/level1_holdout_seed42.parquet`, seed 42) |
| Class balance (train) | 7,356,573 TRUE / 12,469,475 FALSE (37/63%) |
| Loss | `BCEWithLogitsLoss(pos_weight=1.6950)` |
| Optimizer | AdamW lr 1e-3, weight_decay 0.0 |
| Scheduler | CosineAnnealingLR over total_steps |
| Batch size | 8,192 |
| Epochs | 10 |
| Wall clock | 95 s (4080) |
| Best epoch | 8 (holdout bal_acc 0.9996) |
| Decision threshold | 0.4675 (calibrated on train scores; sigmoid prob) |

---

## Per-epoch holdout trajectory

| ep | train_loss | train_bal | hold_bal | TRUE_R | FALSE_R | thresh |
|---:|----------:|----------:|----------:|-------:|--------:|-------:|
| 0 | 0.0258 | 0.9982 | 0.9981 | 0.9976 | 0.9986 | 0.5539 |
| 1 | 0.0054 | 0.9989 | 0.9987 | 0.9984 | 0.9991 | 0.4896 |
| 2 | 0.0037 | 0.9992 | 0.9990 | 0.9989 | 0.9992 | 0.4366 |
| 3 | 0.0028 | 0.9994 | 0.9992 | 0.9991 | 0.9993 | 0.4164 |
| 4 | 0.0021 | 0.9996 | 0.9994 | 0.9993 | 0.9995 | 0.4607 |
| 5 | 0.0016 | 0.9997 | 0.9995 | 0.9994 | 0.9995 | 0.4367 |
| 6 | 0.0012 | 0.9998 | 0.9996 | 0.9995 | 0.9997 | 0.4720 |
| 7 | 0.0010 | 0.9999 | 0.9996 | 0.9995 | 0.9997 | 0.4535 |
| **8** | **0.0008** | **0.9999** | **0.9996** | 0.9995 | 0.9997 | **0.4675** |
| 9 | 0.0007 | 0.9999 | 0.9996 | 0.9995 | 0.9997 | 0.4384 |

Already at 0.9981 hold_bal after epoch 0; plateaus at 0.9996 by
epoch 6. Train–holdout gap stays under 0.0003 throughout — no
meaningful overfitting at this data scale (918K params vs
19.8M training pairs).

---

## Comparison: all four models, matched 10% holdout

Per Methodology Resolutions 2 and 6, all four are evaluated on the
same `level1_holdout_seed42` 10% holdout (seed 42); the JEPA
distance threshold and the linear/MLP probability thresholds are
each calibrated on the corresponding train portion.

| Model | Params | Holdout bal_acc | TRUE_R | FALSE_R | n_wrong |
|---|---:|---:|---:|---:|---:|
| Linear probe | 1,537 | 0.9530 | 0.9386 | 0.9675 | 105,140 |
| **JEPA distance (skip_h128_seed42 ep9)** | 270K + 197K | **0.9841** | 0.9792 | 0.9890 | **32,259** |
| **MLP concat** | 918K | **0.9996** | 0.9995 | 0.9997 | **~880** |
| MLP concat+abs+had | 1,705K | 0.9996 | 0.9995 | 0.9997 | ~880 |

Observations:

- **Linear → MLP concat = +4.66pp**, **JEPA distance → MLP concat =
  +1.55pp.** Both are huge gains for a probe on frozen embeddings.
- **MLP concat ≈ MLP concat+abs+had.** The 2-layer MLP learns the
  abs-diff/Hadamard interactions on its own from the concat input;
  exposing them as explicit features doesn't help. Consistent with
  capacity not being the bottleneck on either side.
- **The JEPA already does most of the work** (linear → JEPA = +3.11pp;
  JEPA → MLP = +1.55pp). The remaining headroom for a nonlinear
  pair classifier is real but smaller than the gain from the JEPA's
  learned embedding.

---

## Vacuity analysis: did the MLP learn "TRUE-by-vacuity"?

The failure-overlap analysis identified 32,496 pairs that all 8
trained JEPA runs miss (`always_wrong_trained.parquet`), 99.99% of
them TRUE pairs where E2 is a tautological or weakly-constraining
equation. These are the "TRUE-by-vacuity" cases the JEPA distance
metric structurally cannot represent (the distance encodes "are
these two equations similar" — for trivial / weak E2s, structural
similarity isn't the deciding signal).

**3,209 of those 32,496 pairs landed in the 10% holdout** by chance
(32,496 × 0.10 ≈ 3,250). These are the diagnostic subset — pairs
the JEPA gets 0 / 3,209 correct by definition. The MLP, never
explicitly told these are special, learned them anyway:

| Model | Acc on aw ∩ holdout | n_correct | n_total | Mean predicted prob |
|---|---:|---:|---:|---:|
| Linear probe | 17.11% | 549 | 3,209 | 0.2437 |
| JEPA distance | 0.00% | 0 | 3,209 | — |
| **MLP concat** | **91.37%** | **2,932** | **3,209** | **0.8946** |
| MLP concat+abs+had | 91.15% | 2,925 | 3,209 | 0.8996 |

### Per-spine breakdown (MLP concat, E2 spine)

| E2 spine | n_pairs | n_correct | accuracy | mean MLP prob |
|---|---:|---:|---:|---:|
| family_c (high-var weak axioms) | 1,729 | 1,612 | 0.9323 | 0.9096 |
| left | 655 | 557 | 0.8504 | 0.8353 |
| mixed | 484 | 428 | 0.8843 | 0.8670 |
| trivial (`x = x`, eq 1) | 179 | 179 | **1.0000** | 1.0000 |
| right | 162 | 156 | 0.9630 | 0.9402 |
| no_lhs_var_in_rhs | 0 | 0 | — | — |

**Eq 1 (`x = x`, the trivial reflexive) is 100% solved** by the MLP
on this subset (179/179). The hardest cases are `left`-spine E2s
(85%) — equations like `x ◇ y = z ◇ y` where the implication is
TRUE only when the magma collapses on the unconstrained variable.
But even these are above 85%.

### Why this works mechanistically

The linear probe's 17.11% accuracy is itself informative: even a
LINEAR function on `concat(z_e1, z_e2)` can detect "E2 is the
trivial reflexive" (87% on that subset) — that's a single
near-constant offset in z_e2. But it can't detect the harder
patterns (family_c at 22%, left at 1%, mixed at 0.2%, right at 1%)
because those require nonlinear interactions between z_e1 and
z_e2: "given E1's specific structure, does E2 represent a
generalization that E1 entails?" The 2-layer MLP captures these
interactions through its hidden representations.

Crucially, the JEPA's pooled-rep / projection / L2-norm pipeline
preserves the information for the MLP to recover — the geometric
distance metric is the bottleneck, not the embeddings.

---

## What this means

1. **Embeddings >> distance metric.** The JEPA's geometric
   assumption (implication ≈ proximity on a single unit sphere) is
   the limiting factor. Replacing it with a learned classifier on
   the same embeddings closes 90%+ of the remaining error.

2. **TRUE-by-vacuity IS in the embeddings.** The MLP recovers
   91.37% of the always-wrong-trained subset without ever being
   told these are special. The weakness pattern is encoded — the
   distance metric just can't read it because vacuity is a global
   property of E2, not a relational similarity between E1 and E2.

3. **Concat alone is sufficient.** The 2-layer MLP on
   `concat(z_e1, z_e2)` matches the version with abs-diff and
   Hadamard product as additional inputs. The relational
   interactions are discoverable, not hand-engineerable.

4. **The JEPA isn't "wrong" — it's solving a different problem.**
   The JEPA trains by contrastive pulling/pushing, which gives a
   metric structure that's useful (clusters equivalence classes,
   exposes algebraic relationships). The MLP trains for
   classification, which leverages the embeddings without
   geometric constraints. These are complementary, not
   competing — the MLP's bal_acc improvement does not invalidate
   the JEPA work; it shows what the JEPA learned in latent space
   that the chosen decision rule was missing.

---

## What's next

- **Phase 5: channel ablations.** Train the MLP on skip-channel-only
  (256+256=512 dim) vs proj-channel-only (512+512=1024 dim). The
  failure-overlap analysis showed that skip and proj carry slightly
  different information; this would surface which one carries the
  vacuity signal.
- **Ensemble with the JEPA distance.** Combine JEPA distance and
  MLP probability via a simple rule (e.g., predict TRUE if either
  votes TRUE, with calibrated thresholds). Should approach 100%
  on the holdout.
- **Distill the MLP back into the JEPA.** If the MLP's discovered
  features are in the embeddings, a JEPA loss term that targets
  the always-wrong set directly (e.g., a vacuity head) might lift
  the JEPA's own bal_acc closer to 0.9996 without needing an
  external classifier at inference.
