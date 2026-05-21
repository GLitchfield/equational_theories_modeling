# Probing diagnostics — `aux_h128_seed1_diag`

What the project's best stable model (h=128, structural-only aux,
peak bal_acc 0.9621 at ep6) actually encodes in its embeddings.
Linear probes on frozen embeddings; if a linear classifier can
recover a feature, the embedding encodes it linearly.

The probing plan (`probing_experiments_plan.md`) covered six
experiments. All six ran successfully (~12 min total). This report
narrates the findings; raw numbers are in `summary_table.md` and the
JSON files at `outputs/probing/results/exp{1..6}*.json`.

---

## TL;DR

1. **Cohomology features are already in the embeddings, learned for free.**
   The encoder was trained on contrastive + structural-aux supervision
   only — it never saw the cohomology profiles during training. But
   per-prime cohom summaries (count_p, mean_h2_p) probe at
   **80-88% accuracy on trained embeddings vs 55-64% on random-init**,
   gain of +25-32pp across all 8 cohom targets.
2. **The contrastive task forced the encoder to learn cohom structure.**
   The 4 failed cohom-as-aux-head experiments (`aux_h256_cohom_*`)
   asked the encoder to learn what it was already learning anyway. The
   negative result on those experiments was misinterpreted as "cohom
   isn't useful"; the right interpretation was "cohom is already free."
3. **Information is acquired in epoch 0 then plateaus.** Cohom-summary
   probe accuracy at ep0=0.840, ep3=0.846, ep6=0.847. The contrastive
   loss compresses structural information into the embedding within
   the first epoch; subsequent training only refines pair-level geometry.
4. **A linear classifier on `concat(z_e1, z_e2)` hits 95% balanced
   accuracy on ground-truth TRUE/FALSE prediction** — within 1pp of
   the JEPA's 96.21% headline. So the trained distance metric
   isn't the only way to use these embeddings; a linear pair classifier
   recovers nearly the same performance.
5. **The output projection + L2-norm destroys some cohom info.**
   `count_p2` drops 0.98 → 0.88 between the 256-dim pooled rep and the
   512-dim L2-normalized output. Other cohom summaries lose 1-4pp. If
   we wanted maximally-cohom-aware embeddings, we'd skip the L2-norm.
6. **Equivalence-class NN: 52% top-1, 67% top-10** (vs 35% / 56% random,
   11% / 39% baseline). Trained embeddings cluster equivalent equations,
   but the metric is hard to interpret because 81% of classes are
   singletons (no possible NN match) and the largest class has 1,496
   members (~32% of the corpus).

---

## Methodology recap

Per the resolved methodology (top of `probing_experiments_plan.md`):

- **Embeddings**: 8 sets — pooled (256-dim) and L2-norm output (512-dim)
  for each of `aux_h128_seed1_diag` ep0/ep3/ep6 plus a `random_init`
  baseline (fresh JEPA, no checkpoint loaded, seed 0).
- **Probes**: 5-fold cross-validation. `LogisticRegression(C=1.0,
  max_iter=500)` for classification; `Ridge(alpha=1.0)` for regression.
  Parallelized across folds with `n_jobs=-1` (~12× speedup vs sequential).
- **Metrics**: accuracy + balanced accuracy for classification; R² for
  regression. Majority-class baseline reported alongside every probe.
- **Bins for cohom summary**: reused `COHOM_COUNT_BINS` and
  `COHOM_MEAN_BINS` from `scripts/build_equation_invariants.py` so
  results are comparable to the cohom aux-head experiments.
- **Pair sampling for Exp 5**: 100K pairs stratified 50K TRUE / 50K
  FALSE from `data/processed/implication_pairs.parquet`.
- **Equivalence classes for Exp 6**: derived as SCCs of the implication
  graph from the same pair table. Skipped k-fold (broken at 1,415
  classes) in favor of nearest-neighbor accuracy with cosine similarity.

A handful of probes had `n_failed_folds=1` (warning logged, fold's
score replaced by NaN, mean computed over remaining 4 folds). This
hits classes with very few members (e.g. spine_type class 5 has 1
sample of 4,694) and doesn't materially affect the headline numbers.

---

## Experiment 1 — Structural sanity check

The 10 structural targets the encoder WAS trained on as aux heads.
Expected: probe at ~100% on trained embeddings; if not, embedding
extraction is broken.

| target | majority | random_init | ep6 trained | gain over random |
|---|---:|---:|---:|---:|
| spine_type (6-class) | 0.331 | 0.810 | 1.000 | +0.190 |
| lhs_structure (binary) | 0.669 | 0.999 | 1.000 | +0.001 |
| num_variables (6-class) | 0.445 | 0.982 | 0.996 | +0.014 |
| magma_bit_0 (binary) | 0.669 | 1.000 | 0.999 | -0.001 |
| magma_bit_1 (binary) | 0.741 | 0.770 | 0.996 | +0.226 |
| magma_bit_2 (binary) | 0.741 | 0.784 | 0.999 | +0.215 |
| magma_bit_3 (binary) | 0.859 | 0.860 | 0.997 | +0.137 |
| magma_bit_4 (binary) | 0.669 | 1.000 | 0.999 | -0.001 |
| magma_bit_6 (binary) | 0.926 | 0.926 | 0.991 | +0.065 |
| magma_bit_7 (binary) | 0.926 | 0.926 | 0.996 | +0.070 |

**Interpretations:**

- All trained probes ≥ 99%. Embedding extraction works.
- Several targets (lhs_structure, magma_bit_0/4) are at 99-100% even on
  random-init embeddings — the input node features encode them so
  trivially that any linear projection preserves them.
- Targets with random-init at 50-78% and trained at 99% (spine_type,
  magma_bit_1/2/3, the 6-magma bits) show real learning: the encoder
  actively reorganizes these features during training.

**Pooled vs output**: structural targets are essentially identical
between the 256-dim pooled rep and the 512-dim L2-norm output (max
drop 0.009pp on magma_bit_6). The aux heads hit 100% during training
because they probed the pooled rep; the output rep loses essentially
no structural separability through the projection.

---

## Experiment 2 — Per-prime cohom summaries

The headline experiment. 8 binned classification targets covering
`count_p` and `mean_h2_p` for p ∈ {2, 3, 5, 7}. The encoder was
NEVER trained on these.

| target | majority | random_init | ep0 | ep3 | ep6 |
|---|---:|---:|---:|---:|---:|
| count_p2 (5-class) | 0.405 | 0.636 | 0.867 | 0.878 | 0.880 |
| mean_h2_p2 (5-class) | 0.333 | 0.553 | 0.857 | 0.866 | 0.868 |
| count_p3 (5-class) | 0.340 | 0.574 | 0.836 | 0.841 | 0.843 |
| mean_h2_p3 (6-class) | 0.340 | 0.566 | 0.832 | 0.837 | 0.838 |
| count_p5 (6-class) | 0.328 | 0.553 | 0.807 | 0.814 | 0.810 |
| mean_h2_p5 (6-class) | 0.328 | 0.606 | 0.831 | 0.836 | 0.837 |
| count_p7 (6-class) | 0.334 | 0.568 | 0.846 | 0.850 | 0.851 |
| mean_h2_p7 (6-class) | 0.334 | 0.622 | 0.841 | 0.848 | 0.849 |

**Headline**: every cohom summary probes at 81-88% on trained
embeddings, vs 55-64% on random init, vs majority at 33-41%. The
gap from random-init to trained is +25-32pp uniformly. **Cohomology
features are linearly recoverable from the encoder's embeddings**
even though the encoder was never trained to predict them.

**This reframes the cohom-aux-head experiments.** The four
`aux_h256_cohom_*` runs added cohom heads as auxiliary supervision
on the assumption that cohom needed to be explicitly anchored to
prevent the contrastive loss from suppressing it. The probes show
the contrastive loss *isn't* suppressing cohom — it's actively
encoding it, just not because of any aux pressure. So:

- The cohom aux heads were redundant supervision (they got >99% val
  accuracy from ep1, consistent with this finding — the encoder
  could already predict cohom).
- The negative bal_acc result on the cohom aux experiments was about
  the auxiliary loss itself interfering with contrastive geometry, not
  about cohom-vs-no-cohom feature representation.
- Adding cohom on the input-feature path (not the aux-head path) is
  unlikely to help further: the encoder already represents cohom info
  to ~85% linear-probe accuracy from contrastive supervision alone.

**Multi-epoch evolution** is striking:
- random_init mean (8 targets): 0.585
- ep0 mean: **0.840** (+0.255 in 1 epoch of training)
- ep3 mean: 0.846 (+0.006)
- ep6 mean: 0.847 (+0.001)

The cohom representation is essentially complete by ep0. Subsequent
epochs refine pair-level contrastive geometry (which is why bal_acc
goes from 0.93 at ep0 to 0.96 at ep6) without changing per-equation
linear separability of cohom features.

**Pooled vs output** — cohom IS partly destroyed by the projection:

| target | pooled (256-d) | output (512-d, L2) | drop |
|---|---:|---:|---:|
| count_p2 | 0.980 | 0.880 | -0.099 |
| count_p3 | 0.878 | 0.843 | -0.035 |
| count_p5 | 0.849 | 0.810 | -0.039 |
| mean_h2_p2 | 0.885 | 0.868 | -0.017 |

count_p2 in particular is much more accessible at the pooled stage
(98% vs 88%). This means the projection head + L2-norm sacrifices
some cohom information to satisfy the contrastive geometry's
unit-sphere constraints.

---

## Experiment 3 — Cell-level cohom (87 individual triples)

For each (a, b, p) triple, two probes: binary "satisfies this cell"
and regression "dim H² at this cell" (only on satisfying equations,
n ≥ 50). 56 of 87 cells were probable (others had positive rates
outside [1%, 99%] or fewer than 50 satisfying equations).

| embedding | mean binary bal_acc | mean H² R² |
|---|---:|---:|
| random_init | 0.552 | 0.221 |
| ep0 | 0.749 | 0.475 |
| ep6 | 0.753 | 0.479 |

**Trained embeddings probe cell-level cohomology at 75% mean balanced
accuracy.** The signal is real but weaker per-cell than per-summary —
which makes sense because individual cells are noisier and many
satisfying-vs-not classifications correlate with broader structural
features (spine_type, etc.).

The TOP 5 binary cells are all `(0, 0, p)` — these correspond to
the equation satisfying the constant-zero magma over F_p, which is
nearly synonymous with `is_trivial` (probes at 100% on every embedding).

The BOTTOM 5 cells are sparse p=7 cells with 50-160 satisfying
equations and bal_acc at the 0.500 floor — too few examples and
not enough discriminative signal.

The best dim-H² regression cell is `(0, 0, p7)` at R² = 0.975 on
trained ep6 — again, this is the trivial-magma case where the H²
value is well-predicted by structural features.

---

## Experiment 4 — Non-anchored invariants

Features the model wasn't trained on (neither aux targets nor input
features). What did contrastive discover on its own?

**Classification (binary):**

| target | maj_bal | random_init bal | ep6 bal | gain |
|---|---:|---:|---:|---:|
| is_trivial | 0.500 | 1.000 | 1.000 | 0.000 |
| is_singleton_forcing | 0.500 | 1.000 | 0.998 | -0.002 |
| is_balanced (0.7% positive) | 0.500 | 0.500 | 0.500 | 0.000 |
| leftmost_match | 0.500 | 0.628 | 0.998 | +0.370 |
| rightmost_match | 0.500 | 0.581 | 0.992 | +0.411 |

- `is_trivial` and `is_singleton_forcing` probe at ~100% on RANDOM-INIT
  embeddings. These are essentially derivable from the input node
  features (parser-level structure) — no learning required.
- `is_balanced` is unreachable: 0.7% positive rate, all probes at the
  0.500 floor. Even balanced metrics can't help when the positive
  class has only 33 examples in the corpus.
- `leftmost_match` / `rightmost_match` show real training-driven
  learning: random-init at 58-63%, trained at 99%. The contrastive
  loss discovered these features (whether the leftmost/rightmost leaf
  variable on LHS matches RHS) without being told to.

**Regression (R²):**

| target | random_init | ep6 |
|---|---:|---:|
| op_count_asymmetry | 0.808 | 0.960 |
| spine_depth | 0.271 | 0.849 |

`spine_depth` is the most striking gain in the entire experiment:
random-init at 0.27, trained at 0.85 (+0.58). The contrastive loss
encoded spine depth essentially-correctly.

---

## Experiment 5 — Pairwise cohom

100K pairs stratified 50K TRUE / 50K FALSE. Three pair representations
tested: `concat(z_e1, z_e2)` (1024-dim), `z_e1 ⊙ z_e2` (Hadamard,
512-dim), `|z_e1 - z_e2|` (abs-diff, 512-dim).

Subset sizes:
- Both equations have ≥1 non-null cohom cell ("clean"): 39,443
- One or both all-null cohom ("confounded"): 60,557 (61% of pairs!)
- With shared triples (for compatibility regression): 22,029

The confounded subset is the majority — many pairs involve at least
one of the 1,507 all-null equations. This validates the methodology
choice to report subsets separately.

**Ground-truth TRUE/FALSE probe** (linear classifier on pair embeddings):

| pair_rep | random_init | ep6 |
|---|---:|---:|
| concat | 0.829 | **0.951** |
| hadamard | 0.659 | 0.611 |
| absdiff | 0.687 | 0.699 |

**Big finding**: `concat` of trained embeddings + linear classifier
hits **0.951 balanced accuracy** on ground-truth implication —
within 1.1pp of the JEPA's 0.9621 trained-distance prediction. The
contrastive distance metric isn't the only way to use these
embeddings; a linear pair classifier recovers nearly the same
performance.

But Hadamard and absdiff are MUCH worse on trained embeddings (61-70%).
Both are symmetric pair functions: `hadamard(z1, z2) = hadamard(z2, z1)`,
which can't represent the directional implication `z_1 → z_2 ≠ z_2 → z_1`.
The contrastive distance the JEPA was trained on IS asymmetric (it
goes through the predictor MLP), so a symmetric pair function loses
the direction information. concat preserves direction.

Interestingly, random_init concat already hits 0.829 — random
projections of the input features (which encode equation structure)
are enough to predict implication at 83% with a linear classifier
on the concatenated representation. The trained gain of +12pp is
real but smaller than I would have expected.

**has_obstruction (clean subset only)**:

| pair_rep | random_init bal | ep6 bal |
|---|---:|---:|
| concat | 0.546 | 0.692 |
| hadamard | 0.500 | 0.500 |
| absdiff | 0.501 | 0.524 |

Concat trained at 0.69 balanced accuracy on has_obstruction — real
signal but well below the 0.95 ground-truth result. The cohomological
obstruction relationship between equation pairs is partly recoverable
from the embeddings but not as cleanly as the implication relationship
itself. This makes sense: the encoder optimized for predicting
implication, not for cohom-similarity directly.

**Compatibility regression** (R² for fraction of shared cells with
matching dim H², dropping pairs with no shared triples — 77,971 dropped):

| pair_rep | random_init R² | ep6 R² |
|---|---:|---:|
| concat | 0.123 | 0.231 |
| hadamard | 0.051 | 0.185 |
| absdiff | 0.101 | 0.330 |

Modest but consistent gains. Best is `absdiff` at R² = 0.330 — the
absolute difference of embeddings carries some cohomological-similarity
information, but the signal is weak (R² < 0.4 even at best).

---

## Experiment 6 — Equivalence class via nearest-neighbor

1,415 equivalence classes derived from the implication graph. Class
sizes are extremely skewed: **1,145 singletons** (81%), median = 1,
p95 = 4, max = **1,496 members** (~32% of corpus).

| embedding | top-1 (base) | top-5 (base) | top-10 (base) |
|---|---:|---:|---:|
| random_init | 0.348 (0.112) | 0.527 (0.319) | 0.563 (0.392) |
| ep0 | 0.526 (0.112) | 0.636 (0.319) | 0.667 (0.392) |
| ep3 | 0.530 (0.112) | 0.642 (0.319) | 0.676 (0.392) |
| ep6 | 0.522 (0.112) | 0.644 (0.319) | 0.673 (0.392) |

Trained ep6 hits 0.522 top-1 vs 0.112 baseline (4.7× lift) and
0.673 top-10 vs 0.392 baseline (1.7× lift). Real signal — embedding
geometry partially clusters equivalent equations.

**Caveats** that limit interpretation:
- 1,145 singletons (81% of classes) are unreachable: their nearest
  neighbor by definition cannot share their class. They drag the
  metric down regardless of embedding quality. A "top-1 conditional
  on having ≥1 same-class neighbor" metric would be more informative
  but would also exclude the bulk of the data.
- The largest class has 1,496 members (~32% of all equations). Any
  equation in this class has ~32% chance that a random neighbor is
  in-class. This inflates the metric for whatever fraction of the
  corpus is in that giant class.
- The baseline computation uses class-proportional matching (`1 -
  (1 - p_c)^k`), so it accounts for the size skew correctly. The
  trained gains are net of that.

The flat ep0/ep3/ep6 curve again says: equivalence-class structure
is captured early, then plateaus.

---

## Cross-cutting observations

### 1. The contrastive loss does most of the structural work

Across all experiments, ep0 (after one epoch of training) is within
1pp of ep6 on every probe except the bal_acc itself. The contrastive
loss compresses input features into a structurally-informative
embedding very quickly; subsequent training refines pair-level
geometry to push bal_acc from 0.93 → 0.96 without changing
per-equation feature representation.

### 2. Cohomology was already free

The cohom-as-aux-head experiments (`aux_h256_cohom_*`) added 8
cohom-summary aux heads under the hypothesis that explicit
supervision was needed to anchor cohom features. But probing shows
cohom features were already at 80-88% accessibility from contrastive
training alone. The aux heads were redundant supervision that turned
out to add nothing useful (peak bal_acc 0.94-0.95 vs the structural-
only baseline's 0.95-0.96).

### 3. The output projection sacrifices some cohom info for contrastive geometry

The 256-dim pooled rep (pre-projection) probes 5-10pp better on
cohom summaries than the 512-dim L2-norm output. The projection +
L2-norm step is necessary for the contrastive loss (unit-sphere
geometry, bounded distance) but it actively destroys some cohom
linear separability. This is a tradeoff inherent to the architecture
rather than a training failure.

### 4. The pair classifier matches the JEPA distance metric

A linear classifier on `concat(z_e1, z_e2)` reaches 0.951 balanced
accuracy — within 1.1pp of the trained JEPA's 0.962 distance-based
prediction. So the embeddings encode the implication relationship
in a directly linearly accessible way; the predictor MLP + distance
metric learns marginal additional structure, but the bulk of the
signal is in the embedding pair concatenation.

This suggests the JEPA architecture's predictor module isn't as
critical as one might expect — most of the work is in the encoder.

### 5. Equivalence class structure is partially captured but hard to measure

NN top-1 of 0.52 (vs random 0.35) is a real but modest signal. The
methodology (singletons, giant class) makes the headline number
hard to compare against an absolute reference. A more useful
follow-up metric would be NN accuracy restricted to equations in
classes with ≥3 members — gives a cleaner "is the geometry
clustering equivalences" signal.

---

## Implications

For the project as a whole:

1. **Stop adding cohom as aux-head supervision.** The contrastive
   loss already encodes cohom features at near-saturation (80-88%).
   Adding aux heads either does nothing (if they saturate) or hurts
   (if they distort the contrastive geometry — which the
   `aux_h256_cohom_*` runs confirmed).
2. **The encoder is the load-bearing component.** Pair classification
   from concat embeddings nearly matches the trained distance metric.
   Improvements in bal_acc beyond 0.96 should focus on encoder
   capacity, encoder regularization, or pair-specific features —
   not on the predictor or the contrastive loss form.
3. **Cohomology as an INPUT feature could still help** — the L2-norm
   step destroys some cohom info, so feeding cohom directly to the
   encoder (as a node attribute or graph-level conditioning) would
   bypass the destruction. This is a different intervention from
   aux-head supervision and is the only "use cohom directly" path
   the probing results don't refute.
4. **Multi-epoch training has diminishing returns on representation,
   but still moves bal_acc.** The +0.03pp from ep0 to ep6 in bal_acc
   is real (and reproduced across both seeds at h=128), but the
   per-equation features are essentially complete by ep0. So if
   compute is tight, ep1 + a checkpoint may be sufficient for
   downstream embedding extraction.
