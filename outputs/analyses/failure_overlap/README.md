# Failure-overlap analysis across runs

Question: when models fail to predict an equation pair correctly,
are those failures the **same pairs** across runs (systematic
difficulty), or **different random pairs** (model-specific noise)?

Answer: **strongly systematic**, with a clean and interpretable
failure mode (TRUE pairs where E2 is a trivial or very-weak
axiom). The headline numbers are partitioned by training health
to keep the broken-cluster effect separate from the systematic-
difficulty effect — see "Partitioning" below.

Reproducer: `scripts/analyze_failure_overlap.py`

## Partitioning

Of 13 runs (excluding SIGReg / LeJEPA variants, which use a
fundamentally different regularizer and aren't comparable here):

- **8 runs trained successfully** (best epoch > 0):
  - `aux_h128_diag` (ep=3, bal_acc=0.9582)
  - `aux_h128_seed1_diag` (ep=6, bal_acc=0.9619)
  - `aux_h256_cohom_diag` (ep=6, bal_acc=0.9490)
  - `aux_h256_diag` (ep=7, bal_acc=0.9636)
  - `skip_h128_seed1_diag` (ep=5, bal_acc=0.9779)
  - `skip_h128_seed42_20ep_diag` (ep=15, bal_acc=0.9777)
  - `skip_h128_seed42_diag` (ep=9, bal_acc=0.9840)
  - `stable_m1p5_vic0p1_diag` (ep=6, bal_acc=0.9487)
- **5 runs are pathological** (best epoch == 0; the model
  never improved past random init — either h=256 simplicity-bias
  drift or aux_weight=5.0 gradient-clip pathology):
  - `aux_h128_cohom_w5_diag` (ep=0, bal_acc=0.9362)
  - `aux_h256_cohom_w5_diag` (ep=0, bal_acc=0.9331)
  - `aux_h256_seed1_diag` (ep=0, bal_acc=0.9488)
  - `stable_h256_m1p5_vic0p05_diag` (ep=0, bal_acc=0.9463)
  - `stable_h256_m1p5_vic0p1_diag` (ep=0, bal_acc=0.9329)

All aggregate headlines below are reported for both partitions.
Sub-cluster Jaccards (skip vs skip, cohom_w5 vs cohom_w5)
don't need this partitioning — they're already conditioned on
a recipe.

## Per-pair failure-count distribution

For each of 22,028,942 pairs, count how many runs got it wrong.
Both partitions shown side-by-side:

| k runs failed | n pairs (all runs) | n pairs (trained only) |
|---:|---:|---:|
| 0 | 18,443,324 | 19,406,835 |
| 1 | 1,005,351 | 971,304 |
| 2 | 609,548 | 799,897 |
| 3 | 397,088 | 484,478 |
| 4 | 338,209 | 202,720 |
| 5 | 316,388 | 88,693 |
| 6 | 289,785 | 26,415 |
| 7 | 243,528 | 16,104 |
| 8 | 209,690 | 32,496 |
| 9 | 90,923 | 0 |
| 10 | 29,102 | 0 |
| 11 | 17,036 | 0 |
| 12 | 12,988 | 0 |
| 13 | 25,982 | 0 |

**Always-wrong (k == n_runs):** 25,982 pairs across all 13 runs (of which 25,980 are TRUE pairs); 32,496 pairs across 8 trained runs (of which 32,493 are TRUE).

**Independence baseline:** mean per-run failure rate is 0.0463 (all runs) / 0.0330 (trained only). Under independence, the expected count at k=13 (all) is 9.92e-11 and at k=8 (trained) is 3.07e-05 — both effectively zero. The observed always-wrong sets are radically non-independent.

## Pairwise Jaccard between failure sets

Jaccard = |intersection| / |union|; 1.0 = identical, 0 = disjoint. Full 13×13 matrix in `pairwise_jaccard.csv`; upper triangle shown below.

| | aux128.cohw5 | aux128 | aux128.s1 | aux256.coh | aux256.cohw5 | aux256 | aux256.s1 | skip.s1 | skip.s42.20 | skip.s42 | stab.h256.05 | stab.h256.1 | stab.h128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aux128.cohw5 | 1.000 | 0.385 | 0.363 | 0.220 | 0.762 | 0.109 | 0.442 | 0.093 | 0.074 | 0.057 | 0.403 | 0.587 | 0.394 |
| aux128 |  | 1.000 | 0.402 | 0.206 | 0.360 | 0.161 | 0.331 | 0.098 | 0.078 | 0.083 | 0.362 | 0.343 | 0.376 |
| aux128.s1 |  |  | 1.000 | 0.220 | 0.336 | 0.170 | 0.369 | 0.094 | 0.078 | 0.080 | 0.367 | 0.345 | 0.355 |
| aux256.coh |  |  |  | 1.000 | 0.186 | 0.483 | 0.158 | 0.059 | 0.048 | 0.058 | 0.233 | 0.201 | 0.221 |
| aux256.cohw5 |  |  |  |  | 1.000 | 0.084 | 0.448 | 0.087 | 0.077 | 0.054 | 0.374 | 0.583 | 0.354 |
| aux256 |  |  |  |  |  | 1.000 | 0.092 | 0.051 | 0.041 | 0.061 | 0.164 | 0.109 | 0.147 |
| aux256.s1 |  |  |  |  |  |  | 1.000 | 0.120 | 0.097 | 0.077 | 0.362 | 0.429 | 0.312 |
| skip.s1 |  |  |  |  |  |  |  | 1.000 | 0.443 | 0.392 | 0.109 | 0.102 | 0.107 |
| skip.s42.20 |  |  |  |  |  |  |  |  | 1.000 | 0.452 | 0.095 | 0.095 | 0.093 |
| skip.s42 |  |  |  |  |  |  |  |  |  | 1.000 | 0.088 | 0.073 | 0.091 |
| stab.h256.05 |  |  |  |  |  |  |  |  |  |  | 1.000 | 0.415 | 0.348 |
| stab.h256.1 |  |  |  |  |  |  |  |  |  |  |  | 1.000 | 0.342 |
| stab.h128 |  |  |  |  |  |  |  |  |  |  |  |  | 1.000 |

**Aggregate Jaccard, partitioned:**

- **Mean off-diagonal across 8 trained runs: 0.184.** This is the headline for "systematic difficulty across well-trained models."
- Mean off-diagonal across all 13 runs (trained + pathological): 0.232.
- Mean off-diagonal among pathological runs only: 0.480. Higher than the trained-only mean because broken models default to similar majority-class predictions and so share most of their errors.

**Reading the matrix:**

- **Skip runs cluster together** (skip.s1 ↔ skip.s42 ↔ skip.s42.20: 0.39-0.45) but have LOW Jaccard with all other runs (0.05-0.12). The skip recipe fails on substantially different pairs than the no-skip recipes.
- **High-aux-weight cohom runs cluster** (aux128.cohw5 ↔ aux256.cohw5 ≈ 0.76). These are the two grad-pathology runs — both mostly predicting majority class, so they share failure patterns.
- **aux256 (lucky regime-B run)** has unusually low Jaccard with everything; its failure set is unique. Consistent with the finding that its 0.9635 came from a different mechanism than the other models.

## Always-wrong characterization

Across the 8 trained runs, **32,496 pairs are always-wrong**, of which **32,493 (99.99%) are TRUE pairs**. The systematic failure mode is "TRUE pairs that look like FALSE pairs to every trained model."

Equation 1 (`x = x`, the trivial reflexive) accounts for **1,629 of the 32,496 always-wrong-trained pairs** — pairs `(E1, x=x)` where E1 → x=x is tautologically TRUE but the model misses.

### Top-10 E2 (equation being implied) by always-wrong frequency

| eq_id | pairs always-wrong | text |
|---:|---:|---|
| 1 | 1629 | `x = x` |
| 4374 | 322 | `x ◇ (y ◇ z) = w ◇ (y ◇ z)` |
| 4378 | 318 | `x ◇ (y ◇ z) = w ◇ (u ◇ z)` |
| 4360 | 294 | `x ◇ (y ◇ z) = x ◇ (w ◇ z)` |
| 4362 | 291 | `x ◇ (y ◇ z) = y ◇ (x ◇ z)` |
| 4367 | 280 | `x ◇ (y ◇ z) = y ◇ (w ◇ z)` |
| 4693 | 256 | `(x ◇ y) ◇ z = (w ◇ u) ◇ z` |
| 4351 | 251 | `x ◇ (y ◇ y) = z ◇ (y ◇ y)` |
| 4622 | 251 | `(x ◇ x) ◇ y = (z ◇ z) ◇ y` |
| 4689 | 247 | `(x ◇ y) ◇ z = (w ◇ y) ◇ z` |

All of the top 10 are either the trivial reflexive (eq 1) or high-variable-count equations with multiple distinct variables on both sides — equations that are implied by almost every more-specific equation because they're so weakly constraining.

### Spine breakdown (always-wrong-trained set)

When the always-wrong pair is `(E1, E2)`, the spine of E2 (the
equation being implied):

| E2 spine | always-wrong count | % of always-wrong |
|---|---:|---:|
| family_c | 17,705 | 54.5% |
| left | 6,645 | 20.4% |
| mixed | 4,798 | 14.8% |
| right | 1,719 | 5.3% |
| trivial | 1,629 | 5.0% |

By E1's spine type, what fraction of pairs starting with that
spine are always-wrong:

| E1 spine | always-wrong / total | rate |
|---|---:|---:|
| no_lhs_var_in_rhs | 0 / 3,829,488 | 0.00% |
| trivial | 0 / 4,693 | 0.00% |
| family_c | 7,443 / 7,297,615 | 0.10% |
| left | 6,533 / 3,824,795 | 0.17% |
| mixed | 14,324 / 5,946,031 | 0.24% |
| right | 4,196 / 1,126,320 | 0.37% |

`trivial` and `no_lhs_var_in_rhs` E1s are never (or almost never) always-wrong — when the LHS equation is structurally simple, the model handles it. The rates are highest on `right`, `mixed`, and `left` E1s.

## Files in this directory

- `README.md` — this writeup
- `failure_count_per_pair.parquet` — every pair with `n_runs_failed_all` and `n_runs_failed_trained` (one row per pair)
- `always_wrong_all.parquet` — pairs where all 13 runs got it wrong (25,982 rows)
- `always_wrong_trained.parquet` — pairs where all 8 trained runs got it wrong (32,496 rows)
- `pairwise_jaccard.csv` — N×N Jaccard matrix
- `failure_count_distribution.csv` — histogram of n_runs_failed for both partitions

## Implications

1. **The failure mode is interpretable: TRUE-by-vacuity.** Models can't reliably detect TRUE pairs where E2 is a tautology (eq 1) or a very weak axiom (high-variable family_c equations). The contrastive distance metric encodes "are these two equations structurally similar?" but the actual implication relation is "does E1's set of models include E2's set of models?", which for trivial / weak E2s requires reasoning beyond structural similarity.
2. **Skip recipe fails differently from aux recipes.** Skip runs share most of their failures with each other but barely overlap with aux baselines. Likely complementary errors — an ensemble of skip + aux would probably move on the always-wrong set.
3. **The 32,496 always-wrong-trained pairs are a stable target.** Any future intervention that improves this set is making real progress. A simple rule "if E2 is `x = x`, predict TRUE" would correctly fix 1,629 of them at zero training cost.
