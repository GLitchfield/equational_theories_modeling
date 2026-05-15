# Failure-overlap analysis across 13 no-holdout runs

Question: when models fail to predict an equation pair correctly, are
those failures the **same pairs** across runs (systematic difficulty),
or **different random pairs** (model-specific noise)?

Answer: **strongly systematic**, with a clean and interpretable
failure mode. Specifically:
- **25,982 pairs (0.12% of the corpus) are wrong in ALL 13 runs.**
  Independence would predict ~0; observed/iid-expected ratio for k=13
  is ~260,000×.
- **99.99% of those always-wrong pairs are TRUE pairs** (25,980 of
  25,982) — the systematic failure is "TRUE pairs that look like
  FALSE pairs to every model".
- **Equation 1 (`x = x`) alone accounts for 1,587 always-wrong pairs**
  (6.1% of the always-wrong set, 17% of all pairs involving eq 1).
- The systematic failures concentrate where E2 is **family_c spine**
  (high-variable-count "weak" equations like `x◇(y◇z) = w◇(u◇z)`)
  or **trivial spine** (like `x = x`). These are equations implied by
  almost everything by tautology / generality, but the contrastive
  distance metric can't reliably detect "implied by vacuity".

## Inputs

13 runs, each at its best-bal_acc epoch. Predictions on the full
22,028,942-pair table (full corpus, no holdout):

| Run | Best epoch | bal_acc on full corpus |
|---|---:|---:|
| skip_h128_seed42_diag | 9 | 0.9840 |
| skip_h128_seed1_diag | 5 | 0.9779 |
| skip_h128_seed42_20ep_diag | 15 | 0.9777 |
| aux_h256_diag | 7 | 0.9636 |
| aux_h128_seed1_diag | 6 | 0.9619 |
| aux_h128_diag | 3 | 0.9582 |
| aux_h256_cohom_diag | 6 | 0.9490 |
| aux_h256_seed1_diag | 0 | 0.9488 |
| stable_m1p5_vic0p1_diag | 6 | 0.9487 |
| stable_h256_m1p5_vic0p05_diag | 0 | 0.9463 |
| aux_h128_cohom_w5_diag | 0 | 0.9362 |
| stable_h256_m1p5_vic0p1_diag | 0 | 0.9329 |
| aux_h256_cohom_w5_diag | 0 | 0.9331 |

Excluded: SIGReg-family runs (`stable_h256_lejepa_3ep_diag`,
`stable_h256_m1p5_sigreg*`) since they used a fundamentally different
regularizer.

## Per-pair failure count distribution

For each of 22M pairs, count how many of the 13 runs got it wrong:

| k runs failed | n pairs | % | iid-expected at p=0.046 | observed/iid |
|---:|---:|---:|---:|---:|
| 0 | 18,443,324 | 83.7% | 11,893,080 | 1.55× |
| 1 | 1,005,351 | 4.6% | 7,507,478 | 0.13× |
| 2 | 609,548 | 2.8% | 2,187,266 | 0.28× |
| 3 | 397,088 | 1.8% | 389,430 | **1.02×** |
| 4 | 338,209 | 1.5% | 47,274 | 7.2× |
| 5 | 316,388 | 1.4% | 4,132 | 76.6× |
| 6 | 289,785 | 1.3% | 268 | 1,083× |
| 7 | 243,528 | 1.1% | 13 | 18,747× |
| 8 | 209,690 | 1.0% | 0.5 | 443,255× |
| 9 | 90,923 | 0.4% | ~0 | huge |
| 10 | 29,102 | 0.13% | ~0 | huge |
| 11 | 17,036 | 0.08% | ~0 | huge |
| 12 | 12,988 | 0.06% | ~0 | huge |
| 13 | **25,982** | **0.12%** | ~0 | huge |

The distribution is **bimodal**: most pairs are at k=0 (always
correct) or piled up at high k (≥4 runs failed). Only at k=3 do
observed and iid expected agree. Above k=3, failures are radically
non-independent — pairs failed by 4+ runs are the same hard pairs
across many models.

## Always-wrong (k=13) characterization

- **25,982 pairs** (0.118% of corpus)
- **25,980 are TRUE pairs** (99.99%); only 2 are FALSE pairs
- **3,108 unique equations** appear in the always-wrong set (~66% of
  the corpus's 4,694 equations)
- Top equations by always-wrong frequency:

| eq_id | pairs always-wrong | text |
|---:|---:|---|
| 1 | 1,587 | `x = x` (the trivial reflexive) |
| 4362 | 290 | `x ◇ (y ◇ z) = y ◇ (x ◇ z)` |
| 4360 | 282 | `x ◇ (y ◇ z) = x ◇ (w ◇ z)` |
| 4689 | 241 | `(x ◇ y) ◇ z = (w ◇ y) ◇ z` |
| 4622 | 233 | `(x ◇ x) ◇ y = (z ◇ z) ◇ y` |
| 4378 | 229 | `x ◇ (y ◇ z) = w ◇ (u ◇ z)` |
| 4374 | 227 | `x ◇ (y ◇ z) = w ◇ (y ◇ z)` |
| 4341 | 211 | `x ◇ (y ◇ y) = x ◇ (z ◇ z)` |
| 4367 | 206 | `x ◇ (y ◇ z) = y ◇ (w ◇ z)` |
| 4675 | 203 | `(x ◇ y) ◇ z = (x ◇ w) ◇ z` |

All of the top 10 are either the trivial reflexive (eq 1) or
high-variable-count equations with multiple distinct variables on
both sides — equations that are **implied by almost every more
specific equation** because they're so weakly constraining.

### Spine breakdown

When the always-wrong pair is `(E1, E2)`, the spine of E2 (the
equation being implied):

| E2 spine | always-wrong count | % of always-wrong |
|---|---:|---:|
| family_c | 14,429 | 55.5% |
| left | 5,238 | 20.2% |
| mixed | 3,436 | 13.2% |
| trivial | 1,587 | 6.1% |
| right | 1,292 | 5.0% |
| no_lhs_var_in_rhs | 0 | 0.0% |

`family_c` (the most weakly-constraining spine, with multiple
distinct variables on both sides) accounts for over half. Plus the
1,587 trivial-spine pairs are basically all pairs implying `x = x`.

By eq1's spine type, what fraction of pairs starting with that spine
are always-wrong?

| E1 spine | always-wrong / total | rate |
|---|---:|---:|
| trivial | 0 / 4,693 | 0.00% |
| no_lhs_var_in_rhs | 0 / 3,829,488 | 0.00% |
| right | 3,422 / 1,126,320 | 0.30% |
| mixed | 10,572 / 5,946,031 | 0.18% |
| left | 5,240 / 3,824,795 | 0.14% |
| family_c | 6,748 / 7,297,615 | 0.09% |

`trivial` and `no_lhs_var_in_rhs` E1s are NEVER always-wrong — when
the LHS equation is structurally simple, the model handles it.

## Pairwise Jaccard between failure sets

How similar are run pairs' failure sets? Jaccard = |intersection| /
|union|; 1.0 = identical, 0 = disjoint.

| | aux128 | aux128.s1 | aux128.cohw5 | aux256 | aux256.s1 | aux256.coh | aux256.cohw5 | skip.s1 | skip.s42 | skip.s42.20 | stab.h256.05 | stab.h256.1 | stab.h128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aux128 | 1.000 | 0.402 | 0.385 | 0.161 | 0.331 | 0.206 | 0.360 | 0.098 | 0.083 | 0.078 | 0.362 | 0.343 | 0.376 |
| aux128.s1 | | 1.000 | 0.363 | 0.170 | 0.369 | 0.220 | 0.336 | 0.094 | 0.080 | 0.078 | 0.367 | 0.345 | 0.355 |
| aux128.cohw5 | | | 1.000 | 0.109 | 0.442 | 0.220 | **0.762** | 0.093 | 0.057 | 0.074 | 0.403 | 0.587 | 0.394 |
| aux256 | | | | 1.000 | 0.092 | 0.483 | 0.084 | 0.051 | 0.061 | 0.041 | 0.164 | 0.109 | 0.147 |
| aux256.s1 | | | | | 1.000 | 0.158 | 0.448 | 0.120 | 0.077 | 0.097 | 0.362 | 0.429 | 0.312 |
| aux256.coh | | | | | | 1.000 | 0.186 | 0.059 | 0.058 | 0.048 | 0.233 | 0.201 | 0.221 |
| aux256.cohw5 | | | | | | | 1.000 | 0.087 | 0.054 | 0.077 | 0.374 | 0.583 | 0.354 |
| skip.s1 | | | | | | | | 1.000 | **0.392** | **0.443** | 0.109 | 0.102 | 0.107 |
| skip.s42 | | | | | | | | | 1.000 | **0.452** | 0.088 | 0.073 | 0.091 |
| skip.s42.20 | | | | | | | | | | 1.000 | 0.095 | 0.095 | 0.093 |
| stab.h256.05 | | | | | | | | | | | 1.000 | 0.415 | 0.348 |
| stab.h256.1 | | | | | | | | | | | | 1.000 | 0.342 |

**Observations:**

- **Mean off-diagonal Jaccard: 0.232.** Failures are systematically
  shared across runs (random would be ~0.04 given the per-run failure
  rates).
- **Skip runs cluster together** (skip.s1 ↔ skip.s42 ↔ skip.s42.20:
  0.39-0.45) but have **low Jaccard with all other runs** (0.05-0.12).
  The skip recipe is wrong about substantially different pairs than
  the no-skip recipes are.
- **High-aux-weight cohom runs cluster** (aux128.cohw5 ↔ aux256.cohw5:
  **0.762**, the highest off-diagonal Jaccard). These two badly-
  trained runs (aux_weight=5.0 caused gradient pathology) make the
  same mistakes — likely because they're both predicting close to
  majority class.
- **aux256 (the lucky regime-B run)** has very low Jaccard with
  almost everything — its failure set is unique. Consistent with the
  finding that its 0.9635 came from a different mechanism than the
  other models.

## Files in this directory

- `README.md` — this file
- `failure_count_per_pair.parquet` — every pair with how many of 13
  runs got it wrong (37 MB; one row per pair)
- `always_wrong_pairs.parquet` — the 25,982 pairs that all 13 runs
  failed (just eq1, eq2, ground_truth)
- `pairwise_jaccard.csv` — 13×13 Jaccard matrix
- `failure_count_distribution.csv` — histogram of n_runs_failed

## Implications

1. **The failure mode is interpretable: TRUE-by-vacuity.** Models
   can't reliably detect TRUE pairs where E2 is a tautology (eq 1)
   or a very weak axiom (high-variable family_c equations). The
   contrastive distance metric encodes "are these two equations
   structurally similar?" but the actual implication relation is
   "does E1's set of models include E2's set of models?", which
   for trivial / weak E2s requires reasoning beyond structural
   similarity.
2. **Skip recipe fails differently from aux recipes.** Skip runs
   share most of their failures with each other (0.4-0.45 Jaccard)
   but barely overlap with aux baselines (0.05-0.12). This means
   the +1.5pp bal_acc gain isn't just "fewer failures of the same
   type" — it's "different failure profile entirely". An ensemble
   of skip + aux models would likely be substantially better than
   either alone.
3. **The 25,982 always-wrong pairs are a stable target.** Any
   future intervention that improves this set is making real
   progress. A simple rule-based prediction "if E2 is the trivial
   reflexive (`x = x`), predict TRUE" would correctly fix ~1,587
   of them at zero training cost.
