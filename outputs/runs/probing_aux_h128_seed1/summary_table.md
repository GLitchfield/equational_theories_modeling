# Probing summary table
All probe runs at the four key embeddings: `random_init` (untrained, no-info floor), ep0/ep3/ep6 of `aux_h128_seed1_diag`. Multi-class targets report plain accuracy; binary targets report balanced accuracy (plain in parens). Regression targets report R² (baseline R²=0).

## Experiment 1 — Structural sanity check (10 aux-head targets)
| target | majority_acc | random_init | ep0 | ep3 | ep6 |
|---|---:|---:|---:|---:|---:|
| spine_type | 0.331 | 0.810 | 0.999 | 0.999 | 1.000 |
| lhs_structure | 0.669 | 0.999 (0.999) | 1.000 (1.000) | 0.999 (0.999) | 1.000 (1.000) |
| num_variables | 0.445 | 0.982 | 0.998 | 0.995 | 0.996 |
| magma_bit_0 | 0.669 | 1.000 (1.000) | 1.000 (1.000) | 0.999 (0.999) | 0.999 (0.999) |
| magma_bit_1 | 0.741 | 0.581 (0.770) | 0.998 (0.999) | 0.994 (0.997) | 0.992 (0.996) |
| magma_bit_2 | 0.741 | 0.628 (0.784) | 1.000 (1.000) | 0.999 (0.999) | 0.998 (0.999) |
| magma_bit_3 | 0.859 | 0.508 (0.860) | 0.930 (0.980) | 0.989 (0.997) | 0.991 (0.997) |
| magma_bit_4 | 0.669 | 1.000 (1.000) | 1.000 (1.000) | 0.999 (0.999) | 0.999 (0.999) |
| magma_bit_6 | 0.926 | 0.500 (0.926) | 0.945 (0.992) | 0.944 (0.992) | 0.939 (0.991) |
| magma_bit_7 | 0.926 | 0.500 (0.926) | 0.984 (0.998) | 0.977 (0.997) | 0.976 (0.996) |

## Experiment 2 — Per-prime cohom summaries (binned classification)
| target | majority_acc | random_init | ep0 | ep3 | ep6 |
|---|---:|---:|---:|---:|---:|
| count_p2 | 0.405 | 0.636 | 0.867 | 0.878 | 0.880 |
| mean_h2_p2 | 0.333 | 0.553 | 0.857 | 0.866 | 0.868 |
| count_p3 | 0.340 | 0.574 | 0.836 | 0.841 | 0.843 |
| mean_h2_p3 | 0.340 | 0.566 | 0.832 | 0.837 | 0.838 |
| count_p5 | 0.328 | 0.553 | 0.807 | 0.814 | 0.810 |
| mean_h2_p5 | 0.328 | 0.606 | 0.831 | 0.836 | 0.837 |
| count_p7 | 0.334 | 0.568 | 0.846 | 0.850 | 0.851 |
| mean_h2_p7 | 0.334 | 0.622 | 0.841 | 0.848 | 0.849 |

## Experiment 3 — Cell-level cohom (aggregate over 56 probable cells)
| embedding | mean binary bal_acc | mean H² R² | best binary cell | best R² cell |
|---|---:|---:|---|---|
| random_init | 0.552 | 0.221 | (0,0,p2) (1.000) | (3,5,p7) (0.629) |
| ep0 | 0.749 | 0.475 | (0,0,p2) (1.000) | (0,0,p7) (0.978) |
| ep3 | 0.753 | 0.472 | (1,0,p2) (0.999) | (0,0,p7) (0.974) |
| ep6 | 0.753 | 0.479 | (0,0,p2) (0.999) | (0,0,p7) (0.975) |

## Experiment 4 — Non-anchored invariants
Classification targets (binary; bal_acc in parens):

| target | majority_acc / bal | random_init | ep0 | ep3 | ep6 |
|---|---:|---:|---:|---:|---:|
| is_trivial | 1.000 / 0.500 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| is_singleton_forcing | 0.826 / 0.500 | 1.000 (1.000) | 0.999 (0.998) | 0.999 (0.998) | 0.999 (0.998) |
| is_balanced | 0.993 / 0.500 | 0.993 (0.500) | 0.993 (0.500) | 0.993 (0.500) | 0.993 (0.500) |
| leftmost_match | 0.741 / 0.500 | 0.784 (0.628) | 1.000 (1.000) | 0.999 (0.999) | 0.999 (0.998) |
| rightmost_match | 0.741 / 0.500 | 0.770 (0.581) | 0.999 (0.998) | 0.997 (0.994) | 0.996 (0.992) |

Regression targets (R²):

| target | random_init | ep0 | ep3 | ep6 |
|---|---:|---:|---:|---:|
| op_count_asymmetry | 0.808 | 0.960 | 0.958 | 0.960 |
| spine_depth | 0.271 | 0.844 | 0.844 | 0.849 |

## Experiment 5 — Pairwise cohom (100K pairs, stratified 50/50)
Subset sizes — clean (both have non-null cohom): 39,443, confounded (one or both all-null): 60,557, pairs with shared triples for compat regression: 22,029

Ground truth (TRUE/FALSE) probe — balanced acc by pair representation:

| pair_rep | random_init | ep0 | ep3 | ep6 |
|---|---:|---:|---:|---:|
| concat | 0.829 | 0.948 | 0.950 | 0.951 |
| hadamard | 0.659 | 0.697 | 0.608 | 0.611 |
| absdiff | 0.687 | 0.699 | 0.700 | 0.699 |

has_obstruction (clean subset only) — balanced acc:

| pair_rep | random_init | ep0 | ep3 | ep6 |
|---|---:|---:|---:|---:|
| concat | 0.546 | 0.692 | 0.686 | 0.692 |
| hadamard | 0.500 | 0.500 | 0.500 | 0.500 |
| absdiff | 0.501 | 0.526 | 0.522 | 0.524 |

compatibility regression — R² (drops pairs without shared triples):

| pair_rep | random_init | ep0 | ep3 | ep6 |
|---|---:|---:|---:|---:|
| concat | 0.123 | 0.232 | 0.231 | 0.231 |
| hadamard | 0.051 | 0.125 | 0.185 | 0.185 |
| absdiff | 0.101 | 0.354 | 0.330 | 0.330 |

## Experiment 6 — Equivalence class via nearest-neighbor
1,415 classes derived from the implication graph; **1,145 are singletons** (no possible NN match), largest class = 1,496 members.

| embedding | top-1 (base) | top-5 (base) | top-10 (base) |
|---|---:|---:|---:|
| random_init | 0.348 (0.112) | 0.527 (0.319) | 0.563 (0.392) |
| ep0 | 0.526 (0.112) | 0.636 (0.319) | 0.667 (0.392) |
| ep3 | 0.530 (0.112) | 0.642 (0.319) | 0.676 (0.392) |
| ep6 | 0.522 (0.112) | 0.644 (0.319) | 0.673 (0.392) |

## Pooled vs L2-norm output (trained ep6 only)
Where the pooled rep (256-dim, what aux heads see) outperforms the L2-norm output (512-dim, what contrastive uses), the projection+norm step is destroying linear separability of that feature.

| target | pooled | output | drop |
|---|---:|---:|---:|
| Exp2/count_p2 | 0.980 | 0.880 | -0.099 |
| Exp2/mean_h2_p2 | 0.885 | 0.868 | -0.017 |
| Exp2/count_p3 | 0.878 | 0.843 | -0.035 |
| Exp2/count_p5 | 0.849 | 0.810 | -0.039 |
| Exp2/count_p7 | 0.862 | 0.851 | -0.012 |
