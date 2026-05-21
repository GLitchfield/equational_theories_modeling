# failure_overlap

Cross-run failure-overlap analysis across 8 trained JEPA models. Identifies pairs the contrastive embedding fundamentally cannot separate — the residual the embedding loss can't address.

## Headline

- **32,496 always-wrong-trained pairs** across all 8 models (99.99% TRUE).
- Dominated by the **TRUE-by-vacuity** pattern: equations implied tautologically by every more-specific axiom (e.g., `E2: x=x`-style targets that hold trivially under any premise).
- A 6.6× over-represented cluster within this set is `both`-direction equations (`Vars(lhs) = Vars(rhs)`), which are also the equations plain Knuth-Bendix completion can't orient.
- The MLP head trained on frozen v1.0 embeddings recovers **91.37%** of this subset — the signal *is* in the embeddings, but not in a way the JEPA's distance metric can read.

## Contents

| Path | What |
|---|---|
| `outputs/README.md` | Detailed writeup |
| `outputs/always_wrong_all.parquet` | Pairs every model gets wrong |
| `outputs/always_wrong_trained.parquet` | Restricted to trained models (excludes random-init) |
| `outputs/failure_count_distribution.csv` | Histogram of "wrong-on-N-models" counts |
| `outputs/failure_count_per_pair.parquet` | Per-pair failure count |
| `outputs/pairwise_jaccard.csv` | Pairwise Jaccard distance between models' failure sets |
| `scripts/analyze_failure_overlap.py` | Reproducer |

This study informed two follow-up findings:
1. The MLP-classifier plan (recover the TRUE-by-vacuity signal frozen v1.0 embeddings carry).
2. The orientability cluster connection — `both`-direction equations are simultaneously the failure mode of plain KB completion and an over-represented cluster in always-wrong-trained.
