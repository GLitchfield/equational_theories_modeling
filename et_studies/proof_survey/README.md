# proof_survey

Survey of the Equational Theories Project's Lean proofs, to inform the architecture of the proof generator.

## Headline

Three-track recommendation:
1. **62.5% arg-permutation closes** → structured-prediction model emitting `(hypothesis_id, permutation_tuple)`. *Superseded by the deterministic matcher* in `et_studies/matcher_deliverable/`.
2. **15.3% rewrite chains** → small transformer over a ~100-token Lean tactic vocab, conditioned on v1.0 embeddings.
3. **22% Vampire/Egg machine output** → don't learn; delegate to the upstream tool.
4. (Remainder) 0.5% genuinely complex / corner cases.

FALSE side: Stage 2 accepts only `Fin n` matrix counterexamples → brute-force search over small `Fin n` magmas.

Vocab is small: 119 tokens cover 95% of TRUE proof usage. Proof length bimodal — median 10, mean 102 (Vampire/Egg long tail).

## Contents

| Path | What |
|---|---|
| `outputs/README.md` | Detailed write-up with architecture recommendation |
| `outputs/inventory.json` | TRUE/FALSE proof counts + category distribution |
| `outputs/true_characterization.json` | TRUE-proof shape distribution |
| `outputs/false_characterization.json` | FALSE-proof shape distribution |
| `outputs/clustering_results.json` | Proof-shape clustering output |
| `outputs/orientation_notes.md` | Orientation findings on `both`-direction equations |
| `outputs/tokenization_results.json` | Vocab analysis (119 tokens cover 95%) |
| `outputs/intermediate_state_feasibility.md` | Feasibility note for the rewrite-chain decoder track |
| `data/` | `normalized_proofs.parquet`, `true_corpus.parquet`, `facts_corpus.parquet` (gitignored) |
| `scripts/` | `survey_*.py` (7 reproducers) |

## Archived plan

`archive/proof_corpus_survey_plan.md`.
