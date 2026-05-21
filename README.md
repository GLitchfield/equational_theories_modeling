# Equational Implication

Learn the geometric structure of equational implication over magmas. The 22M Lean-verified implication pairs from the Equational Theories Project drove training of a JEPA over equation parse trees; the result is a per-equation embedding that captures both algebraic structure and the harder "TRUE-by-vacuity" pattern (equations implied tautologically by every more-specific axiom). The current phase moves from embeddings into end-to-end proof generation for the ETP Stage 2 competition.

## Repository layout

```
data/                       # raw + processed (shared, stays at root)
lib/                        # shared primitives (parser, matching, unification, LPO)
embeddings/                 # JEPA work, frozen v1.0 embeddings, MLP classifier
et_studies/                 # frozen studies (each with its own outputs/, scripts/, data/, code/)
proof_generation_pilot/     # active phase — methods/ ported in; pilot plan pending
tests/                      # flat test layout, repo-root
archive/                    # archived plans (every completed plan ends up here)
```

Each top-level area has its own README. See [`et_studies/README.md`](et_studies/README.md) for an index of the nine completed studies with one-line headlines.

## Five durable deliverables

1. **v1.0 embeddings** at [`embeddings/outputs/embeddings/v1.0/`](embeddings/outputs/embeddings/v1.0/) — frozen 768-dim equation embeddings; see [`MODEL_CARD.md`](embeddings/outputs/embeddings/v1.0/MODEL_CARD.md).
2. **Proof-corpus characterization** at [`et_studies/proof_survey/outputs/README.md`](et_studies/proof_survey/outputs/README.md) — surveyed ETP's 10,676 explicit-implication proofs and 1,698 facts entries; ends with the three-track architecture recommendation for proof generation.
3. **Rewrite-chain decision** at [`et_studies/rewrite_chain_study/outputs/synthesis_report.md`](et_studies/rewrite_chain_study/outputs/synthesis_report.md) — 83.13% deterministic-search coverage at d_max=6, 99.7% rule set within {h, ← h}, 87% unimodal. Recommendation: neural-guided search targeting the 17% hard tail.
4. **Completion-based decision procedure** at [`et_studies/completion_exploration/outputs/synthesis_report.md`](et_studies/completion_exploration/outputs/synthesis_report.md) — KB completion (plain + ordered) as a unified TRUE/FALSE decision procedure. Headline: 13.46% of 22M pairs decided soundly. Ordered completion lifts TRUE coverage 3× (2.13% → 6.06% of GT TRUE); plain KB FALSE coverage is 17.82% of GT FALSE.
5. **Deterministic-stack coverage measurement** at [`et_studies/coverage_measurement/outputs/coverage_measurement_synthesis.md`](et_studies/coverage_measurement/outputs/coverage_measurement_synthesis.md) — combined matcher + rewrite engine + completion: 6.54% of GT TRUE soundly covered. Structural ceiling 46.07%; the deterministic stack captures 14.20% of that ceiling. Dominant within-ceiling failure mode is `frontier_died` (85.74%) — structural reachability of safe first-order rewriting, not search-budget.

## Active phase

**Proof generation pilot.** The three deterministic methods (matcher, completion, safe rewrite engine) have been ported into `proof_generation_pilot/methods/`. A pilot plan markdown will land at `proof_generation_pilot/pilot_plan.md` and define the experimental cascade + FALSE-search strategy.

## v1.0 embeddings at a glance

- **Spec:** 4694 × 768 float32, L2-normalized.
- **Source:** `skip_h128_seed42_diag` ep9 (h=128 GATv2 encoder with parameter-free skip connection from the pooled rep). Source checkpoint travels with the artifact at `embeddings/outputs/embeddings/v1.0/source_checkpoint.pt`.
- **JEPA distance bal_acc on the matched 10% pair holdout:** 0.9841 (0.9840 on the full 22M corpus).
- **Frozen-embedding MLP-head bal_acc on the same holdout:** 0.9996, recovering 91.37% of the "always-wrong-trained" subset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Symlink the external data sources under `data/raw/`:

```bash
mkdir -p data/raw
ln -s /path/to/equational_theories/data                  data/raw/etp
ln -s /path/to/equational_theories_data/assembly/output  data/raw/invariants
```

Files used:

- `data/raw/etp/equations.txt` — 4,694 equations (line N = equation ID N)
- `data/raw/etp_export/export_raw_implications_5_9_2026.csv` — authoritative ground-truth label for every off-diagonal pair (4694×4694 int8 matrix, codes ±3/±4 mapping to TRUE/FALSE; see CHANGELOG 2026-05-09)
- `data/raw/etp/2025-08-11-vampire.json.gz` — *legacy* Vampire-derived labels (no longer the source of truth; left 1,062 pairs unresolved)
- sibling project `equational_theories_data/invariants/` — per-equation invariants (spine type, LHS structure, variable count, magma signature, cohomology summaries)

Do not write to `data/raw/`. All processing goes to `data/processed/` (shared caches) or `data/splits/` (committed split indices) or per-study `et_studies/X/data/` (gitignored).

## Build the shared caches

```bash
python lib/scripts/parse_equations.py             # data/processed/equation_graphs.pt   (~6 MB)
python lib/scripts/build_pair_table.py            # data/processed/implication_pairs.parquet  (~35 MB)
python lib/scripts/build_equation_invariants.py   # data/processed/equation_invariants.parquet
```

All three are deterministic and re-runnable; their outputs are gitignored.

## Regenerate v1.0 embeddings

```bash
python embeddings/scripts/save_embeddings.py \
    embeddings/outputs/embeddings/v1.0/source_checkpoint.pt \
    --output-dir embeddings/outputs/embeddings/v1.0 --force
```

Full reproduction recipe + validation numbers in [`embeddings/outputs/embeddings/v1.0/MODEL_CARD.md`](embeddings/outputs/embeddings/v1.0/MODEL_CARD.md).

## See also

- [`CLAUDE.md`](CLAUDE.md) — full project-state guide (operational rules, JEPA internals worth knowing, the always-loaded working memory).
- [`CHANGELOG.md`](CHANGELOG.md) — deviations from each plan and why.
- [`archive/`](archive/) — every completed plan archived in chronological order.
