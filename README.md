# Equational Implication

Learn the geometric structure of equational implication over magmas. The 22M Lean-verified implication pairs from the Equational Theories Project drove training of a JEPA over equation parse trees; the result is a per-equation embedding that captures both algebraic structure and the harder "TRUE-by-vacuity" pattern (equations implied tautologically by every more-specific axiom).

The **JEPA training phase**, **proof-corpus survey phase**, **rewrite-chain study**, **completion exploration**, and **coverage measurement** are all complete. Five durable deliverables:

1. **v1.0 embeddings** at [outputs/embeddings/v1.0/](outputs/embeddings/v1.0/) — frozen 768-dim equation embeddings; see [`MODEL_CARD.md`](outputs/embeddings/v1.0/MODEL_CARD.md).
2. **Proof-corpus characterization** at [outputs/analyses/proof_survey/README.md](outputs/analyses/proof_survey/README.md) — surveyed the ETP's 10,676 explicit-implication proofs and 1,698 facts entries, ending with a three-track architecture recommendation for the proof generator.
3. **Rewrite-chain category decision** at [outputs/analyses/rewrite_chain_study/synthesis_report.md](outputs/analyses/rewrite_chain_study/synthesis_report.md) — 83.13% deterministic-search coverage at d_max=6, 99.7% rule set within {h, ← h}, 87% unimodal. Recommendation: neural-guided search targeting the 17% hard tail. Engine in `src/rewrite/`; plan archived at [archive/rewrite_chain_study_plan.md](archive/rewrite_chain_study_plan.md).
4. **Completion-based decision procedure** at [outputs/analyses/completion_study/synthesis_report.md](outputs/analyses/completion_study/synthesis_report.md) — KB completion (plain + ordered) as a unified TRUE/FALSE decision procedure. Headline: 13.46% of 22M pairs decided soundly (TRUE from ordered, FALSE from plain). Ordered completion lifts TRUE coverage 3× (2.13% → 6.06% of GT TRUE); plain KB FALSE coverage is 17.82% of GT FALSE, with 3.28% of GT FALSE coming with a Stage-2-ready finite counter-model (Fin n ≤ 6) for free. Completion engines in `src/completion/`; plan archived at [archive/completion_exploration_plan.md](archive/completion_exploration_plan.md).
5. **Deterministic-stack coverage measurement** at [outputs/analyses/coverage_measurement/coverage_measurement_synthesis.md](outputs/analyses/coverage_measurement/coverage_measurement_synthesis.md) — combined TRUE-side coverage of matcher + rewrite engine + completion. Headline: 6.54% of GT TRUE soundly covered (completion 92.8% of the combined; matcher 5.6% via variable-disjoint substitution; rewrite engine 1.6% within-ceiling residual). Structural ceiling for rewriting methods on this corpus: 46.07% of GT TRUE. The deterministic stack captures 14.20% of that ceiling. Vampire/Egg proof categories bottom at 2.29%/0.54% engine solve rate, confirming safe rewriting can't substitute for narrowing/superposition. Safe rewrite primitive in `src/coverage/`; plan archived at [archive/coverage_measurement_plan.md](archive/coverage_measurement_plan.md).

**Active plan:** [proof_investigation_plan.md](proof_investigation_plan.md) — Phase 8 intermediate-term analysis on hard TRUE proofs (Vampire/Egg/complex categories), testing whether the dense-row residual that frontier-dies under safe rewriting introduces intermediate terms with fresh variables / learnable patterns.

The deterministic matcher and the phase-1 proof decoder (both for the variable-substitution category) are complete and archived. The matcher gives the competition-track solution at 59.8% directly-proved coverage with zero false positives. The decoder is the research-track companion — established that the JEPA v1.0 embeddings preserve the proof witness (96.32% σ-exact-match on both-novel test, vs 5.6% random-embedding control). All prior plans are archived in [archive/](archive/).

## v1.0 embeddings at a glance

- **Spec:** 4694 × 768 float32, L2-normalized.
- **Source:** `skip_h128_seed42_diag` ep9 (h=128 GATv2 encoder with parameter-free skip connection from the pooled rep). Source checkpoint travels with the artifact at `outputs/embeddings/v1.0/source_checkpoint.pt`.
- **JEPA distance bal_acc on the matched 10% pair holdout:** 0.9841 (0.9840 on the full 22M corpus).
- **Frozen-embedding MLP-head bal_acc on the same holdout:** 0.9996, recovering 91.37% of the "always-wrong-trained" subset.

## Proof-corpus survey at a glance

- **TRUE corpus (10,676 explicit-implication proofs):**
  - **62.5% arg-permutation** (`λ x y => h x y x` style — predict a short permutation tuple)
  - 15.3% rewrite chains (short tactic sequences)
  - 14.1% Vampire-superposition machine output (delegate to upstream tool)
  - 7.5% Egg-combinator machine output (delegate to upstream tool)
  - 0.5% genuinely complex / corner cases
- **FALSE corpus:** Stage 2 judge accepts only `Fin n` matrix counterexamples, so FALSE generation is brute-force search over small `Fin n` magmas.
- **Vocab:** 119 tokens cover 95% of TRUE proof usage. Median proof length is 10 tokens.
- **Architecture recommendation** (three-track ensemble + FALSE search) in [`outputs/analyses/proof_survey/README.md`](outputs/analyses/proof_survey/README.md).

See [CLAUDE.md](CLAUDE.md) for the full project-state guide and [`CHANGELOG.md`](CHANGELOG.md) for the history of deviations from the various plans.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

This project depends on raw data from the Equational Theories Project and (eventually) computed invariants from a sibling extraction project. Symlink them under `data/raw/`:

```bash
mkdir -p data/raw
ln -s /path/to/equational_theories/data                  data/raw/etp
ln -s /path/to/equational_theories_data/assembly/output  data/raw/invariants
```

Files used:

- `data/raw/etp/equations.txt` — 4,694 equations (line N = equation ID N)
- `data/raw/etp_export/export_raw_implications_5_9_2026.csv` — authoritative ground-truth label for every off-diagonal pair (4694×4694 int8 matrix, codes ±3/±4 mapping to TRUE/FALSE; see CHANGELOG 2026-05-09)
- `data/raw/etp/2025-08-11-vampire.json.gz` — *legacy* Vampire-derived labels (left 1,062 pairs unresolved); no longer the source of truth
- sibling project `equational_theories_data/invariants/` — per-equation invariants (spine type, LHS structure, variable count, magma signature, cohomology summaries). Used by the JEPA's auxiliary prediction heads during training, and by downstream analyses (e.g., failure-overlap's spine breakdown). Built into `data/processed/equation_invariants.parquet` via `scripts/build_equation_invariants.py`.

Do not write to `data/raw/`. All processing goes to `data/processed/` (caches) or `data/splits/` (committed split indices).

### Build the processed caches

Once the `data/raw/` symlinks are in place:

```bash
python scripts/parse_equations.py             # data/processed/equation_graphs.pt   (~6 MB)
python scripts/build_pair_table.py            # data/processed/implication_pairs.parquet  (~35 MB)
python scripts/build_equation_invariants.py   # data/processed/equation_invariants.parquet  (per-equation invariants table)
```

All three are deterministic and re-runnable; their outputs are gitignored. See the script docstrings for the schema and CLI flags.

### Regenerate the v1.0 embeddings from the source checkpoint

```bash
python scripts/save_embeddings.py \
    outputs/embeddings/v1.0/source_checkpoint.pt \
    --output-dir outputs/embeddings/v1.0 --force
```

See [`outputs/embeddings/v1.0/MODEL_CARD.md`](outputs/embeddings/v1.0/MODEL_CARD.md) for the full reproduction recipe and validation numbers.

## Layout

See [archive/jepa_engineering_plan.md](archive/jepa_engineering_plan.md) §"File Structure".
