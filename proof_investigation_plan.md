# Proof Investigation: Intermediate Term Analysis on Hard TRUE Proofs

**Phases 1–6.** This is a fresh investigation, not a continuation of Coverage Measurement (which closed at Phase 7 with its bounding synthesis). Proof Investigation uses Phase 6 of Coverage Measurement as input and asks a different question: **what's in the proofs the deterministic stack misses, and what would a successor method need to produce?**

## Methodology resolutions (locked in 2026-05-20)

1. **Pool — all 722 eligible pairs.** 392 Vampire + 319 Egg + 11 complex. Complex is descriptive-only (too thin for entropy). The hypothesis tests stratify per-category; the imbalance doesn't break them. Original success criterion of "≥1,000 hard pairs" → "all eligible hard pairs, parsing failures aside."

2. **Phase 1 verification of intermediate visibility.** The proof-survey categories were assigned by Lean-source structure, so a "Vampire-derived" label means the Lean source has explicit clauses rather than `simp only`. Concretely check at exploration: do `e_vampire_superposition` / `e_egg_combinator` / `e_complex_construction` Lean sources have explicit intermediates (in `have`, `calc`, named clauses)? Spot-check one pair per category.

3. **Partition handling.** If hidden-intermediate proofs surface, partition the sample into "visible" (gets the higher-variable test) and "hidden" (descriptive-only). If hidden-intermediate exceeds ~30% in any category, the test's reach is meaningfully bounded — call it out in the synthesis.

4. **Lean-vs-original framing.** The SAIR deliverable is Lean-validated proofs, so Lean-visible intermediates are the operational unit. "Does the existing Lean proof use intermediate terms with more variables than the goal?" maps to "does a successor method need to introduce fresh variables?" That's the right framing — we don't need the raw Vampire output.

5. **Project-local paths.** `archive/coverage_measurement_plan.md` for context. Phase 6 inputs at `data/processed/coverage_measurement/` and `data/processed/proof_corpus/true_corpus.parquet`. New outputs in `outputs/analyses/proof_investigation/` and `data/processed/proof_investigation/`.

6. **Per-E1 entropy at extraction-time.** ~722 hard pairs spread over ~543 within-ceiling E1s → mean ~1.3/E1. Most E1s won't have ≥10 hard proofs for the entropy test. At Phase 4 (bulk extraction), surface the per-E1 hard-proof count distribution. Phase 5 then restricts the entropy test to qualifying E1s and reports category-level entropy as a coarser proxy.

## Context

Coverage measurement Phases 1–6 mapped the deterministic stack's reach: 6.54% of GT TRUE combined, with the engine contributing 0.107pp (1.6% of combined). Phase 6 identified the structural shape of the residual:

- Engine reaches substitution proofs at 100%, rewrite-chain at 62% within-ceiling.
- Engine reaches Vampire-labeled at 2.29%, Egg at 0.54%, complex at 0%.
- Frontier_died dominates 85.74% of within-ceiling failures.
- Solve rate anti-correlates with row density (-0.371); engine catches sparse rows well, dense rows poorly.

Dense-row failures concentrate in proof categories that originally required introducing intermediate terms absent from E2.lhs and E2.rhs. The safe engine doesn't have this capability. Phase 8 studies what those intermediate additions actually look like in the original proofs — testing the higher-variable hypothesis directly and characterizing what a successor method would need to produce.

## Goals

Three hypothesis tests:

1. **Higher-variable hypothesis**: do intermediate terms use more variables than E2.lhs ∪ E2.rhs? Fraction of dense-row, hard-labeled proofs with at least one intermediate where max_vars exceeds max_vars(E2)?

2. **Pattern learnability**: within a single E1's hard proofs, do intermediates have low-entropy structural patterns? Low entropy means a learnable representation of "what to introduce."

3. **Density connection**: does intermediate complexity (var count, depth) scale with row density at the E1 level?

Plus one descriptive deliverable: per-pair structured records of all intermediate terms, suitable for downstream analysis or training data.

## Scope

**Pair selection**: dense-row (density ≥ 51), TRUE-labeled, in the within-ceiling residual, frontier_died under the safe engine. Stratify by proof category — Vampire, Egg, complex — and sample ~300 pairs per category for ~1,000 total.

**Out of scope**: variable-disjoint pairs (structural ceiling, no rewriting proof exists), sparse-row pairs (engine handles them), FALSE/refutation proofs.

## Phases

### Phase 8.0 — Repository exploration

Goal: understand what proof data exists in `~/git/equational_theories/` and in what formats.

Steps:
1. Map directory structure: `find ~/git/equational_theories -type d | head -50`
2. Locate proof storage per category. Look for Lean tactic scripts and raw external-prover outputs (TSTP, JSON, S-expr, etc.):
   ```
   find ~/git/equational_theories -name '*.lean' | head
   find ~/git/equational_theories \( -name '*.p' -o -name '*.tstp' -o -name '*.json' \) | head
   grep -r 'vampire' ~/git/equational_theories --include='*.lean' -l | head
   ```
3. Identify how Phase 6's proof-category labels map to specific files. The labels are sourced from somewhere in the repo or in equational_theories_data/ — find the canonical mapping.
4. Document findings in `phase8/exploration_notes.md`. Cover: file formats per category, directory paths, pair-identifier format, label-source location.

**Checkpoint**: review exploration_notes.md before proceeding. Confirm at least one pair's proof is locatable per target category (Vampire, Egg, complex).

### Phase 8.1 — Single-pair end-to-end trace

Goal: validate the extraction approach on one pair per category before scaling.

Steps:
1. Pick one hard pair per category from Phase 6's sample, ideally from dense-row E1s.
2. Manually trace each proof: identify every intermediate term, its variables, its derivation source.
3. Write `phase8/single_pair_traces.md` with pretty-printed term structures (AST or s-expr).
4. Identify the parsing challenge per format. What's recoverable, what's lost in translation?

**Checkpoint**: review traces. If a format loses critical information (e.g., Lean translations that flatten intermediates), flag and decide whether to source original prover output.

### Phase 8.2 — Parser implementation

Goal: per-format parsers producing a uniform intermediate-term record.

Target schema (one row per intermediate term):
```
pair_id, e1_id, e2_id, proof_category, intermediate_idx,
term_str, var_set, var_count, term_depth, operator_count,
fresh_vars (vars not in E2), derivation_inference, parent_indices
```

Steps:
1. Implement parser per identified format. Start with the accessible ones (Lean tactic regex; TSTP via `tptp_parser` if originals are retained).
2. Unit-test each on Phase 8.1's traces.
3. Log parsing failures with reasons. Track skip rate per format.

**Checkpoint**: ≥80% parse success per format before bulk extraction.

### Phase 8.3 — Bulk extraction

Goal: parse the ~1,000 selected pairs.

Steps:
1. Build pair selection list from Phase 6 outputs (within-ceiling residual + density filter + frontier_died + category stratification).
2. Run parsers, parallelizing per-pair where possible.
3. Write `phase8/intermediates.parquet` (one row per intermediate term).
4. Write `phase8/per_pair_summary.parquet` (one row per pair with aggregates).

Sanity check: spot-check 20 random rows against original proof files. Distribution of intermediates per pair, format coverage, fresh-variable rates.

### Phase 8.4 — Aggregation and hypothesis tests

1. **Higher-variable test**:
   - Per pair: `var_lift = max_vars(intermediates) - max_vars(E2.lhs ∪ E2.rhs)`
   - Distribution of var_lift stratified by category and density bucket.
   - Headline: fraction with var_lift ≥ 1, mean conditional on positive.

2. **Pattern learnability test**:
   - For each E1 with ≥10 hard proofs, compute the distribution of intermediate "shapes" (operator skeleton with variables abstracted to positions).
   - Shannon entropy of shape distribution per E1.
   - Compare to sparse-row E1 baseline (sparse-row E1 proofs' intermediates as control).

3. **Density connection test**:
   - At the E1 level, correlate row density with mean intermediate complexity (var count, depth, operator count).
   - Spearman and Pearson reported.

Output: `phase8/aggregates.md` with headline numbers, supporting tables, distribution plots (PNGs in `phase8/plots/`).

### Phase 8.5 — Writeup

Output: `phase8/synthesis.md`.

Structure:
1. Three hypothesis tests' outcomes — headline numbers up front, one paragraph each.
2. Higher-variable hypothesis verdict: supported / partially / not.
3. Learnability verdict: exists / doesn't, with E1-level granularity if patterns vary.
4. Implications for dense-row coverage: candidate method directions, if any.
5. Limitations: sample size, format-specific extraction biases, anything that affects the conclusions.

## Style notes

- Direct prose. Avoid "not X, but Y" / "X isn't Y, it's Z" patterns.
- Strong claims stand without rhetorical scaffolding.
- Tables and plots over wordy explanations where data supports it.
- Honest about limitations; resist over-conclusion.

## Open questions to surface during execution

- Proof-format variability per category beyond expectation → revisit extraction scope before doing the full 1,000.
- var_lift ≥ 1 fraction below ~20% → flag early; the framing changes.
- E1s with very low intermediate-shape entropy → candidates for follow-on prototype, surface specifically with the E1 IDs and example intermediates.
- Lean translations that flatten or normalize intermediates (losing the original step structure) → surface format-specific biases that affect interpretation.

## Inputs available

- `/home/gavin/git/equational_theories/` — ETP repo with proofs.
- `/home/gavin/git/equational_theories_data/` — project repo with corpus, Phase 6 outputs, residual partitions.
- Phase 6 outputs: pair selections, density buckets, category labels, within-ceiling residual.
- Coverage measurement plan v2 at `/mnt/user-data/outputs/coverage_measurement_plan.md` for context.

## Success criteria

- ≥1,000 hard pairs with extracted intermediate-term records.
- Three hypothesis tests reported with headline numbers and stratified breakdowns.
- Synthesis writeup ready for inclusion in Phase 7 synthesis or as a standalone follow-on.
- One concrete recommendation for downstream method development based on findings (or explicit "no learnable signal found" if the data goes that way).
