# Coverage Measurement Plan

**Purpose:** Measure the combined TRUE-side coverage of the deterministic symbolic stack (matcher + rewrite engine + completion) over the 22M-pair corpus before committing to a proof-generator build. The number gates the build and sizes the opportunity for embeddings-as-search-heuristic.

**Status:** v2 with methodology resolutions locked in (2026-05-19).

## Methodology resolutions (locked in 2026-05-19)

After scrutiny against the rewrite engine's actual semantics (`src/rewrite/matching.py::substitute` leaves unbound rule vars as literal names), v2 incorporated the structural-ceiling, weighted-aggregation, and stratification fixes. These additional choices are now resolved:

1. **Sample sizing: full pipeline at N=30 first, then scale to N=50 only if needed.** Run all phases end-to-end at N=30 (~77K pairs); get the headline. Scale to N=50 only if results are noisy near a decision-table boundary.

2. **Pool size: `os.cpu_count() // 2`.** Leaves headroom; avoids hyperthreading thrash on CPU-bound work. Probed at runtime, not hard-coded.

3. **Engine scope: residual only.** Matcher subsumption verified separately by the ~100-pair spot-check at `d_max=2` in Phase 2; no need to re-run engine on completion-solved pairs.

4. **Wall timeout: 30s default; 60s sub-experiment at `d_max=8` on a 100-pair subset.** If 60s vs 30s changes results by < 1pp at d=8, stay at 30s for the main run.

5. **Depth sweep: `{4, 6, 8}` as planned.** Add `{10}` only if the curve at d=8 hasn't plateaued.

6. **Soundness target: zero TRUE-wrong predictions.** The engine should never claim solved on a GT FALSE pair (it's sound by construction). Any disagreement is an implementation bug, not a method failure.

## Goal

Produce three numbers from one focused measurement run:

1. **Combined TRUE coverage** — what fraction of GT TRUE the assembled deterministic methods cover, as a union, expressed both absolutely and as a fraction of the structural ceiling (see below).
2. **Residual size and structure** — how much TRUE remains uncovered, partitioned into the structural-impossibility segment (variable-disjoint E1s) and the within-ceiling residual.
3. **Engine failure-mode breakdown** — within the within-ceiling residual, the rewrite engine's failure modes (no valid rule, depth exhausted, wall timeout, frontier died). This is what a learned heuristic would have to address.

The completion exploration mapped completion's reach. This plan extends the mapping to the rewrite engine and unions the three deterministic methods. No new methods are built — existing components only.

## Scope

**In:** TRUE-side coverage of matcher + rewrite engine + completion over the 22M-pair labeled corpus (order-≤4 ETP equations).

**Out:**
- FALSE-side beyond completion. Completion already covers 17.82% of GT FALSE soundly; further FALSE coverage requires finite-model search (Mace4-style, standard technique), to be sized if and when the proof-generator build proceeds. FALSE has finite-model search as a natural extension that does *not* inherit the rewriting ceiling described below; TRUE does not have an equivalent standard extension.
- Order-5. The symbolic methods work on order-5 natively but the corpus here is order-≤4.
- Lean emission. This measurement asks whether methods *decide* pairs, not whether they *emit* Lean proofs.

## What's already in hand

| Component | Status |
|---|---|
| Completion per-pair predictions (plain + ordered) | Measured in Phase 3 / Phase 4. Per-pair parquets on disk. |
| Matcher coverage | Measured (49,586 / 8.17M GT TRUE). Parquet exists. |
| Rewrite engine | Implemented (`src/rewrite/engine.py`), validated against the rewrite-chain corpus (83% at `d_max=6/side`). Never run over the full corpus. |
| Soundness-check discipline | Phase 3 pattern: predicted-vs-ground-truth, zero disagreements is the target. Reusable. |

## Structural ceiling

The rewrite engine inherits Phase 0's variable-disjoint ceiling cleanly, and acknowledging this up front sharpens both the methodology and the decision rule.

The engine uses first-order matching with σ taken at match-time only; rule variables not bound by σ are substituted literally. For E1 = `s = t` where `Vars(t) ⊄ Vars(s)`, applying the rule `s → t` leaks rule variable names into the rewritten term — they appear as free symbols downstream steps treat as constants. That's a leakage bug in the rule application, not a Birkhoff completeness gap; Birkhoff would handle it via narrowing or paramodulation, neither of which the engine implements. So variable-disjoint E1s contribute nothing as rule sources, in either direction.

From Phase 0: 1,656 equations (35.28%) are variable-disjoint. From Phase 4b's row-density data: mean ~2,662 GT TRUE consequents per variable-disjoint E1. Therefore ~1,656 × 2,662 ≈ **4.4M GT TRUE pairs (≈54% of GT TRUE) live in variable-disjoint E1 rows and are structurally unreachable by the rewrite engine and completion.**

The matcher catches a small slice within this region via direct var-to-var substitution-instance matches; the magnitude is bounded above by its total reach (~0.6% of GT TRUE) and the variable-disjoint subset of that is necessarily smaller. To be measured rather than assumed.

**Combined-coverage ceiling: ≈46% of GT TRUE plus the matcher's variable-disjoint addendum.**

This means: the embeddings-as-heuristic story applies *within the rewritable ≈46% half*. Within that half it can prune branching-blowup or extend depth productively. The remaining ≈54% has no production-ready prover in this project — the matcher's substitution slice aside — and extending coverage there requires either superposition (substantial reimplementation; out of scope) or a research extension of learned proving beyond variable substitution (also out of scope for this measurement). The JEPA distance metric can predict TRUE/FALSE on the variable-disjoint set, but predicting is not proving for a generator that must emit Lean.

## The missing measurement

The rewrite engine has never been run as a decision attempt over arbitrary `(E1, E2)` pairs at scale. The 83% figure was specifically the rewrite-chain proof category at `d_max=6/side`. Over the full GT TRUE corpus — dominated by transitive consequences — its coverage is unknown. The empirical question is what fraction of the rewritable half (the ≈46% of GT TRUE in non-variable-disjoint E1 rows) the engine actually covers within a sane budget, and what failure modes dominate the within-ceiling residual.

## Methodology

### Pair set

The TRUE residual relative to completion:

```
residual = {(E1, E2) ∈ GT TRUE : E1 ∉ ordered_completed ∪ plain_only_completed}
```

`ordered_completed` is the 2,129 from Phase 4. `plain_only_completed` is the 9 equations plain KB completed but ordered KB lost — negligible magnitude but kept in the union for cleanliness. Roughly 2,556 equations × variable row sizes ≈ 5–7M GT TRUE pairs in the residual.

The residual splits naturally for analysis:
- **Variable-disjoint segment** (E1 ∈ Phase 0's 1,656 set, intersected with the residual): structurally unreachable by the engine; measured to confirm.
- **Within-ceiling segment** (E1 not variable-disjoint, not completed): the real target of the engine's measurement.

### Stratified sample

Per non-completed E1, sample `N` pairs uniformly from E1's GT TRUE row (cap at row size for sparse rows).

- Stratify by E1 to ensure structural diversity (equation shapes, sizes, classes).
- Initial `N = 50` per E1 → ~128K total. Adjustable based on engine wall-time.
- Random seed pinned for reproducibility.
- Each sampled pair carries E1's row density as a column, for the weighted aggregation below.

### Engine harness

Wrap `src/rewrite/engine.py` for batch use:

- Input per call: `(eq1_id, eq2_id, d_max, wall_seconds)`.
- Output per pair: `(solved, depth_at_solve, wall_time, terminate_reason)` where:
  ```
  terminate_reason ∈ {
    solved,
    no_valid_rule_from_e1,   # E1 produces no rule applicable in either direction
                              # (variable-disjoint or otherwise rule-incapable)
    depth_exhausted,          # search ran to d_max without finding the path
    wall_timeout,             # SIGALRM fired
    frontier_died             # bidirectional frontier collapsed (no new states)
  }
  ```
- `no_valid_rule_from_e1` is a primary category, not lumped into `frontier_died`, because variable-disjoint E1s will fail immediately on rule construction and the failure mode is structural, not depth-dependent.
- SIGALRM-based per-pair wall timeout (30s default — matches Phase 4 discipline).
- Parallelizable: `multiprocessing.Pool` over pairs since each call is pure.

### Depth sweep

Run the sample at three depth budgets to characterize the coverage-depth curve, not just a single point:

- `d_max ∈ {4, 6, 8}` baseline. Add 10 if the curve hasn't plateaued at 8.
- The curve's *shape* matters as much as the peak: a steep curve to high `d_max` says the within-ceiling residual is "deep but tractable" (depth extension helps; learned heuristic targets pruning to reach further depth); a plateau says "stalls" (depth doesn't help, learned heuristic needs structurally different paths).
- Variable-disjoint pairs short-circuit on `no_valid_rule_from_e1` regardless of `d_max`, so the depth sweep's wall-time cost is borne mostly by the within-ceiling segment.

### Row-density-weighted aggregation

Per-E1 N=50 sampling does not extrapolate to pair-level coverage without weighting. For headline coverage numbers:

```
pair_level_coverage = Σ_E1 (sample_success_rate_E1 × row_density_E1) / total_GT_TRUE
```

An E1 with 5 TRUE consequents and an E1 with 4,000 contribute differently to the pair-level headline. The sample gives the success rate per E1; the row density gives the weight.

Report two numbers:
- **Sample-level coverage** (unweighted, average across sampled E1s) — useful for interpretability per equation.
- **Pair-level coverage** (row-density-weighted) — the headline number, comparable to the existing completion / matcher figures.

## Phases

### Phase 1 — Pair set and sampling (~½ day)

- Filter against existing completion parquets to build the GT TRUE residual set.
- Annotate each E1 with its variable-disjoint flag (from Phase 0 outputs) and row density.
- Stratified sample, `N = 50` per non-completed E1, fixed seed.
- Outputs: `data/processed/coverage_measurement/residual.parquet`, `sample.parquet`.

### Phase 2 — Engine harness and preconditions (~½ day)

Preconditions before harness wrap:
- **Engine purity audit.** Inspect `src/rewrite/engine.py` and `src/rewrite/matching.py` for module-level state (caches, globals, mutable defaults) that would race under multiprocessing. Refactor if needed.
- **Matcher subsumption spot-check.** Confirm empirically (on ~100 matcher-solved pairs) that the engine at `d_max=2` solves all of them. If not, the matcher's contribution is not subsumed and Phase 5's join logic needs to treat the methods as fully independent.

Then wrap:
- `engine.py` for batch use per the interface above.
- Verify on ~100 c_rewrite_chain pairs the study solved at `d_max=6` before running at scale.
- Implement SIGALRM, multiprocessing, parquet output.

### Phase 3 — Depth sweep (½–1 day, mostly wall time)

- Run at `d_max ∈ {4, 6, 8}` over the sample.
- One parquet per setting: `engine_results_d{4,6,8}.parquet`.
- Expected runtime per setting at ~16 cores: a few hours. Total: half a day to a day.

### Phase 4 — Soundness check

For any pair the engine claims solved, verify the pair is GT TRUE. Target: zero disagreements (the engine is sound by construction; any disagreement is an implementation bug). Same discipline as Phase 3 of the completion exploration.

Spot-check: for a sample of "solved" pairs, replay the reported path and confirm it executes correctly to a common normal form.

Output: `soundness_check.json` (zero disagreements expected).

### Phase 5 — Combined coverage analysis (~½ day)

Join engine results with existing completion (Phase 3 / Phase 4) and matcher parquets. Compute:

- Per-method coverage: matcher-only, engine-only, completion-only.
- Pairwise and three-way overlaps.
- Combined union as a fraction of GT TRUE — both sample-level and **pair-level (row-density-weighted)**.
- Combined coverage as a fraction of the ceiling (≈46% + matcher addendum). This is the more meaningful denominator for evaluating how much of the *reachable* half the deterministic stack captures.
- Stratification cuts:
  - By E1 properties: row density, variable-disjoint flag, equivalence class size.
  - **By proof-type taxonomy from the survey (substitution / rewrite-chain / Vampire / Egg).** The Vampire-derived bucket (14% of TRUE) is exactly where first-order-only rewriting bites hardest; its coverage rate is a sharp diagnostic for the engine's reach.

Outputs: `combined_coverage.json`, `per_pair_method_coverage.parquet`.

### Phase 6 — Residual characterization (~½ day)

Partition the residual:

- **Structural-impossibility segment.** Pairs where E1 is variable-disjoint. Expected to dominate by `no_valid_rule_from_e1` in the engine outputs. The matcher's coverage of this segment is the only deterministic-method foothold; report it.
- **Within-ceiling residual.** Pairs where E1 is not variable-disjoint and no method solved. Break down by:
  - Engine `terminate_reason` (depth_exhausted vs wall_timeout vs frontier_died).
  - Proof-type taxonomy (Vampire-derived concentration is the diagnostic).
  - Stratification by E1 — which non-variable-disjoint equations dominate?
- Coverage-depth curve from the sweep.

Outputs: `residual_characterization.json`, coverage-vs-depth plot.

### Phase 7 — Synthesis and decision (~½ day)

Short report (`coverage_measurement_synthesis.md`) covering:

- Headline combined-coverage numbers: pair-level absolute and fraction-of-ceiling.
- Residual partition: structural-impossibility vs within-ceiling, with sizes.
- Engine failure-mode breakdown within the within-ceiling residual.
- Decision implications for the proof-generator build and the embeddings-as-heuristic opportunity.

**Effort estimate end-to-end: 3–4 days, dominated by Phase 3's engine runs.**

## Decision use

The measurement does not commit to building the proof generator — it gates that decision and sizes the embeddings' opportunity, against the ≈46% structural ceiling rather than 100%.

| Pair-level combined coverage | Implication |
|---|---|
| < 10% of GT TRUE (< ~22% of ceiling) | Deterministic stack barely improves on completion alone; reconsider the build's TRUE-side value. |
| 10–25% of GT TRUE (~22–54% of ceiling) | Modest fraction of the reachable half. Real residual within the ceiling; embeddings-as-heuristic has substantial opportunity *within the rewritable region*. |
| 25–40% of GT TRUE (~54–87% of ceiling) | Strong capture of the reachable half. Build clearly justified; embeddings' role within the ceiling shrinks as coverage approaches the cap. |
| > 40% of GT TRUE (≥ ~87% of ceiling) | Deterministic stack nearly saturates the rewritable half. Within-ceiling embeddings role is marginal. |

In all cases the variable-disjoint ≈54% remains outside any rewriting-based prover this project has built. That bound is the project's actual TRUE-side limit, not a measurement artifact, and any path past it is a separate research question (superposition, learned proving extensions) — not what building the proof generator answers.

Where the embeddings would land *within the within-ceiling residual* depends on its character:

- **Branching-blowup-dominated** → learned heuristic prunes search; transition model gives multi-step lookahead. Well-scoped opportunity.
- **Stall-dominated** → learned heuristic needs to find structurally different proof paths, not just prune. Harder.
- **Small** → embeddings stay a research asset, not a production component. Not a failure of the embeddings; the project's finding that the rewritable problem is mostly already symbolic, made concrete.

## Open questions for iteration

1. **Sample size N per E1.** 50 is a starting point. Sensitivity check at N=30 worth running first to confirm results stabilize before committing wall time at N=50.
2. **Depth sweep settings.** `{4, 6, 8}` is a coarse curve. Intermediate values worth adding if the curve looks interesting?
3. **Wall timeout per pair.** 30s matches Phase 4. May be too short at `d_max=8`; consider 60s at the deeper setting.
4. **Ceiling re-check.** The ≈46% number is from Phase 4b's row-density estimate; the measurement will produce the exact figure (by direct row-density sum over variable-disjoint E1s in the residual). Report both expected and measured.
5. **Matcher's variable-disjoint catch.** Worth a direct count: of the matcher's 49,586 matches, how many have variable-disjoint E1? That number is the matcher's contribution above the engine's ≈46% ceiling.
