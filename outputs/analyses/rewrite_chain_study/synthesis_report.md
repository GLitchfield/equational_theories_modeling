# Rewrite-Chain Proof Study — Synthesis Report

**Status:** complete (2026-05-17).
**Corpus:** 1,636 `c_rewrite_chain` proofs from `data/processed/proof_corpus/normalized_proofs.parquet`, filtered against `true_corpus.parquet`'s category mask on `(e1, e2)`. Bodies are Lean 4.20.1 ETP proofs.
**Reproducer scripts:** `scripts/build_rewrite_chain_dataset.py`, `scripts/survey_rewrite_chain.py`, `scripts/run_rewrite_engine.py`, `scripts/analyze_rewrite_traces.py`, `scripts/run_multimodality.py`.

## Headline

The c_rewrite_chain category is **mostly deterministically search-solvable, with a hard 17% tail**.

| Measure | Number |
|---|---|
| Deterministic search closes (d_max=6, both sides) | **1,360 / 1,636 = 83.13%** |
| Proofs that stay purely within `{h, ← h}` rule set | **1,631 / 1,636 = 99.7%** |
| Shortest proofs that are unique (one chain at min depth) | **434 / 499 sampled = 87%** |

The first two numbers — the coverage curve from Task 4 and the within-{E1} rule-set fraction from Task 2 — are the decision drivers per the plan.

## The two decision-driving numbers

### Coverage vs depth (Task 4)

Bidirectional iterative-deepening BFS over E1 (forward + reverse), term-size cap `max(50, 5×max(|lhs2|, |rhs2|))`, per-pair states budget 100K, per-side `d_max=6`. All 1,636 pairs in 619s (~3 pairs/s amortized).

| total_depth ≤ | closed | fraction |
|---|---|---|
| 1 | 48 | 2.93% |
| 2 | 871 | **53.24%** |
| 3 | 1,102 | 67.36% |
| 4 | 1,192 | 72.86% |
| 5 | 1,270 | 77.63% |
| 6 | 1,306 | **79.83%** |
| 8 | 1,336 | 81.66% |
| 10 | 1,356 | 82.89% |
| 12 (max) | 1,360 | **83.13%** |
| not closed | 276 | 16.87% |

The shape: **>50% of the corpus closes at total_depth ≤ 2**, the curve flattens fast after depth 5. Cost per pair on the solved part is cheap (median 2 states / 0.03 ms, p95 232 / 2.1 ms); cost grows by orders of magnitude past depth 8 (D=10: median 33K states / 894 ms).

### Rule-set fraction (Task 2)

Of all `rw`/`rwa`/`nth_rewrite` arguments tokenized across the 1,636 proofs (9,770 tokens total):

- `h` (forward direction, E1): 4,573 (46.8%)
- `← h` (reverse direction): 5,182 (53.0%)
- other names: 15 (0.15%)

Per-proof: **1,631 / 1,636 = 99.7% stay purely within {h, ← h}**. Only 5 proofs reach for outside lemmas (`idem`, `op_idem`, prior `Equation*_implies_*` theorems, etc).

This is the strongest result of the study: the rewrite-chain category uses essentially nothing outside the single rule E1. The branching factor of any search is bounded by `|rewrites(t, E1) + rewrites(t, ← E1)|`, which is small on the typical ETP term (size ~10).

## Characterization (Tasks 1, 3, 5)

### Tactic structure (Task 1)

Chain length distribution (number of rewrite-tactic steps per proof):

| stat | value |
|---|---|
| min | 0 |
| median | 2 |
| p75 | 8 |
| p90 | 13 |
| p95 | 24 |
| p99 | 46 |
| max | 80 |

Proof token length: median 44 tokens, p95 259. Most proofs are short.

15.3% (250 proofs) use structural tactics (`calc`, `have`, nested blocks) — i.e., not a single straight-line `rw`-sequence. The other 84.7% are flat rewrite chains.

Tactic mix (per-proof presence): `intro` 91.9%, `repeat` 85.9%, `rw` 83.2%, `try` 55.6%, `nth_rewrite` 25.5%, `assumption` 21.5%, `apply` 16.3%, `have` 14.9%. The vast majority is `intro/repeat/try` boilerplate around the core `rw` work.

### Intermediate-state structure (Task 3)

From Phase 6 analysis of the engine traces:

- **91.9% of solved proofs (1,250 / 1,360) require both sides to move** — these are genuine meet-in-the-middle proofs, not one-sided rewriting toward the other endpoint. Validates the bidirectional algorithm choice (not just a speed-up).
- **Term-size trajectory stays bounded**: on a 500-pair sample, max term size along the proof path is median 9, p95 19, max 43 — well under the default size cap (50). Only 37.6% of proofs visit a term LARGER than the larger endpoint, and the typical blowup is 6 nodes.
- "Intermediate states are equations" holds by construction — the engine tracks `(current_lhs, current_rhs)` as a pair, so every visited state is itself an equation.

### Multimodality (Task 5)

For 499 closed pairs (stratified over total_depth ∈ [1, 6]; states budget 200K), enumerate ALL distinct chains of length = shortest-path depth:

| n_chains | count | fraction |
|---|---|---|
| exactly 1 | 434 | **86.97%** |
| 2-3 | 40 | 8.02% |
| 4-9 | 6 | 1.20% |
| 10-99 | 1 | 0.20% |
| 100+ | 0 | 0.00% |
| budget hit | 18 | 3.61% |

n_chains distribution (completed): median 1, p75 1, p95 2, max 12.

The category is **essentially unimodal**. A learned model would have a well-defined target — imitation learning is viable; a verifier-centric objective is not forced by multimodality.

## The hard 17% tail (276 pairs)

The 276 pairs the deterministic engine does NOT close at d_max=6 split by failure mode:

| failure mode | count | reading |
|---|---|---|
| states budget exhausted (~100K) | 109 (40%) | combinatorial blow-up within d_max=6 — model heuristic could prune |
| frontier died early (<100 states) | 116 (42%) | need d_max > 6 OR rules outside {E1} |
| in between (exhausted depth without budget) | 51 (18%) | need d_max > 6 |

The "frontier died" cluster overlaps strongly with the 5 proofs Phase 4 flagged as using outside lemmas. The rest of the "died early" tail is likely d_max-limited.

The "states budget" cluster is where neural guidance most clearly helps: branching factor is the problem, not depth.

## Prover-architecture recommendation

The three options from the plan:

1. **Deterministic search.** Engine alone, no model.
2. **Neural-guided search.** Engine handles the easy cases; model targets the tail as a heuristic.
3. **Closed-loop generation.** Model produces proofs end-to-end; engine is a verifier or absent.

The numbers favor **option 2: neural-guided search** for this category, with these qualifications:

- Option 1 is already correct for ~80% of the corpus (the depth-≤6 cheap region). For a project that can accept 80% coverage on this category, deterministic search alone is the right baseline — no model needed. The matcher story repeats: a category framed as "needs a learned model" in the survey turns out to be mostly mechanical.
- Option 2 is the right answer for the remaining 17% tail. The arithmetic:
  - Multimodality is low (87% unimodal) → imitation training has a clean target.
  - Rule set is tight (99.7% within {E1}) → action space is tiny.
  - Failure modes are concrete (states-budget blow-up + d_max exhaustion) → the model's job is well-defined: rank candidate next-steps to prune branching.
  - Cost asymmetry: easy pairs are 0.03 ms each; tail pairs are 100s-1000s of ms. Model overhead per call is plausibly cheap relative to what it saves.
- Option 3 is overkill. The engine + model is the same architecture but reversed: model proposes, engine verifies. With multimodality this low and the rule set this tight, generation-first wastes compute on what search already nails.

The natural next phase, if a prover is built: a **neural-guided rewrite engine** where a learned policy ranks candidate rewrites at each BFS step. Imitation training data comes from the 1,360 search traces already produced. The model is graded on whether it closes the 276 hard-tail proofs that pure search misses — that 17% is the move-the-needle metric, not the easy 83%.

Two cheaper "free" improvements to try first, before building the model:
- **Bump d_max from 6 to 8 or 10.** At least 51 of 276 tail pairs are pure-depth-limited; some of the 116 "died early" might also be. The per-pair cost at d_max=8 is acceptable for the tail given the easy pairs cap out at d_max=2-3.
- **Smaller size cap on the explosion tail.** The 109 budget-exhausted pairs likely include cases where the BFS frontier blows up via term-size growth before finding the goal. Tighter caps tested as a sweep.

Either of these could chip 5-10pp off the tail before the model is needed at all.

## Limits of this study

- **Within-corpus only.** The engine was run on c_rewrite_chain proofs of true implications. The 11M pair non-implication negatives in the full ETP set were not tested — the engine doesn't have a refutation primitive, only a proof one.
- **Lean version mismatch ignored.** ETP bodies are Lean 4.20.1; the rewrite engine reimplements first-order matching, so the version mismatch with the Stage-2 judge (Lean 4.30.0-rc2) is irrelevant for this study but matters for any prover that produces Lean source.
- **No semantic equivalence.** The engine compares terms structurally. Two terms that are α-equivalent but written with different variable names would not be detected as the same state. The matcher's variable-renaming canonicalization is not applied here. This could slightly inflate the hard tail, though the effect should be small since the rule set is fixed.
- **Multimodality measured only at minimum depth.** Paths of length D* + 1, D* + 2 etc. were not enumerated. The "essentially unimodal" claim refers to shortest proofs; longer proofs would multiply.
- **Forward-only enumeration was wrong on first attempt.** The `enumerate_chains` function was first written as forward-only DFS, which undercounts when E1's RHS has variables not in its LHS (e.g., `x = x ◇ y` — forward rewriting introduces the rule variable `y` literally rather than instantiating to the subject variable needed). The reported numbers come from the bidirectional-all-parents BFS reimplementation; the forward-only version is in the git history as a learning artifact.

## Artifacts

| Phase | Script | Output |
|---|---|---|
| 1 | `scripts/build_rewrite_chain_dataset.py` | `data/processed/proof_corpus/rewrite_chain_dataset.parquet` |
| 3-4 | `scripts/survey_rewrite_chain.py` | `outputs/analyses/rewrite_chain_study/tactic_structure.json`, `ruleset_analysis.json` |
| 5 | `scripts/run_rewrite_engine.py` | `outputs/analyses/rewrite_chain_study/coverage.json`, `data/processed/proof_corpus/rewrite_engine_traces.parquet` |
| 6 | `scripts/analyze_rewrite_traces.py` | `outputs/analyses/rewrite_chain_study/task3_intermediate_states.json` |
| 7 | `scripts/run_multimodality.py` | `outputs/analyses/rewrite_chain_study/task5_multimodality.json` |
| 8 | (this file) | `outputs/analyses/rewrite_chain_study/synthesis_report.md` |

Engine: `src/rewrite/engine.py` (search, find_path, enumerate_chains) + `src/rewrite/matching.py` (first-order matching, all_rewrites, term_size, substitute). 36 tests in `tests/test_rewrite_engine.py` + `tests/test_rewrite_matching.py`, all green.
