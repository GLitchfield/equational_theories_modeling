# Coverage Measurement — Synthesis Report

**Status:** complete (2026-05-19).
**Corpus:** 4,694 magma equations (ETP order-≤4); 22,028,942 off-diagonal implication pairs from the May 2026 ETP export (8,173,585 TRUE / 13,855,357 FALSE).
**Code:** `src/coverage/safe_search.py` (direction-restricted bidirectional BFS); harness in `scripts/coverage_measurement/`.

## Headline

**Three numbers.**

| number | value | reading |
|---|---|---|
| Structural ceiling for rewriting methods | **46.07% of GT TRUE** (3,765,754 pairs) | GT TRUE pairs in non-variable-disjoint E1 rows. The other 53.93% live in variable-disjoint E1 rows and are outside any first-order rewriting method on this project. |
| Combined deterministic stack / structural ceiling | **14.20%** | The stack captures roughly one-seventh of what first-order rewriting can in principle reach. |
| Combined sound TRUE coverage of GT TRUE | **6.54%** (534,591 pairs) | Absolute. Sound by zero-disagreement check against the 22M ground truth. |

**Architectural distribution of the 6.54%:**

| contributor | n pairs | % of GT TRUE | % of combined |
|---|---|---|---|
| Completion (ordered + plain-only) | 495,801 | 6.066% | **92.8%** |
| Matcher (variable-disjoint addendum) | 30,010 | 0.367% | 5.6% |
| Safe rewrite engine (within-ceiling residual) | 8,780 | 0.107% | 1.6% |
| **Combined** | **534,591** | **6.540%** | — |

Completion is the deterministic stack's TRUE engine. The matcher and rewrite engine are small marginal contributors.

## TRUE/FALSE asymmetry: the build-decision frame

Read across completion exploration and coverage measurement:

| side | combined sound coverage | base of contribution |
|---|---|---|
| TRUE | 6.54% of GT TRUE | mostly completion (93%) |
| FALSE | 17.82% of GT FALSE | only plain KB's 574 fully-completed E1s (sound), 18% of which yield Stage-2-ready finite counter-models |

**FALSE side is 2.7× larger by pair count and sits on validated completion logic that already exists.** It needs no additional prover infrastructure — completion produces the verdict and, for ~18% of the cases, the finite counter-model directly. The build for FALSE is engineering: emit the counter-model into the Stage-2 format. No new algorithms.

**TRUE side concentrates the engineering complexity for less coverage.** Three independent provers (completion + matcher + engine) need to be integrated, each requires its own Lean-emission story, and the variable-disjoint substitution slice (matcher's unique 0.367%) has a categorically different emission path than the rewriting-based slice. That's separate soundness audits on each, separate Lean-formatting, separate corpus management — for an additional 0.5pp over completion alone.

This reframes the proof-generator build: **the TRUE-side investment is small in coverage and large in complexity; the FALSE-side investment is large in coverage and small in complexity.** A build that keeps the FALSE side and treats TRUE as "completion only, accept the 6pp coverage" is a much smaller engineering ask than the full three-prover TRUE-side stack.

## Where each method's reach ends — a structural account

### 1. Variable-disjoint E1s (1,656 equations, 53.93% of GT TRUE)

E1 with `Vars(lhs)` and `Vars(rhs)` incomparable. Examples: `x ◇ y = z ◇ w`. Neither direction satisfies variable inclusion under first-order matching; any rule constructed from such an E1 would leak rule-variable names downstream, producing unsound rewrites. The safe rewrite engine refuses (returns `NO_VALID_RULE_FROM_E1`); the completion exploration's Phase 0 excluded these a priori; ordered completion's per-instance LPO doesn't help either (same variable-inclusion gate).

What does reach this set: the matcher's substitution-instance check, which establishes `E2 = σ(E1)` for some var-to-var σ (sound by Birkhoff substitution, regardless of variable arrangement). That's the matcher's 30,010 unique pairs = 0.367% of GT TRUE — small in size, but reached by nothing else.

For the residual within this 53.93%: nothing in the project handles it. A learned narrowing suggester or a Vampire-style superposition implementation would be required; both are out of scope.

### 2. Within-ceiling residual: frontier_died at 85.74%

E1 with at least one variable-inclusion-valid direction (the other 46.07% of the corpus, after subtracting variable-disjoint). For these, safe_search has a valid rule to apply but only produces a meet-in-the-middle proof for 14.19% of the sampled (E1, E2) pairs at d_max=8. The other 85.74% fail via **frontier_died**: both BFS frontiers exhaust without meeting. Depth sweep `{4, 6, 8}` shows the curve plateaus at d=4 (+0.05pp from d=4 → d=8) — depth is not the constraint. A 60-second wall-timeout sensitivity sub-experiment changes nothing (zero new solves vs 30s).

The dominant failure mode is **structural reachability of safe first-order rewriting**: the reachable set from the safe-direction rules just isn't large enough to bridge lhs2 and rhs2 for most (E1, E2) pairs. A pruning heuristic doesn't expand the reachable set; it traverses the same set faster. That observation rules out a class of embeddings-as-heuristic framings that the original plan considered.

### 3. Row-density anti-correlation: the within-ceiling sharpener

Within the 14.19% sample-level engine coverage, the engine systematically concentrates on **sparse-row E1s**:

| row density | n E1s | pair-level solve rate |
|---|---|---|
| 1–10 TRUE consequents | 222 | **68.6%** |
| 11–50 | 181 | 27.6% |
| 51–200 | 168 | 13.7% |
| 201–1000 | 179 | 4.7% |
| 1001+ | 150 | **0.2%** |

Pearson `corr(row_density, solve_rate) = −0.371`. The engine reaches *many rows shallowly* — 638/900 within-ceiling E1s yield at least one solve — but covers little of their content. Sparse-row equations (which already have few consequents) are over-represented in the engine's wins; dense-row equations (which dominate by pair count) are under-represented. That's exactly what makes the sample-level 14.19% collapse to a pair-level 0.107%.

Mechanism: dense-row E1s are more structurally complex; their TRUE consequents are reached via longer chains than safe-direction rewriting can produce.

### 4. Proof-survey category cross-cut: the categorical signal

Joining the within-ceiling sample to the proof corpus's category labels gives the cleanest diagnosis of what reaches and what doesn't:

| category | n in sample | engine solve rate |
|---|---|---|
| `a_variable_substitution` | 785 | **100.00%** |
| `a_trivial_rfl` | 20 | 100.00% |
| `c_rewrite_chain` | 123 | 61.79% |
| `uncategorized_transitive` (no shortest-proof label) | 18,511 | 10.51% |
| `e_vampire_superposition` | 350 | **2.29%** |
| `e_egg_combinator` | 186 | **0.54%** |
| `e_complex_construction` | 7 | 0.00% |

The engine subsumes substitution-shape proofs perfectly. It handles most rewrite-chain proofs within the within-ceiling subset. It essentially cannot touch Vampire-superposition or Egg-combinator pairs — exactly the categories whose proofs require capabilities (narrowing for Vampire, congruence-closure-with-canonicalization for Egg) that safe first-order rewriting doesn't implement.

This is the categorical confirmation of the structural diagnosis: the 85.74% frontier_died residual on the within-ceiling sample is concentrated in proof categories that fundamentally need different algorithms than the rewrite engine implements.

## Where embeddings could land — what the data permits and rules out

The plan v2 considered embeddings-as-heuristic in two roles:

- **As a search pruning heuristic for safe rewriting.** Ruled out empirically. The depth sweep plateau and wall-timeout insensitivity together show that the engine's residual is not depth-bound or search-budget-bound — it's reachable-set-bound. A pruning heuristic doesn't expand the reachable set; pruning a set that already fits is irrelevant.
- **As a transition model for multi-step lookahead.** Not directly evaluated here, but the same reachable-set limit applies — a transition model that predicts the next state from the same safe-rule set still operates within that set.

Two roles the data permits but that the project hasn't yet built:

- **Triage layer.** Per-pair, predict which method-class to attempt (matcher / completion / engine / "needs Vampire-style"). With the categorical signal above and the JEPA v1.0 embeddings already characterizing equation structure, this is feasible. Returns: avoid wasting compute on Vampire-category pairs with the engine; route them to matcher or to "not currently solvable."
- **Learned narrowing suggester.** Suggest σ-extensions that bridge gaps in the reachable set — turning safe rewriting into something closer to narrowing without implementing full superposition. Would require both a model and the engine integration to consume its suggestions. Out of scope for this measurement.

## Build decision

**Coverage measurement's bottom-line read for the proof-generator build:**

- TRUE side: completion already covers 6.07%. The full deterministic stack adds 0.47pp (matcher + engine combined) for substantial integration complexity. The 86% structural residual within the rewriting-reachable half is overwhelmingly Vampire/Egg/transitive proofs that need algorithms the project hasn't built. **The TRUE-side build is small in coverage and large in complexity.**
- FALSE side: completion already covers 17.82%. The build is "emit the existing verdicts and finite counter-models into Stage-2 format." **The FALSE-side build is large in coverage and small in complexity.**

The honest contribution the project can make from here is **bounding** rather than **extending**. The measurement produces:

- Three structural ceilings: variable-disjoint (53.93% of GT TRUE, outside all rewriting), within-ceiling residual (86% of the rewritable half, beyond safe rewriting's reachable set), category-bounded (Vampire/Egg need superposition/narrowing).
- A clean account of why each method tops out where it does.
- The architectural slot for each prover (completion as TRUE+FALSE engine, matcher as variable-disjoint substitution prover, engine as the dense-row depth-1 + within-ceiling sparse-row contributor).

That bounding result is a stronger story than the build's deliverable would be. The deterministic stack's TRUE-side contribution is essentially **completion**; the FALSE-side contribution is **completion + a Stage-2 emission layer**. A proof generator that adds the rewrite engine and matcher as TRUE-side provers gains 0.47pp of coverage at the cost of three separate Lean-emission stories and three independent soundness audits.

## Artifacts

| Phase | Script | Output |
|---|---|---|
| 1 | `scripts/coverage_measurement/phase1_build_sample.py` | `residual.parquet`, `sample.parquet`, `phase1_residual_summary.json` |
| 2 | `src/coverage/safe_search.py`, `phase2_engine_harness.py` | `phase2_preconditions.json`, tests in `test_coverage_safe_search.py` (8 tests) |
| 3 | `phase2_engine_harness.py` × {d4, d6, d8, variable_disjoint_d6} | `engine_results_*.parquet`, `phase3_depth_sweep.json` |
| 4 | `phase4_soundness_check.py` | `phase4_soundness.json` (PASS) |
| 5 | `phase5_combined_coverage.py` | `phase5_combined_coverage.json` |
| 6 | `phase6_residual_characterization.py` | `phase6_residual_characterization.json` |
| 7 | (this file) | `coverage_measurement_synthesis.md` |

Tests: 8 new for `safe_search`, 383 total in the project suite, all passing.

## Closing reading

The completion exploration produced *how much completion alone reaches*. The coverage measurement extended that to *how much the full deterministic stack reaches when matcher and rewrite engine are added*. The answer: 0.47pp of additional TRUE coverage above completion, distributed structurally — the matcher catches a categorically distinct variable-disjoint slice; the rewrite engine catches sparse-row within-ceiling pairs and substitution-shape proofs that the matcher already handles. The engine's marginal lift over matcher + completion on this corpus is 1.6% of combined coverage.

The 14.20% combined / structural ceiling ratio is the cleaner methodology number: the deterministic stack covers one-seventh of what first-order rewriting can in principle reach. The 86% gap is overwhelmingly proof categories needing narrowing/superposition (Vampire/Egg) and transitive consequences that exceed safe rewriting's reachable set. Both gaps are bounded out of scope rather than partially filled.

The bounding result is the project's actual deliverable here.
