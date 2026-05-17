# Rewrite-Chain Proof Study

## Goal

Characterize the rewrite-chain proof category in depth, before committing to a prover architecture. The proof survey was a broad pass over all proof types; this is a deep dive into one category — the ~1,636 `c_rewrite_chain` proofs in the directly-proved set.

The study answers one decisive question: **does the rewrite-chain category need a learned model, or is it deterministically searchable?** Everything else in the study is characterization that either explains the answer or informs a model design if one turns out to be needed.

This is a study, not a build. It produces a report that picks the prover option. It does not build the prover.

## Methodology resolutions (locked in 2026-05-17)

After scrutiny, the following were resolved before implementation. The
plan body has been updated to reflect them; this section is the
canonical record.

1. **First-order matching is a new function, not a reuse of the
   matcher's structural-match.** Rewriting with E1 binds E1's
   variables to arbitrary subterms (compound terms included), which
   the matcher's `structural_match` explicitly prohibits (variable
   substitution category is variable-to-variable by definition;
   rewriting has no such restriction). Implementation: a recursive
   walk identical in shape to the matcher's, with one rule changed
   — pattern-variable-vs-anything BINDS instead of requiring
   variable-vs-variable. Same consistency-dict discipline (a rule
   variable appearing twice must bind the same subterm). Pattern
   side is E1's LHS or RHS; subject side is a goal subterm with E2's
   variables treated as constants. Implement as its own function,
   probably in `src/rewrite/matching.py`.
2. **Source full bodies from `data/processed/proof_corpus/normalized_proofs.parquet`,
   column `n1_whitespace`.** `true_corpus.parquet` has `first_80`
   (truncated); insufficient for parsing rewrite tactics. Filter to
   the c_rewrite_chain category by joining against `true_corpus.parquet`
   on `(e1, e2)` keys.
3. **Term-size cap: `max(50, 5 × max(|lhs2|, |rhs2|))`, CLI-tunable.**
   Fixed-50 and 5×-relative roughly coincide for typical ~10-node
   ETP equations; the `max` handles occasional larger ones. Sensitivity
   sweeps are part of the analysis.
4. **Depth bound: `d_max = 6` per side, CLI-tunable.** With bidirectional
   search that covers chains up to length 12 — comfortably beyond the
   category's short-chain median. Implications unsolved at depth 6
   go in the "hard tail" for separate characterization.
5. **Bidirectional (meet-in-the-middle) BFS.** Expand reachable sets
   from both `lhs2` and `rhs2`; succeed when they intersect. For short
   chains the saving is large (~10³ vs ~10⁶ states).
6. **Task 3 reads from Task 4's engine.** Without LeanDojo, simulating
   rewrites in Python IS the only way to get intermediate states, and
   that's exactly what Task 4's engine produces. Build engine first;
   run on a sample; Task 3 analyzes the traces. Effectively merge
   Task 3 into the analysis layer that consumes the engine output.
7. **Verify the 1,636 count at runtime** against `true_characterization.json`'s
   `category_distribution` — the JSON already shows
   `c_rewrite_chain: 1636`, so this should just confirm.
8. **Rule-set detection for Task 2:** tokenize the arguments inside
   `rw [...]` and `rwa [...]`, classify each token into
   `{h, ← h, other_name}`. Any `other_name` flags the proof as
   reaching outside the {E1} rule set. The fraction that stays purely
   within {E1} is the strongest predictor of whether deterministic
   search is tractable.
9. **Multimodality sample (Task 5): start at 100, bump to 500** if
   Task 4's search is cheap (it probably will be — small terms,
   depth-6 bidirectional BFS is fast).
10. **Report search cost as distributions** — median, p95, and a
    per-depth breakdown. The "90% solved by depth 2, hard 10% needs
    depth 6+" shape is the informative picture; a mean would hide it.

The two numbers the prover-architecture decision turns on:
**the coverage-vs-depth curve from Task 4** and **the within-{E1}
rule-set fraction from Task 2**. Everything else characterizes or
explains those.

## Why this first

The matcher already showed that a category the survey framed as needing a learned model (variable substitution) was in fact deterministically solvable. The rewrite-chain category might be the same story: short chains over a near-singleton rewrite rule set is squarely in breadth-first-search territory. If a depth-bounded search closes most of the 1,636 implications, the prover is a search algorithm and the neural model is solving a hard tail, not a 15% slice. You cannot pick between deterministic search, neural-guided search, and closed-loop generation without this measurement.

## Scope

The 1,636 proofs tagged `c_rewrite_chain` in `true_characterization.json` (verify count at runtime — Methodology Resolution 7). Full bodies live in `data/processed/proof_corpus/normalized_proofs.parquet`'s `n1_whitespace` column (Methodology Resolution 2 — `true_corpus.parquet`'s `first_80` is truncated and insufficient). Filter the bodies parquet against the category mask in `true_corpus.parquet` to get exactly the c_rewrite_chain subset. The ETP corpus is Lean 4.20.1.

## Study Tasks

### Task 1: Tactic structure

Parse each rewrite-chain proof and extract its step structure.

Report:
- Distribution of chain length (number of rewrite steps per proof).
- Tactic mix: how often each of `rw`, `rwa`, `calc`, `conv`, `simp_rw`, `nth_rw` appears.
- How many proofs are a single straight-line `rw` sequence vs structured (`calc` blocks, `conv` navigation, nested tactics).
- Proof length in tokens, for this category specifically (the survey's median <20 was across all categories).

### Task 2: Rule-set analysis

In an ETP implication proof `theorem E_a_implies_E_b (G) [Magma G] (h : E_a G) : E_b G`, the only hypothesis in scope is `h` (the equation E1). Determine what the proofs actually rewrite with.

Detection methodology (Methodology Resolution 8): tokenize the arguments inside each `rw [...]` and `rwa [...]` invocation. Classify each token into `{h, ← h, other_name}`. Any `other_name` flags the proof as reaching outside the {E1} rule set.

Report:
- Do proofs rewrite ONLY with `h` (E1), or do they pull in other lemmas (mathlib, `Magma` structure, congruence lemmas)?
- **The fraction of proofs that stay purely within {E1, ← E1}** — this is the strongest predictor of whether deterministic search is tractable, and one of the two decision-driving numbers.
- Direction usage: `rw [h]` (forward) vs `rw [← h]` (reverse) vs both within one proof.
- If the rule set is effectively just {E1, both directions}, the search branching factor is small and bounded. If proofs reach for outside lemmas, the rule set — and the search space — is larger.

### Task 3: Intermediate-state characterization (reads Task 4's engine output)

For a sample of rewrite-chain proofs, trace the intermediate goals step by step. Per Methodology Resolution 6, this task reads from Task 4's engine traces rather than running independent Lean replay — simulating the rewrites in Python (which the engine does anyway) IS the only way to extract intermediate states without LeanDojo. Logical ordering: build engine first (Task 4), run on a sample, dump intermediate states, then this task's characterization reads those traces.

Report:
- Confirm intermediate states are equations (the survey's feasibility report claimed this for the category — verify it directly via engine traces).
- Term-size trajectory: does the term grow, shrink, or stay bounded across the chain? A rewrite chain that blows up term size mid-proof makes search states large and search harder. A chain that stays size-bounded is search-friendly.
- Whether the proof is a straight-line transformation (one side of E2 rewritten step by step toward the other) or has structure (auxiliary `have`s, case splits — these would show up as proofs the engine doesn't close via pure rewriting).

### Task 4: Deterministic rewrite search — the decisive measurement

Build a deterministic rewrite engine in Python and measure how much of the category it closes.

The engine uses **first-order matching** (Methodology Resolution 1) — a new function in `src/rewrite/matching.py`, a strict generalization of the matcher's structural-match where pattern variables bind to arbitrary subterms (compound terms included), not just other variables. Given a goal equation and the rewrite rule E1, generate all one-step rewrites by matching an instance of E1's LHS (or RHS) against each subterm of the current term and substituting. A proof of `lhs2 = rhs2` is a path `lhs2 →* c ←* rhs2` — both sides reduced to a common term by rewriting with E1.

Search strategy: **bidirectional (meet-in-the-middle) iterative-deepening BFS** (Methodology Resolution 5). Expand reachable sets from both `lhs2` and `rhs2`; succeed when they intersect. Per-side depth bound `d_max = 6` by default (Methodology Resolution 4), CLI-tunable. Term-size cap `max(50, 5 × max(|lhs2|, |rhs2|))` per state (Methodology Resolution 3), CLI-tunable.

Run on each of the 1,636 implications.

Report:
- **Coverage vs depth bound**: at per-side depth 1, 2, 3, … 6, what fraction of the 1,636 implications does the search close? The coverage curve is the headline result and **one of the two decision-driving numbers**.
- **Search cost as distributions** (Methodology Resolution 10): median, p95, and a per-depth breakdown of states expanded and wall time per implication. The "90% solved by depth 2, hard 10% needs depth 6+" shape is what informs the prover decision; a mean would hide it.
- The hard tail: the implications NOT closed at depth 6. Characterize them — are they long chains, large-branching, or do they need rules outside {E1} (cross-check against Task 2's per-implication rule-set classification)?

Interpretation:
- Search closes ~all of the category at small depth → the rewrite-chain prover is deterministic. No model. The matcher story repeats.
- Search closes most, stalls on a tail → neural-guided search; the model targets the tail and provides a heuristic.
- Search closes little → the category genuinely needs a learned prover (closed-loop generation).

### Task 5: Multimodality measurement

For a sample of implications the search solves, enumerate ALL proofs up to the depth bound, not just the first. Per Methodology Resolution 9, **start at sample size 100; bump to 500 if Task 4's search is cheap** (which it likely will be — small terms, depth-6 bidirectional BFS is fast).

Report:
- Distribution of distinct valid proofs per implication. Is it 1-3, or dozens, or hundreds?
- For implications with many valid proofs, how different are they (length, rule applications) — minor reorderings or genuinely distinct strategies?

This quantifies the multimodality problem. A category with 2-3 valid proofs per implication is tractable for imitation training; a category with dozens means a verifier-centric objective is necessary if a model is built at all.

## Output: Characterization Report

A report answering:

1. **Is the rewrite-chain category deterministically searchable?** The coverage-vs-depth curve from Task 4. This is the decisive result.
2. Chain length, tactic mix, rule set, intermediate-state structure — the characterization from Tasks 1-3.
3. Multimodality — how many valid proofs per implication (Task 5).
4. A prover-architecture recommendation: deterministic search / neural-guided search / closed-loop generation, justified by the measurements.

The report stands in the same relation to the prover as the proof survey stood to the three-track architecture — it picks the option from evidence rather than assumption.

## Notes for Claude Code

- This is characterization, not construction. Build the rewrite engine (Task 4 needs it), but do not build the prover.
- **First-order matching is a NEW function**, not a reuse of `src/matcher/matcher.py::structural_match` (Methodology Resolution 1). The matcher requires variable-to-variable mappings; rewriting needs variable-to-arbitrary-subterm mappings. Implement in `src/rewrite/matching.py` as its own function. Same recursive shape, same consistency-dict discipline, different binding rule when the pattern side is a variable.
- Bidirectional iterative-deepening BFS with a per-side depth bound (default 6) and a term-size cap (default `max(50, 5*max(|lhs2|, |rhs2|))`). Both CLI-tunable for sensitivity sweeps.
- Task 4 is the decisive task. Tasks 1-3 explain the Task 4 result; Task 5 informs the training objective if a model is needed. **Task 3 reads from Task 4's engine output** (Methodology Resolution 6) — build the engine first, run on a sample, dump traces, then Task 3 analyzes them.
- Report the coverage-vs-depth curve as the headline, and the within-{E1} rule-set fraction from Task 2 alongside it. Those two numbers drive the prover decision.
- Save the rewrite engine and the search results — if the decision is neural-guided search, the engine is the search backbone the model will plug into, and the search traces are training data.
- Source bodies from `data/processed/proof_corpus/normalized_proofs.parquet` `n1_whitespace` column, filtered by joining `true_corpus.parquet`'s category mask on `(e1, e2)` (Methodology Resolution 2).
