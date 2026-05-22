# Phase 1 — Synthesis

Dataset built from 1,000 variable-disjoint matcher-solvable TRUE implication pairs (uniform random from the ~30,010 pool). Three proof-producing tools run per pair; per-step `(state, tactic, next_state)` transitions captured via PyPantograph. Outputs at `proof_generation_pilot/outputs/phase1_1k/`.

## Headline

| Tool | Solved | mean wall (ms) | p99 wall (ms) | mean transitions/pair | distinct tactics |
|---|---|---|---|---|---|
| matcher | **1000/1000 (100%)** | 1.5 | 2 | 2.0 | 173 |
| vampire | **1000/1000 (100%)** | 23.3 | 30 | 7.0 | 1,450 |
| grind | **447/1000 (44.7%)** | 6.1 | 9 | 2.0 | 2 |

- **3,000 proof rows + 11,000 transition rows** total
- Pipeline wall-clock for the 1k run: **32.7 seconds** end-to-end (PyPantograph server starts in ~2s, amortized)

## Per-tool findings

### Matcher — 100% on its native slice

The matcher solves every pair as expected (the sample was selected from its solved set). Its proofs are uniform tactic-mode `intro G _ h; exact fun ... => h ...` — 2 transitions per pair. The variability lives in the `exact` body's σ permutation: 173 distinct second-tactic strings across 1,000 pairs (≈5.8 pairs per distinct σ pattern), reflecting how concentrated the σ space is.

### Vampire — 100%, uniform 7-step proof shape

Vampire proves every pair, always in ~23 ms (p99 30 ms, well under the 1-second budget). Every proof has **exactly 7 transitions**: the scaffolding is invariant, only the equation indices + Skolem witnesses + statements differ. Normalized tactic templates (collapsing `eqN`, `sKN`, magnitudes):

```
1000 × intro G _ h
1000 × by_contra nh
1000 × simp only [not_forall] at nh
 433 × obtain ⟨sK, sK, sK, nh⟩ := nh         (3-var goals)
 367 × obtain ⟨sK, sK, nh⟩ := nh             (2-var)
 131 × obtain ⟨sK, sK, sK, sK, nh⟩ := nh     (4-var)
 ... × have eqN ... := mod_symm nh           (vary by statement)
 ... × have eqN ... := superpose eqN eqN     (vary by indices)
1000 × subsumption eqN eqN
```

Vampire's grammar is much richer than matcher's: **1,450 distinct tactics** across 7,000 transitions. Most variability sits in the two `have` lines per proof (one for the negated conjecture, one for an instance of the axiom). The fixed scaffolding (intro / by_contra / simp / final subsumption) is identical for every pair on this slice.

### grind — 44.7%, identical 2-tactic shape

Grind solves under half the pairs. Failures look like:

```
[grind] Issues
  [issue] failed to create E-match local theorem for
        ∀ (x y z w u : G), x = (y ◇ z) ◇ (w ◇ u)
```

Pattern: when an axiom's LHS is a bare variable `x`, grind's E-matching has no useful pattern to match against, so it can't instantiate the axiom productively. This is structural to grind's heuristic, not a tunable parameter.

The 553 grind failures are tracked in `proofs.parquet` with `status='error'` and the full grind diagnostic in `validation_log`. They aren't "tool unavailable" — they're "tool tried and gave up," which is a meaningful negative training example.

## Transition-data grammar

Distinct tactic counts per tool give a first read on what a downstream model would need to span:

- **grind: 2 distinct tactics.** Fixed scaffolding only — model wouldn't learn anything tool-specific.
- **matcher: 173 distinct.** Bounded by the σ-permutation space.
- **vampire: 1,450 distinct.** Per-pair-specific equation indices, Skolem witnesses, full statement texts.

State-text sizes (chars in PyPantograph's `state_before` string):

| Tool | mean | p50 | max |
|---|---|---|---|
| matcher | 59 | 58 | 60 |
| grind | 59 | 58 | 60 |
| vampire | 130 | 102 | 315 |

Vampire states grow with each `have` line (carrying more local equations).

## What the dataset supports

- **Supervised training of a state→tactic predictor.** Each transition row is one training example: condition on `state_before`, predict `tactic`. Vampire dominates the supervised signal both by row count (7,000 of 11,000) and by tactic diversity (1,450 of ~1,625 distinct tactics).
- **Inference via search.** At inference the model proposes tactics, applies them through PyPantograph, gets the next state, repeats. Beam/MCTS-style. Each pair's proof is a 2–7 step trajectory.
- **Negative examples.** Grind's 553 failures are full goal-state traces where a tactic *didn't* close the proof — useful for training models to recognize when an approach won't work.

## What it doesn't support (yet)

- **Non-variable-disjoint pairs.** This sample's a single proof category by construction. Phase 2 broadens to the 19,576-pair matcher∩completion intersection and the wider TRUE corpus.
- **Multi-tool agreement / disagreement on the same pair.** All three tools target the same sample, but only matcher and Vampire have 100% coverage; grind's overlap is the 447 pairs it solved. Cross-tool analysis is limited to those 447 + structural comparisons on the other 553.
- **Stage 2 submission.** Output is Lean 4.29.1; Stage 2 judge runs Lean 4.30.0-rc2. A one-Lean-release porting step at the submission boundary is the next infrastructure obligation (per Q3 in the plan).

## Surprises

1. **Vampire's perfectly uniform 7-step shape on this slice.** Predicted ~5–20 transitions per pair; actual = 7 for all 1,000. This sample is concentrated enough that Vampire's pipeline always lands on one-superposition + closing subsumption. Wider sample (Phase 2) will likely show more variety.
2. **Grind's structural failure mode.** Predicted "atomic from PyPantograph's view"; actual is also "atomic AND often unproductive." The 44.7% is a genuine coverage finding, not a Phase 1 setup bug.
3. **Pipeline throughput.** Predicted "few overnight runs" for 1k × 3 tools given Vampire's ~1s timeout; actual = 32 seconds. Vampire converges in <30ms on this easy slice. Scaling to Phase 2's wider sample will exercise the budget more.

## Phase 2 hooks

Phase 1 deliberately fixed three things that Phase 2 will reopen:

- **Sample scope** — broaden beyond variable-disjoint matcher-solvable.
- **Completion** — wire it in once the sample includes non-variable-disjoint pairs where it structurally contributes (rewrite chains, real multi-step proofs LeanDojo-style).
- **Stage 2 porting** — one-pair-at-a-time port from Lean 4.29.1 → 4.30.0-rc2 for submission.

## Files

- `proofs.parquet` — 3,000 rows; one per (pair, tool); columns: pair_id, eq1_id, eq2_id, tool, proof_lean, status, n_transitions, wall_clock_ms, validated, validation_log
- `transitions.parquet` — 11,000 rows; one per (state_before, tactic, state_after) step; columns: pair_id, eq1_id, eq2_id, tool, step_idx, state_before, tactic, state_after, is_terminal

Reproducer: `proof_generation_pilot/scripts/build_phase1_dataset.py` against `proof_generation_pilot/data/processed/phase1_sample_1k.parquet`.
