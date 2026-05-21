# Proof Generation Pilot — Plan

**Status:** in-design. Iterating on decisions before any code is written.

**Goal of the pilot:** generate Lean proofs for ETP Stage 2 — TRUE implications and `Fin n` FALSE counterexamples — using a combination of deterministic methods (matcher, completion, safe rewrite engine) and learned models. This plan covers the first phase only: building the initial dataset that will feed downstream model design.

## Phase 1 — Initial dataset

### What

Build a dataset of ~1,000 (pair, tool, proof, intermediate-states) tuples across four proof-producing tools, run over the same sample of variable-disjoint TRUE implication pairs. Validate all generated Lean against the chosen Lean environment.

### Goals

- Calibrate how the four tools differ in proof shape on the same easy class.
- Produce a corpus of validated Lean proofs in the target submission format.
- Discover practical issues (Lean version skew, tool environment setup, intermediate-state extraction) before scaling.

### Sample selection — RESOLVED

**Definition.** "Variable-disjoint pair (E1, E2)" = pair where E1's `Vars(lhs)` and `Vars(rhs)` are **incomparable** (neither is a subset). Per-equation property of E1; defined in `et_studies/completion_exploration/scripts/phase0_variable_inclusion.py` and computed in `et_studies/completion_exploration/data/equation_orientability.parquet`.

This is the slice where no orientation of E1 yields a valid rewrite rule (variable-inclusion condition fails), so the safe rewrite engine returns `no_valid_rule_from_e1` and proofs must go through Birkhoff substitution.

**Population numbers:**
- 1,656 of 4,694 equations are variable-disjoint E1s (sub-classes: 1,044 literally disjoint + 612 overlapping-incomparable).
- 4,407,831 TRUE pairs have a variable-disjoint E1 (53.93% of GT TRUE).
- ~30,010 of those are matcher-solvable (E2 is a direct σ-substitution instance of E1).

**Sample.** Uniform random N=1,000 from the **~30,010 matcher-solvable variable-disjoint TRUE pairs**. Reasoning: this is the intersection where all four tools have a fighting chance, so the resulting dataset reflects real tool-strength differences on a single shared class. Stratification can be revisited after seeing the spread.

### Tool list

For each sampled pair, run the four tools and produce a Lean proof:

1. **Matcher** (our deterministic, `proof_generation_pilot/methods/matcher/`) — finds σ; one-line term-mode proof.
2. **Completion** (our deterministic, `proof_generation_pilot/methods/completion/`) — plain + ordered KB; produces a rewrite chain.
3. **ETP Vampire** — re-run on each pair; translate output to Lean. (TBD: see open items.)
4. **ETP Egg** — re-run on each pair; translate output to Lean. (TBD: see open items.)

### Decisions log

- **Q1** (variable-disjoint definition): **incomparable lhs/rhs vars on E1**. Confirmed against the Phase 0 code definition.
- **Q2** (sampling): **uniform random N=1,000 from the ~30,010 matcher-solvable variable-disjoint TRUE pairs.** Sample stays at the original slice. Completion is still in the 4-tool design even though it structurally cannot handle variable-disjoint E1 (returns `NO_RULES_FROM_INPUT`) — its empty output is itself data (calibration of where each tool helps), and downstream policy uses whichever tool actually produced a proof.
- **Q3** (Lean validation environment): **Lean 4.20.1** (ETP's version). Reasoning: compatibility with ETP's Vampire/Egg adapters and tooling, and LeanDojo is more likely to work on the older version. Caveat: Stage 2 judge runs 4.30.0-rc2 — a porting step (4.20.1 → 4.30.0-rc2) will be needed at the submission boundary in a later phase. Later Lean versions may make downstream work easier; revisit when it becomes the bottleneck.
- **Q4** (Vampire / Egg sourcing): **Vampire re-runs via ETP's existing driver; "Egg" substituted by `lean-egg`** (the Lean tactic, not the external Rust MagmaEgg).
   - **Vampire**: re-run on the 1k sample using `equational_theories/Generated/VampireProven/src/vampire_proofs.py` — ETP's TPTP encoder + `leanify()` translator already produce tactic-mode Lean that uses ETP's `superpose` tactic (`Superposition.lean`). Install the Vampire binary locally (the driver expects `~/Downloads/vampire`). Tactic-mode output gives LeanDojo intermediate states.
   - **Egg → lean-egg**: original plan had MagmaEgg (Rust crate in external repo). Recon (`outputs/q12_recon.md`) showed MagmaEgg's runner is in a separate Rust repo with undocumented input format — significant setup cost for an uncertain pilot benefit, and its native Lean output is term-mode (no intermediate states for LeanDojo without rewrapping). **Substituted with `lean-egg`** — Rossel's Lean tactic, already used in ETP's `Subgraph.lean` for 86 proofs. As a Lean tactic it lives in the Lean toolchain, requires no separate runner, and `theorem ... := by egg [h]`-style proofs are tactic-mode → LeanDojo-friendly. If we later want MagmaEgg-specific data we add it as a follow-on.
   - Open sub-question: once all 4 tools have outputs on the same pair, do we use Vampire/lean-egg's proof for pairs where a "simpler" tool (matcher) also succeeded? Deferred until we see what the outputs look like.
- **Q5** (intermediate proof-state capture): **use LeanDojo** to programmatically step through proofs and record the goal state after each tactic. Run against Lean 4.20.1 (Q3) where LeanDojo support is more likely to work. Rationale: intermediate states are needed to determine the **minimum token grammar** for the downstream proof-generation model — we need to see the actual token stream Lean produces (tactic names, identifier conventions, goal-state rendering) to know what vocabulary the model has to span. This makes the schema requirement specific: capture **raw textual** tactic + goal-state output from LeanDojo, not just a structured AST. Setup is a Phase 1 step: install LeanDojo, configure for Lean 4.20.1, build the ETP-flavored Lean project that imports the necessary Magma/equation definitions, smoke-test on one proof end-to-end before sampling begins.
- **Q6** (Lean equation statements): **reuse ETP's source** directly. ETP defines each of the 4,694 equations as a Lean predicate (likely in `equational_theories/Equations.lean` or similar — Q12 recon to locate). The pair's Lean goal statement is `Equation<e1_id> G → Equation<e2_id> G` (with `G : Type` + `[Magma G]` instance), shared across all four tools — only the proof body differs. Reuse keeps us aligned with ETP's canonical form and avoids divergent parses; our `lib/data/parser.py` stays scoped to symbolic-side work (matching, completion) and isn't used to generate Lean.
- **Q7** (Matcher → Lean output style): **term-mode body wrapped in `def submission := ...`**. Concretely: `def submission : Goal := fun G _ h => h <σ-permuted args>`. No `by` block; pure term-mode lambda introduces G, the Magma instance, and the hypothesis h, then applies h to the σ-permuted arguments. Implication for intermediate-state capture (Q5): a term-mode proof has no per-tactic state transitions, so for matcher outputs the "intermediate states" reduce to (initial goal, final proved). That's degenerate for the matcher (it's a one-step proof anyway) but is a relevant consideration for Q8. **Side finding:** the matcher's Lean output adapter already exists — `matcher_proofs.parquet` has `lean_proof_term` (e.g., `fun x => h x x`) and `lean_proof_tactic` columns. The adapter work for the matcher reduces to "wrap the existing term in `def submission := ...`."
- **Q8** (Completion → Lean adapter): **keep completion in the 4-tool design even though it returns `NO_RULES_FROM_INPUT` on the entire variable-disjoint slice.** Both tools' outputs are captured per pair — matcher will produce a σ-substitution proof; completion will produce a "no proof" diagnostic. Downstream policy uses whichever tool actually produced a proof (matcher in this case); completion's empty output is calibration data. No 10-pair pilot needed for this sample — the code already determines the outcome. The adapter work is therefore minimal: capture completion's return status and any partial output (rule set built, time spent) without forcing it into a Lean proof shape. If we later run completion on a different sample where it CAN contribute, we'd build a real Completion → Lean adapter then.
- **Q9** (Code layout): under `proof_generation_pilot/`:
   - `methods/matcher/to_lean.py` — wraps the existing `matcher_proofs.parquet:lean_proof_term` in `def submission := ...`.
   - `methods/completion/to_lean.py` — captures completion's status + diagnostics for the variable-disjoint slice (no real Lean output to produce here, but the adapter exists for symmetry and for future samples where completion *can* contribute).
   - `methods/vampire/` — new subdir; thin wrapper around ETP's `vampire_proofs.py` (TPTP encoder + `leanify()` translator already exist). Installs/configures the binary.
   - `methods/lean_egg/` — new subdir; even thinner. lean-egg is a Lean tactic, so there's no Python runner per se — just a Lean-file template that emits `theorem ... := by egg [h]` and we invoke Lean to validate. (Originally planned as `methods/egg/` for MagmaEgg; swapped to lean-egg per Q4 update.)
   - `scripts/sample_pairs.py` — builds the 1k sample from the 30K matcher-solvable variable-disjoint pool with a fixed seed.
   - `scripts/build_phase1_dataset.py` — orchestration: per-pair, run all 4 tools, validate in Lean 4.20.1 via LeanDojo, capture intermediate states, write parquet rows.
   - `data/processed/phase1_dataset/` — output parquet location.
- **Q10** (Storage format): **one combined parquet** with a `tool` column. ~4,000 rows (1k pairs × 4 tools) — joins across tools cheap, uniform schema. Schema sketch:
   ```
   pair_id          int       # local 0-based index into the 1k sample
   e1_id            int       # equation IDs from ETP
   e2_id            int
   tool             str       # 'matcher' | 'completion' | 'vampire' | 'egg'
   proof_lean       str       # full Lean source, or empty if the tool produced no proof
   status           str       # 'proved' | 'no_proof' | 'timeout' | 'error'
   intermediate_states list[str]  # ordered goal-state texts from LeanDojo
   tactic_used      list[str]  # ordered tactic texts paired with each transition
   wall_clock_ms    int
   validated        bool      # did the proof_lean compile and close the goal under Lean 4.20.1?
   validation_log   str       # stderr / error trace if not validated
   ```
- **Q11** (Sample-size budget): **stepped: 100 → 1k.** Build the pipeline end-to-end on 100 pairs first to surface install/integration issues + estimate wall-clock per tool. Scale to 1k once the 100-pair run is clean. 1k is the Phase 1 target, not a hard cap; later phases will broaden it.
- **Q12** (ETP wrapper recon): **research task, runs first before implementation.** Read `~/git/equational_theories/` to locate (per tool): (a) the runner / invocation script, (b) the input serialization (TPTP for Vampire; e-graph spec for Egg), (c) the output → Lean translator (if it exists), (d) the targeted Lean version. If a translator is missing or unusable, escalate as "build our own" before Phase 1 implementation.

## Phase 1 work breakdown (post-decisions)

1. **Q12 recon.** Read ETP's Vampire/Egg drivers; produce a short writeup at `proof_generation_pilot/outputs/q12_recon.md`. If the Lean translators don't exist, decide whether to build them.
2. **Tooling install.** Vampire, Egg, LeanDojo. Verify each runs end-to-end on one equation.
3. **Lean project shim.** Build a small Lean 4.20.1 project under `proof_generation_pilot/` that imports ETP's equation definitions. LeanDojo points at this.
4. **Sample build.** `scripts/sample_pairs.py` → 100-row + 1k-row variants from the 30K pool, seeded.
5. **Per-tool adapters.** Matcher (trivial), Completion (diagnostic capture), Vampire/Egg (depends on Q12).
6. **Orchestration pipeline.** `scripts/build_phase1_dataset.py` — runs the 4 tools per pair, validates in Lean, captures intermediate states, writes the parquet.
7. **100-pair smoke run.** Resolve any issues surfaced. Measure wall clock per tool.
8. **1k full run.** Production Phase 1 dataset.
9. **Synthesis.** Short writeup at `proof_generation_pilot/outputs/phase1_synthesis.md` — what each tool produced, intermediate-state vocab discovered, surprises, recommended next steps.

## Phase 2+ (placeholder)

Subsequent phases — full dataset construction, model design, training — depend on what Phase 1 reveals. Plan extends once Phase 1 closes.
