# Proof Generation Pilot — Plan

**Status:** in-design. Iterating on decisions before any code is written.

**Goal of the pilot:** generate Lean proofs for ETP Stage 2 — TRUE implications and `Fin n` FALSE counterexamples — using a combination of deterministic methods (matcher, completion, safe rewrite engine) and learned models. This plan covers the first phase only: building the initial dataset that will feed downstream model design.

## Phase 1 — Initial dataset

### What

Build a dataset of **per-transition `(state_before, tactic, state_after)` training examples** from three proof-producing tools (matcher, Vampire, grind) run over a 1,000-pair sample of variable-disjoint TRUE implications. Validate all generated Lean against Lean 4.29.1; capture goal states via PyPantograph at each tactic step.

The unit of supervision is the transition, not the full proof. A single proof contributes multiple training examples (matcher ~2, grind ~1, Vampire ~5–20 — total 5–20k transitions from the 1k pilot). Downstream model is a state-conditioned tactic predictor; inference time uses beam search / MCTS-style exploration over tactic candidates, applying each candidate to Lean to get the next state, repeat until goal is closed. Closer to how research transformer provers actually work (GPT-f / COPRA / ReProver) than atomic seq2seq proof generation.

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

For each sampled pair, run **three** tools and produce a tactic-mode Lean proof wrapped in `def submission := by ...`:

1. **Matcher** (`proof_generation_pilot/methods/matcher/`) — finds σ; tactic-mode `by intro G _ h ...; exact h <σ-args>`. PyPantograph captures the post-intro state.
2. **Vampire** (via ETP's `vampire_proofs.py` + `Superposition.lean`) — re-run per pair; tactic-mode `by` block with multiple `have eqN ... := superpose ...` lines + closing tactic. PyPantograph captures per-superposition state.
3. **`grind`** (Lean core tactic) — `by grind [h]` single-tactic proof. PyPantograph captures the post-grind state. (Replaces the earlier lean-egg plan, which is officially deprecated in favor of grind.)

**Completion is deferred to Phase 2.** Completion structurally produces no proof on variable-disjoint inputs (`NO_RULES_FROM_INPUT`), so including it here adds work without contributing proofs. Phase 2 will broaden the sample to non-variable-disjoint pairs where completion naturally yields rewrite chains — the right home for the stratification question.

### Decisions log

- **Q1** (variable-disjoint definition): **incomparable lhs/rhs vars on E1**. Confirmed against the Phase 0 code definition.
- **Q2** (sampling): **uniform random N=1,000 from the ~30,010 matcher-solvable variable-disjoint TRUE pairs.** Sample stays at the original slice. Reasoning: variable-disjoint Birkhoff substitution was chosen for clean experimental focus on a single proof category. Stratifying to give completion something to do would add heterogeneity that complicates downstream "what proof shape did the model learn" analysis. Phase 2 broadens to non-variable-disjoint pairs.
- **Q3** (Lean validation environment): **Lean 4.29.1.** Matches ETP upstream (commit `3c7f9e77` "chore: Bump to version 4.29") and matches Pantograph dev branch (current pin v4.29.1). Aligning everything at 4.29.1 avoids the toolchain pinning friction we hit when targeting 4.20.x. Stage 2 judge runs 4.30.0-rc2 — one Lean release ahead, so the eventual porting step (4.29.1 → 4.30.0-rc2) at the submission boundary will be much smaller than the 4.20 → 4.30 leap we originally faced. **Originally chose 4.20.1**, migrated to 4.29.1 after PyPantograph/Pantograph protocol drift made the 4.20-era stack unusable; ETP's upstream having already done the 4.20 → 4.29 bump made the migration cheap (`git pull` + cache refresh + rebuild).
- **Q4** (Vampire / Egg sourcing): **Vampire re-runs via ETP's existing driver; "Egg" substituted by `lean-egg`** (the Lean tactic, not the external Rust MagmaEgg).
   - **Vampire**: re-run on the 1k sample using `equational_theories/Generated/VampireProven/src/vampire_proofs.py` — ETP's TPTP encoder + `leanify()` translator already produce tactic-mode Lean that uses ETP's `superpose` tactic (`Superposition.lean`). Install the Vampire binary locally (the driver expects `~/Downloads/vampire`). Tactic-mode output gives LeanDojo intermediate states.
   - **Egg → `grind`** (Lean core). Two prior substitutions: MagmaEgg (external Rust crate, undocumented input format, expensive setup) → lean-egg (Lean tactic, used in ETP's `Subgraph.lean`) → **`grind`** (Lean's built-in e-graph-based tactic). lean-egg is officially deprecated; its own README recommends `grind`. Same e-graph reasoning, no external Lake dep, actively maintained as Lean core. Same atomic shape from PyPantograph's view (one tactic call = one transition); the value of this slot is "e-graph-based proof shape in the dataset," not per-step traces.
   - Open sub-question: once all 4 tools have outputs on the same pair, do we use Vampire/lean-egg's proof for pairs where a "simpler" tool (matcher) also succeeded? Deferred until we see what the outputs look like.
- **Q5** (intermediate proof-state capture): **use PyPantograph directly at Pantograph commit `3ce335eb` (Lean 4.29.1)** — skip LeanDojo entirely. Capture the goal state after each tactic step for tactic-mode proofs. Rationale: intermediate states are needed to determine the **minimum token grammar** for the downstream proof-generation model. Schema requirement is specific: capture **raw textual** tactic + goal-state output, not just a structured AST. Setup is a Phase 1 step: pin Pantograph submodule, build the REPL via `lake build repl` against Lean 4.29.1, install PyPantograph against that pinned source, smoke-test on one proof end-to-end before sampling begins.

   **Why PyPantograph and not LeanDojo:**
   - LeanDojo v1 supports Lean 4.29.1 but is deprecated (no future bug fixes).
   - LeanDojo-v2 is the active project but tracks Pantograph `dev`, currently pinned to Lean 4.29.1 — version mismatch with ETP at 4.20.1, and Pantograph compatibility is strict per its own docs.
   - LeanDojo-v2 also bundles a heavy ML stack (PyTorch, transformers, deepspeed, peft, trl, openai) for its training/inference agents (`agent/`, `trainer/`, `prover/`, `lean_progress/`). We use *none* of that for Phase 1.
   - LeanDojo-v2's Lean-interaction core (`lean_dojo_v2/lean_dojo/`) is itself a wrapper around PyPantograph. Using PyPantograph directly gives us the same primitive with a much smaller install footprint and no version-mismatch risk.

   **Known issues with this approach + the broader Phase 1 design:**
   - **Pantograph submodule pinning is manual.** No tagged release maps cleanly to Lean 4.29.1; we identify the commit by date (Pantograph's `lean-toolchain` changed to v4.20.1 at commit `3ce335eb` on 2025-06-17, then to v4.21.0 two weeks later). We point PyPantograph's `src` submodule at that commit and rebuild. One-time work, but not a stable supported configuration.
   - **PyPantograph itself may need an older revision** to match the Pantograph 4.20.1 era (mid-2025 PyPantograph). API stability between PyPantograph versions and the pinned Pantograph commit is untested — possible integration friction.
   - **Per-step state richness varies by tool.** Vampire's tactic-mode `by` block with `have eqN ... := superpose ...` lines yields the richest per-step transitions. Matcher and lean-egg each give one meaningful transition (the post-`intro` / post-`egg` state). So PyPantograph's per-step capture effectively yields:
     - **Matcher:** 1k × 1 transition (initial → post-`intro` → closed = ~2 captured states)
     - **Vampire:** 1k × multi-step traces (~5–20 transitions per pair) — the real grammar signal
     - **lean-egg:** 1k × 1 transition (initial → post-`egg` → closed = ~2 captured states)
   - Net: per-step grammar data is dominated by Vampire's superposition traces. Matcher and lean-egg contribute anchor / single-step data. Q5 grammar-discovery rationale is mostly Vampire-grounded; the other two tools are present for proof-shape comparison rather than per-step structure.
- **Q6** (Lean equation statements): **reuse ETP's source** directly. ETP defines each of the 4,694 equations as a Lean predicate (likely in `equational_theories/Equations.lean` or similar — Q12 recon to locate). The pair's Lean goal statement is `Equation<e1_id> G → Equation<e2_id> G` (with `G : Type` + `[Magma G]` instance), shared across all four tools — only the proof body differs. Reuse keeps us aligned with ETP's canonical form and avoids divergent parses; our `lib/data/parser.py` stays scoped to symbolic-side work (matching, completion) and isn't used to generate Lean.
- **Q7** (Matcher → Lean output style): **tactic-mode body wrapped in `def submission := by ...`**. Revised from the original term-mode plan once Q5 settled on PyPantograph for intermediate-state capture — term-mode proofs are atomic from PyPantograph's view, but `by intro G _ h <vars>; exact h <σ-args>` gives one meaningful intermediate state (the post-`intro` goal) at zero cost. Same proof, same semantics, just in tactic-mode form. The matcher's existing `lean_proof_tactic` column in `matcher_proofs.parquet` is already tactic-mode-ish (`exact fun x => h x x`); the adapter wraps it in `by intro ...; <existing tactic>`.
- **Q8** (Completion → Lean adapter): **completion is deferred to Phase 2.** On the variable-disjoint slice it would just produce `NO_RULES_FROM_INPUT` for all 1k pairs (no proofs, no Lean output, no states to capture). Carrying it through Phase 1 as a diagnostic-only tool adds adapter work, install verification, and a fourth `tool` value in the schema without producing actual proof data. Cleaner to drop completion from Phase 1 entirely and revisit in Phase 2 when the sample broadens to non-variable-disjoint pairs (where completion produces real rewrite chains that become tactic-mode `rw` sequences — the right setting for both the tool *and* the stratification question). The `proof_generation_pilot/methods/completion/` code stays in place from the refactor; we just don't exercise it in Phase 1.
- **Q9** (Code layout): under `proof_generation_pilot/`:
   - `methods/matcher/to_lean.py` — wraps `matcher_proofs.parquet:lean_proof_tactic` in `def submission := by intro G _ h <vars>; <existing tactic>`.
   - `methods/vampire/` — new subdir; thin wrapper around ETP's `vampire_proofs.py` (TPTP encoder + `leanify()` translator already exist). Installs/configures the binary.
   - `methods/grind/` — new subdir; thin. `grind` is a Lean core tactic (no external dep), so this is just a Lean-file template emitting `def submission := by grind [h]` invoked through Lean for validation. (Replaces the earlier `methods/lean_egg/`.)
   - `methods/completion/` — exists from the refactor but **not used in Phase 1**. Phase 2 wires it in.
   - `scripts/sample_pairs.py` — builds the 1k sample from the 30K matcher-solvable variable-disjoint pool with a fixed seed.
   - `scripts/build_phase1_dataset.py` — orchestration: per-pair, run all 4 tools, validate in Lean 4.29.1 via LeanDojo, capture intermediate states, write parquet rows.
   - `data/processed/phase1_dataset/` — output parquet location.
- **Q10** (Storage format): **per-transition rows.** The training unit is `(state_before, tactic, state_after)`, not the full proof. A single proof contributes multiple training examples — matcher ~2, grind ~1, Vampire ~5–20. Total **5–20k transition rows** from the 1k pilot, much richer than 1k atomic per-proof examples. Two parquets:

   `proofs.parquet` — one row per (pair, tool):
   ```
   pair_id          int       # local 0-based index into the 1k sample
   e1_id, e2_id     int
   tool             str       # 'matcher' | 'vampire' | 'grind'
   proof_lean       str       # full Lean source ('def submission := by ...')
   status           str       # 'proved' | 'no_proof' | 'timeout' | 'error'
   n_transitions    int       # number of rows in transitions.parquet for this proof
   wall_clock_ms    int
   validated        bool      # did proof_lean compile and close the goal?
   validation_log   str
   ```

   `transitions.parquet` — one row per (state, tactic, next_state) step:
   ```
   pair_id, e1_id, e2_id  int
   tool             str
   step_idx         int       # 0-based position within this proof
   state_before     str       # raw goal-state text from PyPantograph
   tactic           str       # raw tactic text applied
   state_after      str       # raw goal-state text after the tactic
   is_terminal      bool      # state_after is `no goals` / closed
   ```

   Joinable on `(pair_id, tool)`. Downstream training reads `transitions.parquet` directly. The supervised signal is "given `state_before`, predict `tactic`"; `state_after` is the verification target after applying the tactic to Lean.
- **Q11** (Sample-size budget): **stepped: 100 → 1k.** Build the pipeline end-to-end on 100 pairs first to surface install/integration issues + estimate wall-clock per tool. Scale to 1k once the 100-pair run is clean. 1k is the Phase 1 target, not a hard cap; later phases will broaden it.

## Known issues with the current Phase 1 design

Surfaced during decision walk-through; tracked here so they're visible before implementation begins.

1. **Per-step state richness varies by tool.** Vampire produces multi-step traces (~5–20 transitions per pair); matcher gives 2 transitions (post-`intro`, post-`exact`); grind gives 1 transition (post-`grind`). Vampire dominates the per-transition supervised signal; matcher and grind contribute fewer but valid transition tuples. Total per-pair: ~8–23 transitions; total dataset: ~5–20k transitions.

2. **Lean 4.29.1 → 4.30.0-rc2 porting** for Stage 2 submission is a deferred step (one Lean release ahead — should be a small port).

3. **Vampire `leanify()` may raise `NotImplementedError`** on uncommon proof patterns (superposition + backward demodulation + equality resolution are handled; others may not be). Per-pair failures will be tracked and characterized in Phase 1 outputs.

4. **Vampire setup cost.** Vampire binary install — verified working (v4.9, ETP's `leanify()`-compatible format). `grind` requires no separate install (Lean core).

5. **ETP's Vampire driver was written for Lean 4.20.x ETP.** ETP upstream bumped to 4.29.1 — the driver may need minor updates if `superpose`'s signature or `Superposition.lean` semantics changed. To be verified at first end-to-end Vampire run per pair.

These are issues to *manage*, not blockers. The plan proceeds; we revisit if any turn out worse than expected during the 100-pair smoke run.

## Phase 1 setup status

| Item | Status |
|---|---|
| elan + Lean 4.29.1 | ✓ installed |
| Vampire 4.9 binary at `~/Downloads/vampire` | ✓ installed; smoke test produces clean leanify-compatible output |
| Pantograph dev + PyPantograph latest (v0.3.15) | ✓ built; `pip install -e ~/git/PyPantograph`; protocol works against 4.29.1 |
| ETP at upstream main (4.29.1) + Mathlib cache + `lake build` | ✓ 4729 modules compiled |
| Lean project shim at `proof_generation_pilot/lean_project/` | ✓ `PilotShim` lib builds (584 jobs) against ETP 4.29.1 |
| `grind` Lean core tactic | ✓ no install needed (built into Lean 4.29.1) |
| **End-to-end PyPantograph smoke** | ✓ goal_start + 3 goal_tactic calls succeed on `Equation2 → Equation1`; states captured at each step |

## Deferred to Phase 2

- **Completion as a proof producer.** Move sample to non-variable-disjoint pairs (or a stratified mix) where completion structurally contributes via rewrite chains. The Phase 1 completion code stays in `proof_generation_pilot/methods/completion/` ready to wire in.
- **Stratification / sample-shape questions.** Mixing variable-disjoint and non-variable-disjoint pairs introduces heterogeneity that Phase 1's clean single-category focus deliberately avoids. Phase 2 owns this.
- **Birkhoff substitution inside completion** (option (d) from the design discussion). Real engineering that converges on Vampire-style paramodulation. Out of scope unless Phase 2 surfaces a specific need.

## Phase 1 work breakdown (post-decisions)

1. **Q12 recon.** Read ETP's Vampire/Egg drivers; produce a short writeup at `proof_generation_pilot/outputs/q12_recon.md`. If the Lean translators don't exist, decide whether to build them.
2. **Tooling install.** Vampire, Egg, LeanDojo. Verify each runs end-to-end on one equation.
3. **Lean project shim.** Build a small Lean 4.29.1 project under `proof_generation_pilot/` that imports ETP's equation definitions. LeanDojo points at this.
4. **Sample build.** `scripts/sample_pairs.py` → 100-row + 1k-row variants from the 30K pool, seeded.
5. **Per-tool adapters.** Matcher (wrap existing `lean_proof_tactic` in `def submission := by intro G _ h <vars>; <tactic>`), Vampire (thin wrapper around ETP's `vampire_proofs.py`), lean-egg (Lean-file template emitting `def submission := by egg [h]`).
6. **Orchestration pipeline.** `scripts/build_phase1_dataset.py` — runs the 4 tools per pair, validates in Lean, captures intermediate states, writes the parquet.
7. **100-pair smoke run.** Resolve any issues surfaced. Measure wall clock per tool.
8. **1k full run.** Production Phase 1 dataset.
9. **Synthesis.** Short writeup at `proof_generation_pilot/outputs/phase1_synthesis.md` — what each tool produced, intermediate-state vocab discovered, surprises, recommended next steps.

## Phase 2 — broader dataset + model-ready signal

**Status:** in-design. Phase 1 closed cleanly; now resolving Phase 2 decisions before any implementation.

### What Phase 1 taught us that reshapes Phase 2

1. **Compute is no longer the bottleneck.** 32s for 1k × 3 tools means 10k → ~5 min, 100k → ~1 hr, 1M → ~9 hr. Sample design and storage become the binding constraints, not execution.
2. **The variable-disjoint matcher-solvable slice is structurally homogeneous.** Vampire's proofs are uniformly 7 tactics; matcher's σ-space has only 334 distinct patterns; grind fails on ~55% by structural-heuristic limit. Phase 2 needs deliberately heterogeneous sampling, not just "more of the same."
3. **Completion was inert on Phase 1's slice** (variable-disjoint = NO_RULES_FROM_INPUT for every pair). Phase 2 reopens completion's role on samples where it contributes.
4. **Vampire's proofs aren't directly Stage-2-submittable.** They use ETP's `Superposition.lean` tactics (`superpose`, `mod_symm`, `subsumption`) the judge environment doesn't bundle. Translation is a future-phase item.
5. **Per-step grammar after canonicalization is ~140 templates** (not 1,450) for Vampire. This is the *real* learning surface — tractable for a small transformer.

### Goal

Produce a dataset that gives a downstream proof-generation model:
- **Genuine structural variety** in proof shape (multi-step rewriting + substitution + e-graph reasoning + paramodulation, not all collapsed to one shape)
- **Coverage of multiple equation orientability classes** (variable-disjoint AND `lhs_to_rhs`/`rhs_to_lhs`/`both`)
- **Negative training signal** (proofs that didn't work, tactic-state pairs where a step failed) at a comparable scale to positives
- A scale large enough (10k–100k transitions per training class) to fine-tune a small transformer to test the architecture before committing to larger runs

### Open decisions

These are the next questions to resolve. Each gets folded into a "Decisions log" once locked.

- **P2Q1 — Sample design.** Which slice(s) does Phase 2 target?
   - (A) 19,576-pair matcher∩completion intersection (matcher AND completion both prove)
   - (B) Wider TRUE corpus: stratified by orientability class × proof-survey category (substitution / rewrite-chain / Vampire-shaped / Egg-shaped)
   - (C) Specifically the *hard* pairs not solved by matcher (the 8.17M − 49,586 = ~8.1M pairs where matcher fails) — where multi-step proofs actually live
   - (D) A mix: small N from each of several slices to compare structural diversity, then commit the main sample
   - Recommend: (D) for a small pre-sample characterization, then a deliberate choice.

- **P2Q2 — Sample size.** Phase 1 was 1k; throughput says 10k–100k is feasible. What scale gives diminishing returns?
   - Drives by how many distinct *templates* the sample exposes — if 10k captures all 140 Vampire templates + all 334 σ patterns + comparable completion shape diversity, no need for 100k.

- **P2Q3 — Tool set additions.** Bring completion back (it'll contribute on non-var-disjoint pairs)? Add MagmaEgg (the Rust runner we deferred in Phase 1)? Both? Neither?

- **P2Q4 — Vampire→core translator.** Build it in Phase 2 (so Vampire proofs become Stage-2-submittable now), or defer to Phase 3?

- **P2Q5 — Lean version policy.** Stay at 4.29.1 (current stable for our stack), or bump to 4.30.0-rc2 when ETP upstream does the next bump (closer to Stage 2)? The bump is small but unblocked by ETP, not us.

- **P2Q6 — Pre-sample characterization.** Run small N=100 against 2–3 candidate samples (e.g., (A), a completion-only slice, a wider stratified slice) to measure structural diversity before committing the Phase 2 main sample. Worth doing before P2Q1 finalizes?

- **P2Q7 — Model design / training.** Does Phase 2 stop at "produce the dataset" and Phase 3 owns model design, or does Phase 2 train a baseline tokenizer + small transformer on the Phase 1 dataset to surface dataset-design issues early?

- **P2Q8 — Storage at scale.** If we go 10k+, the transitions parquet hits hundreds of MB (state-text strings dominate). Split per-tool? Compress state strings as separate column with dedup? Move data dir under `et_studies/X/data/` (gitignored) instead of `proof_generation_pilot/outputs/` (currently committed)?

### Phase 2 work breakdown (sketch, post-decisions)

1. **Pre-sample characterization** — small-N runs against candidate slices to inform P2Q1
2. **Tool additions** — bring completion (and optionally MagmaEgg) back online, smoke-test
3. **Sample build** — Phase 2 main sample per the P2Q1 / P2Q2 resolution
4. **Orchestration update** — adapt build_phase1_dataset.py for additional tools / larger scale (resumability, per-pair caching)
5. **Production run** — full Phase 2 dataset
6. **Synthesis writeup** — what's structurally new vs Phase 1; readiness for model training
7. **(Conditional on P2Q4)** — Vampire→core translator
8. **(Conditional on P2Q7)** — baseline tokenizer + model design notes
