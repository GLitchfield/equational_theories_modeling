# Proof Generation Pilot — Plan

**Status:** in-design. Iterating on decisions before any code is written.

**Goal of the pilot:** generate Lean proofs for ETP Stage 2 — TRUE implications and `Fin n` FALSE counterexamples — using a combination of deterministic methods (matcher, completion, safe rewrite engine) and learned models. This plan covers the first phase only: building the initial dataset that will feed downstream model design.

## Phase 1 — Initial dataset

### What

Build a dataset of ~1,000 (pair, tool, proof, intermediate-states) tuples across **three** proof-producing tools (matcher, Vampire, lean-egg) run over the same sample of variable-disjoint TRUE implication pairs. Validate all generated Lean against the chosen Lean environment. Uniform tactic-mode output (`def submission := by ...`) so PyPantograph captures per-step states consistently.

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
- **Q3** (Lean validation environment): **Lean 4.20.1** (ETP's version). Reasoning: compatibility with ETP's Vampire/Egg adapters and tooling, and LeanDojo is more likely to work on the older version. Caveat: Stage 2 judge runs 4.30.0-rc2 — a porting step (4.20.1 → 4.30.0-rc2) will be needed at the submission boundary in a later phase. Later Lean versions may make downstream work easier; revisit when it becomes the bottleneck.
- **Q4** (Vampire / Egg sourcing): **Vampire re-runs via ETP's existing driver; "Egg" substituted by `lean-egg`** (the Lean tactic, not the external Rust MagmaEgg).
   - **Vampire**: re-run on the 1k sample using `equational_theories/Generated/VampireProven/src/vampire_proofs.py` — ETP's TPTP encoder + `leanify()` translator already produce tactic-mode Lean that uses ETP's `superpose` tactic (`Superposition.lean`). Install the Vampire binary locally (the driver expects `~/Downloads/vampire`). Tactic-mode output gives LeanDojo intermediate states.
   - **Egg → `grind`** (Lean core). Two prior substitutions: MagmaEgg (external Rust crate, undocumented input format, expensive setup) → lean-egg (Lean tactic, used in ETP's `Subgraph.lean`) → **`grind`** (Lean's built-in e-graph-based tactic). lean-egg is officially deprecated; its own README recommends `grind`. Same e-graph reasoning, no external Lake dep, actively maintained as Lean core. Same atomic shape from PyPantograph's view (one tactic call = one transition); the value of this slot is "e-graph-based proof shape in the dataset," not per-step traces.
   - Open sub-question: once all 4 tools have outputs on the same pair, do we use Vampire/lean-egg's proof for pairs where a "simpler" tool (matcher) also succeeded? Deferred until we see what the outputs look like.
- **Q5** (intermediate proof-state capture): **use PyPantograph directly at Pantograph commit `3ce335eb` (Lean 4.20.1)** — skip LeanDojo entirely. Capture the goal state after each tactic step for tactic-mode proofs. Rationale: intermediate states are needed to determine the **minimum token grammar** for the downstream proof-generation model. Schema requirement is specific: capture **raw textual** tactic + goal-state output, not just a structured AST. Setup is a Phase 1 step: pin Pantograph submodule, build the REPL via `lake build repl` against Lean 4.20.1, install PyPantograph against that pinned source, smoke-test on one proof end-to-end before sampling begins.

   **Why PyPantograph and not LeanDojo:**
   - LeanDojo v1 supports Lean 4.20.1 but is deprecated (no future bug fixes).
   - LeanDojo-v2 is the active project but tracks Pantograph `dev`, currently pinned to Lean 4.29.1 — version mismatch with ETP at 4.20.1, and Pantograph compatibility is strict per its own docs.
   - LeanDojo-v2 also bundles a heavy ML stack (PyTorch, transformers, deepspeed, peft, trl, openai) for its training/inference agents (`agent/`, `trainer/`, `prover/`, `lean_progress/`). We use *none* of that for Phase 1.
   - LeanDojo-v2's Lean-interaction core (`lean_dojo_v2/lean_dojo/`) is itself a wrapper around PyPantograph. Using PyPantograph directly gives us the same primitive with a much smaller install footprint and no version-mismatch risk.

   **Known issues with this approach + the broader Phase 1 design:**
   - **Pantograph submodule pinning is manual.** No tagged release maps cleanly to Lean 4.20.1; we identify the commit by date (Pantograph's `lean-toolchain` changed to v4.20.1 at commit `3ce335eb` on 2025-06-17, then to v4.21.0 two weeks later). We point PyPantograph's `src` submodule at that commit and rebuild. One-time work, but not a stable supported configuration.
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
   - `scripts/build_phase1_dataset.py` — orchestration: per-pair, run all 4 tools, validate in Lean 4.20.1 via LeanDojo, capture intermediate states, write parquet rows.
   - `data/processed/phase1_dataset/` — output parquet location.
- **Q10** (Storage format): **one combined parquet** with a `tool` column. **~3,000 rows** (1k pairs × 3 tools) — joins across tools cheap, uniform schema. Schema sketch:
   ```
   pair_id          int       # local 0-based index into the 1k sample
   e1_id            int       # equation IDs from ETP
   e2_id            int
   tool             str       # 'matcher' | 'vampire' | 'lean_egg'
   proof_lean       str       # full Lean source ('def submission := by ...'); empty if no proof
   status           str       # 'proved' | 'no_proof' | 'timeout' | 'error'
   intermediate_states list[str]  # ordered goal-state texts from PyPantograph
   tactic_used      list[str]  # ordered tactic texts paired with each transition
   wall_clock_ms    int
   validated        bool      # did the proof_lean compile and close the goal under Lean 4.20.1?
   validation_log   str       # stderr / error trace if not validated
   ```
- **Q11** (Sample-size budget): **stepped: 100 → 1k.** Build the pipeline end-to-end on 100 pairs first to surface install/integration issues + estimate wall-clock per tool. Scale to 1k once the 100-pair run is clean. 1k is the Phase 1 target, not a hard cap; later phases will broaden it.

## Known issues with the current Phase 1 design

Surfaced during decision walk-through; tracked here so they're visible before implementation begins.

1. **Per-step state richness varies by tool.** Vampire produces multi-step traces (~5–20 transitions per pair); matcher gives one transition (post-`intro`); lean-egg gives one transition (post-`egg`). Vampire dominates the grammar-discovery signal. Atomic captures from matcher and lean-egg are still useful as anchor data but don't carry multi-step structure.

2. **Pantograph version pinning is manual and unsupported.** No tagged release of Pantograph or PyPantograph maps to Lean 4.20.1. We pin by commit hash (`3ce335eb`, 2025-06-17) and rebuild. Acceptable for Phase 1 but not a stable long-term posture.

3. **Lean 4.20.1 → 4.30.0-rc2 porting** for Stage 2 submission is a deferred step. Not a Phase 1 problem but a known later-phase obligation per Q3.

4. **Vampire `leanify()` may raise `NotImplementedError`** on uncommon proof patterns (superposition + backward demodulation + equality resolution are handled; others may not be). Per-pair failures will be tracked and characterized in Phase 1 outputs.

5. **Vampire setup cost.** Vampire binary install — verified working (v4.9, ETP's `leanify()`-compatible format) before Phase 1 sampling. `grind` requires no separate install (Lean core).

6. **ETP's `lean-toolchain` patched locally** from `v4.20.1` → `v4.20.0` to match Mathlib's pinned cache (same Lean binary; the v4.20.1 toolchain string was a republish of the same v4.20.0 release). The patch is in `~/git/equational_theories/lean-toolchain` — a local modification to the sibling repo. Not pushed upstream; not in our repo. To reproduce: `cp .lake/packages/mathlib/lean-toolchain ./lean-toolchain` in the ETP checkout after `lake update`.

These are issues to *manage*, not blockers. The plan proceeds; we revisit if any turn out worse than expected during the 100-pair smoke run.

## Phase 1 setup status

| Item | Status |
|---|---|
| elan + Lean 4.20.0 + Lean 4.20.1 | ✓ installed (Lean 4.20.0 binary is the same as v4.20.1 — toolchain string republish) |
| Vampire 4.9 binary at `~/Downloads/vampire` | ✓ installed; smoke test produces clean leanify-compatible output |
| Pantograph + PyPantograph (Pantograph pinned to commit `3ce335eb`) | ✓ built; `pip install -e ~/git/PyPantograph` succeeded; `Server()` smoke import OK |
| ETP `lake update` + Mathlib cache fetch + `lake build` | ✓ 4334 modules compiled (Mathlib cache used) |
| Lean project shim at `proof_generation_pilot/lean_project/` | ✓ `PilotShim` lib builds; `Smoke.lean` proves `Equation2 → Equation1` |
| `grind` Lean core tactic | ✓ no install needed (built into Lean 4.20.1) |

## Deferred to Phase 2

- **Completion as a proof producer.** Move sample to non-variable-disjoint pairs (or a stratified mix) where completion structurally contributes via rewrite chains. The Phase 1 completion code stays in `proof_generation_pilot/methods/completion/` ready to wire in.
- **Stratification / sample-shape questions.** Mixing variable-disjoint and non-variable-disjoint pairs introduces heterogeneity that Phase 1's clean single-category focus deliberately avoids. Phase 2 owns this.
- **Birkhoff substitution inside completion** (option (d) from the design discussion). Real engineering that converges on Vampire-style paramodulation. Out of scope unless Phase 2 surfaces a specific need.

## Phase 1 work breakdown (post-decisions)

1. **Q12 recon.** Read ETP's Vampire/Egg drivers; produce a short writeup at `proof_generation_pilot/outputs/q12_recon.md`. If the Lean translators don't exist, decide whether to build them.
2. **Tooling install.** Vampire, Egg, LeanDojo. Verify each runs end-to-end on one equation.
3. **Lean project shim.** Build a small Lean 4.20.1 project under `proof_generation_pilot/` that imports ETP's equation definitions. LeanDojo points at this.
4. **Sample build.** `scripts/sample_pairs.py` → 100-row + 1k-row variants from the 30K pool, seeded.
5. **Per-tool adapters.** Matcher (wrap existing `lean_proof_tactic` in `def submission := by intro G _ h <vars>; <tactic>`), Vampire (thin wrapper around ETP's `vampire_proofs.py`), lean-egg (Lean-file template emitting `def submission := by egg [h]`).
6. **Orchestration pipeline.** `scripts/build_phase1_dataset.py` — runs the 4 tools per pair, validates in Lean, captures intermediate states, writes the parquet.
7. **100-pair smoke run.** Resolve any issues surfaced. Measure wall clock per tool.
8. **1k full run.** Production Phase 1 dataset.
9. **Synthesis.** Short writeup at `proof_generation_pilot/outputs/phase1_synthesis.md` — what each tool produced, intermediate-state vocab discovered, surprises, recommended next steps.

## Phase 2+ (placeholder)

Subsequent phases — full dataset construction, model design, training — depend on what Phase 1 reveals. Plan extends once Phase 1 closes.
