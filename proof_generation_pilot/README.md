# proof_generation_pilot/

The active phase. Targets the ETP Stage 2 competition format: generate Lean proofs for TRUE implications + `Fin n` matrix counterexamples for FALSE implications.

**Status: Phase 1 complete; Phase 2 in design.** Phase 1 built a 1,000-pair dataset of per-tactic `(state, tactic, next_state)` transitions from three proof-producing tools (matcher, Vampire, `grind`) on variable-disjoint TRUE implications, with every proof stepped through Lean 4.29.1 via PyPantograph; see [`outputs/phase1_synthesis.md`](outputs/phase1_synthesis.md). Phase 2 (a broader, structurally varied sample) is resolving its open decisions in [`pilot_plan.md`](pilot_plan.md), alongside the uniform proof-output language in [`proof_spec.md`](proof_spec.md).

## Methods in place

| Path | Source study | Headline |
|---|---|---|
| `methods/matcher/` | `et_studies/matcher_deliverable/` | First-order matcher: 59.8% of directly-proved set, 0 false positives across 13.8M GT FALSE |
| `methods/completion/` | `et_studies/completion_exploration/` | Plain + ordered Knuth-Bendix completion: 13.46% combined sound coverage of 22M pairs |
| `methods/safe_engine/` | `et_studies/coverage_measurement/` (safe-rewrite primitive) | Direction-restricted bidirectional BFS over `lib/rewrite/` primitives; 1.6% of combined TRUE coverage |
| `methods/vampire/` | new in Phase 1 | Vampire 4.9 → Lean adapter (TPTP encoding + ETP's superposition `leanify`); 1000/1000 on the Phase 1 sample |
| `methods/grind/` | new in Phase 1 | `intro G _ h; grind` using Lean's built-in e-graph tactic; 447/1000 on the Phase 1 sample |

Combined sound TRUE-side coverage of the deterministic stack is **6.54% of GT TRUE** (vs 46.07% structural ceiling — see `et_studies/coverage_measurement/`). The pilot's job: lift coverage beyond this ceiling, ideally via neural-guided search or proof-corpus learning, while preserving zero-false-positive correctness.

## Contents

| Path | What |
|---|---|
| `methods/matcher/` | Deterministic first-order matcher (Track 1 of the proof-survey recommendation) |
| `methods/completion/` | Plain KB (`kb.py`) + ordered/unfailing KB (`ordered.py`) |
| `methods/safe_engine/` | Bidirectional safe-rewrite search (`safe_search.py`) |
| `methods/vampire/`, `methods/grind/`, `methods/matcher/to_lean.py` | Per-tool adapters that emit `def submission := by ...` Lean source |
| `lean_project/` | `PilotShim` Lake project (Lean 4.29.1) that imports ETP + Mathlib for PyPantograph; requires ETP as a sibling checkout |
| `scripts/sample_pairs.py` | Draws the seeded Phase 1 samples (`data/processed/phase1_sample_{100,1k}.parquet`) |
| `scripts/build_phase1_dataset.py` | Phase 1 orchestration: runs the three tools per pair, steps proofs through PyPantograph, writes `proofs.parquet` + `transitions.parquet` |
| `data/processed/` | Pilot samples (Phase 1 samples; Phase 2 10-pair characterization sample) |
| `outputs/` | `phase1_1k/` (production run), `phase1_smoke/` (100-pair smoke), `phase1_test_5/`, `phase1_synthesis.md`, `q12_recon.md` (ETP Vampire/Egg recon) |
| `notebooks/` | Exploratory notebooks (currently empty) |

Running the Phase 1 pipeline needs the external tools listed under "Proof-generation pilot tooling" in the root README: Lean 4.29.1 with ETP built, PyPantograph, and Vampire (on `PATH` or via `VAMPIRE_BIN`).

## Shared dependencies

The matcher, completion, and safe-engine methods import from `lib/`:
- `lib/data/parser.py` — equation parsing
- `lib/rewrite/matching.py`, `lib/rewrite/unification.py`, `lib/rewrite/lpo.py` — first-order primitives

The unsafe rewrite engine variant (used by the rewrite-chain study) lives at `et_studies/rewrite_chain_study/code/engine.py` and is not part of the pilot — only the *safe* variant ported here.

## Tests

Pilot-method tests live at the repo-root `tests/` directory (flat layout): `test_matcher.py`, `test_completion_kb.py`, `test_completion_ordered.py`, `test_coverage_safe_search.py`.
