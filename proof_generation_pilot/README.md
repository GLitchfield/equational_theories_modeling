# proof_generation_pilot/

The active phase. Targets the ETP Stage 2 competition format: generate Lean proofs for TRUE implications + `Fin n` matrix counterexamples for FALSE implications.

**Status: methods migrated in; pilot plan pending.** The three deterministic methods (matcher, completion, safe rewrite engine) have been carved out of their original studies and moved here as the starting point. A pilot plan markdown will land at `proof_generation_pilot/pilot_plan.md` and define the experimental cascade + FALSE-search strategy.

## Methods in place

| Path | Source study | Headline |
|---|---|---|
| `methods/matcher/` | `et_studies/matcher_deliverable/` | First-order matcher: 59.8% of directly-proved set, 0 false positives across 13.8M GT FALSE |
| `methods/completion/` | `et_studies/completion_exploration/` | Plain + ordered Knuth-Bendix completion: 13.46% combined sound coverage of 22M pairs |
| `methods/safe_engine/` | `et_studies/coverage_measurement/` (safe-rewrite primitive) | Direction-restricted bidirectional BFS over `lib/rewrite/` primitives; 1.6% of combined TRUE coverage |

Combined sound TRUE-side coverage of the deterministic stack is **6.54% of GT TRUE** (vs 46.07% structural ceiling — see `et_studies/coverage_measurement/`). The pilot's job: lift coverage beyond this ceiling, ideally via neural-guided search or proof-corpus learning, while preserving zero-false-positive correctness.

## Contents

| Path | What |
|---|---|
| `methods/matcher/` | Deterministic first-order matcher (Track 1 of the proof-survey recommendation) |
| `methods/completion/` | Plain KB (`kb.py`) + ordered/unfailing KB (`ordered.py`) |
| `methods/safe_engine/` | Bidirectional safe-rewrite search (`safe_search.py`) |
| `scripts/` | Pilot scripts (currently empty; populated when the pilot plan lands) |
| `data/processed/` | Per-pilot derived data (currently empty) |
| `outputs/` | Pilot outputs (currently empty) |
| `notebooks/` | Exploratory notebooks (currently empty) |

## Shared dependencies

All three methods import from `lib/`:
- `lib/data/parser.py` — equation parsing
- `lib/rewrite/matching.py`, `lib/rewrite/unification.py`, `lib/rewrite/lpo.py` — first-order primitives

The unsafe rewrite engine variant (used by the rewrite-chain study) lives at `et_studies/rewrite_chain_study/code/engine.py` and is not part of the pilot — only the *safe* variant ported here.

## Tests

Pilot-method tests live at the repo-root `tests/` directory (flat layout): `test_matcher.py`, `test_completion_kb.py`, `test_completion_ordered.py`, `test_coverage_safe_search.py`.
