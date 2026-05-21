# rewrite_chain_study

Deep characterization of the 1,636 `c_rewrite_chain` proofs in the ETP corpus — proofs that use Lean's `calc` block of `← h` / `h` rewrites to walk from E1's LHS through intermediate terms to E2's RHS.

## Headline

| Question | Answer |
|---|---|
| What's the deterministic-search ceiling? | **83.13% coverage at d_max=6**, going to 1,360/1,636 proofs |
| What rule set do these proofs use? | **99.7% within {h, ← h}** — almost never use other lemmas |
| Are the proof chains multimodal? | **87% unimodal** (one shortest chain). 13% have ≥2 shortest chains, mostly small alternates |
| Architectural recommendation | Neural-guided search targeting the 17% hard tail, after first trying d_max=8/10 and tighter size caps |

The 17% hard tail is the meaningful research target for proof generation. Pure search exhausts at d_max=10; the remainder needs either heuristic pruning or model-guided exploration.

## Contents

| Path | What |
|---|---|
| `outputs/synthesis_report.md` | Full study writeup with prover-architecture recommendation |
| `outputs/coverage.json` | Coverage by depth (d_max=4,6,8,10) |
| `outputs/ruleset_analysis.json` | Rule-set distribution |
| `outputs/tactic_structure.json` | Per-proof tactic-step structure |
| `outputs/task3_intermediate_states.json` | Intermediate-term shapes |
| `outputs/task5_multimodality.json` | Multimodality breakdown |
| `code/engine.py` | The unsafe rewrite engine used by this study (search/find_path/enumerate_chains) |
| `scripts/build_rewrite_chain_dataset.py` | Dataset builder |
| `scripts/run_rewrite_engine.py` | Engine harness |
| `scripts/run_multimodality.py` | Multimodality analysis |
| `scripts/analyze_rewrite_traces.py` | Trace analysis |
| `data/rewrite_chain_dataset.parquet`, `rewrite_engine_traces.parquet` | Per-proof data (gitignored) |

The unsafe engine variant lives here in the study. The pilot uses a **safe** variant at `proof_generation_pilot/methods/safe_engine/` — direction-restricted bidirectional BFS, distinct algorithm.

## Archived plan

`archive/rewrite_chain_study_plan.md`.
