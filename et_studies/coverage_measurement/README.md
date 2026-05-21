# coverage_measurement

TRUE-side coverage measurement of the full deterministic stack — matcher + safe rewrite engine + completion — over the 22M-pair corpus. Bounds what the engineering can deliver before learned models are introduced.

## Headline

| Metric | Value |
|---|---|
| Combined sound TRUE coverage | **6.54% of GT TRUE** |
| Structural ceiling for rewriting methods on this corpus | 46.07% of GT TRUE |
| Combined coverage / ceiling | **14.20%** — the engine plateau |
| Dominant within-ceiling failure mode | `frontier_died` at **85.74%** of within-ceiling residual |

Architectural distribution of the 6.54%:
- Completion: 92.8%
- Matcher: 5.6% (the variable-disjoint slice the matcher uniquely catches)
- Safe rewrite engine: 1.6%

Depth sweep showed depth is not the constraint — coverage plateaus at d=4. The bottleneck is structural reachability of safe first-order rewriting on these equations, not search budget.

Cross-cut against the proof-survey categories: Vampire/Egg proof categories bottom at 2.29% / 0.54% engine solve rate respectively. This confirms safe rewriting cannot substitute for narrowing / superposition; those categories need the upstream tools the proof survey recommends delegating to.

The honest deliverable here is a **bounding result**, not an extension. The pilot's job is to go beyond this ceiling without losing the zero-false-positive correctness.

## Contents

| Path | What |
|---|---|
| `outputs/coverage_measurement_synthesis.md` | Full study writeup |
| `outputs/phase1_residual_summary.json` | Initial residual characterization |
| `outputs/phase2_preconditions.json` | Engine precondition stats |
| `outputs/phase3_depth_sweep.json` | Depth-vs-coverage curve |
| `outputs/phase4_soundness.json` | Soundness check (zero false positives) |
| `outputs/phase5_combined_coverage.json` | Combined matcher+engine+completion stats |
| `outputs/phase6_residual_characterization.json` | Per-E1 stratification, category cross-cut, variable-disjoint stratum |
| `scripts/phase{1,2,4,5,6}_*.py` | Per-phase reproducers (phase 3 was a depth sweep over phase 2 outputs) |
| `data/sample.parquet`, `engine_results_*.parquet` | Per-pair engine results (gitignored) |

The safe-rewrite primitive was promoted to the pilot at `proof_generation_pilot/methods/safe_engine/safe_search.py`. 8 unit tests cover it.

## Archived plan

`archive/coverage_measurement_plan.md`.
