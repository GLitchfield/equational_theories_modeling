# proof_investigation

Phase 8 intermediate-term analysis on hard TRUE proofs. **PAUSED** — completed Phase 1 (repository exploration + format verification), then paused when project scope changed. May resume.

## Status

| Phase | State |
|---|---|
| Phase 1: repository exploration + format verification | Complete (see `outputs/phase1_exploration_notes.md`) |
| Phase 2: single-pair end-to-end trace per category | Pending |
| Phase 3: parser implementation | Pending |
| Phase 4: bulk extraction of all 722 eligible pairs | Pending |
| Phase 5: aggregation + three hypothesis tests | Pending |
| Phase 6: synthesis writeup | Pending |

The Phase 1 exploration also produced a durable structural finding that was carved out into its own study at `et_studies/etp_structure_analysis/` — the ~11K seed proofs + Closure.lean derives 8M+ transitive labels architecture.

## Research question

Given the dense-row residual that frontier-dies under safe rewriting (the bottleneck identified in `et_studies/coverage_measurement/`), test three hypotheses on hard TRUE proofs (Vampire / Egg / complex categories):
1. **Higher-variable** — do intermediate terms introduce fresh variables beyond E2?
2. **Pattern learnability** — are intermediate-term shapes low-entropy per E1?
3. **Density connection** — does intermediate-term complexity correlate with row density?

The output would characterize what a successor method (e.g., narrowing- or superposition-based) would need to produce to extend beyond the 14.20% engine plateau.

## Contents

| Path | What |
|---|---|
| `proof_investigation_plan.md` | The full study plan (paused) |
| `outputs/phase1_exploration_notes.md` | Phase 1 findings |
| `data/` | Per-phase derived data (currently empty — Phase 2+ not run) |

## Resuming

When the user picks this back up, paths inside the plan markdown predate the refactor — they reference `outputs/analyses/proof_investigation/` and `data/processed/proof_investigation/`. New locations:
- Outputs: `et_studies/proof_investigation/outputs/`
- Data: `et_studies/proof_investigation/data/`
- Scripts (when written): `et_studies/proof_investigation/scripts/`
