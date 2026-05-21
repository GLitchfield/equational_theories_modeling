# et_studies/

Frozen studies on the Equational Theories Project corpus. Each study has its own subdirectory with `outputs/` (results), `scripts/` (reproducers), `data/` (per-study processed parquets, mostly gitignored), and where applicable `code/` (study-specific modules).

## Index

| Study | Headline | Synthesis |
|---|---|---|
| **proof_survey** | Surveyed the ETP's 10,676 explicit-implication proofs; recommended a three-track architecture (matcher / rewrite-chain decoder / upstream-tool delegation) + brute-force `Fin n` FALSE search | [outputs/README.md](proof_survey/outputs/README.md) |
| **matcher_deliverable** | Deterministic first-order matcher: 59.8% of directly-proved set, **0 false positives** across 13.8M GT FALSE pairs | [outputs/README.md](matcher_deliverable/outputs/README.md) |
| **proof_decoder** | Phase 1 research decoder: both-novel σ-exact 96.32% (JEPA) vs 5.6% (random ctl). Established that JEPA embeddings preserve the proof witness for the variable-substitution category | (no synthesis .md; see `outputs/runs/proof_decoder_jepa/diagnostics_report.md` in `embeddings/outputs/`) |
| **rewrite_chain_study** | Deep characterization of the 1,636 c_rewrite_chain proofs: 83.13% deterministic-search coverage at d_max=6, 99.7% rules within {h, ← h}, 87% unimodal | [outputs/synthesis_report.md](rewrite_chain_study/outputs/synthesis_report.md) |
| **completion_exploration** | KB completion (plain + ordered): 13.46% sound combined coverage of 22M pairs (TRUE from ordered, FALSE from plain); ordered lifts TRUE 3× | [outputs/synthesis_report.md](completion_exploration/outputs/synthesis_report.md) |
| **coverage_measurement** | TRUE-side coverage of matcher + rewrite engine + completion: 6.54% of GT TRUE soundly covered; structural ceiling 46.07%; engine plateau at 14.20% of ceiling; frontier_died at 85.74% of within-ceiling residual | [outputs/coverage_measurement_synthesis.md](coverage_measurement/outputs/coverage_measurement_synthesis.md) |
| **failure_overlap** | Cross-run failure-overlap analysis (8 trained models): stable 32,496 always-wrong-trained pairs (99.99% TRUE), TRUE-by-vacuity dominant | [outputs/README.md](failure_overlap/outputs/README.md) |
| **etp_structure_analysis** | Carved out from proof_investigation Phase 1: ETP closure architecture — ~11K direct proofs + Closure.lean derives 8M+ transitive labels from ~4 MB source evidence | [outputs/etp_proof_structure_note.md](etp_structure_analysis/outputs/etp_proof_structure_note.md) |
| **proof_investigation** | Phase 8 intermediate-term analysis. **PAUSED** — paused after Phase 1 exploration when scope changed; may resume. Plan in `proof_investigation_plan.md` at the study root | [outputs/phase1_exploration_notes.md](proof_investigation/outputs/phase1_exploration_notes.md) |

## What goes here

A study is a self-contained research effort with a defined question, a synthesis report describing what was learned, and the scripts + data needed to reproduce its outputs. Once a study lands in `et_studies/`, it's **frozen** — the outputs document what was true at the time; the scripts may be left "broken-but-readable" if subsequent refactors change paths or APIs.

When a study finds something durable that future work should reuse (a matcher, a completion engine, a probing technique), the durable artifact gets promoted out of the study — into `lib/`, `proof_generation_pilot/`, or `embeddings/` as appropriate. The study itself stays as the historical record.

## Archived plans

Each study's plan markdown is in `archive/`: `proof_corpus_survey_plan.md`, `matcher_plan.md`, `proof_decoder_plan.md`, `rewrite_chain_study_plan.md`, `completion_exploration_plan.md`, `coverage_measurement_plan.md`. The active plan that's paused is at `et_studies/proof_investigation/proof_investigation_plan.md`.
