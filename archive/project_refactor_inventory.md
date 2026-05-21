# Refactor Phase 1 — Inventory + Mapping + Resolutions

**Repo**: `~/git/equational_theories_modeling/` (the modeling repo; sibling `~/git/equational_theories_data/` stays as-is and is referenced via the existing `data/raw/invariants` symlink).

**Decisions baked in (from review):**

- **`lib/`** for shared primitives (not `core/`).
- **Engine variants**: safe → `proof_generation_pilot/methods/safe_engine/`; unsafe (`src/rewrite/engine.py`) → `et_studies/rewrite_chain_study/code/`.
- **Studies promoted**: `failure_overlap/`, `matcher_deliverable/`, `proof_decoder/`, `rewrite_chain_study/`, `etp_structure_analysis/` (carved out of proof_investigation).
- **Naming**: `et_studies/proof_survey/` (drop `_corpus_`).
- **Tests**: stay at repo-root `tests/`, mirroring the new tree internally.
- **Scripts**: per-area assignment (JEPA → embeddings, per-study phase scripts → that study, method runners → pilot).
- **Large outputs**: `git mv` for tracked, plain `mv` for gitignored; flagged per directory.
- **`archive/`**: stays at root (historical plans).
- **`wandb/`**: stays at root (ML tracking infrastructure).
- **`probing*/` consolidation**: merge `outputs/probing/` + `outputs/probing_results/` → `embeddings/outputs/probing/`.

---

## 1. Inventory

### Top-level directories (current)

| dir | files | size | notes |
|---|---|---|---|
| `src/` | 73 | 664 K | all source code; will split across `lib/`, `embeddings/src/`, `proof_generation_pilot/methods/`, `et_studies/rewrite_chain_study/code/` |
| `scripts/` | 53 | 660 K | mixed: 15 JEPA-era, 13 method-runners, ~25 per-study phase scripts |
| `tests/` | 49 | 1.0 M | 23 test modules; stay at root, mirror tree internally |
| `data/raw/` | 2 | 56 M | ground truth + ETP equation list; stays at root |
| `data/processed/` | 27 | 47 M | mixed per-study artifacts + a few shared parquets |
| `data/splits/` | 3 | 7.5 M | JEPA holdout indices → embeddings |
| `outputs/analyses/` | — | 39 M | 8 study output dirs |
| `outputs/checkpoints/` | — | 1.4 G | JEPA + decoder checkpoints, gitignored |
| `outputs/runs/` | — | 1.6 G | JEPA per-run records, gitignored |
| `outputs/embeddings/` | — | 17 M | v1.0 embeddings deliverable |
| `outputs/probing/` | — | 130 M | probing artifacts (gitignored .npy) |
| `outputs/probing_results/` | — | 4.0 K | probing JSON results |
| `outputs/figures/` | — | 480 K | plots |
| `outputs/failure_analysis/` + `outputs/failure_analysis_m1p5_vic0p1/` | — | 243 M | JEPA failure analyses |
| `outputs/results/` | — | 68 K | misc JEPA |
| `configs/` | 36 | 172 K | JEPA training configs |
| `archive/` | 15 | 236 K | 12 archived plans + dual_encoder dir |
| `notebooks/` | 1 | 4.0 K | empty placeholder |
| `wandb/` | 326 | 323 M | ML tracking (gitignored) |
| `tests/` | 49 | 1.0 M | (counted above) |

### Outputs sub-summary

```
outputs/
├── analyses/                39M   — frozen-study artifacts (8 study dirs)
├── checkpoints/             1.4G  — JEPA + decoder checkpoints (gitignored)
├── runs/                    1.6G  — JEPA per-run records (gitignored .parquet/.npy)
├── failure_analysis/        127M  — JEPA failure_analysis (gitignored)
├── failure_analysis_m1p5_vic0p1/   116M  — variant of above
├── probing/                 130M  — probing embeddings (.npy gitignored)
├── probing_results/         4.0K  — probing JSON (committed)
├── embeddings/              17M   — v1.0 deliverable (npy gitignored, .pt + MODEL_CARD committed)
├── figures/                 480K  — plots (gitignored)
├── results/                 68K   — misc
└── *.log                    780K  — training logs at root
```

---

## 2. Import graph

### `src/` internal dependencies

```
src/data/parser.py                                    (no internal deps)
    ↑
    used by: src/{matcher, rewrite, completion, coverage, data/dataset}

src/data/dataset.py  → src.data.parser
src/data/splits.py                                    (no internal deps)

src/rewrite/matching.py     → src.data.parser
src/rewrite/lpo.py          → src.data.parser, src.rewrite.unification
src/rewrite/unification.py  → src.data.parser
src/rewrite/engine.py       → src.data.parser, src.rewrite.matching       ← UNSAFE engine

src/coverage/safe_search.py → src.data.parser, src.rewrite.{matching, unification}

src/matcher/matcher.py      → src.data.parser

src/completion/kb.py        → src.data.parser, src.rewrite.{lpo, matching, unification}
src/completion/ordered.py   → src.completion.kb, src.data.parser, src.rewrite.{lpo, matching, unification}

src/model/encoder.py        (torch only)
src/model/predictor.py      (torch only)
src/model/jepa.py           → src.model.{encoder, predictor}
src/model/aux_heads.py      (torch only)
src/model/pair_classifier.py (torch only)

src/decoder/tokenizer.py    (stdlib only)
src/decoder/decoder.py      → src.decoder.tokenizer
src/decoder/eval_helpers.py → src.decoder.tokenizer

src/training/losses.py      (torch only)
src/training/sigreg.py      (torch only)
src/training/scheduler.py   (stdlib only)
src/training/train.py       → src.data.dataset, src.model.{aux_heads, encoder, jepa, predictor}, src.training.{losses, scheduler, sigreg}
src/training/recording.py   → src.data.dataset, src.model.{encoder, jepa, predictor}, src.training.train

src/probing/probes.py       (torch + sklearn, no internal deps)
src/baselines/              (empty __init__.py only)
src/evaluation/             (empty __init__.py only)
```

**Dependency layers (bottom-up):**

1. **Pure stdlib / 3rd-party leaves**: `data/parser.py`, `data/splits.py`, `decoder/tokenizer.py`, `model/{encoder,predictor,aux_heads,pair_classifier}.py`, `training/{losses,scheduler,sigreg}.py`, `probing/probes.py`.
2. **Layer 1**: `data/dataset.py`, `rewrite/{matching,lpo,unification}.py`, `matcher/matcher.py`, `decoder/{decoder,eval_helpers}.py`, `model/jepa.py`.
3. **Layer 2**: `rewrite/engine.py` (unsafe), `coverage/safe_search.py`, `completion/kb.py`, `training/train.py`.
4. **Layer 3**: `completion/ordered.py`, `training/recording.py`.

### Script categorization

**JEPA-era scripts (import `src.{model, training, evaluation, baselines, decoder, probing}`):**

`analyze_mlp_vacuity.py`, `austin_eval.py`, `build_decoder_dataset.py`*, `embedding_spectrum.py`, `eval_predictions_on_split.py`, `evaluate_checkpoint.py`, `extract_probe_embeddings.py`, `record_run.py`, `run_experiment.py`, `run_probes.py`, `save_embeddings.py`, `save_predictions.py`, `train.py`, `train_decoder.py`, `train_pair_classifier.py`

*`build_decoder_dataset.py` also imports symbolic methods (matcher) — it's a bridge script (decoder dataset built from matcher proofs).

**Symbolic-method scripts (import `src.{matcher, completion, coverage, rewrite}`):**

`analyze_rewrite_traces.py`, `build_decoder_dataset.py`, `phase2_run_completion.py`, `phase3_decision_procedure.py`, `phase4_decision_procedure.py`, `phase4_run_ordered_completion.py`, `phase4a_classify_failed_cps.py`, `phase5_finite_models.py`, `run_matcher.py`, `run_multimodality.py`, `run_rewrite_engine.py`, `scripts/coverage_measurement/phase{1,2,4,5,6}_*.py`

**Pure-data / analysis scripts (no `src.{methods, model}` imports):**

`analyze_austin.py`, `analyze_distances.py`, `analyze_equation_concentration.py`, `analyze_failure_overlap.py`, `build_equation_invariants.py`, `build_pair_table.py`, `build_rewrite_chain_dataset.py`, `build_splits.py`, `eval_decoder_majority_baseline.py`, `extract_run_diagnostics.py`, `parse_equations.py`, `phase0_variable_inclusion.py`, `probe_decoder_linear.py`, `survey_*.py` (7 survey scripts)

---

## 3. Target layout

```
~/git/equational_theories_modeling/
│
├── lib/                                  # NEW — shared primitives
│   ├── data/                             # was src/data/  (parser, dataset, splits)
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   ├── dataset.py
│   │   └── splits.py
│   ├── rewrite/                          # was src/rewrite/{matching,lpo,unification}.py
│   │   ├── __init__.py
│   │   ├── matching.py
│   │   ├── lpo.py
│   │   └── unification.py
│   └── __init__.py
│
├── embeddings/                           # NEW — JEPA + decoder + MLP
│   ├── src/
│   │   ├── baselines/                    # was src/baselines/
│   │   ├── model/                        # was src/model/
│   │   ├── training/                     # was src/training/
│   │   ├── evaluation/                   # was src/evaluation/
│   │   ├── decoder/                      # was src/decoder/
│   │   └── probing/                      # was src/probing/
│   ├── scripts/                          # 15 JEPA-era scripts
│   ├── configs/                          # was configs/ (all JEPA configs)
│   ├── data/
│   │   └── splits/                       # was data/splits/
│   ├── outputs/
│   │   ├── checkpoints/                  # was outputs/checkpoints/   (1.4 GB, mv)
│   │   ├── runs/                         # was outputs/runs/          (1.6 GB, mv)
│   │   ├── embeddings/                   # was outputs/embeddings/    (17 M, partial git mv)
│   │   ├── probing/                      # was outputs/{probing, probing_results}/  (consolidated)
│   │   ├── failure_analysis/             # was outputs/failure_analysis/  (mv)
│   │   ├── failure_analysis_m1p5_vic0p1/ # was same  (mv)
│   │   ├── figures/                      # was outputs/figures/
│   │   ├── results/                      # was outputs/results/
│   │   └── *.log                         # was outputs/*.log
│   └── README.md                         # JEPA-phase recap, key results
│
├── et_studies/                           # NEW — frozen analyses
│   ├── README.md                         # index of studies
│   ├── proof_survey/
│   │   ├── outputs/                      # was outputs/analyses/proof_survey/
│   │   ├── scripts/                      # was scripts/survey_*.py
│   │   └── README.md
│   ├── matcher_deliverable/              # matcher's parquet + readme (NOT method code)
│   │   ├── outputs/                      # was outputs/analyses/matcher/
│   │   ├── data/                         # the 49,586-row matcher_proofs.parquet
│   │   ├── scripts/                      # was scripts/run_matcher.py if specific
│   │   └── README.md
│   ├── proof_decoder/
│   │   ├── outputs/                      # was outputs/analyses/proof_decoder/
│   │   ├── data/                         # decoder dataset
│   │   ├── scripts/                      # was scripts/{train_decoder, eval_decoder_*, probe_decoder_linear}
│   │   └── README.md
│   ├── rewrite_chain_study/
│   │   ├── outputs/                      # was outputs/analyses/rewrite_chain_study/
│   │   ├── data/                         # was data/processed/proof_corpus/rewrite_*
│   │   ├── code/                         # was src/rewrite/engine.py + matching primitives copy? See resolution
│   │   ├── scripts/                      # was scripts/{run_rewrite_engine, run_multimodality, analyze_rewrite_traces, build_rewrite_chain_dataset, survey_rewrite_chain}
│   │   └── README.md
│   ├── completion_exploration/
│   │   ├── outputs/                      # was outputs/analyses/completion_study/
│   │   ├── data/                         # was data/processed/completion_study/
│   │   ├── scripts/                      # was scripts/{phase0_variable_inclusion, phase2_run_completion, phase3_decision_procedure, phase4_run_ordered_completion, phase4_decision_procedure, phase4a_classify_failed_cps, phase4b_row_density_and_spotcheck, phase5_finite_models}
│   │   └── README.md
│   ├── coverage_measurement/
│   │   ├── outputs/                      # was outputs/analyses/coverage_measurement/
│   │   ├── data/                         # was data/processed/coverage_measurement/
│   │   ├── scripts/                      # was scripts/coverage_measurement/*
│   │   └── README.md
│   ├── proof_investigation/
│   │   ├── outputs/                      # was outputs/analyses/proof_investigation/  (minus etp_proof_structure_note.md)
│   │   ├── data/                         # was data/processed/proof_investigation/
│   │   ├── proof_investigation_plan.md   # was repo-root proof_investigation_plan.md
│   │   └── README.md                     # state: paused after Phase 1
│   ├── etp_structure_analysis/           # carved out of proof_investigation
│   │   ├── outputs/
│   │   │   └── etp_proof_structure_note.md   # was outputs/analyses/proof_investigation/etp_proof_structure_note.md
│   │   └── README.md
│   └── failure_overlap/
│       ├── outputs/                      # was outputs/analyses/failure_overlap/
│       ├── scripts/                      # was scripts/analyze_failure_overlap.py
│       └── README.md
│
├── proof_generation_pilot/               # NEW — active phase (placeholder for now)
│   ├── methods/
│   │   ├── matcher/                      # was src/matcher/
│   │   ├── completion/                   # was src/completion/  (kb + ordered)
│   │   └── safe_engine/                  # was src/coverage/  (renamed safe_search.py module)
│   ├── scripts/                          # method runners (run_matcher, run_rewrite_engine? see resolution)
│   ├── data/processed/                   # empty for now
│   ├── outputs/                          # empty for now
│   ├── notebooks/                        # empty for now
│   ├── pilot_plan.md                     # FROM SANDBOX, not yet in repo
│   └── README.md
│
├── data/                                 # SHARED ground truth (stays at root)
│   └── raw/                              # was data/raw/  (etp, etp_export, invariants symlink)
│
├── tests/                                # STAYS AT ROOT, mirrors new tree
│   ├── lib/
│   │   ├── data/
│   │   │   ├── test_parser.py            # was tests/test_parser.py
│   │   │   ├── test_dataset.py
│   │   │   └── test_splits.py
│   │   └── rewrite/
│   │       ├── test_matching.py          # was tests/test_rewrite_matching.py
│   │       ├── test_lpo.py
│   │       └── test_unification.py
│   ├── embeddings/
│   │   ├── test_aux_heads.py
│   │   ├── test_encoder.py
│   │   ├── test_jepa.py
│   │   ├── test_losses.py
│   │   ├── test_pair_classifier.py
│   │   ├── test_predictor.py
│   │   ├── test_sigreg.py
│   │   ├── test_scheduler.py
│   │   ├── test_train.py
│   │   ├── test_decoder.py
│   │   ├── test_decoder_eval.py
│   │   └── test_decoder_tokenizer.py
│   ├── pilot/
│   │   ├── test_matcher.py
│   │   ├── test_completion_kb.py
│   │   ├── test_completion_ordered.py
│   │   └── test_safe_search.py           # was tests/test_coverage_safe_search.py
│   └── et_studies/
│       └── rewrite_chain_study/
│           └── test_engine.py            # was tests/test_rewrite_engine.py
│
├── archive/                              # STAYS AT ROOT  (12 archived plans, dual_encoder/)
├── wandb/                                # STAYS AT ROOT
├── CHANGELOG.md                          # STAYS AT ROOT
├── CLAUDE.md                             # STAYS AT ROOT  (project guidance)
├── README.md                             # STAYS AT ROOT  (project overview / hub)
├── pyproject.toml                        # STAYS AT ROOT
├── requirements.txt                      # STAYS AT ROOT
└── project_refactor_plan.md              # MOVES after refactor → archive/
```

---

## 4. Mapping table

### `src/` — code

| current | target | move type | notes |
|---|---|---|---|
| `src/data/parser.py` | `lib/data/parser.py` | git mv | shared primitive (37 references) |
| `src/data/dataset.py` | `lib/data/dataset.py` | git mv | JEPA uses it but `dataset` is a primitive concept |
| `src/data/splits.py` | `lib/data/splits.py` | git mv | shared splits utility |
| `src/data/__init__.py` | `lib/data/__init__.py` | git mv | |
| `src/rewrite/matching.py` | `lib/rewrite/matching.py` | git mv | shared by completion + safe_engine + unsafe engine |
| `src/rewrite/lpo.py` | `lib/rewrite/lpo.py` | git mv | shared by completion + safe_engine |
| `src/rewrite/unification.py` | `lib/rewrite/unification.py` | git mv | shared by completion + safe_engine |
| `src/rewrite/engine.py` | `et_studies/rewrite_chain_study/code/engine.py` | git mv | UNSAFE engine; powered the rewrite-chain study only |
| `src/rewrite/__init__.py` | `lib/rewrite/__init__.py` | git mv | shared init |
| `src/matcher/matcher.py` | `proof_generation_pilot/methods/matcher/matcher.py` | git mv | |
| `src/matcher/__init__.py` | `proof_generation_pilot/methods/matcher/__init__.py` | git mv | |
| `src/completion/kb.py` | `proof_generation_pilot/methods/completion/kb.py` | git mv | |
| `src/completion/ordered.py` | `proof_generation_pilot/methods/completion/ordered.py` | git mv | |
| `src/completion/__init__.py` | `proof_generation_pilot/methods/completion/__init__.py` | git mv | |
| `src/coverage/safe_search.py` | `proof_generation_pilot/methods/safe_engine/safe_search.py` | git mv | |
| `src/coverage/__init__.py` | `proof_generation_pilot/methods/safe_engine/__init__.py` | git mv | |
| `src/model/*.py` | `embeddings/src/model/*.py` | git mv | JEPA architecture |
| `src/training/*.py` | `embeddings/src/training/*.py` | git mv | |
| `src/decoder/*.py` | `et_studies/proof_decoder/code/*.py` OR `embeddings/src/decoder/*.py` | git mv | **see resolution** |
| `src/probing/probes.py` | `embeddings/src/probing/probes.py` | git mv | |
| `src/baselines/__init__.py` | `embeddings/src/baselines/__init__.py` | git mv | empty placeholder |
| `src/evaluation/__init__.py` | `embeddings/src/evaluation/__init__.py` | git mv | empty placeholder |

### `scripts/` — code

| current | target |
|---|---|
| `train.py` | `embeddings/scripts/train.py` |
| `train_decoder.py` | `et_studies/proof_decoder/scripts/train_decoder.py` |
| `train_pair_classifier.py` | `embeddings/scripts/train_pair_classifier.py` |
| `record_run.py` | `embeddings/scripts/record_run.py` |
| `run_experiment.py` | `embeddings/scripts/run_experiment.py` |
| `run_probes.py` | `embeddings/scripts/run_probes.py` |
| `evaluate_checkpoint.py` | `embeddings/scripts/evaluate_checkpoint.py` |
| `eval_predictions_on_split.py` | `embeddings/scripts/eval_predictions_on_split.py` |
| `extract_probe_embeddings.py` | `embeddings/scripts/extract_probe_embeddings.py` |
| `extract_run_diagnostics.py` | `embeddings/scripts/extract_run_diagnostics.py` |
| `save_embeddings.py` | `embeddings/scripts/save_embeddings.py` |
| `save_predictions.py` | `embeddings/scripts/save_predictions.py` |
| `embedding_spectrum.py` | `embeddings/scripts/embedding_spectrum.py` |
| `austin_eval.py` | `embeddings/scripts/austin_eval.py` |
| `analyze_austin.py` | `embeddings/scripts/analyze_austin.py` |
| `analyze_distances.py` | `embeddings/scripts/analyze_distances.py` |
| `analyze_mlp_vacuity.py` | `embeddings/scripts/analyze_mlp_vacuity.py` |
| `analyze_equation_concentration.py` | `embeddings/scripts/analyze_equation_concentration.py` |
| `build_decoder_dataset.py` | `et_studies/proof_decoder/scripts/build_decoder_dataset.py` |
| `eval_decoder_majority_baseline.py` | `et_studies/proof_decoder/scripts/eval_decoder_majority_baseline.py` |
| `probe_decoder_linear.py` | `et_studies/proof_decoder/scripts/probe_decoder_linear.py` |
| `parse_equations.py` | `lib/scripts/parse_equations.py` or `embeddings/scripts/parse_equations.py` — **see resolution** |
| `build_pair_table.py` | `lib/scripts/build_pair_table.py` — produces shared pair-table |
| `build_equation_invariants.py` | `lib/scripts/build_equation_invariants.py` — produces shared invariants |
| `build_splits.py` | `embeddings/scripts/build_splits.py` |
| `survey_*.py` (7 files) | `et_studies/proof_survey/scripts/survey_*.py` |
| `analyze_failure_overlap.py` | `et_studies/failure_overlap/scripts/analyze_failure_overlap.py` |
| `build_rewrite_chain_dataset.py` | `et_studies/rewrite_chain_study/scripts/build_rewrite_chain_dataset.py` |
| `run_rewrite_engine.py` | `et_studies/rewrite_chain_study/scripts/run_rewrite_engine.py` |
| `run_multimodality.py` | `et_studies/rewrite_chain_study/scripts/run_multimodality.py` |
| `analyze_rewrite_traces.py` | `et_studies/rewrite_chain_study/scripts/analyze_rewrite_traces.py` |
| `run_matcher.py` | `et_studies/matcher_deliverable/scripts/run_matcher.py` |
| `phase0_variable_inclusion.py` | `et_studies/completion_exploration/scripts/phase0_variable_inclusion.py` |
| `phase2_run_completion.py` | `et_studies/completion_exploration/scripts/phase2_run_completion.py` |
| `phase3_decision_procedure.py` | `et_studies/completion_exploration/scripts/phase3_decision_procedure.py` |
| `phase4_run_ordered_completion.py` | `et_studies/completion_exploration/scripts/phase4_run_ordered_completion.py` |
| `phase4_decision_procedure.py` | `et_studies/completion_exploration/scripts/phase4_decision_procedure.py` |
| `phase4a_classify_failed_cps.py` | `et_studies/completion_exploration/scripts/phase4a_classify_failed_cps.py` |
| `phase4b_row_density_and_spotcheck.py` | `et_studies/completion_exploration/scripts/phase4b_row_density_and_spotcheck.py` |
| `phase5_finite_models.py` | `et_studies/completion_exploration/scripts/phase5_finite_models.py` |
| `scripts/coverage_measurement/phase{1,2,4,5,6}_*.py` | `et_studies/coverage_measurement/scripts/phase{1,2,4,5,6}_*.py` |

### `tests/` — stays at root, mirrors tree

| current | target |
|---|---|
| `test_parser.py` | `tests/lib/data/test_parser.py` |
| `test_dataset.py` | `tests/lib/data/test_dataset.py` |
| `test_splits.py` | `tests/lib/data/test_splits.py` |
| `test_rewrite_matching.py` | `tests/lib/rewrite/test_matching.py` |
| `test_rewrite_lpo.py` | `tests/lib/rewrite/test_lpo.py` |
| `test_rewrite_unification.py` | `tests/lib/rewrite/test_unification.py` |
| `test_rewrite_engine.py` | `tests/et_studies/rewrite_chain_study/test_engine.py` |
| `test_matcher.py` | `tests/pilot/test_matcher.py` |
| `test_completion_kb.py` | `tests/pilot/test_completion_kb.py` |
| `test_completion_ordered.py` | `tests/pilot/test_completion_ordered.py` |
| `test_coverage_safe_search.py` | `tests/pilot/test_safe_search.py` (rename) |
| `test_encoder.py` | `tests/embeddings/test_encoder.py` |
| `test_predictor.py` | `tests/embeddings/test_predictor.py` |
| `test_jepa.py` | `tests/embeddings/test_jepa.py` |
| `test_aux_heads.py` | `tests/embeddings/test_aux_heads.py` |
| `test_pair_classifier.py` | `tests/embeddings/test_pair_classifier.py` |
| `test_losses.py` | `tests/embeddings/test_losses.py` |
| `test_sigreg.py` | `tests/embeddings/test_sigreg.py` |
| `test_scheduler.py` | `tests/embeddings/test_scheduler.py` |
| `test_train.py` | `tests/embeddings/test_train.py` |
| `test_decoder.py` | `tests/et_studies/proof_decoder/test_decoder.py` |
| `test_decoder_tokenizer.py` | `tests/et_studies/proof_decoder/test_decoder_tokenizer.py` |
| `test_decoder_eval.py` | `tests/et_studies/proof_decoder/test_decoder_eval.py` |

### `data/` — data files

| current | target | move type |
|---|---|---|
| `data/raw/etp` | `data/raw/etp` | (no move — stays at root) |
| `data/raw/etp_export` | `data/raw/etp_export` | (no move) |
| `data/raw/invariants` | `data/raw/invariants` | (no move; sibling-repo symlink) |
| `data/processed/equation_graphs.pt` | `lib/data/processed/equation_graphs.pt` OR `embeddings/data/processed/` | gitignored; mv; **see resolution** |
| `data/processed/equation_invariants.parquet` | `lib/data/processed/equation_invariants.parquet` | git mv; shared |
| `data/processed/implication_pairs.parquet` | `lib/data/processed/implication_pairs.parquet` | gitignored?; shared |
| `data/processed/proof_corpus/` | split across studies — see resolution | mostly git mv |
| `data/processed/completion_study/` | `et_studies/completion_exploration/data/processed/` | git mv (parquets) |
| `data/processed/coverage_measurement/` | `et_studies/coverage_measurement/data/processed/` | git mv |
| `data/processed/proof_investigation/` | `et_studies/proof_investigation/data/processed/` | git mv (currently 4 KB, mostly empty) |
| `data/splits/` | `embeddings/data/splits/` | git mv |

`data/processed/proof_corpus/` contents (need to be split across studies):

| file | destination |
|---|---|
| `matcher_proofs.parquet` | `et_studies/matcher_deliverable/data/` |
| `rewrite_chain_dataset.parquet` | `et_studies/rewrite_chain_study/data/` |
| `rewrite_engine_traces.parquet` | `et_studies/rewrite_chain_study/data/` |
| `normalized_proofs.parquet` | `et_studies/proof_survey/data/` |
| `true_corpus.parquet` | `et_studies/proof_survey/data/` |
| `facts_corpus.parquet` | `et_studies/proof_survey/data/` |
| `decoder_phase1_dataset.parquet` | `et_studies/proof_decoder/data/` |

### `outputs/` — analysis artifacts

| current | target | move type |
|---|---|---|
| `outputs/analyses/proof_survey/` | `et_studies/proof_survey/outputs/` | git mv |
| `outputs/analyses/matcher/` | `et_studies/matcher_deliverable/outputs/` | git mv |
| `outputs/analyses/proof_decoder/` | `et_studies/proof_decoder/outputs/` | git mv |
| `outputs/analyses/rewrite_chain_study/` | `et_studies/rewrite_chain_study/outputs/` | git mv |
| `outputs/analyses/completion_study/` | `et_studies/completion_exploration/outputs/` | git mv |
| `outputs/analyses/coverage_measurement/` | `et_studies/coverage_measurement/outputs/` | git mv |
| `outputs/analyses/proof_investigation/` | `et_studies/proof_investigation/outputs/` (minus etp_proof_structure_note.md) | git mv |
| `outputs/analyses/proof_investigation/etp_proof_structure_note.md` | `et_studies/etp_structure_analysis/outputs/etp_proof_structure_note.md` | git mv |
| `outputs/analyses/failure_overlap/` | `et_studies/failure_overlap/outputs/` | git mv |
| `outputs/checkpoints/` | `embeddings/outputs/checkpoints/` | **mv** (1.4 G, gitignored) |
| `outputs/runs/` | `embeddings/outputs/runs/` | **mv** (1.6 G, partially gitignored — parquets/.npy in `.gitignore`) |
| `outputs/embeddings/` | `embeddings/outputs/embeddings/` | git mv .pt/.md, mv .npy |
| `outputs/failure_analysis/` | `embeddings/outputs/failure_analysis/` | **mv** (127 M, gitignored) |
| `outputs/failure_analysis_m1p5_vic0p1/` | `embeddings/outputs/failure_analysis_m1p5_vic0p1/` | **mv** (116 M, gitignored) |
| `outputs/probing/` | `embeddings/outputs/probing/` | **mv** for .npy, git mv for committed |
| `outputs/probing_results/` | `embeddings/outputs/probing/results/` | git mv — consolidate |
| `outputs/figures/` | `embeddings/outputs/figures/` | **mv** (gitignored) |
| `outputs/results/` | `embeddings/outputs/results/` | git mv |
| `outputs/*.log` (2 files) | `embeddings/outputs/logs/` | git mv |
| `outputs/refactor/` | (this file itself — stays put OR moves to root after) | **see resolution** |

### Root-level files

| current | target | notes |
|---|---|---|
| `CHANGELOG.md` | (stays at root) | project-wide changelog |
| `CLAUDE.md` | (stays at root) | project guidance |
| `README.md` | (stays at root, rewritten as hub) | overview, links to areas |
| `pyproject.toml` | (stays at root) | package manifest |
| `requirements.txt` | (stays at root) | |
| `project_refactor_plan.md` | `archive/project_refactor_plan.md` after Phase 8 completes | move when done |
| `proof_investigation_plan.md` | `et_studies/proof_investigation/proof_investigation_plan.md` | with the paused investigation |
| `configs/` | `embeddings/configs/` | JEPA-only configs |

### Stays-at-root (not moved)

`archive/`, `wandb/`, `notebooks/`, `.git/`, `.venv/`, `.claude/`, `.python-version`, `.pytest_cache/`, `pyproject.toml`, `requirements.txt`, `CHANGELOG.md`, `CLAUDE.md`, `README.md`, `tests/`, `data/raw/`.

---

## 5. Resolution list

Ambiguities and design decisions surfaced by walking the mapping. Each needs a yes/no or a pick.

### R1. Where does `src/decoder/` go — `embeddings/` or `et_studies/proof_decoder/`?

The σ-decoder code is methodologically part of the embeddings effort (decoded representations from JEPA embeddings), but the *outputs* are a frozen study. Three options:

- **(a) `embeddings/src/decoder/`** — by-mechanism: it's a model on top of embeddings.
- **(b) `et_studies/proof_decoder/code/`** — by-deliverable: it's a frozen study; everything related lives together.
- **(c) split — `embeddings/src/decoder/` for code + tests, `et_studies/proof_decoder/{outputs,scripts}/` for the deliverable** — by-role.

Recommendation: **(b)**. The decoder is a frozen study and the embedding model it sits on top of is the v1.0 deliverable. The decoder is downstream, not generic. Keeping code + scripts + outputs + tests together as a frozen unit matches the studies pattern.

### R2. `parse_equations.py` and `build_pair_table.py` — `lib/scripts/` or `embeddings/scripts/`?

These produce shared artifacts (`equation_invariants.parquet`, `implication_pairs.parquet`) used across embeddings + methods + studies. Two options:

- **(a) `lib/scripts/`** — they produce shared data, so they belong with `lib/`.
- **(b) `embeddings/scripts/`** — they were written for JEPA training and most uses are there.

Recommendation: **(a)**. They build *shared* artifacts; the right semantics is "library-level data generation."

### R3. `data/processed/equation_graphs.pt` and `implication_pairs.parquet` — `lib/data/processed/` or `embeddings/data/processed/`?

Same question as R2. The `.pt` is used by the GNN encoder (`embeddings/src/model/encoder.py`); `implication_pairs.parquet` is the per-pair label table used across the project.

Recommendation: **`lib/data/processed/`** for both. They're shared inputs. Symbolic methods don't use them directly but analysis scripts often do.

### R4. `archive/dual_encoder/` — has its own subdirectory inside `archive/` with code

`archive/dual_encoder/` contains in-progress code from the deferred dual-encoder plan. It's tracked but inactive. Stays in `archive/` (no move).

### R5. `outputs/refactor/phase1_inventory.md` (this file) — where does the refactor inventory live?

Likely either: (a) stays in `outputs/refactor/` as a transient artifact deleted after refactor completes; (b) moves to `archive/project_refactor_inventory.md` alongside the plan when refactor is archived.

Recommendation: (b). Keep both as the historical record of the refactor decision.

### R6. The unsafe engine — does it need a *copy* of its primitives?

`src/rewrite/engine.py` (going to `et_studies/rewrite_chain_study/code/`) imports `from src.rewrite.matching import all_rewrites, term_size`. After the move, those primitives live in `lib/rewrite/matching.py`. Two options:

- **(a) Update import** — `engine.py` imports from `lib.rewrite.matching`. The study's "code" depends on `lib`; reasonable since `lib/` is the canonical home of those primitives.
- **(b) Vendor copy** — copy `matching.py` into `et_studies/rewrite_chain_study/code/` so the study is self-contained. Pure historical reading; no shared dependency.

Recommendation: **(a)**. Keep the import dependency on `lib/`. Studies are read-only artifacts but they're still part of the live codebase; they can depend on the live `lib/` shared primitives. Self-containment isn't a stated requirement and the duplication has its own cost.

### R7. Pilot plan — where does `proof_generation_pilot/pilot_plan.md` come from?

The plan isn't in the repo yet (user note: "currently in the planning sandbox"). The refactor creates the directory; once Phase 8 completes and the pilot plan is moved into the repo, it lands at `proof_generation_pilot/pilot_plan.md`.

No action this phase. Surface in Phase 8 verification: confirm `proof_generation_pilot/pilot_plan.md` exists after the user moves it in.

### R8. `proof_investigation_plan.md` location

Currently at repo root. Paused investigation; user wrote "may come back to it." Options:

- **(a) `et_studies/proof_investigation/proof_investigation_plan.md`** — alongside its outputs, marked paused.
- **(b) Keep at root** — active plan that's just on hiatus.

Recommendation: **(a)**. Move to the study's directory with the rest of its assets. If the investigation resumes, the plan can move back to root then. CLAUDE.md / README.md update to mark "no active plan" until that point.

### R9. CLAUDE.md / README.md content updates

After refactor, both need updating to:
- Reflect "no active plan" or new active plan (refactor itself, then the pilot).
- Update internal links to new paths.
- Move the "Previous plans (X, Y, Z all archived)" listing to reference `et_studies/` paths.

Defer to Phase 7 (write READMEs) — but flag that root CLAUDE.md and README.md also need full content update, not just the directory READMEs the plan lists.

### R10. Pilot plan placeholder in proof_generation_pilot/

Until the pilot plan arrives, the `proof_generation_pilot/` directory will have `methods/`, `scripts/`, empty `data/processed/`, `outputs/`, `notebooks/`, plus the README. No `pilot_plan.md`. Worth flagging that the directory is functional skeleton, not yet active.

### R11. Test directory mirroring — is the deep mirroring worth it?

The mapping proposes `tests/lib/data/test_parser.py`, `tests/embeddings/test_encoder.py`, `tests/pilot/test_matcher.py`, etc. This means deep mirroring (4 levels). Alternative: flat `tests/test_*.py` with the file names disambiguating.

Recommendation: **mirror only one level deep**:
- `tests/lib/`
- `tests/embeddings/`
- `tests/pilot/`
- `tests/et_studies/`

That gives clear scope grouping without 4-level nesting. Within each, files stay flat (`tests/embeddings/test_encoder.py`, not `tests/embeddings/model/test_encoder.py`).

### R12. `notebooks/` is empty

Currently has only `__init__` (empty). Move to `proof_generation_pilot/notebooks/` as the empty placeholder for the active phase, or delete entirely?

Recommendation: **delete and recreate per-area** (`embeddings/notebooks/`, `proof_generation_pilot/notebooks/`) as needed. Empty cross-cutting `notebooks/` doesn't add value.

### R13. `outputs/refactor/` for the inventory

I'm putting this file at `outputs/refactor/phase1_inventory.md`. After the refactor, `outputs/` itself moves (to `embeddings/outputs/`). So either:

- **(a)** Move `outputs/refactor/` to `archive/refactor_phase1_inventory.md` after refactor.
- **(b)** Keep `outputs/refactor/` at root as the refactor's working directory through Phases 1-8.

Recommendation: **(b)** through the refactor. Phase 8 archives both `project_refactor_plan.md` and `outputs/refactor/*` together into `archive/`.

---

## 6. Smoke-test checklist (for Phase 8)

After the move:

```
# Import sanity
python -c "from lib.data.parser import parse_equation; print(parse_equation('x = x'))"
python -c "from lib.rewrite.matching import all_rewrites; print('ok')"
python -c "from proof_generation_pilot.methods.matcher.matcher import structural_match; print('ok')"
python -c "from proof_generation_pilot.methods.completion.kb import complete; print('ok')"
python -c "from proof_generation_pilot.methods.safe_engine.safe_search import safe_search; print('ok')"
python -c "from embeddings.src.model.jepa import JEPA; print('ok')"

# Test suite
pytest tests/ -q

# Symbolic-method smoke (10 pairs, no I/O surprises)
python -c "
import sys; sys.path.insert(0, '.')
from lib.data.parser import parse_equation
from proof_generation_pilot.methods.completion.kb import complete
eq = parse_equation('x = x ◇ x')
r = complete([(eq.lhs, eq.rhs)])
print(r.status.value, len(r.rules))
"

# Embeddings smoke — verify v1.0 still loads
python -c "
import torch
sd = torch.load('embeddings/outputs/embeddings/v1.0/source_checkpoint.pt')
print(list(sd.keys())[:3])
"

# Disk verification
du -sh lib/ embeddings/ et_studies/ proof_generation_pilot/ tests/
```

---

## 7. Ordering recommendation (for Phases 2-8)

The cross-imports between methods (`completion → rewrite`, `coverage → rewrite`, etc.) mean we need to be careful about ordering. Suggested execution:

1. **Phase 2** (create skeleton): empty target dirs with `.gitkeep`. Trivial.
2. **Phase 3** (studies): move `outputs/analyses/{study}/` → `et_studies/{study}/outputs/` and `data/processed/{study}/` → `et_studies/{study}/data/`. Plus their phase scripts. **No source-code imports affected** — studies are frozen artifacts.
3. **Phase 4a** (`lib/`): move `src/data/` and `src/rewrite/{matching,lpo,unification}.py` to `lib/`. Update imports in: `src/completion/`, `src/coverage/`, `src/matcher/`, `src/rewrite/engine.py`, `src/training/`, `src/decoder/`, `src/probing/`, `src/baselines/`. Run tests.
4. **Phase 4b** (pilot methods): move `src/matcher/`, `src/completion/`, `src/coverage/` into `proof_generation_pilot/methods/`. Update imports in scripts that reference them. Run tests.
5. **Phase 4c** (rewrite_chain unsafe engine): move `src/rewrite/engine.py` into `et_studies/rewrite_chain_study/code/`. Update imports in scripts that use it. Run tests.
6. **Phase 5** (embeddings): move `src/{baselines, model, training, evaluation, decoder, probing}/` to `embeddings/src/`. Move scripts + configs + outputs. Move tests. Run tests.
7. **Phase 6** (path-reference audit): grep all repo for hardcoded paths (`outputs/`, `data/processed/`, `src/`) and update.
8. **Phase 7** (READMEs).
9. **Phase 8** (verify).

Splitting Phase 4 into 4a/4b/4c is what keeps the import updates manageable — each sub-phase has a small, well-defined import-change set.

---

## What's next

Review this inventory + mapping + resolutions. Once R1-R13 have decisions, Phases 2-8 execute mechanically per the ordering above. The largest single risk is Phase 4 import updates (60+ files reference `src/data/parser` and `src/rewrite/*`); the rest is bulk file moves with verification.
