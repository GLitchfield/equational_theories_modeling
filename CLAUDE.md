# CLAUDE.md — JEPA for Equational Implication

The active plan is `aux_heads_implementation_plan.md` — adding auxiliary prediction heads (spine type, balance, singleton-forcing, variable count) to the encoder to anchor structural features against the simplicity-bias drift that surfaced at h=256. Read it first when starting a session.

Current best is `stable_m1p5_vic0p1_diag` at bal_acc 0.9492 (h=128, VICReg 0.1/0.1, ep6 monitor). The h=256 width sweep showed `stable_h256_m1p5_vic0p05_diag` reaching 0.9469 at ep0 then degrading to 0.93 — TRUE-recall climbs to 0.99, FALSE-recall drops to 0.87. That late-training TRUE-bias drift is the active problem the aux-heads plan is designed to address.

The original Phase 1 → Phase 4 engineering plan is at `archive/jepa_engineering_plan.md`. Phase 1 (architecture sweep + loss form) is complete. The original "failure analysis" plan (`trash/testing_plan_phase2.md`) is also complete — it produced the diagnostic infrastructure (per-step grad/loss/distance instrumentation, no-holdout training mode) and the run records under `outputs/runs/stable_*` that the current aux-heads plan builds on.

`CHANGELOG.md` records deviations from the plan and explains *why* the plan reads the way it does — read it whenever a plan or commit references a CHANGELOG entry.

When you deviate from the active plan, update it to match reality AND add a CHANGELOG.md entry explaining why, in the same commit as the deviating code.

## Operational rules

- Do NOT modify files in `data/raw/`. All processing goes to `data/processed/` or `data/splits/`.
- Use `wandb` for ALL experiment tracking. Every training run must be logged.
- Always report metrics per holdout level separately: L1, L2 one-unknown, L2 both-unknown, L3. Never average across levels — the gaps BETWEEN levels are the headline finding.
- Save holdout split indices to `data/splits/` so experiments are reproducible. Every experiment log must reference which split file was used.
- Run baselines (1–4 in `archive/jepa_engineering_plan.md`) before training the JEPA. Without baselines you can't interpret the JEPA results.
- The equation parser is the critical path — if parsing is wrong, everything downstream is wrong. Test it thoroughly against known equations before any training.
- Hardware: RTX 4080, 16 GB VRAM. Monitor and reduce batch size if OOM.
- Paper figures must be reproducible from notebooks in `notebooks/`. No one-off scripts.
- Keep checkpoints small — encoder weights + full JEPA state, skip optimizer state unless resuming from a crash.
- Every completed training run gets recorded artifacts (`outputs/runs/<name>/` with `diagnostics_report.md`, `diagnostic_extract.json`, `per_step_trace.csv`, `per_epoch_eval.csv`) AND a commit with the message pattern `Phase N: record <run_name> — <one-line takeaway>`. Run records are part of the run's deliverable, not a separate request.

## Regularization options

VICReg (`src/training/losses.py::vicreg_terms`) is the production regularizer. Active via `loss.vicreg_var_weight` and `loss.vicreg_cov_weight` in any config. Works at h=128 (0.9492 bal_acc), degrades into TRUE-bias drift at h=256.

SIGReg (`src/training/sigreg.py`) is implemented and verified (9 unit tests pass) but **shelved as not-a-fit for our task**. Three variants tested at h=256 — see commits `4fec7a3` → `cf5e00a` and `outputs/runs/stable_h256_*sigreg*/`, `outputs/runs/stable_h256_lejepa_3ep_diag/`. Failure modes: with L2-norm, SIGReg's pre-norm Gaussian target → uniform on sphere, conflicts with contrastive's TRUE-clustering need (cl_TRUE pinned at 1.97 = random unit-sphere distance²); without L2-norm, SIGReg's globally-coherent spread direction beats contrastive in clip-saturated steps (cl_TRUE = K = orthogonal-pair distance). Code stays in repo as a known-correct implementation for future setups where the prediction loss isn't margin-based contrastive. Don't re-attempt without revisiting the linked diagnostics_report.md files.

The `model.l2_normalize_outputs` config flag (default `true`) routes through `JEPA.forward(normalize_outputs=...)` to drop the L2-norm on encoder/predictor outputs — added for the SIGReg LeJEPA-style attempt; safe to leave at default for any contrastive-with-margin recipe.

## Data layout

- `data/raw/etp/` → symlink to the Equational Theories Project's `data/` directory. Source of `equations.txt`, the `Austin_implications.txt` curated subset, and other ETP artifacts.
- `data/raw/etp_export/` → exports from the ETP project (not symlinked). Currently holds `export_raw_implications_5_9_2026.csv`, the **authoritative ground truth** for every off-diagonal implication pair (see CHANGELOG 2026-05-09). The pair table is built from this, not from the older `2025-08-11-vampire.json.gz` (which left 1,062 pairs unresolved).
- `data/raw/invariants/` → symlink to the sibling invariants project's `assembly/output/` directory. Currently empty until that project is run.
- `data/processed/` → derived caches (parsed-tree PyG `Data` cache, pair-table parquet). Not committed.
- `data/splits/` → holdout split index files. Committed, for reproducibility.
