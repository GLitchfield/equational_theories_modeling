# CLAUDE.md — JEPA for Equational Implication

The active plan is `testing_plan_phase2.md` — failure analysis on the post-VICReg model (current best `vicreg_encoder_v0p25c0p25_tw1p0_v2` at bal_acc 0.9112). Read it first when starting a session.

The original Phase 1 → Phase 4 engineering plan is at `archive/jepa_engineering_plan.md`. It set up the architecture sweep that produced the current best model. Phase 1 is complete; Phases 2–4 (level-2/3 holdouts, architecture refinement, probing as a separate phase) are on hold pending the failure analysis. Revisit if the failure analysis points back at any of those threads.

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

## Data layout

- `data/raw/etp/` → symlink to the Equational Theories Project's `data/` directory. Source of `equations.txt`, the `Austin_implications.txt` curated subset, and other ETP artifacts.
- `data/raw/etp_export/` → exports from the ETP project (not symlinked). Currently holds `export_raw_implications_5_9_2026.csv`, the **authoritative ground truth** for every off-diagonal implication pair (see CHANGELOG 2026-05-09). The pair table is built from this, not from the older `2025-08-11-vampire.json.gz` (which left 1,062 pairs unresolved).
- `data/raw/invariants/` → symlink to the sibling invariants project's `assembly/output/` directory. Currently empty until that project is run.
- `data/processed/` → derived caches (parsed-tree PyG `Data` cache, pair-table parquet). Not committed.
- `data/splits/` → holdout split index files. Committed, for reproducibility.
