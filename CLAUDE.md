# CLAUDE.md — JEPA for Equational Implication

The active plan is `skip_connection_plan.md` — add a parameter-free skip connection from the GNN's pooled representation (256-dim) to the embedding, concatenated with the projected output and L2-normalized. Tests the simpler hypothesis "the projection is throwing away features the GNN already learned." Tabled and archived: `dual_encoder_plan.md` (would have added a cohomology MLP as a second encoder pathway); the skip-connection result will inform whether the dual encoder is still needed. Read the plan first when starting a session.

Prior plans completed, in `archive/`: `aux_heads_implementation_plan.md` (delivered `aux_h128_seed1_diag` at 0.9621 — current stable best), `probing_experiments_plan.md` (delivered the probing run record showing cohom is already in the embeddings to 80-88%), `dual_encoder_plan.md` (deferred pending skip-connection result; in-progress code is in `archive/dual_encoder/`).

**Current stable best is `skip_h128_seed1_diag` at bal_acc 0.9775 (h=128, 10 structural aux heads, VICReg 0.1/0.1, margin 1.5, skip connection from pooled to embedding with per-channel LayerNorms, embed_dim=768, ep5 monitor).** Beats the previous best `aux_h128_seed1_diag` (0.9621) by +1.54pp; skip wins at every single epoch by 1-4pp. Mechanism is geometric/structural (more dims, stable LayerNorm'd skip reference) rather than feature-preservation as the plan predicted — re-probing after training showed cohom probe accuracy gained only +0.4pp on average (none reaching the planned 95% threshold), even though bal_acc jumped substantially. See `outputs/runs/skip_h128_seed1_diag/diagnostics_report.md` for the mechanism analysis.

Reproducibility on the skip recipe is single-seed so far — `skip_h128_seed42_diag` is the natural next run before treating 0.9775 as definitive. The h=256 single-seed `aux_h256_diag` hit 0.9635 via a regime-B phase transition that did NOT reproduce under seed=1 (which reverted to 0.9492); h=256 is considered an unreliable recipe and future experiments should default to h=128. The original h=256 simplicity-bias drift problem (TRUE-recall → 0.99, FALSE-recall → 0.87) is unresolved on the aux-head path: 4 cohomology aux-head experiments (`aux_h256_cohom_diag`, `aux_h256_cohom_w5_diag`, `aux_h128_cohom_w5_diag`, plus the structural-only baselines) confirmed that per-equation classification aux heads do not constrain pair-level contrastive geometry at any tested aux_weight.

The original Phase 1 → Phase 4 engineering plan is at `archive/jepa_engineering_plan.md`. Phase 1 (architecture sweep + loss form) is complete. The original "failure analysis" plan (`trash/testing_plan_phase2.md`) is also complete — it produced the diagnostic infrastructure (per-step grad/loss/distance instrumentation, no-holdout training mode) and the run records under `outputs/runs/stable_*` that the current aux-heads plan builds on.

`CHANGELOG.md` records deviations from the plan and explains *why* the plan reads the way it does — read it whenever a plan or commit references a CHANGELOG entry.

When you deviate from the active plan, update it to match reality AND add a CHANGELOG.md entry explaining why, in the same commit as the deviating code.

**Always read planning markdown files critically before implementing.** When the user adds a plan or experiment-design `.md`, scrutinize it like a code reviewer scrutinizes a PR: look for broken methodology (missing baselines, class-imbalance metrics, ill-defined CV), inconsistencies with existing code (wrong constants, paths, function signatures), and ambiguities a reasonable implementer could get wrong. Raise concerns ranked by severity BEFORE starting implementation; wait for the user's resolution on criticals at minimum. Don't be performatively skeptical — only flag issues that would actually change the implementation or interpretation.

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

VICReg (`src/training/losses.py::vicreg_terms`) is the production regularizer. Active via `loss.vicreg_var_weight` and `loss.vicreg_cov_weight` in any config. Works robustly at h=128 (0.9621 bal_acc with structural aux heads, 0.9582 stable from prior seed); degrades into TRUE-bias drift at h=256.

SIGReg (`src/training/sigreg.py`) is implemented and verified (9 unit tests pass) but **shelved as not-a-fit for our task**. Three variants tested at h=256 — see commits `4fec7a3` → `cf5e00a` and `outputs/runs/stable_h256_*sigreg*/`, `outputs/runs/stable_h256_lejepa_3ep_diag/`. Failure modes: with L2-norm, SIGReg's pre-norm Gaussian target → uniform on sphere, conflicts with contrastive's TRUE-clustering need (cl_TRUE pinned at 1.97 = random unit-sphere distance²); without L2-norm, SIGReg's globally-coherent spread direction beats contrastive in clip-saturated steps (cl_TRUE = K = orthogonal-pair distance). Code stays in repo as a known-correct implementation for future setups where the prediction loss isn't margin-based contrastive. Don't re-attempt without revisiting the linked diagnostics_report.md files.

The `model.l2_normalize_outputs` config flag (default `true`) routes through `JEPA.forward(normalize_outputs=...)` to drop the L2-norm on encoder/predictor outputs — added for the SIGReg LeJEPA-style attempt; safe to leave at default for any contrastive-with-margin recipe.

Auxiliary heads (`src/model/aux_heads.py`) attach single-layer linear probes to the encoder's pooled rep. Active when `loss.aux_weight > 0` and `data.aux_labels_path` is set. `HEAD_SPECS` defines all 18 heads (10 structural + 8 cohomology); `loss.aux_heads` selects the active subset — accepts `"structural"`, `"cohom"`, `"all"` (named subsets in `HEAD_SUBSETS`), or an explicit list of head names. Defaults to all heads when omitted. Reproducing pre-cohom runs requires `loss.aux_heads: structural`.

## Data layout

- `data/raw/etp/` → symlink to the Equational Theories Project's `data/` directory. Source of `equations.txt`, the `Austin_implications.txt` curated subset, and other ETP artifacts.
- `data/raw/etp_export/` → exports from the ETP project (not symlinked). Currently holds `export_raw_implications_5_9_2026.csv`, the **authoritative ground truth** for every off-diagonal implication pair (see CHANGELOG 2026-05-09). The pair table is built from this, not from the older `2025-08-11-vampire.json.gz` (which left 1,062 pairs unresolved).
- `data/raw/invariants/` → symlink to the sibling invariants project's `assembly/output/` directory. Currently empty until that project is run.
- `data/processed/` → derived caches (parsed-tree PyG `Data` cache, pair-table parquet). Not committed.
- `data/splits/` → holdout split index files. Committed, for reproducibility.
