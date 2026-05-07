# CLAUDE.md — JEPA for Equational Implication

The full plan is `jepa_engineering_plan.md`. Read it first when starting a session. `CHANGELOG.md` records deviations from the plan and explains *why* the plan reads the way it does — read it whenever the plan references a CHANGELOG entry.

When you deviate from the plan, update the plan to match reality AND add a CHANGELOG.md entry explaining why, in the same commit as the deviating code.

## Operational rules

- Do NOT modify files in `data/raw/`. All processing goes to `data/processed/` or `data/splits/`.
- Use `wandb` for ALL experiment tracking. Every training run must be logged.
- Always report metrics per holdout level separately: L1, L2 one-unknown, L2 both-unknown, L3. Never average across levels — the gaps BETWEEN levels are the headline finding.
- Save holdout split indices to `data/splits/` so experiments are reproducible. Every experiment log must reference which split file was used.
- Run baselines (1–4 in the plan) before training the JEPA. Without baselines you can't interpret the JEPA results.
- The equation parser is the critical path — if parsing is wrong, everything downstream is wrong. Test it thoroughly against known equations before any training.
- Hardware: RTX 4080, 16 GB VRAM. Monitor and reduce batch size if OOM.
- Paper figures must be reproducible from notebooks in `notebooks/`. No one-off scripts.
- Keep checkpoints small — encoder weights + full JEPA state, skip optimizer state unless resuming from a crash.

## Data layout

- `data/raw/etp/` → symlink to the Equational Theories Project's `data/` directory. Source of `equations.txt` and `2025-08-11-vampire.json.gz`.
- `data/raw/invariants/` → symlink to the sibling invariants project's `assembly/output/` directory. Currently empty until that project is run.
- `data/processed/` → derived caches (parsed-tree PyG `Data` cache, decompressed pair tables). Not committed.
- `data/splits/` → holdout split index files. Committed, for reproducibility.
