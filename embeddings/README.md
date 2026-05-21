# embeddings/

All JEPA training, MLP classifier head, and frozen v1.0 equation embeddings. The JEPA phase is **complete and wrapped**; downstream work consumes the v1.0 embeddings as a frozen input.

## Headline results

| Artifact | Metric | Number |
|---|---|---|
| v1.0 embeddings (JEPA distance bal_acc, matched holdout) | 0.9841 |
| Frozen-embedding MLP head (matched holdout) | 0.9996 |
| MLP recovery on the "always-wrong-trained" subset (TRUE-by-vacuity) | 91.37% |
| Probing accuracy on cohomology features (linear) | 80–88% |

## v1.0 embeddings (the deliverable)

- **Spec:** 4694 × 768 float32, L2-normalized.
- **Source:** `embeddings/outputs/embeddings/v1.0/source_checkpoint.pt` — the `skip_h128_seed42_diag` epoch 9 checkpoint. The .pt and the model card are committed; the .npy embeddings are regenerable from the .pt and gitignored.
- **Model card:** `embeddings/outputs/embeddings/v1.0/MODEL_CARD.md` — sha256s, reproduction recipe, known limitations.
- **Regenerate:** `python embeddings/scripts/save_embeddings.py embeddings/outputs/embeddings/v1.0/source_checkpoint.pt --output-dir embeddings/outputs/embeddings/v1.0 --force`

## Contents

| Path | What |
|---|---|
| `embeddings/src/model/` | GATv2 encoder, predictor, JEPA, auxiliary heads, MLP pair classifier |
| `embeddings/src/training/` | Train loop, losses (VICReg, SIGReg), scheduler, run recording |
| `embeddings/src/probing/` | Linear probes on the encoder's pooled representation |
| `embeddings/src/baselines/`, `evaluation/` | Currently empty placeholder packages |
| `embeddings/scripts/` | Run scripts: `train.py`, `run_experiment.py`, `evaluate_checkpoint.py`, `save_embeddings.py`, `save_predictions.py`, `extract_probe_embeddings.py`, etc. |
| `embeddings/configs/` | YAML training configs — `default.yaml` + `experiments/*.yaml` (34 experiments) |
| `embeddings/outputs/checkpoints/` | Per-epoch checkpoints (gitignored) |
| `embeddings/outputs/runs/` | Per-run records: `diagnostics_report.md`, `diagnostic_extract.json`, `per_step_trace.csv`, `per_epoch_eval.csv` |
| `embeddings/outputs/embeddings/v1.0/` | The frozen v1.0 deliverable |
| `embeddings/outputs/probing/` | Probing results (committed JSONs) + embeddings (gitignored .npy) |
| `embeddings/outputs/failure_analysis/`, `failure_analysis_m1p5_vic0p1/` | Per-run failure characterizations |
| `embeddings/outputs/results/`, `figures/`, `logs/` | Misc summaries, plots, training logs |

## Reproduction at a glance

```bash
# 1. Build shared caches (lib/scripts/)
python lib/scripts/parse_equations.py
python lib/scripts/build_pair_table.py
python lib/scripts/build_equation_invariants.py

# 2. Train (uses an experiment config in embeddings/configs/experiments/)
python embeddings/scripts/run_experiment.py embeddings/configs/experiments/skip_h128_seed42_diag.yaml

# 3. After training, regenerate v1.0 embeddings from the source checkpoint
python embeddings/scripts/save_embeddings.py embeddings/outputs/embeddings/v1.0/source_checkpoint.pt \
    --output-dir embeddings/outputs/embeddings/v1.0 --force
```

## Archived plans

All JEPA-era plans live in `archive/`: `jepa_engineering_plan.md`, `aux_heads_implementation_plan.md`, `probing_experiments_plan.md`, `dual_encoder_plan.md`, `skip_connection_plan.md`, `mlp_classifier_plan.md`.
