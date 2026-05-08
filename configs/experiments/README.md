# Experiments

Each YAML in this directory is a self-contained experiment recipe. The launcher trains it, picks the best (highest `val/balanced_accuracy`) epoch, and writes a JSON record under `outputs/results/` so results from different runs can be compared at a glance.

## Reproducing an experiment

```bash
python scripts/run_experiment.py configs/experiments/<name>.yaml
```

Convention: `<name>.yaml` produces

- wandb run named `<name>`
- checkpoints in `outputs/checkpoints/<name>/epoch_NNN.pt`
- result record at `outputs/results/<name>.json`

The launcher refuses to overwrite an existing result file; pass `--force` to re-run.

## Comparing results

```bash
python scripts/record_run.py --summary
```

Prints a table over every JSON in `outputs/results/`. Sortable by hand; columns are `name | epoch | bal_acc | true_recall | false_recall | threshold | loss | commit`.

## Anatomy of a YAML

Every experiment YAML is **standalone** — a full copy of `configs/default.yaml` with the variant edits applied. Reading the YAML alone tells you everything about the run; you don't have to mentally diff against a base file. The trade-off is duplication: if `default.yaml` changes, existing experiments don't move with it (which is the point — they remain reproducible).

The header comment in each YAML states the **hypothesis** and links to the result JSON.

## Adding a new experiment

1. Copy an existing YAML (e.g. `baseline.yaml`) to `configs/experiments/<new-name>.yaml`.
2. Edit the header comment: state the hypothesis and what changed.
3. Edit the relevant fields (e.g. `loss.true_weight`, `encoder.hidden_dim`, …).
4. **Do not** change `wandb.name` by hand — the launcher overrides it from the YAML filename.
5. Run `python scripts/run_experiment.py configs/experiments/<new-name>.yaml`.
6. Add a row to the table below in the same commit as the result JSON.

## Experiment list

| Name | Hypothesis | Result | Notes |
|---|---|---|---|
| `baseline` | Vanilla GATv2 + JEPA + contrastive (`true_weight=1.0`, `margin=1.0`) clears the Phase 1 sanity bar (≥80% bal_acc on L1). | [`outputs/results/baseline.json`](../../outputs/results/baseline.json) (pending — re-run via launcher) — historical hand-named record at [`outputs/results/baseline_phase1_l2norm.json`](../../outputs/results/baseline_phase1_l2norm.json) | First framework run; matches the locked-in pre-framework baseline configuration. |
| `true_weight_1p7` | Setting `loss.true_weight` to the inverse-prevalence value (1.7) equalizes per-class loss contribution against the 62.9 % FALSE / 37.1 % TRUE corpus skew and should narrow the recall gap. | [`outputs/results/true_weight_1p7.json`](../../outputs/results/true_weight_1p7.json): bal_acc 0.8141 (vs baseline 0.8121), true_recall 0.8173 ↑ from 0.7535, false_recall 0.8110 ↓ from 0.8708. Recall gap fully closed. | Outcome as predicted: recalls equalized, balanced accuracy essentially unchanged — `true_weight` redistributes errors but doesn't add capability. |
