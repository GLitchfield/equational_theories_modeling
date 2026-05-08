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

## Reference run

After the first round, **`true_weight_1p7` (10 epochs, bal_acc 0.8141)** is the reference variants are compared against — the original `baseline` config undertrained the TRUE class against the 63 / 37 corpus skew, and `true_weight_1p7_30ep` confirmed that more epochs of the same recipe don't lift the balanced-accuracy ceiling. New experiments diff against `true_weight_1p7` and use 10 epochs unless they're explicitly probing training length.

## Experiment list

| Name | Hypothesis | Result | Notes |
|---|---|---|---|
| `baseline` | Vanilla GATv2 + JEPA + contrastive (`true_weight=1.0`, `margin=1.0`) clears the Phase 1 sanity bar (≥80% bal_acc on L1). | [`outputs/results/baseline.json`](../../outputs/results/baseline.json) (pending — re-run via launcher) — historical hand-named record at [`outputs/results/baseline_phase1_l2norm.json`](../../outputs/results/baseline_phase1_l2norm.json) | First framework run; matches the locked-in pre-framework baseline configuration. |
| `true_weight_1p7` ← **reference** | Setting `loss.true_weight` to the inverse-prevalence value (1.7) equalizes per-class loss contribution against the 62.9 % FALSE / 37.1 % TRUE corpus skew and should narrow the recall gap. | [`outputs/results/true_weight_1p7.json`](../../outputs/results/true_weight_1p7.json): bal_acc 0.8141 (vs baseline 0.8121), true_recall 0.8173 ↑ from 0.7535, false_recall 0.8110 ↓ from 0.8708. Recall gap fully closed. | Outcome as predicted: recalls equalized, balanced accuracy essentially unchanged — `true_weight` redistributes errors but doesn't add capability. |
| `true_weight_1p7_30ep` | Same config, 30 epochs instead of 10. Probes whether the ~0.81 ceiling is real or just undertraining (10-epoch loss/bal_acc were still trending). | [`outputs/results/true_weight_1p7_30ep.json`](../../outputs/results/true_weight_1p7_30ep.json): bal_acc 0.8148 (+0.0007 over 10-ep), true_recall 0.8426, false_recall 0.7870. Best epoch 28; val/bal_acc plateaued around epoch 10 and oscillated within ±0.005 thereafter. | Ceiling confirmed: 3× the compute bought ~0.001 — noise-level. The current architecture + contrastive loss tops out near 0.815. Train loss continued downward, suggesting mild overfitting. Next lever should be capability or loss form, not more training. |
| `embed_dim_128` | Doubling `embed_dim` (64 → 128) widens the encoder's final projection and both predictor linears. Tests whether the 0.815 ceiling is partly an embedding-bottleneck artefact. GAT layers themselves are unchanged. | [`outputs/results/embed_dim_128.json`](../../outputs/results/embed_dim_128.json): bal_acc **0.8295** (+0.0154 over `true_weight_1p7`), true_recall 0.8395, false_recall 0.8194. Best epoch 9 (final); val/bal_acc sparkline `▁▃▅▆▆▇▇███` still climbing — run is undertrained. | Embedding bottleneck confirmed: the 0.815 plateau was capacity-bound at 64 dims. Wider projection lifts both per-class recalls; recall gap re-opens slightly (0.006 → 0.020) with TRUE pulling ahead. Worth re-running with more epochs to find the actual ceiling at 128 dims. |
