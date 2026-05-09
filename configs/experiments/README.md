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

Locked-in encoder frame after the width sweep:

- `embed_dim=512` — the {64, 128, 256, 512} sweep slope went +1.5 / +1.7 / +0.6 pp per doubling; 512 captured the diminishing-returns gain. Locked.
- `encoder.hidden_dim=128` — widening to 256 went *backwards* and neither lower LR nor more attention heads recovered. Locked.
- `predictor.hidden_dim=128` — widening to 256 gave noise-level improvement, aligns with the plan's predictor-as-bottleneck design.
- `num_heads=4`, `num_layers=4` — unchanged from the plan default; depth (`num_layers`) is the next sweep direction.
- `true_weight=1.7` — inverse-prevalence upweighting of TRUE pairs.

**Current best: `embed_dim_512` at bal_acc 0.8518.** Future variants compare against this. New experiments use 10 epochs unless explicitly probing training length.

## Experiment list

| Name | Hypothesis | Result | Notes |
|---|---|---|---|
| `baseline` | Vanilla GATv2 + JEPA + contrastive (`true_weight=1.0`, `margin=1.0`) clears the Phase 1 sanity bar (≥80% bal_acc on L1). | [`outputs/results/baseline.json`](../../outputs/results/baseline.json) (pending — re-run via launcher) — historical hand-named record at [`outputs/results/baseline_phase1_l2norm.json`](../../outputs/results/baseline_phase1_l2norm.json) | First framework run; matches the locked-in pre-framework baseline configuration. |
| `true_weight_1p7` ← **reference** | Setting `loss.true_weight` to the inverse-prevalence value (1.7) equalizes per-class loss contribution against the 62.9 % FALSE / 37.1 % TRUE corpus skew and should narrow the recall gap. | [`outputs/results/true_weight_1p7.json`](../../outputs/results/true_weight_1p7.json): bal_acc 0.8141 (vs baseline 0.8121), true_recall 0.8173 ↑ from 0.7535, false_recall 0.8110 ↓ from 0.8708. Recall gap fully closed. | Outcome as predicted: recalls equalized, balanced accuracy essentially unchanged — `true_weight` redistributes errors but doesn't add capability. |
| `true_weight_1p7_30ep` | Same config, 30 epochs instead of 10. Probes whether the ~0.81 ceiling is real or just undertraining (10-epoch loss/bal_acc were still trending). | [`outputs/results/true_weight_1p7_30ep.json`](../../outputs/results/true_weight_1p7_30ep.json): bal_acc 0.8148 (+0.0007 over 10-ep), true_recall 0.8426, false_recall 0.7870. Best epoch 28; val/bal_acc plateaued around epoch 10 and oscillated within ±0.005 thereafter. | Ceiling confirmed: 3× the compute bought ~0.001 — noise-level. The current architecture + contrastive loss tops out near 0.815. Train loss continued downward, suggesting mild overfitting. Next lever should be capability or loss form, not more training. |
| `embed_dim_128` | Doubling `embed_dim` (64 → 128) widens the encoder's final projection and both predictor linears. Tests whether the 0.815 ceiling is partly an embedding-bottleneck artefact. GAT layers themselves are unchanged. | [`outputs/results/embed_dim_128.json`](../../outputs/results/embed_dim_128.json): bal_acc **0.8295** (+0.0154 over `true_weight_1p7`), true_recall 0.8395, false_recall 0.8194. Best epoch 9 (final); val/bal_acc sparkline `▁▃▅▆▆▇▇███` still climbing — run is undertrained. | Embedding bottleneck confirmed: the 0.815 plateau was capacity-bound at 64 dims. Wider projection lifts both per-class recalls; recall gap re-opens slightly (0.006 → 0.020) with TRUE pulling ahead. Worth re-running with more epochs to find the actual ceiling at 128 dims. |
| `embed_dim_256` | Doubling embed_dim again (128 → 256). Sweep step 3 of 3; tests whether projection capacity has more headroom or returns are already diminishing. With predictor.hidden_dim left at 128, the predictor (256 → 128 → 256) becomes a real bottleneck — aligned with the plan's "predictor smaller than encoder" intent. | [`outputs/results/embed_dim_256.json`](../../outputs/results/embed_dim_256.json): bal_acc **0.8461** (+0.0166 over `embed_dim_128`, +0.0320 over `true_weight_1p7`), true_recall 0.8283, false_recall 0.8639. Best epoch 8 (of 10). | No diminishing returns yet — sweep slope is essentially linear: 64→128 +1.5 pp, 128→256 +1.7 pp. FALSE recall jumped past TRUE this round (gap flipped); threshold tightened to 1.045. Run is closer to saturation than `embed_dim_128` was (best at ep 8, ep 9 slightly lower). Total params 203K → 269K (+33%). |
| `embed_dim_256_pred_h256` | Same encoder as `embed_dim_256`; predictor.hidden_dim widens 128 → 256 to remove the symmetric→bottleneck topology shift introduced at embed=256. If bal_acc lifts, the predictor's narrow middle was the binding constraint. If it stays at ~0.846, the encoder is the constraint and the next probe should target it. | [`outputs/results/embed_dim_256_pred_h256.json`](../../outputs/results/embed_dim_256_pred_h256.json): bal_acc 0.8480 (+0.0019 over `embed_dim_256`), true_recall 0.8292, false_recall 0.8667. Best epoch 9 (final). | **Predictor was NOT the constraint.** +65K predictor params bought ~0.002 — within sampling noise. Per-class recalls essentially unchanged. The 0.846 ceiling is encoder-bound. Next probe: widen `encoder.hidden_dim` (the GAT stack itself), not the predictor. The plan's "predictor as bottleneck" design holds — tightening it doesn't hurt bal_acc, so we can keep it cheap. |
| `hidden_dim_256` | Widen `encoder.hidden_dim` 128 → 256 (the actual GAT stack), keeping `embed_dim=256` and the cheap predictor (`predictor.hidden_dim=128`). Tests whether the relational stack's per-layer capacity was the binding constraint at the 0.846 ceiling. | [`outputs/results/hidden_dim_256.json`](../../outputs/results/hidden_dim_256.json): bal_acc **0.8169** (-0.0292 vs `embed_dim_256` 0.8461), true_recall 0.7573 ↓ from 0.8283, false_recall 0.8766 ↑ from 0.8639. Best epoch 8. | **Negative result — bigger encoder went backwards.** Train loss converged HIGHER than the smaller model's (~0.55 vs ~0.50 avg), so this is underfit/under-optimized, not overfit. Suspect causes: LR 1e-3 may be too high for ~3× the params; per-head dim doubled to 64 with `num_heads` fixed at 4; warmup_steps=1000 unchanged. The "wider GAT stack" hypothesis isn't disproven — the experiment didn't isolate it cleanly enough. Re-run with adjusted LR / heads / epochs before drawing a conclusion. |
| `hidden_dim_256_lr5e4_warmup2k` | Re-run `hidden_dim_256` with LR halved (1e-3 → 5e-4) and warmup doubled (1000 → 2000 steps). Tests whether the prior negative result was purely an optimization artifact. Decision rule: bal_acc > 0.846 → wider stack helps (lock in); ≈ 0.846 → wider stack matches narrower (lock in cheaper hidden_dim=128); ≪ 0.846 → either LR still wrong or wider stack genuinely doesn't help for this task. | [`outputs/results/hidden_dim_256_lr5e4_warmup2k.json`](../../outputs/results/hidden_dim_256_lr5e4_warmup2k.json): bal_acc **0.8115** (-0.0054 vs `hidden_dim_256`, -0.0346 vs `embed_dim_256`), true_recall 0.7555, false_recall 0.8674. Best epoch 9 (final). | **LR was not the constraint.** Lowering LR made it slightly *worse* and train loss went up — we're now under-training more, not over-training less. Rules out "LR too high for big model." Remaining suspects: (a) per-head dim doubled to 64 with `num_heads=4` fixed — try `num_heads=8` to keep per-head=32; (b) wider GAT genuinely doesn't help for this task structure. TRUE recall stuck at ~0.755 in both wide-GAT runs despite `true_weight=1.7`, suggesting the wider architecture has a different inductive bias that resists TRUE-pair pulling. |
| `hidden_dim_256_h8` | At hidden_dim=256, restore per-head dim to 32 by doubling num_heads (4 → 8). Pure topology change — identical 668K encoder params, just more / smaller attention heads. Isolates whether the per-head doubling was the cause of the wider-GAT failure. | [`outputs/results/hidden_dim_256_h8.json`](../../outputs/results/hidden_dim_256_h8.json): bal_acc **0.8041** (-0.0128 vs `hidden_dim_256` h=4, -0.0420 vs `embed_dim_256`), true_recall 0.7452, false_recall 0.8631. Best epoch 6 (peaked mid-run). | **Per-head dim was not the issue.** Splitting attention into more, smaller heads made it slightly worse and converged earlier. Train loss higher still (0.611 vs 0.550 vs 0.501 across the wide-GAT progression) — every variation of hidden_dim=256 trains to a worse minimum. Combined with the LR test, conclusion holds: wider GAT layers genuinely don't help for this task at this scale. **Lock in `hidden_dim=128`.** |
| `embed_dim_512` | Extend the embedding-width sweep one more step (256 → 512), now that hidden_dim is locked at 128. Tests whether the projection still has headroom or returns finally diminish. Decides where to lock embed_dim before moving to depth (num_layers). | [`outputs/results/embed_dim_512.json`](../../outputs/results/embed_dim_512.json): bal_acc **0.8518** (+0.0057 vs `embed_dim_256`, +0.0038 vs `embed_dim_256_pred_h256`), true_recall 0.8336, false_recall 0.8699. Best epoch 8 (of 10). Train avg loss 0.490 — lowest yet. | **Diminishing returns observed.** Sweep slope: 64→128 +1.5 pp, 128→256 +1.7 pp, **256→512 +0.6 pp**. Real but modest gain at +49% params. Run is well-optimized (low train loss, late-but-not-final best ep). Decision pending: lock at 256 (cheaper, captures most of the gain) or 512 (squeeze the extra 0.6 pp). |
