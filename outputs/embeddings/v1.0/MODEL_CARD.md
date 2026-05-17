# Equational Implication Embeddings — v1.0

Per-equation embeddings of the 4,694 equations in the Equational Theories
Project corpus, intended as a stable input artifact for downstream
analyses. This is the **JEPA-phase deliverable**; the JEPA work has
been wrapped (see `archive/`) and the embeddings below are the
canonical version going forward.

If a downstream task surfaces a deficiency, we may revisit — but the
v1.0 embeddings are considered good enough to build on (matched-holdout
JEPA distance bal_acc 0.9841; a frozen-embedding MLP on top reaches
0.9996 with 91.37% recovery of the always-wrong-trained vacuity subset).

## Files

| File | Shape / type | sha256 | Size | Tracked? |
|---|---|---|---|---|
| `MODEL_CARD.md` | text | — | — | ✓ |
| `source_checkpoint.pt` | JEPA state dict + config | `0389edfd…` | 2,991,490 B | ✓ |
| `embeddings.npy` | `(4694, 768) float32` | `c9ae46ab…` | 14,420,096 B | gitignored |
| `eq_ids.npy` | `(4694,) int32` | `69914ed9…` | 18,904 B | gitignored |

`embeddings.npy` and `eq_ids.npy` share row order:
`embeddings[i]` is the embedding of equation `eq_ids[i]`. Eq IDs
are 1-indexed and stored ascending (`eq_ids == np.arange(1,
4695)`), so a downstream script can index directly by row
without consulting `eq_ids.npy` in the common case.

`source_checkpoint.pt` is the trained JEPA state dict from
`outputs/checkpoints/skip_h128_seed42_diag/epoch_009.pt`, copied
here so the artifact is fully self-contained — anyone with this
directory plus `data/processed/equation_graphs.pt` can regenerate
the `.npy` files. The `.npy` files are gitignored (per the repo's
`outputs/embeddings/**/*.npy` rule) since they're large and
deterministic from the checkpoint.

## Specification

| Property | Value |
|---|---|
| Embedding dim | 768 |
| Normalization | L2-normalized (unit sphere) |
| Encoder pathway | `JEPA.context_encoder.forward(batch, normalize=True)` |
| Encoder architecture | GATv2, hidden_dim=128, 4 layers, 4 heads, skip_pooled=true, proj_dim=512 |
| Concatenation | `[proj_norm(output_proj(pooled)) ; skip_norm(pooled)]` then L2-norm |
| Trained against | Contrastive loss + VICReg (0.1 / 0.1) + 10 structural aux heads (margin 1.5) |

The 768 dims consist of a 512-dim projection channel (the trained
projection from the GNN's pooled rep) and a 256-dim skip channel
(per-channel LayerNorm of the pooled rep itself). See
`archive/skip_connection_plan.md` for the architecture rationale.

## Source

| Field | Value |
|---|---|
| Source checkpoint | `source_checkpoint.pt` (copied from `outputs/checkpoints/skip_h128_seed42_diag/epoch_009.pt`) |
| Checkpoint sha256 | `0389edfd253febc2c54d4b8326f58788f4b7a31eb9152316d99a68fd340b0b64` |
| Run | `skip_h128_seed42_diag` (wandb id `rs8578ak`) |
| Run record | `outputs/runs/skip_h128_seed42_diag/diagnostics_report.md` |
| Training config | `configs/experiments/skip_h128_seed42_diag.yaml` |
| Pair table | `data/processed/implication_pairs.parquet` (from `data/raw/etp_export/export_raw_implications_5_9_2026.csv`) |
| Seed | 42 |
| Extraction date | 2026-05-16 |

## Reproduce

From a fresh clone (the source checkpoint travels with the artifact):

```bash
python scripts/save_embeddings.py \
    outputs/embeddings/v1.0/source_checkpoint.pt \
    --output-dir outputs/embeddings/v1.0 --force
mv outputs/embeddings/v1.0/equation_embeddings.npy outputs/embeddings/v1.0/embeddings.npy
mv outputs/embeddings/v1.0/equation_embeddings_ids.npy outputs/embeddings/v1.0/eq_ids.npy
```

`scripts/save_embeddings.py` runs the context encoder over every
equation in `data/processed/equation_graphs.pt`, then writes the
L2-normalized embeddings + sorted ID list. Output is deterministic
given the checkpoint and the parse-tree cache.

## Validation

Reported metrics, all measured on the same matched 10% pair holdout
(`data/splits/level1_holdout_seed42.parquet`, seed 42):

| Metric | Value | Source |
|---|---|---|
| JEPA distance bal_acc (matched holdout) | 0.9841 | `outputs/runs/skip_h128_seed42_diag/matched_holdout_eval.json` |
| JEPA distance bal_acc (full 22M corpus) | 0.9840 | `outputs/runs/skip_h128_seed42_diag/failure_summary.json` |
| Linear probe bal_acc (matched holdout) | 0.9530 | `outputs/runs/mlp_classifier_linear/eval_summary.json` |
| MLP-head bal_acc (matched holdout) | 0.9996 | `outputs/runs/mlp_classifier_concat/eval_summary.json` |
| MLP-head accuracy on always-wrong-trained ∩ holdout | 91.37% | `outputs/runs/mlp_classifier_concat/vacuity_eval.json` |

So: a frozen-embedding 2-layer MLP recovers 91.37% of the pairs that
all 8 trained JEPA runs miss — the "TRUE-by-vacuity" pattern (high-
variable family_c equations and the trivial reflexive `x = x`). The
vacuity information is encoded in these embeddings; the JEPA's
geometric distance metric is what couldn't read it. Downstream
classifiers operating on these embeddings can.

See `outputs/runs/mlp_classifier_concat/diagnostics_report.md` for
the full Phase 3/4 analysis.

## Known limitations

1. **Effective rank of the skip channel is ~15.** The 256-dim skip
   channel (LayerNorm of mean+max pool) carries only ~15 effective
   dimensions across the corpus — the underlying GNN pooled rep is
   low-rank because the parse trees are small (~5–8 nodes mean).
   Not blocking, but if a downstream task wants to use the skip
   channel directly, expect that dimensional structure.
2. **Vacuity is in the embeddings but is nonlinear.** A linear probe
   on `concat(z_e1, z_e2)` only captures 17% of the always-wrong-
   trained subset (and only the trivial reflexive cleanly).
   Downstream classifiers that need the vacuity signal should be at
   least 2-layer.
3. **Seed sensitivity at h=128+skip is +0.61pp.** A second-seed run
   (`skip_h128_seed1_diag`) reached 0.9779 (vs this run's 0.9840).
   The v1.0 embeddings are from the better seed; if a strict
   sensitivity analysis is needed, the seed=1 ones are at
   `outputs/runs/skip_h128_seed1_diag/equation_embeddings.npy`.
