# MLP Classifier on Frozen Embeddings

## Goal

Train an MLP classifier on frozen JEPA embeddings to answer two questions:

1. **What is the predictive ceiling of the embeddings?** The JEPA's distance metric gets 98.40% on the full 22M corpus. A nonlinear classifier on the same embeddings might do better — the gap measures how much information the embeddings contain that the distance metric doesn't exploit.

2. **Can the MLP discover "TRUE by vacuity"?** The failure-overlap analysis identified **32,496 always-wrong-trained pairs** (`outputs/analyses/failure_overlap/always_wrong_trained.parquet`) — pairs that all 8 well-trained runs miss, 99.99% of them TRUE pairs where E2 is a weak/tautological equation. The distance metric can't represent "E2 is implied by everything." The MLP sees both embeddings and can learn the weakness pattern without geometric constraints.

## Methodology resolutions (locked in 2026-05-16)

After scrutiny, the following were resolved before implementation. The
plan body has been updated to reflect them; this section is the
canonical record.

1. **Always-wrong set: use the 32,496 trained-only set, not the older
   25,982-across-13 set.** Per Rule B (`feedback_run_aggregate_partitioning.md`),
   the trained partition (8 runs that peaked past ep0) is the
   methodologically correct target — the 25,982-across-13 number was
   inflated by 5 pathological runs that mostly predict majority class.
   Source: `outputs/analyses/failure_overlap/always_wrong_trained.parquet`.
2. **Matched holdout for the JEPA-vs-MLP comparison.** Both the frozen
   JEPA's distance metric and the MLP are evaluated on the same 10% pair
   holdout (seed 42). The 0.9840 number from the full 22M re-eval isn't
   a fair baseline because the JEPA was trained on all of it; the MLP
   gets a held-out 10%. Re-evaluating the frozen JEPA on the same 10%
   gives the apples-to-apples baseline. Expect it to be ≈0.9840 since
   the JEPA trained on the other 90%.
3. **Train on 90% of ALL pairs (including always-wrong-trained pairs).**
   The goal is measuring the embedding's predictive ceiling, not
   generalization to unseen hard cases. The ~3,250 always-wrong pairs
   that land in the 10% holdout (32,496 × 0.10 ≈ 3,250) form the
   vacuity-diagnostic subset.
4. **Vacuity criterion: high overall bal_acc AND high always-wrong
   accuracy, simultaneously.** A model that "learned vacuity" gets high
   accuracy on always-wrong ∩ holdout WITHOUT sacrificing overall
   FALSE recall. The 80%/50% thresholds in the original plan are
   meaningless on a 99.99%-TRUE set (predict-all-TRUE gets 100% there).
   The headline is the joint condition.
5. **Encoder pathway: context encoder for both E1 and E2.** The target
   encoder is a training artifact (EMA copy for stable JEPA targets).
   After training, use the context encoder everywhere — same equation
   should produce the same embedding regardless of which side of the
   pair it's on.
6. **Re-run the linear probe on the same embeddings.** The 0.9525
   probe number in the comparison table is from a different model
   (skip_h128_seed1, 0.9779). Run a logistic regression on the
   skip_h128_seed42 ep9 concat embeddings, same train/test split, so
   the linear-probe-vs-MLP comparison is apples-to-apples.
7. **Class imbalance handling.** Use `pos_weight=n_FALSE/n_TRUE` (≈63/37 ≈ 1.70)
   in `BCEWithLogitsLoss`. Without it the MLP may bias toward FALSE.
8. **Split seed: 42.** Matches other splits in the project.

## Architecture

```python
class PairClassifier(nn.Module):
    def __init__(self, embed_dim=768, hidden_dims=[512, 256]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], 1),
        )
    
    def forward(self, z_e1, z_e2):
        """
        Args:
            z_e1: [batch, 768] — frozen embedding of equation 1
            z_e2: [batch, 768] — frozen embedding of equation 2
        Returns:
            [batch, 1] — logit for TRUE (sigmoid for probability)
        """
        pair = torch.cat([z_e1, z_e2], dim=1)  # [batch, 1536]
        return self.net(pair)
```

Total params: 1536×512 + 512 + 512×256 + 256 + 256×1 + 1 ≈ 918K.

Also test with element-wise features:

```python
class PairClassifierWithFeatures(nn.Module):
    def __init__(self, embed_dim=768, hidden_dims=[512, 256]):
        super().__init__()
        # Input: concat + abs_diff + hadamard = 768*4 = 3072
        input_dim = embed_dim * 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], 1),
        )
    
    def forward(self, z_e1, z_e2):
        pair = torch.cat([
            z_e1,
            z_e2,
            torch.abs(z_e1 - z_e2),
            z_e1 * z_e2,
        ], dim=1)  # [batch, 3072]
        return self.net(pair)
```

## Data

### Embedding extraction

Use the best checkpoint: `skip_h128_seed42_diag` epoch 9 (98.40% bal_acc). Extract 768-dim L2-normalized embeddings for all 4,694 equations. Save as numpy array.

```python
embeddings = {}
model = load_checkpoint("skip_h128_seed42_diag/epoch_009.pt")
model.eval()
with torch.no_grad():
    for eq_id, graph in equation_graphs.items():
        z = model.context_encoder(graph)
        embeddings[eq_id] = z.squeeze().cpu().numpy()

X = np.stack([embeddings[eid] for eid in sorted(embeddings)])
np.save("outputs/mlp_classifier/embeddings_skip42.npy", X)
```

### Training pairs

All 22M pairs from the ETP canonical CSV. For each pair, look up z_e1 and z_e2 from the precomputed array. The embedding lookup is an array index, not a forward pass — training is fast.

### Split

Use a 90/10 random PAIR split, **seed 42** (not equation holdout — this measures what the embeddings can predict on known equations). The 10% holdout (~2.2M pairs) is for evaluation. Save the split indices to `data/splits/mlp_pair_split_seed42.parquet` so the linear-probe baseline and all MLP variants train and evaluate on identical sets.

The **frozen JEPA's distance metric is also re-evaluated on the same 10% holdout** (not the full 22M) — the 0.9840 number cited elsewhere is from the full corpus, which the JEPA trained on; the matched-holdout number is the apples-to-apples baseline for the MLP. Threshold is re-calibrated on the 90% training portion to maximize bal_acc, then frozen for the 10% test.

For the vacuity diagnostic, **always-wrong-trained pairs are NOT excluded from training**. The goal is the predictive ceiling, not generalization. The ~3,250 always-wrong pairs that land in the 10% test split (32,496 × 0.10) form the vacuity-diagnostic subset, evaluated separately.

## Training

```python
optimizer = Adam(mlp.parameters(), lr=1e-3)
# Class imbalance: ~63% FALSE / 37% TRUE -> pos_weight ≈ 1.70.
# Compute exactly from the training split (n_neg / n_pos).
pos_weight = torch.tensor(n_false_train / n_true_train)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

for epoch in range(10):
    for batch in dataloader:
        eq1_ids, eq2_ids, labels = batch
        z_e1 = embedding_matrix[eq1_ids]   # lookup, no gradient
        z_e2 = embedding_matrix[eq2_ids]   # lookup, no gradient

        logits = mlp(z_e1, z_e2)
        loss = loss_fn(logits.squeeze(), labels.float())

        loss.backward()   # gradients flow only through MLP
        optimizer.step()
        optimizer.zero_grad()
```

- Batch size: 8192 (embeddings are precomputed, so batches are cheap)
- Epochs: 10 (the MLP is small and the data is large, so convergence should be fast)
- LR: 1e-3 with cosine decay
- No GPU needed for training if embeddings fit in RAM (4,694 × 768 × 4 bytes = 14.4MB — trivially fits)
- Actually the 22M pairs are the bottleneck for RAM. Load the pair table and do epoch-wise sampling or use a DataLoader with workers.

## Evaluation

### Primary metrics

On the 2.2M holdout pairs:
- Balanced accuracy
- TRUE recall, FALSE recall
- Number of confidently wrong predictions (predicted prob > 0.9 on wrong side)
- Calibration curve (predicted probability vs actual frequency)

### Comparison targets

All evaluated on the **same 10% pair holdout, seed 42** (per Methodology Resolution 2).

| Model | bal_acc | TRUE recall | FALSE recall |
|---|---|---|---|
| JEPA distance (skip_h128_seed42, ep9, full 22M re-eval — NOT comparable) | 0.9840 | 0.9788 | 0.9891 |
| **JEPA distance on the matched 10% holdout** (apples-to-apples baseline) | ??? | ??? | ??? |
| Linear probe (sklearn LogisticRegression on concat(z_e1, z_e2), this run) | ??? | ??? | ??? |
| **MLP classifier (concat-only)** | ??? | ??? | ??? |
| **MLP classifier (concat + absdiff + hadamard)** | ??? | ??? | ??? |

The first row (full-corpus JEPA) is shown for context but isn't the comparison target — it was evaluated on data the JEPA trained on. The MLP vs the matched-holdout JEPA is the real measurement.

### Vacuity analysis

Evaluate the trained MLP on the **always-wrong-trained pairs that landed in the 10% holdout** (~3,250 of the 32,496):
- What fraction does the MLP predict TRUE (at the threshold-calibrated decision boundary)?
- What's its mean predicted probability on these pairs?
- Break down by E2 spine type (family_c, trivial, left, mixed, right) — using `data/processed/equation_invariants.parquet`'s spine_type column.

**Vacuity criterion (Methodology Resolution 4):** the MLP "captured vacuity" iff both:
(a) **Overall holdout bal_acc stays ≥ matched-holdout JEPA baseline** (no overall regression), AND
(b) **MLP accuracy on always-wrong ∩ holdout is substantially above the matched JEPA's** (which is 0% on always-wrong-trained by definition).

The 80%/50% thresholds in the original draft were meaningless on a 99.99%-TRUE set ("always predict TRUE" gets 100% there). The joint condition is what distinguishes "learned vacuity" from "biased toward TRUE."

### Per-equation weakness analysis

For each equation as E2, compute: what fraction of pairs where this equation is E2 does the MLP predict TRUE? Plot this against the ground-truth fraction. Equations where the MLP's fraction matches ground truth are well-calibrated. Equations where the MLP underpredicts TRUE (vs ground truth) are the remaining hard cases.

## Ablations

### 1. Concat only vs full features

Compare PairClassifier (1536-dim input, concat only) vs PairClassifierWithFeatures (3072-dim input, concat + absdiff + hadamard). If absdiff/hadamard help, the MLP is using relational features beyond what concatenation provides.

### 2. Smaller MLP

Try a single-layer classifier: Linear(1536, 1). This is a linear probe with both embeddings. Compare to the sklearn LogisticRegression probe from the probing experiments. Should match within noise — if it doesn't, the training setup has a bug.

### 3. Skip channel only vs proj channel only

Train the MLP on just the skip half (256-dim × 2 = 512-dim input) vs just the proj half (512-dim × 2 = 1024-dim input). Which channel contributes more to pair classification? This complements the channel ablation from the JEPA training diagnostics.

## What the results tell us

**MLP ≫ JEPA distance (e.g., 99.5% vs 98.4%):** The embeddings contain substantially more information than the distance metric exploits. The JEPA's geometric assumption (implication ≈ proximity) is the bottleneck, not the embeddings.

**MLP ≈ JEPA distance (e.g., 98.6% vs 98.4%):** The distance metric is already near-optimal for these embeddings. The remaining errors are genuinely beyond what the embeddings encode. Further improvement requires richer embeddings (dual-encoder, more invariants, order-5 training data), not a better classifier.

**MLP captures vacuity (joint condition met: holdout bal_acc ≥ matched JEPA AND high accuracy on always-wrong ∩ holdout):** The weakness pattern IS in the embeddings. The MLP learned "when z_e2 looks like a weak equation, predict TRUE" without sacrificing overall FALSE recall — meaning it's a real discovered feature, not a TRUE-bias.

**MLP doesn't capture vacuity** (either overall bal_acc regresses, OR always-wrong subset accuracy stays low): the weakness pattern isn't linearly/nonlinearly accessible from the embeddings. The embeddings encode structural similarity but not the global property "this equation is implied by everything." Fixing this requires either (a) adding a weakness feature to the embeddings (e.g., fraction of test magmas that satisfy the equation), or (b) a different embedding architecture that encodes global graph properties.

## Notes for Claude Code

- The embeddings are FROZEN. No gradient flows to the encoder. The only trainable parameters are the MLP's weights.
- Use the skip_h128_seed42_diag epoch 9 checkpoint (the 98.40% model). Extract embeddings via `model.context_encoder` (NOT target encoder, NOT predictor — per Methodology Resolution 5) once, save to `outputs/mlp_classifier/embeddings_skip42_ctx_ep9.npy`, reuse across all experiments.
- The embedding lookup should use integer indexing into the numpy array, not dictionary lookup. Build a mapping eq_id → array_index once and save it alongside the embeddings.
- The 90/10 split index file (seed 42) is saved to `data/splits/mlp_pair_split_seed42.parquet` so the linear probe, all MLP variants, and the matched-holdout JEPA evaluation use identical sets. Commit it.
- For the 22M pair training, use a PyTorch DataLoader with num_workers > 0. The pair table fits in memory; the bottleneck is the DataLoader's batch construction.
- Log to wandb under group "mlp_classifier".
- Save the best MLP checkpoint — this is a deployable artifact.
- The always-wrong pair analysis uses `outputs/analyses/failure_overlap/always_wrong_trained.parquet` (32,496 rows, the trained-runs partition). Inner-join with the 10% holdout to get the ~3,250 vacuity-diagnostic pairs.
- Report both threshold-calibrated accuracy (optimized threshold) and default 0.5-threshold accuracy. The MLP's natural threshold may not be 0.5 due to class imbalance (37% TRUE) and the `pos_weight` adjustment.
- Per CLAUDE.md's reproducer rule: the analysis output is the writeup in `outputs/runs/mlp_classifier_*/diagnostics_report.md`, and any committed CSV / parquet must have a script in `scripts/` or `notebooks/` that regenerates it.
