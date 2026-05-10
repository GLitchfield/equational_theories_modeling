# Failure Analysis Plan

## Objective

Train the best model on ALL ~22M pairs for 20 epochs. Evaluate on every pair. Study the pairs the model gets wrong despite having trained on them. These are representation failures — relationships the JEPA geometry cannot capture.

## Current State

After the ETP canonical data cutover, the model reaches 91.1% balanced accuracy with near-perfect recall balance (TRUE 92.5%, FALSE 89.7%). Austin holdout recall is 94%. The data fix alone was worth +2pp and dramatically improved Austin generalization.

## Configuration

```yaml
encoder.hidden_dim: 128
encoder.embed_dim: 512
encoder.num_layers: 4
encoder.num_heads: 4
predictor.hidden_dim: 128
loss: contrastive
margin: 1.0
true_weight: 1.0
vicreg_var_weight: 0.25        # on encoder pre-norm
vicreg_cov_weight: 0.25
vicreg_gamma: 1.0
lr: 1e-3
warmup_steps: 1000
epochs: 20                     # extended for full convergence
batch_size: 2048
ema_tau: 0.996 → 0.999 cosine
seed: 42
```

NO holdout split. Train on ALL pairs. Evaluate on ALL pairs.

## Protocol

### Step 1: Train

```bash
python src/training/train.py \
    --config configs/default.yaml \
    --seed 42 \
    --name failure_analysis \
    --epochs 20 \
    --no-holdout \
    --save-predictions
```

Log to wandb. Monitor per-epoch: train loss, effective rank, and (if feasible) a quick accuracy estimate on a random 1M-pair subsample.

### Step 2: Full evaluation

After training, evaluate the best checkpoint on ALL ~22M pairs. Save per-pair predictions:

```
outputs/failure_analysis/predictions.parquet
```

Fields per pair:
- eq1_id (int)
- eq2_id (int)
- ground_truth (bool)
- predicted (bool)
- distance (float32)

### Step 3: Identify failure set

```python
df = pd.read_parquet("outputs/failure_analysis/predictions.parquet")
df["wrong"] = df["ground_truth"] != df["predicted"]

failures = df[df["wrong"]]
correct = df[~df["wrong"]]

print(f"Total pairs: {len(df)}")
print(f"Correct: {len(correct)} ({len(correct)/len(df)*100:.1f}%)")
print(f"Wrong: {len(failures)} ({len(failures)/len(df)*100:.1f}%)")
print(f"Wrong TRUE (predicted FALSE): {len(failures[failures.ground_truth])}")
print(f"Wrong FALSE (predicted TRUE): {len(failures[~failures.ground_truth])}")

failures.to_parquet("outputs/failure_analysis/failure_set.parquet")
```

At 91% accuracy, expect ~2M wrong pairs out of ~22M. These are the pairs the architecture cannot represent.

---

## Analysis

### Analysis 1: Austin Implications

Cross-reference failures with the ~820 Austin pairs.

```python
austin_pairs = load_austin_pairs()  # from ETP canonical data
austin_failures = failures.merge(austin_pairs, on=["eq1_id", "eq2_id"])
```

Report:
- Total Austin pairs, how many the model gets wrong
- Austin accuracy vs overall accuracy
- Are Austin pairs overrepresented in the failure set relative to their base rate (~3.7%)?

The current model gets 94% of holdout Austin pairs right. The remaining 6% wrong — what do those specific pairs look like? Print their equations.

This is potentially a headline finding: the GNN learned a syntactic marker for infinite-only implication failure, despite never seeing an infinite magma. The 6% it still misses might be the Austin pairs where the syntactic marker is absent or ambiguous.

### Analysis 2: Structural Invariants

For each pair in the failure set, compute or load from the invariants project:
- E1 and E2 spine types, spine compatibility
- E1 and E2 balance, balance asymmetry
- E1 and E2 tree depth, depth difference
- E1 and E2 variable count
- E1 and E2 singleton forcing
- Same equivalence class?

Compare distributions in the failure set vs the full dataset:

```python
# For each feature, compare failure set vs overall
for feature in features:
    overall_dist = df[feature].value_counts(normalize=True)
    failure_dist = failures[feature].value_counts(normalize=True)
    # Chi-squared test or visual comparison
```

Questions:
- Do failures cluster in specific spine type combinations?
- Are failed TRUE pairs disproportionately from the same equivalence class?
- Is there a depth or complexity signature?

### Analysis 3: Magma Signature Separability

For FALSE-labeled failures specifically (model wrongly predicted TRUE):

```python
false_failures = failures[~failures.ground_truth]  # ground truth FALSE, predicted TRUE
```

For each of these pairs, check: does ANY of the 12 test magmas separate them?

If failed FALSE pairs are NOT magma-separable: the model lacks the syntactic information to distinguish them. The counterexamples require magmas not in the test battery.

If failed FALSE pairs ARE magma-separable: the information exists in the parse tree (since magma evaluation depends on tree structure) but the GNN can't extract it. This is a representational failure, not an informational one.

### Analysis 4: Vampire Solve Time

If Vampire metadata (method_id, solve time) is available for these pairs:

```python
vampire = load_vampire_data()
failure_with_time = failures.merge(vampire, on=["eq1_id", "eq2_id"])

# Correlation
print(failure_with_time.groupby("wrong")["time"].describe())
```

Questions:
- Do failed pairs have longer Vampire solve times on average?
- Is there a solve-time threshold above which the model consistently fails?
- Plot: Vampire solve time vs model distance, colored by correct/wrong

### Analysis 5: Distance Distribution

The distance value tells you HOW wrong the model is, not just that it's wrong.

**Wrong FALSE pairs** (predicted TRUE, actually FALSE):
- These have distance BELOW the threshold — the model thinks they're close
- How far below? Just barely (borderline) or far below (confidently wrong)?

**Wrong TRUE pairs** (predicted FALSE, actually TRUE):
- These have distance ABOVE the threshold — the model thinks they're far apart
- How far above?

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Wrong FALSE
wrong_false = failures[~failures.ground_truth]
axes[0].hist(wrong_false.distance, bins=50, alpha=0.7)
axes[0].axvline(threshold, color='r', linestyle='--')
axes[0].set_title("Wrong FALSE pairs (predicted TRUE)\nDistance distribution")

# Wrong TRUE  
wrong_true = failures[failures.ground_truth]
axes[1].hist(wrong_true.distance, bins=50, alpha=0.7)
axes[1].axvline(threshold, color='r', linestyle='--')
axes[1].set_title("Wrong TRUE pairs (predicted FALSE)\nDistance distribution")

plt.savefig("outputs/failure_analysis/figures/distance_distributions.png", dpi=300)
```

Confidently-wrong pairs (far from threshold on the wrong side) are the most interesting — the model has a firm wrong belief about them.

### Analysis 6: Equation-Level Failure Concentration

```python
# How often each equation appears in failed pairs
eq_as_e1 = failures.eq1_id.value_counts().rename("failures_as_e1")
eq_as_e2 = failures.eq2_id.value_counts().rename("failures_as_e2")
eq_failures = pd.concat([eq_as_e1, eq_as_e2], axis=1).fillna(0)
eq_failures["total_failures"] = eq_failures.sum(axis=1)

# Normalize by how many pairs each equation appears in overall
eq_total = pd.concat([df.eq1_id.value_counts(), df.eq2_id.value_counts()], axis=1).fillna(0).sum(axis=1)
eq_failures["failure_rate"] = eq_failures["total_failures"] / eq_total

top20 = eq_failures.sort_values("failure_rate", ascending=False).head(20)
```

Questions:
- Are failures concentrated in a few "hard" equations or spread evenly?
- Print the equation text for the top 20 highest-failure-rate equations
- Do these equations share structural properties?
- Are any equations perfectly predicted (0% failure rate across all their pairs)?

### Analysis 7: Embedding Space Visualization

Extract embeddings for all 4,694 equations from the trained model.

**UMAP plots:**

1. All equations colored by per-equation failure rate (from Analysis 6)
   - High-failure equations should cluster if they share structural properties
   - Scattered high-failure equations suggest diverse failure modes

2. All equations colored by spine type, with high-failure equations marked with larger markers

3. All equations colored by equivalence class (top 20 classes), high-failure equations circled

**Save embeddings for future use:**
```python
embeddings = {}
for eq_id, eq_graph in equation_cache.items():
    z = context_encoder(eq_graph).detach().numpy()
    embeddings[eq_id] = z

np.save("outputs/failure_analysis/equation_embeddings.npy", 
        np.stack([embeddings[i] for i in sorted(embeddings)]))
```

These embeddings are also the input for the probing experiments (linear classifiers predicting invariants from embeddings).

### Analysis 8: Probing Experiments

Train linear probes (logistic regression) on frozen embeddings to test what the model learned:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = embedding_matrix  # [4694, 512]

for invariant_name, y in invariants.items():
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"{invariant_name}: {scores.mean():.3f} ± {scores.std():.3f}")
```

Invariants to probe:
- Spine type (5-class)
- Balance (binary)
- Singleton forcing (binary)
- Equivalence class (1,415-class)
- Each of the 12 magma signature bits (binary)
- Tree depth (regression or binned)
- Variable count (regression or binned)

High probe accuracy = the embedding encodes that feature.
Low probe accuracy = the embedding doesn't capture it.

Compare: which invariants does the model capture well, and which does it miss? The missed invariants suggest directions for enriching the input representation (e.g., cohomological features).

---

## Outputs

```
outputs/failure_analysis/
├── predictions.parquet              # All ~22M pair predictions
├── failure_set.parquet              # Just the wrong pairs
├── equation_embeddings.npy          # [4694, 512] embedding matrix
├── failure_summary.json             # Headline numbers
├── analysis/
│   ├── austin_analysis.json
│   ├── structural_features.parquet
│   ├── magma_separability.parquet
│   ├── vampire_correlation.json
│   ├── equation_concentration.parquet
│   ├── top20_failure_equations.txt
│   └── probing_results.json
└── figures/
    ├── distance_distributions.png
    ├── failure_by_ground_truth.png
    ├── failure_vs_vampire_time.png
    ├── umap_failure_heatmap.png
    ├── umap_spine_with_failures.png
    ├── equation_failure_rate_distribution.png
    └── probing_accuracy_bar_chart.png
```

## Key Numbers to Report

- Training accuracy on all 22M pairs after 20 epochs
- Effective rank at convergence
- Total failure count and percentage
- Failure breakdown: wrong TRUE vs wrong FALSE
- Austin accuracy (overall and on the ~97 holdout pairs)
- Whether failed FALSE pairs are magma-signature-separable
- Top invariants by probing accuracy
- Bottom invariants by probing accuracy (what the model DOESN'T capture)
- Top 20 equations by failure rate + their equation text
- Count of "confidently wrong" pairs (distance far from threshold on wrong side)

## Compute Budget

- Training: 20 epochs × ~5 min/epoch = ~100 min
- Full eval on 22M pairs: ~100 min
- Analysis scripts: ~30 min
- UMAP + visualization: ~15 min
- Probing: ~5 min
- Total: ~4 hours

## Notes for Claude Code

- NO holdout. Train on everything. The `--no-holdout` flag must disable the split.
- For threshold calibration: calibrate on the training predictions. Fine for this purpose.
- Prediction file will be large (~500MB-1GB). Ensure disk space.
- Save the DISTANCE for every pair, not just the binary prediction. Analysis 5 needs it.
- Extract and save the embedding matrix (4694 × 512) — needed for probing and UMAP.
- Use the ETP canonical CSV as the data source (post-cutover, includes Austin pairs).
- All figures at 300 DPI.
- Log to wandb under name "failure_analysis".
- If 22M-pair full eval is too slow, discuss before subsampling. Do not silently reduce.
