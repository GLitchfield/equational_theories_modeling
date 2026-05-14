# Embedding Probing Experiments

## Goal

Test whether the best model's embeddings (h=128+aux, seed=1, 96.21%) already encode cohomological information, structural invariants, and pairwise algebraic features. Linear probes on frozen embeddings — if a linear classifier can recover a feature, the embedding encodes it in a linearly accessible way.

## Methodology decisions (resolved 2026-05-14)

After scrutiny these were locked in before implementation:

1. **Bins** — Use `COHOM_COUNT_BINS` and `COHOM_MEAN_BINS` from `scripts/build_equation_invariants.py`. Probe results are directly comparable to the aux-run targets. The bin schemes that originally appeared in this plan (different from the implementation) are obsolete.
2. **Baselines** — Every probe reports against TWO floors: (a) majority-class accuracy for that target, (b) probe accuracy on embeddings from a freshly-initialized (untrained) model with the same architecture. The gap between random-init and trained is the information training added.
3. **Metrics** — Binary targets report BOTH plain accuracy AND `balanced_accuracy_score`. For sparse targets like `is_balanced` (0.7% positive) balanced accuracy is the only meaningful number. Multi-class targets report plain accuracy.
4. **Pair confound** — Experiment 5's `has_obstruction` is reported on three subsets: (a) both equations have ≥1 non-null cohom cell ("clean"), (b) one or both all-null ("confounded"), (c) all pairs combined. The clean subset answers the research question.
5. **Equivalence-class probe (Experiment 6)** — Switch from k-fold CV (broken at 1,415 classes / 4,694 samples) to nearest-neighbor in embedding space. Report top-1, top-5, top-10 nearest-neighbor accuracy.
6. **Both representations** — Probe both the 256-dim pooled rep (mean+max concat, what the aux heads saw) AND the 512-dim L2-normalized output (what the contrastive loss uses). A feature present in pooled but missing in output reveals projection-head information loss.
7. **`COHOM_TRIPLES`** — Compute locally as `[(a, b, p) for p in [2,3,5,7] for a in range(p) for b in range(p)]`. Don't add to the build script unless future code needs it elsewhere.
8. **Multi-epoch probing** — Probe ep0, ep3, ep6 of `aux_h128_seed1_diag` so we see the information-evolution story (which features emerge when during training).
9. **Compatibility regression** — Drop pairs with no shared triples; report dropped count.

## Setup

### Extract embeddings

Load the best checkpoint (`aux_h128_seed1_diag`, epoch 6). Run all 4,694 equations through the context encoder. Save the 512-dim embedding vectors.

```python
import torch
import numpy as np

# Load model checkpoint
model = load_checkpoint("outputs/checkpoints/aux_h128_seed1_diag/epoch_006.pt")
model.eval()

# Extract embeddings for all equations
embeddings = {}
with torch.no_grad():
    for eq_id, graph in equation_graphs.items():
        z = model.context_encoder(graph)  # [1, 512], L2-normalized
        embeddings[eq_id] = z.squeeze().cpu().numpy()

# Stack into matrix: [4694, 512]
eq_ids = sorted(embeddings.keys())
X = np.stack([embeddings[eid] for eid in eq_ids])
np.save("outputs/probing/embeddings.npy", X)
np.save("outputs/probing/eq_ids.npy", np.array(eq_ids))
```

### Probe infrastructure

Use sklearn LogisticRegression for classification, Ridge for regression. 5-fold cross-validation. Report mean ± std accuracy (classification) or R² (regression).

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

def probe_classification(X, y, name):
    """Linear probe for classification targets."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores

def probe_regression(X, y, name, mask=None):
    """Linear probe for regression targets. Optional mask for null values."""
    if mask is not None:
        X_masked = X[mask]
        y_masked = y[mask]
    else:
        X_masked, y_masked = X, y
    reg = Ridge(alpha=1.0)
    scores = cross_val_score(reg, X_masked, y_masked, cv=5, scoring="r2")
    print(f"{name}: R²={scores.mean():.4f} ± {scores.std():.4f} (n={len(y_masked)})")
    return scores
```

---

## Experiment 1: Structural invariants (sanity check)

These should probe well — the aux heads predicted them at 100% accuracy, so they're definitely in the embeddings.

Targets:
- spine_type (6-class)
- lhs_structure (binary)
- num_variables (6-class)
- 7 magma signature bits (binary each)

Load from: `invariants/03_spine/spine.parquet`, `invariants/01_lhs_structure/lhs_structure.parquet`, `invariants/04_variables/variables.parquet`, `invariants/08_magma_signature/magma_signature.parquet`

```python
# Example for spine_type
spine = pd.read_parquet("invariants/03_spine/spine.parquet")
y_spine = spine.set_index("eq_id").loc[eq_ids, "spine_type"].values
probe_classification(X, y_spine, "spine_type")
```

**Expected result**: 95%+ on all targets. If any target probes below 90%, something is wrong with the embedding extraction.

---

## Experiment 2: Per-prime cohomology summaries

The features we tried as aux targets. Tests whether the GNN already learned them from syntax + contrastive signal alone.

Targets (from `invariants/11_cohomology/cohomology.parquet`):
- count_p2 (binned: {0, 1, 2, 3, 4})
- count_p3 (binned: {0, 1, 2, 3, 4-5, 6+})
- count_p5 (binned: {0, 1, 2, 3, 5, 6-8, 9+})
- count_p7 (binned: {0, 1, 2, 3, 7, 8-12, 13+})

```python
cohom = pd.read_parquet("invariants/11_cohomology/cohomology.parquet")

# Compute count_p for each prime
for p in [2, 3, 5, 7]:
    triples_for_p = [t for t in COHOM_TRIPLES if t[2] == p]
    counts = []
    for eq_id in eq_ids:
        profile = cohom.set_index("eq_id").loc[eq_id, "cohom_profile"]
        count = sum(1 for i, t in enumerate(COHOM_TRIPLES) if t[2] == p and profile[i] is not None)
        counts.append(count)
    y = bin_counts(counts, p)  # apply the binning scheme
    probe_classification(X, y, f"count_p{p}")
```

**Key question**: Do these probe at 90%+ (GNN already knows) or 60-70% (GNN doesn't capture this)?

---

## Experiment 3: Cell-level cohomology (87 individual triples)

For each (a,b,p) triple, two probes:
- Binary: "does this equation satisfy the linear magma at this triple?" 
- Regression (for satisfying equations only): "what is dim H² at this triple?"

```python
for i, (a, b, p) in enumerate(COHOM_TRIPLES):
    # Binary: satisfies or not
    y_binary = np.array([1 if cohom_profiles[eid][i] is not None else 0 for eid in eq_ids])
    
    # Skip if all same class (no variance)
    if y_binary.mean() < 0.01 or y_binary.mean() > 0.99:
        continue
    
    probe_classification(X, y_binary, f"satisfies_({a},{b},{p})")
    
    # Regression: dim H² for satisfying equations
    mask = y_binary == 1
    if mask.sum() > 50:  # need enough samples
        y_h2 = np.array([cohom_profiles[eid][i] for eid in eq_ids])
        probe_regression(X, y_h2, f"dim_h2_({a},{b},{p})", mask=mask)
```

This is 87 × 2 probes = up to 174 probes. Many triples will have too few satisfying equations to probe meaningfully. Report all probes with n > 50 samples.

**Key question**: Do cell-level features probe as well as the per-prime summaries? If cell-level is worse, the summaries captured the useful information. If cell-level reveals specific triples that probe much better or worse than average, those triples carry special discriminative power.

---

## Experiment 4: Non-anchored invariants

Features the model was NEVER trained on — neither as aux targets nor as input. Tests what the contrastive loss discovered on its own.

Targets (from various invariant parquets):
- is_trivial (binary) — `invariants/02_trivial/trivial.parquet`
- is_singleton_forcing (binary) — `invariants/05_singleton_forcing/singleton_forcing.parquet`
- leaf variable identity (categorical) — `invariants/06_leaf_variables/leaf_variables.parquet`
- op_count_asymmetry (regression) — `invariants/07_tree_metrics/tree_metrics.parquet`
- is_balanced (binary, very sparse 0.7%) — `invariants/04_variables/variables.parquet`
- spine_depth (regression, for non-trivial/non-family-C equations) — `invariants/03_spine/spine.parquet`

**Key question**: Which non-anchored features does the contrastive loss discover? High probe accuracy on a non-anchored feature means the contrastive task NEEDED that feature and the GNN found it. Low probe accuracy means the feature isn't useful for implication prediction or the GNN couldn't learn it.

---

## Experiment 5: Pairwise cohomological features

The most important probe — tests whether the embedding PAIR geometry captures algebraic relationships.

Sample 100K pairs from the full corpus (stratified: 50K TRUE, 50K FALSE). For each pair:

```python
# Pair features
z_pair = np.concatenate([X[eq1_idx], X[eq2_idx]])  # [1024]

# Cohomological obstruction: exists a triple where dim H²_E1 > dim H²_E2?
has_obstruction = any(
    profile_e1[i] is not None and profile_e2[i] is not None 
    and profile_e1[i] > profile_e2[i]
    for i in range(87)
)

# Cohomological compatibility: fraction of shared triples with equal dim H²
shared_triples = [(i, profile_e1[i], profile_e2[i]) 
                   for i in range(87) 
                   if profile_e1[i] is not None and profile_e2[i] is not None]
if shared_triples:
    compatibility = sum(1 for _, d1, d2 in shared_triples if d1 == d2) / len(shared_triples)
else:
    compatibility = None
```

Probes:
1. **Predict has_obstruction** from concat(z_e1, z_e2). If the embeddings encode this, the model already "knows" the cohomological relationship.
2. **Predict compatibility score** from concat(z_e1, z_e2). Tests whether the pairwise embedding geometry captures cohomological similarity.
3. **Predict ground truth (TRUE/FALSE)** from concat(z_e1, z_e2). Baseline — how well does a linear classifier on embeddings do compared to the JEPA's distance-based prediction?

**Key question**: Can a linear probe on embedding pairs predict cohomological obstruction? If yes (>80%), the embeddings already encode the pairwise algebraic relationship. If no (<60%), the pairwise algebraic information is NOT in the embeddings and would need to be added (via the dual-encoder approach or per-pair features).

---

## Experiment 6: Equivalence class prediction

The hardest probe — can a linear classifier recover equivalence class membership (1,415 classes) from the embeddings?

```python
# Equivalence class labels (need invariant 09 computed, or derive from the implication graph)
# If not available, skip this experiment
probe_classification(X, y_equiv_class, "equivalence_class_1415")
```

With 1,415 classes and 4,694 equations (average 3.3 per class), this is a very hard classification problem. Even 30% accuracy would be meaningful (vs 0.07% random baseline). 50%+ would indicate strong equivalence class structure in the embeddings.

---

## Output

```
outputs/probing/
├── embeddings.npy                    # [4694, 512] embedding matrix
├── eq_ids.npy                        # [4694] equation IDs
├── results/
│   ├── structural_probes.json        # Experiment 1 results
│   ├── cohom_summary_probes.json     # Experiment 2 results
│   ├── cohom_cell_probes.json        # Experiment 3 results
│   ├── non_anchored_probes.json      # Experiment 4 results
│   ├── pairwise_probes.json          # Experiment 5 results
│   └── equiv_class_probes.json       # Experiment 6 results (if available)
└── figures/
    ├── probe_accuracy_summary.png    # Bar chart of all probe accuracies
    └── cohom_cell_heatmap.png        # Heatmap of per-triple probe accuracy
```

### Summary table format

```
| Experiment | Target | Type | Probe Accuracy | Notes |
|---|---|---|---|---|
| 1 (structural) | spine_type | 6-class | 0.98 ± 0.01 | Sanity check |
| 2 (cohom summary) | count_p5 | 7-class | ??? | Key question |
| 3 (cohom cell) | satisfies_(1,1,3) | binary | ??? | Per-triple |
| 4 (non-anchored) | is_singleton_forcing | binary | ??? | Contrastive discovered? |
| 5 (pairwise) | has_obstruction | binary | ??? | Algebraic relationship |
```

---

## Notes for Claude Code

- Use the epoch 6 checkpoint from `aux_h128_seed1_diag` (the 96.21% model). Path: check `outputs/checkpoints/` for the exact filename.
- The embeddings should be the L2-normalized output (what the contrastive loss sees), not the pre-norm output.
- For logistic regression, use `max_iter=2000` to ensure convergence on the 512-dim input. Use `C=1.0` (default regularization).
- For the pairwise probes (Experiment 5), concatenate z_e1 and z_e2 into a 1024-dim vector. Also try element-wise product and element-wise absolute difference as alternative pair representations — the best representation for pairwise probing isn't always concatenation.
- The cohomology data is in `invariants/11_cohomology/cohomology.parquet` with a `cohom_profile` column (list of 87 int|None values). Import `COHOM_TRIPLES` from the cohomology module to know the ordering.
- Report all results to 4 decimal places with ± standard deviation from 5-fold CV.
- Save all results as JSON for downstream analysis.
- The entire probing suite should run in under 30 minutes with no GPU.
- If equivalence class labels aren't available (invariant 09 not computed yet), skip Experiment 6 and note the dependency.
