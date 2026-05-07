# JEPA for Equational Implication

## Project Overview

Train a Joint Embedding Predictive Architecture (JEPA) to learn the geometric structure of equational implication over magmas. The model takes two equation parse trees as input and learns an embedding space where distance encodes implication relationships.

The primary contribution is the learned embedding space itself — what structure emerges, how it aligns with known algebraic invariants, and what it reveals about the geometry of the implication graph. Classification accuracy is a validation metric, not the headline result.

The training data is 22 million Lean-verified implication pairs from the Equational Theories Project (ETP). The invariant features (spine type, magma signatures, cohomology profiles, equivalence classes) are computed in a separate project and used here for evaluation and interpretability, NOT as model inputs. The model learns from raw parse trees only.

## Architecture

### Overview

```
E1 (parse tree) → Context Encoder → z_context
                                         ↓
                                    Predictor → z_predicted
                                         ↓
                                   Compare with z_target
                                         ↑
E2 (parse tree) → Target Encoder → z_target
```

- **Context Encoder**: GNN (GATv2) over parse tree of E1. Trained by gradient descent.
- **Target Encoder**: Identical architecture. Updated via exponential moving average (EMA) of context encoder. NOT trained by gradient descent.
- **Predictor**: Small MLP mapping z_context → z_predicted. Trained by gradient descent.
- **Loss**: Contrastive — z_predicted close to z_target when TRUE, far when FALSE.

### Why JEPA and not classification

A classifier would output TRUE/FALSE directly. JEPA instead learns an embedding space where the GEOMETRY encodes implication structure. This is better because:
1. Equivalence classes emerge as clusters without supervision
2. The embedding space is interpretable and probeable against known invariants
3. The model learns "what an equation IS" (the encoder) separately from "what implication means" (the predictor)
4. The embedding space can be analyzed, visualized, and probed — a classifier's internals are opaque

### Parse Tree Representation

Each equation has the form `LHS = RHS` where LHS is typically a bare variable `x`.

Represent each equation's RHS as a directed tree (PyTorch Geometric `Data` object):
- **Node types**: `variable` (leaf) or `operation` (internal node, the ◇ operator)
- **Node features**: One-hot encoding of node type + variable identity
  - Variables encoded by order of first appearance (DeBruijn-style): first variable seen = 0, second = 1, etc.
  - This makes the encoding invariant to variable renaming (x,y,z vs a,b,c)
- **Edges**: Parent-to-child directed edges with left/right labels
  - Edge features: one-hot `[is_left, is_right]`

Node feature vector (per node): `[is_op, is_var, var_id_0, var_id_1, var_id_2, var_id_3, var_id_4, var_id_5, is_lhs_var]` = 9 dimensions. The corpus uses six variables (`{u, v, w, x, y, z}`) and up to 6 distinct per equation; see CHANGELOG entry 2026-05-06.

Variables are canonicalized DeBruijn-style by walking LHS pre-order, then RHS pre-order; the first new variable seen gets id 0, the next 1, and so on.

For equations where LHS is a product (Family C, 1,555 equations), encode the full LHS tree as well and add it as a subgraph with a directional root-to-root edge from LHS root to RHS root. The root-link carries `edge_attr = [0, 0]` (neither left nor right — a third implicit class encoded via all-zeros). For Family-C inputs, `is_lhs_var = 0` throughout; the LHS-vs-RHS distinction is carried by the LHS subtree's presence and the root-link direction.

### Encoder Architecture

GATv2-based encoder over equation parse trees. Produces a fixed-size embedding vector for any equation. Target parameters: hidden dim 128, embedding dim 64, 4 layers, 4 heads, 0.1 dropout. Estimated ~1-2M parameters.

### Predictor Architecture

Small MLP mapping z_context → z_predicted. 2 layers, hidden dim 128. Must be SMALLER than the encoder — if too expressive, it bypasses the encoder's need to learn good representations.

### Loss Function

Contrastive loss pulling TRUE pairs together and pushing FALSE pairs apart:
- TRUE pairs: minimize distance between z_predicted and z_target
- FALSE pairs: maximize distance up to a margin

Consider InfoNCE as an alternative if embeddings collapse. Add VICReg regularization if needed.

### EMA Schedule

Target encoder momentum: cosine anneal from τ=0.996 to τ=0.999 over training.

---

## Holdout Strategy

The holdout design is central to the research contribution. A naive random split is nearly meaningless because every equation appears in ~9,000 training pairs. We use three levels of holdout, each testing a different aspect of generalization.

### Level 1: Random Pair Holdout

Hold out 10% of pairs randomly. Both equations in each held-out pair appear in the training data in other combinations.

**What it tests**: Interpolation — can the model combine known ingredients in a new combination?
**Expected accuracy**: 90%+. Failure here means something is broken.

### Level 2: Equation Holdout

Hold out 500 equations entirely (stratified across spine types, class sizes, complexity). Every pair involving a held-out equation is excluded from training.

**What it tests**: Generalization to unseen equations. Requires the encoder to have learned structural features rather than per-equation fingerprints.

**The critical metric**: The Level 1 → Level 2 accuracy gap measures memorization vs structural learning.

**Sub-populations** (report all three separately):
- **One-unknown**: one equation in the pair was held out, one was in training
- **Both-unknown**: both equations were held out
- The drop from one-unknown to both-unknown measures the anchoring effect

### Level 3: Equivalence Class Holdout

Hold out 150 entire equivalence classes. Every equation in those classes excluded from training.

**What it tests**: Generalization across algebraic neighborhoods.
**Expected accuracy**: Lowest of the three. Above chance (50%) means real structural learning.

### Stability Testing

Run Levels 2 and 3 with 3-5 different random seeds for holdout selection. Report mean ± std. High variance means some equations are harder to generalize to than others.

### Summary

| Level | What's held out | Tests | Expected accuracy |
|---|---|---|---|
| 1 | Random 10% of pairs | Interpolation | 90%+ |
| 2 | 500 equations (all their pairs) | Equation-level generalization | 70-85% |
| 2a | ...one equation unknown | Partial generalization | Higher than 2b |
| 2b | ...both equations unknown | Full generalization | Lower than 2a |
| 3 | 150 equivalence classes | Class-level generalization | 60-75% |

---

## Baselines

Establish these before training the JEPA. You need them to interpret results.

### Baseline 1: Random (50%)
Chance level. Any useful model must beat this.

### Baseline 2: Majority class (63%)
Always predict FALSE. Model must beat this to demonstrate it learned beyond the base rate.

### Baseline 3: Feature-based MLP
Train a simple MLP on hand-crafted pair features from the invariants project (spine match, depth difference, balance asymmetry, variable count difference, magma signature separability, same equivalence class). This tells you how much of the problem is captured by KNOWN structural features. The JEPA's value is whatever it learns BEYOND these features.

Evaluate on all three holdout levels. The feature MLP on Level 2 is the most important comparison.

### Baseline 4: GNN classifier (no JEPA)
Same GNN encoder, but classification head instead of contrastive loss: `concat(encoder(E1), encoder(E2)) → MLP → sigmoid`. Isolates the value of the JEPA architecture specifically. If it matches JEPA on accuracy but has worse embedding quality (lower probing scores, less interpretable UMAP), that validates JEPA over plain classification.

---

## Experiment Roadmap

### Phase 1: Infrastructure and Sanity Checks (Day 1)

- Parse all 4,694 equations into PyG Data objects, validate thoroughly
- Build the three-level holdout splits, save indices for reproducibility
- Run baselines (feature MLP, GNN classifier) on all three levels
- Train JEPA Level 1 on 1M pairs — goal: >80% balanced accuracy
- If below 80%: debug (embedding collapse? parser bugs? loss not decreasing?)

### Phase 2: The Generalization Test (Day 2-3)

- Train on full data with Level 2 holdout configuration
- Evaluate on all three levels and sub-populations
- Run stability check (3 seeds for Level 2 holdout selection)
- Report generalization gaps: L1→L2, L2a→L2b, L2→L3
- Compare to baselines at each level

### Phase 3: Architecture and Loss Sweep (Day 3-4)

- Sweep encoder architectures: GATv2 vs GIN vs GraphSAGE
- Sweep dimensions: hidden 64/128/256, embed 32/64/128, layers 3/4/6
- Sweep losses: contrastive vs InfoNCE vs triplet, with/without VICReg
- Evaluate each on Level 2 both-unknown — the hardest, most discriminative test

### Phase 4: Embedding Analysis (Day 5-7)

- Extract embeddings for all 4,694 equations (training AND held-out)
- Probing experiments: linear probes predicting spine type, balance, singleton forcing, equiv class, magma signature — separately for training vs held-out equations
- Clustering: k-means (k=1415) → ARI against true equiv classes
- Visualization: UMAP colored by spine type, equiv class, depth, highlighting held-out equations
- Confusion analysis: categorize failures by invariant type

### Phase 5: Scaling and Ablation (Day 7-8)

- Learning curves: train on 100K, 500K, 1M, 5M, 22M pairs, evaluate all on same Level 2 holdout
- Self-supervised ablation: train on TRUE pairs only (no FALSE), compare embedding geometry
- Both results inform how much data and what kind of supervision the model needs

### Phase 6: Paper Writing (Day 8-10)

Key findings to report:
1. Generalization gaps across three holdout levels
2. What invariants the embeddings recover (probing) and what they DON'T
3. Geometry of the embedding space (UMAP, cluster structure)
4. Scaling analysis
5. Supervised vs self-supervised embedding comparison
6. Comparison to baselines at each level

---

## Environment Setup

### Dependencies

```
torch>=2.1.0
torch-geometric>=2.5.0
torch-scatter
torch-sparse
lark>=1.1.0
wandb
scikit-learn
umap-learn
matplotlib
seaborn
pandas
pyarrow
pyyaml
tqdm
```

### Hardware

- GPU: NVIDIA RTX 4080 (16GB VRAM)
- Expected VRAM usage: 4-12GB
- Training time per epoch (22M pairs, batch 2048): ~30-60 minutes
- Full experiment roadmap: ~10 days on single GPU

---

## File Structure

```
jepa-equational-implication/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── raw/                     # Symlink from data extraction project
│   ├── processed/
│   │   └── equation_graphs.pt
│   └── splits/                  # Saved split indices for reproducibility
│       ├── level1_split.json
│       ├── level2_split_seed42.json
│       └── level3_split_seed42.json
├── src/
│   ├── model/
│   │   ├── encoder.py
│   │   ├── predictor.py
│   │   └── jepa.py
│   ├── data/
│   │   ├── parser.py
│   │   ├── dataset.py
│   │   └── splits.py
│   ├── training/
│   │   ├── train.py
│   │   ├── losses.py
│   │   └── scheduler.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── embeddings.py
│   │   ├── probing.py
│   │   └── visualization.py
│   └── baselines/
│       ├── feature_mlp.py
│       └── gnn_classifier.py
├── configs/
│   ├── default.yaml
│   └── sweeps/
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   ├── parse_equations.py
│   └── build_splits.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   ├── 03_probing_experiments.ipynb
│   ├── 04_scaling_analysis.ipynb
│   └── 05_paper_figures.ipynb
└── outputs/
    ├── checkpoints/
    ├── embeddings/
    ├── probing_results/
    └── figures/
```

---

## Key Design Decisions and Rationale

### Why GATv2 over other GNN architectures
GATv2 (dynamic attention) lets the model learn WHICH parts of the parse tree matter. The spine is structurally important (McKenna's theorem), and attention can learn to weight it. This is an empirical question — Phase 3 tests alternatives.

### Why not encode both E1 and E2 into a single graph
Separate encoders let the model learn "what is an equation" independently from "what is implication." More interpretable, cleaner separation of concerns.

### Why asymmetric encoders (EMA) instead of shared weights
Implication is asymmetric. The EMA creates the right inductive bias — the encoding spaces for hypothesis and conclusion are related but not identical.

### Why contrastive loss instead of classification
Forces the embedding space to have meaningful geometry. A classifier would work for accuracy but produce opaque internals. Baseline 4 (GNN classifier) exists to test this hypothesis explicitly.

### Why three holdout levels
Random pair holdout (Level 1) is trivial for this data. The three levels form a hierarchy of difficulty that tells a much richer story than any single split. The generalization gaps ARE the key finding.

---

## Notes for Claude Code

- This project depends on data from a SEPARATE data extraction project. The `data/raw/` directory should be populated by symlink or copy from that project's outputs.
- Do not modify files in `data/raw/`. All processing goes to `data/processed/` or `data/splits/`.
- Use `wandb` for ALL experiment tracking. Every training run should be logged.
- The RTX 4080 has 16GB VRAM. Monitor VRAM usage and reduce batch size if OOM.
- The equation parser is a critical path item — if parsing is wrong, everything downstream is wrong. Test the parser against known equations before training.
- When evaluating, ALWAYS report metrics per holdout level separately. Never average across levels — the gap between levels is the key finding.
- Within Level 2, always report one-unknown and both-unknown sub-populations separately.
- Save holdout split indices to `data/splits/` so experiments are reproducible. Every experiment log should reference which split file was used.
- Run baselines FIRST before training the JEPA. You need baselines to interpret JEPA results.
- For the paper: every figure should be reproducible from a notebook in `notebooks/`. No one-off scripts that generate figures.
- Keep checkpoints small — save encoder weights and full JEPA state, skip optimizer state unless resuming from crash.
