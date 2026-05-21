---
tags:
- graph-neural-network
- jepa
- mathematical-logic
- equational-theories
pipeline_tag: feature-extraction
license: apache-2.0
---

# Equational Implication Embeddings — v1.0

768-dimensional embeddings for the 4,694 equations of the
[Equational Theories Project](https://github.com/teorth/equational_theories)
corpus, learned by a Joint Embedding Predictive Architecture (JEPA)
trained against 22 million Lean-verified implication labels. Each
embedding is an L2-normalized vector designed so that pairwise distance
encodes equational implication: `E1 → E2` corresponds to small
`||z_e1 − z_e2||`.

The encoder is a small graph neural network (≈ 270K parameters) over
equation parse trees, with a parameter-free skip connection from the
pooled representation to the output so the projection bottleneck
doesn't discard pooled features.

## TL;DR

- 4,694 equations → 768-dim L2-normalized vectors.
- **Balanced accuracy 0.9841** on held-out implication pairs using the
  embedding's pairwise distance alone.
- **0.9996** using the same embeddings with a 2-layer MLP head — the
  embeddings carry more signal than the distance metric exploits.
- Specifically, the MLP recovers 91% of the "TRUE-by-vacuity" cases the
  pure distance metric structurally can't represent (high-variable weak
  axioms and the trivial reflexive `x = x`).

## Files

| File | Shape / type | Size | sha256 |
|---|---|---|---|
| `embeddings.npy` | `(4694, 768) float32`, L2-normalized | 14.4 MB | `c9ae46ab…1a5cdaa8` |
| `eq_ids.npy` | `(4694,) int32`, ascending 1..4694 | 19 KB | `69914ed9…24ecaed6` |
| `source_checkpoint.pt` | torch state dict + training config | 3.0 MB | `0389edfd…340b0b64` |
| `MODEL_CARD.md` | this file | — | — |
| `LICENSE` | Apache 2.0 license text | — | — |

`embeddings[i]` is the embedding of equation `eq_ids[i]`. Eq IDs are
1-indexed and stored ascending, so a downstream script can index by
row without consulting `eq_ids.npy` in the common case
(`embeddings[eq_id - 1]`).

## How to use

### As a static embedding table

```python
import numpy as np

Z = np.load('embeddings.npy')        # (4694, 768) float32, L2-normalized
ids = np.load('eq_ids.npy')          # (4694,) int32

# Vectors are unit-norm, so dot product is cosine similarity.
sim_to_eq42 = Z @ Z[42 - 1]          # eq IDs are 1-indexed; row = id - 1

# Pairwise distance encodes implication: smaller distance => more likely
# that the row-equation implies the column-equation.
def distance(i, j):
    return float(np.linalg.norm(Z[i - 1] - Z[j - 1]))

# Calibrated decision threshold on the L2 distance is approximately 1.25;
# `distance(i, j) < 1.25` predicts "equation i implies equation j".
```

### Re-encode new equations

The source checkpoint can be reloaded and run over arbitrary
ETP-style parse trees:

```python
import torch
from src.model.encoder import EquationEncoder

ckpt = torch.load('source_checkpoint.pt', weights_only=False, map_location='cpu')
enc_cfg = ckpt['config']['encoder']

encoder = EquationEncoder(
    node_features=enc_cfg['node_features'],
    edge_features=enc_cfg['edge_features'],
    hidden_dim=enc_cfg['hidden_dim'],
    embed_dim=enc_cfg['embed_dim'],
    num_layers=enc_cfg['num_layers'],
    num_heads=enc_cfg['num_heads'],
    dropout=enc_cfg['dropout'],
    skip_pooled=enc_cfg.get('skip_pooled', False),
    proj_dim=enc_cfg.get('proj_dim', None),
)

# The trained JEPA contains the context encoder, an EMA target encoder,
# and a predictor. We want only the context encoder's weights.
state = {
    k[len('context_encoder.'):]: v
    for k, v in ckpt['jepa_state_dict'].items()
    if k.startswith('context_encoder.')
}
encoder.load_state_dict(state)
encoder.eval()

# `batch` is a `torch_geometric.data.Batch` of parse-tree `Data` objects;
# the parser is in `src/data/parser.py`.
with torch.no_grad():
    z = encoder(batch)  # (B, 768), L2-normalized
```

## Model overview

| Property | Value |
|---|---|
| Encoder family | Graph Attention Network (GATv2) |
| Parameters | 270,592 |
| Layers | 4 × GATv2Conv, hidden 128, 4 heads, residual + post-LayerNorm |
| Node features | 9-dim (op/var indicator + 6 variable slots + LHS-var flag) |
| Edge features | 4-dim (left/right child + up/down direction) |
| Pooling | concat(mean-pool, max-pool) → 256-dim pooled rep |
| Output head | `[LayerNorm(Linear(pooled)) ; LayerNorm(pooled)]` (512 + 256) → L2-normalize |
| Embedding dim | 768 |
| Dropout | 0.1 (training only) |

The output's "projection half" (first 512 dims) is a learned linear
from the pooled rep; the "skip half" (last 256 dims) is the
LayerNormed pooled rep itself. Both halves participate in the unit
vector. The skip was added to keep features the projection alone was
discarding; in practice it lifted balanced accuracy by ~2pp.

## Intended use

- **Distance-based implication queries.** Compute
  `||z_e1 − z_e2||` and threshold to predict whether E1 → E2.
- **Input features for pair classifiers.** Concatenate `z_e1` and
  `z_e2` and feed to a 2-layer MLP for state-of-the-art accuracy on
  this corpus (0.9996 balanced accuracy on a 10% holdout). A linear
  classifier gives up ~5pp and is not recommended.
- **Probing the corpus's algebraic structure.** Per-equation
  invariants (spine type, variable count, magma signature, low-prime
  cohomology counts) are linearly accessible from these embeddings at
  80–99% accuracy depending on the target.

### Out of scope

- **New equations outside the ETP corpus.** The encoder generalizes
  to any equation expressible in the ETP alphabet (variables
  `u, v, w, x, y, z`, single binary operator `◇`), but its
  performance has only been validated on the 4,694 corpus equations.
- **Equations over richer signatures** (more than one operator, other
  variables, non-equational axioms).
- **Symbolic reasoning beyond pairwise implication.** The model
  produces a metric space, not a proof.

## Training data

- 22,028,942 implication labels over 4,694 equations, from the
  Equational Theories Project's `export_raw_implications_5_9_2026.csv`
  (a 4694 × 4694 dense int8 matrix where each off-diagonal entry
  encodes E1 → E2 as TRUE or FALSE, derived from Lean-verified
  proofs and counterexamples).
- Class balance: 37% TRUE, 63% FALSE.

## Training procedure

JEPA-style self-supervised setup:

- A **context encoder** (the deliverable) embeds E1.
- An **EMA-updated target encoder** (cosine schedule, τ: 0.996 → 0.999)
  embeds E2; its weights are a moving average of the context encoder's.
- A small **predictor MLP** maps `context_encoder(E1)` to a predicted
  embedding in the target space.
- The training signal is a margin-contrastive loss on the predicted vs.
  target embedding distance: pull TRUE pairs together, push FALSE
  pairs apart up to a margin (1.5 on the unit sphere). Augmented with
  VICReg variance / covariance regularization (λ = 0.1 each) and 10
  auxiliary single-layer classification heads on the pooled rep
  (per-equation structural invariants — spine type, LHS structure,
  variable count, magma signature bits).

Hyperparameters: AdamW lr 1e-3 with 1000-step cosine warmup then decay,
weight_decay 0.05, batch 2048, gradient clip 1.0, 10 epochs, seed 42.
Training wall-clock: ~104 minutes on a single RTX 4080.

The architectural choice of the skip connection and the loss-mix
weights came from a long sequence of ablations; the prior history is
in the project's CHANGELOG and the archived plan files. None of that
history is needed to use the model.

## Evaluation

All numbers below are on a random 10% pair holdout (seed 42,
2,202,894 pairs).

| Decision rule | Params | Balanced acc | TRUE recall | FALSE recall |
|---|---:|---:|---:|---:|
| Distance threshold on the embeddings | 0 | 0.9841 | 0.9792 | 0.9890 |
| Linear probe on `concat(z_e1, z_e2)` | 1,537 | 0.9530 | 0.9386 | 0.9675 |
| 2-layer MLP on `concat(z_e1, z_e2)` | 918,529 | 0.9996 | 0.9995 | 0.9997 |

A 2-layer MLP on the same embeddings clears the embedding-distance
baseline by +1.55pp and the linear probe by +4.66pp. The relational
features (abs-diff, Hadamard product) that are sometimes useful for
pair classifiers add nothing here — the 2-layer MLP discovers them on
its own.

### Vacuity (the embeddings carry the "global weakness" signal)

A subset of ≈ 32,500 pairs (≈ 0.15% of the corpus) consistently
fool *every* trained run of this architecture (8 different seeds /
recipes). These are pairs where E2 is structurally simple or weakly
constraining (e.g., `x = x`, or high-variable equations like
`x ◇ (y ◇ z) = w ◇ (u ◇ z)` that hold tautologically of almost any
magma) and so doesn't look "similar" to its E1 — yet E1 → E2 is TRUE
by vacuity, since every magma satisfying E1 also satisfies E2. A
single-threshold metric over a unit sphere structurally cannot
represent this kind of "global weakness of E2" property.

On the ~3,200 of those vacuity pairs that fall in the 10% holdout, the
distance metric gets 0% correct (by construction — they're defined as
the hard cases). A 2-layer MLP on the same embeddings recovers 91.4%,
with 100% on the trivial-reflexive case and 85-96% on the harder
asymmetric cases. So the vacuity information *is* in the embeddings;
the distance metric just can't read it.

## Limitations

- **Corpus is fixed.** Behavior on equations outside the 4,694 ETP
  equations is unvalidated. The encoder is willing to embed any
  parse tree over the right alphabet, but quality is not guaranteed.
- **Skip channel is low effective rank (~15).** The 256-dim skip half
  carries only ~15 effective dimensions across the corpus because the
  parse trees are small (mean ~5–8 nodes); most of the embedding's
  expressivity lives in the 512-dim projection half.
- **Vacuity needs nonlinear access.** A linear function on
  `concat(z_e1, z_e2)` recovers ~17% of the vacuity subset (it gets
  the trivial-reflexive case cleanly but misses the harder spines).
  Tasks where vacuity matters should use at least a 2-layer MLP head.
- **Seed sensitivity ≈ ±0.6pp.** A second-seed run with the same
  recipe landed at 0.9779 balanced accuracy (vs. this artifact's
  0.9841). For comparisons sensitive to the third decimal, average
  over multiple seeds rather than picking the best one.

## Reproduce

The `.npy` files are deterministic given the checkpoint and the parse
tree cache. To regenerate them from the included checkpoint:

```bash
python scripts/save_embeddings.py source_checkpoint.pt --output-dir . --force
mv equation_embeddings.npy embeddings.npy
mv equation_embeddings_ids.npy eq_ids.npy
```

The parse tree cache (`data/processed/equation_graphs.pt`) is built
from `data/raw/etp/equations.txt` via `scripts/parse_equations.py`.

## License

Apache License 2.0. See the `LICENSE` file shipped alongside this
artifact for the full text.

The training labels and equation corpus are derived from the
Equational Theories Project, which is itself Apache 2.0 — so this
artifact's license matches upstream.

## Citation

If you use these embeddings in research, please cite the upstream
corpus:

```
@misc{etp,
  title  = {Equational Theories Project},
  author = {Tao, Terence and contributors},
  year   = {2024},
  url    = {https://github.com/teorth/equational_theories},
  note   = {Lean-verified implication labels and counterexamples for the 4{,}694-equation corpus.}
}
```
