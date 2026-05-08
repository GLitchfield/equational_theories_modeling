# Changelog

Deviations from `jepa_engineering_plan.md` and other notable project decisions, in reverse-chronological order. The plan itself is updated to reflect the new reality; entries here explain *why* a deviation was made and on what evidence, so a fresh session can understand the plan's current state without digging through git history.

---

## 2026-05-08 — Phase 1 baseline locked

`outputs/results/baseline_phase1_l2norm.json` is the project's reference baseline. It captures `epoch_006.pt` from the `phase1_l2norm_10ep` wandb run, evaluated on all 2,202,788 L1 val pairs (no batch cap):

  - balanced_accuracy = 0.8121
  - true_recall = 0.7535
  - false_recall = 0.8708
  - threshold = 1.0613 (L2-normalized embeddings, distances ∈ [0, 2])
  - epoch 6 of 10 (per-epoch peak; epoch 9 is statistically identical at 0.8119)
  - encoder 153,920 params, predictor 16,576 params
  - default config: hidden 128, embed 64, 4 layers, 4 heads, dropout 0.1, batch 2048, lr 1e-3, weight_decay 0.05, margin 1.0, true_weight 1.0
  - git commit: `f601c00`
  - wandb run: `fxwe85fy`

Subsequent variants will be recorded the same way under `outputs/results/<variant_name>.json` via `scripts/record_run.py`. Comparison table: `python scripts/record_run.py --summary`.

---

## 2026-05-07 — L2-normalize encoder + predictor outputs (collapse fix)

The first end-to-end Phase 1 sanity training (1 epoch on 19.8M train pairs, run `phase1_sanity_1ep` on wandb) showed a classic JEPA failure mode: distances grew unboundedly during training, exceeding the contrastive loss's `margin = 1.0` after the LR warmup, at which point the FALSE term `relu(margin - d)² = 0` went silent. With only the TRUE term active, no force kept the embedding scale bounded; train loss saturated near 42 and val balanced accuracy plateaued at 0.635 (barely above the 0.63 majority-class baseline). Calibrated val threshold settled at ~7.99 — about 8× the loss's margin.

**Fix**: L2-normalize the output of the encoder *and* the predictor (`F.normalize(..., dim=-1)`). Both encoder outputs (which the EMA target encoder also produces by construction) and predictor outputs now lie on the unit sphere. Pairwise distances are bounded to `[0, 2]`, so:

- The `margin = 1.0` sits at the midpoint of the achievable range (rather than being trivially exceeded).
- The FALSE term is always active when distances are below 1.
- There is no scale dimension for the optimizer to escape into.

This is the standard practice in modern contrastive learning (SimCLR, BYOL, DINO). User direction was specifically: try L2-normalization first; reach for VICReg only if dimensional collapse appears *after* this fixes the scale issue.

**Code changes**:
- `src/model/encoder.py`: `forward(batch)` returns `F.normalize(self.proj(pooled), dim=-1)`.
- `src/model/predictor.py`: `forward(z)` returns `F.normalize(self.net(z), dim=-1)`.
- This supersedes a 2026-05-07 design note saying the predictor should *not* L2-normalize. Predictor module docstring updated to reflect the new state.

**New tests** (3, all in addition to the existing suite):
- `test_encoder_output_is_unit_norm`
- `test_predictor_output_is_unit_norm`
- `test_forward_outputs_are_unit_norm_and_distance_is_bounded` (JEPA level — verifies distances ≤ 2)

---

## 2026-05-07 — Predictor design clarifications

The plan's §"Predictor Architecture" is brief: "Small MLP, 2 layers, hidden_dim=128, smaller than encoder." Several details needed clarification at implementation time. User direction:

- **"2 layers" = two weight matrices** (BYOL/SimSiam convention): `Linear(64→128) → ReLU → Linear(128→64)`. *Not* three weight matrices. Three would add expressivity, pushing against the design principle of keeping the predictor bottlenecked.
- **No dropout** (the predictor's `dropout=0`, in contrast to the encoder's `0.1`). The predictor must be a clean, deterministic mapping from embedding space to embedding space; dropout would inject noise into `z_predicted` and degrade the contrastive loss signal. BYOL and SimSiam predictors have no dropout for exactly this reason — the precise-bottleneck role is the design principle.
- **No layer normalization, no output L2-normalization**. If embedding collapse occurs at training time, try VICReg regularization first; only L2-normalize the predictor output if VICReg doesn't help.
- **Activation = ReLU**, **bias = on**.

Final structure (`src/model/predictor.py`): 16,576 parameters. Encoder is 152,896, so predictor is ~9× smaller — comfortably within the "smaller than encoder" hard constraint.

The plan stays as-is (still vague on these specifics); this entry is the canonical record of the design choices.

---

## 2026-05-07 — Bidirectional edges with 4-dim edge features (plan amendment)

The plan previously specified parent→child unidirectional edges with a 2-dim `[is_left, is_right]` edge feature, plus a single directional root-to-root edge for Family-C with `edge_attr = [0, 0]`. With unidirectional edges, the root never receives any message from its descendants in any number of message-passing layers — all structural integration was forced into the global pooling step.

User authorized a plan amendment: edges are now **bidirectional** (`parent→child` *and* `child→parent` for every tree link), and the edge feature is 4-dim **`[is_left, is_right, is_downward, is_upward]`**. Family-C cross-link is also bidirectional, with both edges carrying `[0, 0, 0, 0]` (neither left/right nor up/down — an all-zeros pattern that attention can learn to treat as a distinct cross-link edge type).

Plan §"Parse Tree Representation" updated and a new §"Edge Representation" sub-section added (verbatim from the user's spec).

Effects:
- `EDGE_FEATURE_DIM`: 2 → 4 in `src/data/parser.py`.
- Parser emits two edges per tree link (down and up) with the four bits set per the spec.
- Family-C: bidirectional cross-link replacing the previous single directional root-to-root edge.
- Encoder default `edge_features`: 2 → 4 (`GATv2Conv(..., edge_dim=4)`).
- `configs/default.yaml` `encoder.edge_features`: 2 → 4.
- `data/processed/equation_graphs.pt` rebuilt: max edges per equation went 9 → 18, file 6.25 MB → 7.67 MB. Family-C count unchanged at 1,555.
- New test `test_bidirectional_edges_propagate_root_from_leaf_in_one_layer` in `tests/test_encoder.py` validates the entire reason for the change: a single GATv2Conv layer over a 3-node tree, perturbing only a leaf's input features, and asserting the root's output changes.

---

## 2026-05-07 — Encoder parameter count: ~150K for Phase 1 sanity

Implementing the plan's encoder config literally (`hidden_dim=128`, `embed_dim=64`, `num_layers=4`, `num_heads=4`, `dropout=0.1`; GATv2Conv with `concat=True` and `per_head_dim = hidden_dim / num_heads = 32`) yields **152,896 parameters** — an order of magnitude below the plan's "Estimated ~1-2M parameters."

User confirmed the plan's 1-2M estimate was off and that we should proceed with the ~150K config for Phase 1 ("let's just see if this works at all"). Larger models are flagged for later experimentation, presumably under the Phase 3 architecture sweep.

The plan still reads "~1-2M" — that wording is the user's prerogative; this changelog entry exists so future sessions reading the plan don't misinterpret the ~150K implementation as accidental drift.

Per-GATv2-layer breakdown (in=128, out=32, heads=4, edge_dim=2): `lin_l` 16,512 + `lin_r` 16,512 + `lin_edge` 256 + `att` 128 + `bias` 128 = 33,536. × 4 layers = 134,144. Plus input proj (1,280) + final proj 256→64 (16,448) + four LayerNorms (1,024 total) ≈ 152,896.

Bonus consequence: GPU forward on a batch of 2,048 graphs is ~4.6 ms, putting an epoch over 22M pairs at ~3 minutes — well below the plan's "30-60 minutes/epoch" estimate (which presumably assumed the larger encoder).

---

## 2026-05-06 — Parser deviations from the engineering plan

When implementing `src/data/parser.py` (Phase 1, first task), three encoding deviations and one implementation choice came up:

### 1. Variable slots: 4 → 6 (node features 7-dim → 9-dim)

The plan specified `[is_op, is_var, var_id_0..3, is_lhs_var]` (7 dimensions). Inspection of `data/raw/etp/equations.txt` shows the corpus alphabet is `{u, v, w, x, y, z}` — six distinct variables — and 22 equations actually use all six. With only 4 slots, 347 equations (5- and 6-variable cases combined) would lose information.

Updated to `[is_op, is_var, var_id_0..5, is_lhs_var]` = 9 dimensions. The constant `NUM_VAR_SLOTS = 6` is exposed at the top of `src/data/parser.py`. `configs/default.yaml` updated to `encoder.node_features: 9` accordingly.

Evidence: distinct-variable histogram across the 4,694 equations is `{1: 31, 2: 779, 3: 2090, 4: 1447, 5: 325, 6: 22}`.

### 2. Family-C root-to-root edge label: `edge_attr = [0, 0]`

The plan calls for a 2-dim edge feature `[is_left, is_right]` (one-hot) and a "special root-to-root edge" between LHS and RHS subtrees in Family-C equations. The plan does not specify how that edge should be labelled. We use `[0, 0]` — neither left nor right — as a third implicit class encoded via all-zeros. This keeps the 2-dim edge feature unchanged while still distinguishing root-links from tree edges.

Alternative considered: extend `edge_attr` to 3-dim with an explicit `is_root_link` bit. Rejected for now to preserve the plan's edge dimensionality; revisit if attention struggles to distinguish the `[0, 0]` case from masked-out tree edges.

### 3. `is_lhs_var = 0` everywhere in Family-C inputs

The plan's `is_lhs_var` flag marks "the distinguished LHS variable." For Family-C (LHS is a product, not a single variable), no single variable is distinguished. We set `is_lhs_var = 0` throughout for these equations; the LHS-vs-RHS distinction is carried by the LHS subtree's presence plus the directional root-link.

### 4. Hand-rolled parser instead of `lark`

The plan's parser sketch uses a `lark` grammar. Inspection of the corpus shows every sub-product is fully parenthesized and there is no operator-precedence ambiguity; a 50-line recursive-descent parser is faster, dependency-free, and equally correct. `lark` remains in `requirements.txt` for now in case it's useful elsewhere; can be dropped if no other use emerges.

### 5. Canonical variable ordering — traversal explicitly defined

The plan says variables are assigned "by order of first appearance (DeBruijn-style): first variable seen = 0, second = 1, etc." but doesn't specify the traversal order, which matters when LHS and RHS share variables in different positions. The parser uses LHS pre-order, then RHS pre-order. Plan updated to specify this explicitly.

### 6. Dropped `torch-scatter` and `torch-sparse` from `requirements.txt`

The plan's dependency list included `torch-scatter` and `torch-sparse`. Verified empirically that `torch_geometric` 2.7 + `torch` 2.6 handle scatter and sparse ops via PyTorch's built-in primitives — `Data`, `Batch`, `GATv2Conv` (with `edge_dim`), and `global_mean_pool` / `global_max_pool` all work on CPU and GPU without those packages. Only an informational warning that "max" scatter could be accelerated by `torch-scatter`. The packages also need a special wheel index from the PyG project that has to match the exact `torch + cuda` build, making them awkward to reproduce. Dropped from `requirements.txt`; can be added back if profiling shows scatter ops are a bottleneck during training.

### Verification

`tests/test_parser.py` covers the parser deviations (27 tests, full-corpus parse, Family-C count = 1,555). The full pipeline (parser → PyG `Data` → `Batch` → `GATv2Conv` → pooling) was smoke-tested on CPU and CUDA. Cache built via `scripts/parse_equations.py` (6.25 MB, 4,694 entries, max 10 nodes / 9 edges per equation).
