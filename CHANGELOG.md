# Changelog

Deviations from `jepa_engineering_plan.md` and other notable project decisions, in reverse-chronological order. The plan itself is updated to reflect the new reality; entries here explain *why* a deviation was made and on what evidence, so a fresh session can understand the plan's current state without digging through git history.

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
