# Changelog

Deviations from `jepa_engineering_plan.md` and other notable project decisions, in reverse-chronological order. The plan itself is updated to reflect the new reality; entries here explain *why* a deviation was made and on what evidence, so a fresh session can understand the plan's current state without digging through git history.

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

### Verification

`tests/test_parser.py` covers all of the above (27 tests, including a smoke test that parses every line of the 4,694-equation corpus and verifies the Family-C count is exactly 1,555).
