# lib/

Shared primitives used by the embeddings work, the proof generation pilot, and the frozen studies. Code here has no upward dependencies — it imports only from the standard library + third-party packages, never from `embeddings/`, `proof_generation_pilot/`, or `et_studies/`.

## Contents

| Path | What |
|---|---|
| `lib/data/parser.py` | First-order equation parser. Builds the canonical `Term` representation (operations, variables) from the ETP `equations.txt` lines. |
| `lib/data/dataset.py` | PyG `Data` cache for the 4,694 equations. Reads `data/processed/equation_graphs.pt`. |
| `lib/data/splits.py` | Holdout-split index loader. |
| `lib/rewrite/matching.py` | First-order matcher + rewrite-step enumeration. |
| `lib/rewrite/unification.py` | Robinson unification (used by completion's critical-pair computation). |
| `lib/rewrite/lpo.py` | Lexicographic path ordering — strict (orders some pairs) and total (orders every pair, used by ordered completion). |
| `lib/scripts/parse_equations.py` | Generates the parse-tree cache. |
| `lib/scripts/build_pair_table.py` | Generates the 22M-row implication-pair table from the ETP export. |
| `lib/scripts/build_equation_invariants.py` | Generates the per-equation invariants table (spine type, LHS structure, cohomology summaries). |

## Tests

Tests for `lib/` modules live at the repo-root `tests/` directory (flat layout): `test_parser.py`, `test_dataset.py`, `test_splits.py`, `test_rewrite_matching.py`, `test_rewrite_lpo.py`, `test_rewrite_unification.py`.

## What goes here vs. what doesn't

- **Belongs here:** anything that is genuinely shared *and* stable. Parsing, matching, unification, LPO — these have well-defined behavior and don't change with research direction.
- **Doesn't belong here:** training code, study-specific analysis, proof-generation methods, anything tied to a particular research phase. Those live in `embeddings/`, `et_studies/X/`, or `proof_generation_pilot/`.

If a new shared primitive emerges from a study or the pilot, it can be promoted into `lib/` once stable.
