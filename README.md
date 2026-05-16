# JEPA for Equational Implication

Train a Joint Embedding Predictive Architecture to learn the geometric structure of equational implication over magmas. The 22M Lean-verified implication pairs from the Equational Theories Project are used to learn an embedding space where distance encodes implication relationships; the geometry of that space — what algebraic structure it recovers, what it doesn't, how it generalizes to held-out equations — is the primary research output. Classification accuracy is a validation metric, not the headline.

The current active plan is [skip_connection_plan.md](skip_connection_plan.md) — a parameter-free skip connection from the GNN's pooled representation to the embedding output, tested under the hypothesis "the projection is throwing away features the GNN already learned." Prior plans (all complete and archived) are in [archive/](archive/): `aux_heads_implementation_plan.md`, `probing_experiments_plan.md`, the original `jepa_engineering_plan.md` (Phase 1 architecture + loss sweep), and `dual_encoder_plan.md` (deferred pending skip-connection results). The earlier failure-analysis plan, `testing_plan_phase2.md`, is in `trash/` — its findings produced the diagnostic infrastructure and the `outputs/runs/stable_*` records.

**Current stable project best: `skip_h128_seed42_diag` at bal_acc 0.9840** (h=128, structural aux heads, VICReg 0.1/0.1, margin 1.5, skip connection from pooled rep to embedding with per-channel LayerNorms, embed_dim=768, two-seed reproducible). See [CLAUDE.md](CLAUDE.md) for the full project-state guide and [`CHANGELOG.md`](CHANGELOG.md) for the history of deviations from the original plan.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

This project depends on raw data from the Equational Theories Project and (eventually) computed invariants from a sibling extraction project. Symlink them under `data/raw/`:

```bash
mkdir -p data/raw
ln -s /path/to/equational_theories/data                  data/raw/etp
ln -s /path/to/equational_theories_data/assembly/output  data/raw/invariants
```

Files used:

- `data/raw/etp/equations.txt` — 4,694 equations (line N = equation ID N)
- `data/raw/etp_export/export_raw_implications_5_9_2026.csv` — authoritative ground-truth label for every off-diagonal pair (4694×4694 int8 matrix, codes ±3/±4 mapping to TRUE/FALSE; see CHANGELOG 2026-05-09)
- `data/raw/etp/2025-08-11-vampire.json.gz` — *legacy* Vampire-derived labels (left 1,062 pairs unresolved); no longer the source of truth
- sibling project `equational_theories_data/invariants/` — per-equation invariants used by the auxiliary prediction heads (spine type, LHS structure, variable count, magma signature, cohomology summaries). Built into `data/processed/equation_invariants.parquet` via `scripts/build_equation_invariants.py`.

Do not write to `data/raw/`. All processing goes to `data/processed/` (caches) or `data/splits/` (committed split indices).

### Build the processed caches

Once the `data/raw/` symlinks are in place:

```bash
python scripts/parse_equations.py             # data/processed/equation_graphs.pt   (~6 MB)
python scripts/build_pair_table.py            # data/processed/implication_pairs.parquet  (~35 MB)
python scripts/build_equation_invariants.py   # data/processed/equation_invariants.parquet  (needed for aux-head training)
```

All three are deterministic and re-runnable; their outputs are gitignored. See the script docstrings for the schema and CLI flags.

## Layout

See [archive/jepa_engineering_plan.md](archive/jepa_engineering_plan.md) §"File Structure".
