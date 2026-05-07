# JEPA for Equational Implication

Train a Joint Embedding Predictive Architecture to learn the geometric structure of equational implication over magmas. The 22M Lean-verified implication pairs from the Equational Theories Project are used to learn an embedding space where distance encodes implication relationships; the geometry of that space — what algebraic structure it recovers, what it doesn't, how it generalizes to held-out equations — is the primary research output. Classification accuracy is a validation metric, not the headline.

The full design is in [jepa_engineering_plan.md](jepa_engineering_plan.md). Read that first.

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
- `data/raw/etp/2025-08-11-vampire.json.gz` — 22M pair labels, keyed `"i_j" -> {value: 0|1|2}` (0=FALSE, 1=TRUE, 2=unknown)
- `data/raw/invariants/equation_invariants.parquet` — per-equation invariants (used by Baseline 3 and Phase 4 probing; not needed by the JEPA model itself)

Do not write to `data/raw/`. All processing goes to `data/processed/` (caches) or `data/splits/` (committed split indices).

### Build the processed caches

Once the `data/raw/` symlinks are in place:

```bash
python scripts/parse_equations.py     # data/processed/equation_graphs.pt   (~6 MB)
python scripts/build_pair_table.py    # data/processed/implication_pairs.parquet  (~35 MB)
```

Both are deterministic and re-runnable; their outputs are gitignored. See the script docstrings for the schema and CLI flags.

## Layout

See [jepa_engineering_plan.md](jepa_engineering_plan.md) §"File Structure".
