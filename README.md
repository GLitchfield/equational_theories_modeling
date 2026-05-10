# JEPA for Equational Implication

Train a Joint Embedding Predictive Architecture to learn the geometric structure of equational implication over magmas. The 22M Lean-verified implication pairs from the Equational Theories Project are used to learn an embedding space where distance encodes implication relationships; the geometry of that space — what algebraic structure it recovers, what it doesn't, how it generalizes to held-out equations — is the primary research output. Classification accuracy is a validation metric, not the headline.

The current active plan is [testing_plan_phase2.md](testing_plan_phase2.md) — failure analysis on the post-VICReg model. The original engineering plan that set up Phase 1 (architecture + loss sweep) lives at [archive/jepa_engineering_plan.md](archive/jepa_engineering_plan.md); Phase 1 is complete and the current best model (bal_acc 0.9112) reflects that work. Read the active plan first; the archive has the Phase 2–4 ideas that are on hold pending failure-analysis findings.

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

See [archive/jepa_engineering_plan.md](archive/jepa_engineering_plan.md) §"File Structure".
