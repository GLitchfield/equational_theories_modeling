"""Build a unified per-equation invariants table for auxiliary head training.

Combines the four source parquets from the sibling invariants project
into a single ``data/processed/equation_invariants.parquet`` keyed by
``eq_id``, with one column per auxiliary head:

  - ``spine_type``      6-class categorical (int)
  - ``lhs_structure``   binary (int 0/1)
  - ``num_variables``   6-class categorical (int 0-5 corresponding to 1-6 vars)
  - ``magma_bit_0..7``  7 binary columns (one per usable magma signature bit)

Sparse magma bits (5, 8, 9, 10, 11) are dropped — their <5% positive
rates collapse the auxiliary head to a trivial "predict 0" solution
that provides no anchoring pressure.

Source paths follow the layout of the sibling
``equational_theories_data`` project:
``invariants/<NN>_<name>/output/<name>.parquet``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DATA_PROJECT = "/home/gavin/git/equational_theories_data/invariants"

# String -> int mapping for spine_type. 6 classes; deterministic order
# so subsequent runs reproduce the same labels.
SPINE_LABELS = [
    "family_c",
    "mixed",
    "no_lhs_var_in_rhs",
    "left",
    "right",
    "trivial",
]

LHS_LABELS = ["bare_variable", "product"]

# magma_signature is a 12-vector; we keep these 7 indices (>=7% positive
# rate). Indices 5, 8, 9, 10, 11 dropped (1-3% positive — too sparse).
MAGMA_BITS = [0, 1, 2, 3, 4, 6, 7]


def build(data_project: str, out_path: str) -> pd.DataFrame:
    base = Path(data_project)
    spine = pd.read_parquet(base / "03_spine/output/spine.parquet")
    lhs = pd.read_parquet(base / "01_lhs_structure/output/lhs_structure.parquet")
    vars_ = pd.read_parquet(base / "04_variables/output/variables.parquet")
    magma = pd.read_parquet(base / "08_magma_signature/output/magma_signature.parquet")

    n = len(spine)
    assert len(lhs) == n and len(vars_) == n and len(magma) == n, (
        f"row-count mismatch: spine={len(spine)} lhs={len(lhs)} "
        f"vars={len(vars_)} magma={len(magma)}"
    )
    # All four are indexed by eq_id; verify alignment.
    for df, name in [(lhs, "lhs"), (vars_, "vars"), (magma, "magma")]:
        if not spine.index.equals(df.index):
            raise ValueError(f"index mismatch between spine and {name}")

    out = pd.DataFrame(index=spine.index)
    out["spine_type"] = spine["spine_type"].map(
        {label: i for i, label in enumerate(SPINE_LABELS)}
    ).astype("int8")
    out["lhs_structure"] = lhs["lhs_structure"].map(
        {label: i for i, label in enumerate(LHS_LABELS)}
    ).astype("int8")
    # num_variables column has values 1-6; map to 0-5 for cross-entropy.
    out["num_variables"] = (vars_["num_variables"] - 1).astype("int8")

    sig = np.stack(magma["magma_signature"].values).astype("int8")  # (N, 12)
    for bit in MAGMA_BITS:
        out[f"magma_bit_{bit}"] = sig[:, bit]

    # Sanity: no NaN, no out-of-range labels.
    assert out.notna().all().all(), "NaN in invariants output"
    assert out["spine_type"].between(0, len(SPINE_LABELS) - 1).all()
    assert out["lhs_structure"].between(0, len(LHS_LABELS) - 1).all()
    assert out["num_variables"].between(0, 5).all()
    for bit in MAGMA_BITS:
        col = out[f"magma_bit_{bit}"]
        assert col.isin([0, 1]).all(), f"non-binary in magma_bit_{bit}"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build per-equation invariants table")
    parser.add_argument(
        "--data-project",
        default=_DATA_PROJECT,
        help="Path to the equational_theories_data invariants/ directory",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_REPO_ROOT, "data/processed/equation_invariants.parquet"),
        help="Output parquet path",
    )
    args = parser.parse_args(argv)
    out = build(args.data_project, args.output)
    print(f"wrote {args.output}: shape={out.shape}, columns={list(out.columns)}")
    print("\nclass distributions:")
    for c in out.columns:
        vc = out[c].value_counts().sort_index()
        print(f"  {c}: {dict(vc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
