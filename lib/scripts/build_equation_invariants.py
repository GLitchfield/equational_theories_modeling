"""Build a unified per-equation invariants table for auxiliary head training.

Combines the source parquets from the sibling invariants project into a
single ``data/processed/equation_invariants.parquet`` keyed by ``eq_id``,
with one column per auxiliary head:

  - ``spine_type``           6-class categorical (int)
  - ``lhs_structure``        binary (int 0/1)
  - ``num_variables``        6-class categorical (int 0-5 corresponding to 1-6 vars)
  - ``magma_bit_0..7``       7 binary columns (one per usable magma signature bit)
  - ``cohom_count_{p}``      4 per-prime binned counts (p in {2,3,5,7})
  - ``cohom_mean_h2_{p}``    4 per-prime binned mean dim H² (null bin for count==0)

Sparse magma bits (5, 8, 9, 10, 11) are dropped — their <5% positive
rates collapse the auxiliary head to a trivial "predict 0" solution
that provides no anchoring pressure.

Cohomology summaries are derived from the length-87 cohom_profile in
the cohomology source: for each prime p, count_p = #(a,b) pairs that
satisfy the equation over F_p, mean_h2_p = mean dim H² across those
satisfying pairs (or null when count_p == 0). Per-prime bins are
chosen to keep the smallest class >= 30 examples while preserving
the natural integer-valued shape of dim H² means. See ``COHOM_*_BINS``
below for exact edges and class names.

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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
# Sibling equational_theories_data checkout by default; override with $INVARIANTS_ROOT.
_DATA_PROJECT = os.environ.get(
    "INVARIANTS_ROOT",
    os.path.normpath(os.path.join(_REPO_ROOT, os.pardir, "equational_theories_data", "invariants")),
)

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

# Cohomology profile layout: 87 cells per equation, one per (a, b, p)
# triple, in prime-order p in {2, 3, 5, 7} with cell counts 4/9/25/49.
COHOM_PRIMES = [2, 3, 5, 7]
COHOM_BLOCK_SIZES = [4, 9, 25, 49]
COHOM_OFFSETS = np.cumsum([0] + COHOM_BLOCK_SIZES)  # [0, 4, 13, 38, 87]

# Per-prime binning specs. Each entry: list of (lo_inclusive, hi_exclusive, name).
# Bin index = position in the list. Hi=inf for the open-top bin. The lookup
# below maps each value into exactly one bin.
COHOM_COUNT_BINS: dict[int, list[tuple[int, float, str]]] = {
    2: [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 4, "3"), (4, 5, "4")],
    3: [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 4, "3"), (4, np.inf, "4+")],
    5: [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 4, "3"), (4, 6, "4-5"), (6, np.inf, "6+")],
    7: [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 4, "3"), (4, 7, "4-6"), (7, np.inf, "7+")],
}

# mean_h2 bins. Bin 0 is reserved for "null" (count_p == 0, mean undefined).
# Remaining bins are over the value range. The "0" bin is matched as
# strict equality, since dim H² values are integers and exact-zero means
# correspond to "all satisfying pairs are 2-cohomologically trivial."
COHOM_MEAN_BINS: dict[int, list[tuple[float | None, float | None, str]]] = {
    2: [(None, None, "null"), (0.0, 1e-9, "0"), (1e-9, 1.0, "(0,1)"),
        (1.0, 2.0, "[1,2)"), (2.0, np.inf, "[2,4]")],
    3: [(None, None, "null"), (0.0, 1e-9, "0"), (1e-9, 1.0, "(0,1)"),
        (1.0, 2.0, "[1,2)"), (2.0, 4.0, "[2,4)"), (4.0, np.inf, "[4,9]")],
    5: [(None, None, "null"), (0.0, 1e-9, "0"), (1e-9, 1.0, "(0,1)"),
        (1.0, 3.0, "[1,3)"), (3.0, 8.0, "[3,8)"), (8.0, np.inf, "[8,25]")],
    7: [(None, None, "null"), (0.0, 1e-9, "0"), (1e-9, 2.0, "(0,2)"),
        (2.0, 5.0, "[2,5)"), (5.0, 15.0, "[5,15)"), (15.0, np.inf, "[15,50]")],
}


def _bin_count(values: np.ndarray, spec: list[tuple[int, float, str]]) -> np.ndarray:
    """Bucket integer counts into bin indices per ``spec``."""
    out = np.full(len(values), -1, dtype=np.int8)
    for i, (lo, hi, _) in enumerate(spec):
        out[(values >= lo) & (values < hi)] = i
    if (out == -1).any():
        bad = values[out == -1]
        raise ValueError(f"unbinned count values: {bad[:10]} (spec covers all of [0, inf))")
    return out


def _bin_mean(values: np.ndarray, spec: list[tuple[float | None, float | None, str]]) -> np.ndarray:
    """Bucket float means into bin indices per ``spec`` (NaN -> null bin)."""
    out = np.full(len(values), -1, dtype=np.int8)
    null_mask = np.isnan(values)
    out[null_mask] = 0  # bin 0 is "null" by convention
    valid = ~null_mask
    for i, (lo, hi, _) in enumerate(spec):
        if lo is None:
            continue  # null bin already assigned
        out[valid & (values >= lo) & (values < hi)] = i
    if (out == -1).any():
        bad = values[out == -1]
        raise ValueError(f"unbinned mean values: {bad[:10]} (spec must cover [0, inf))")
    return out


def cohom_summaries(profiles: np.ndarray) -> dict[str, np.ndarray]:
    """Compute binned per-prime count + mean_h2 columns from raw profiles.

    Args:
        profiles: ``[N_eq, 87]`` float array; NaN where (a,b,p) does not
            satisfy the equation, integer dim H² where it does.
    Returns:
        Dict of 8 int8 numpy arrays, keys ``cohom_count_{p}`` and
        ``cohom_mean_h2_{p}`` for ``p in {2, 3, 5, 7}``.
    """
    out: dict[str, np.ndarray] = {}
    for p, lo, hi in zip(COHOM_PRIMES, COHOM_OFFSETS[:-1], COHOM_OFFSETS[1:]):
        block = profiles[:, lo:hi]
        cnt = (~np.isnan(block)).sum(axis=1).astype(np.int32)
        with np.errstate(invalid="ignore"):
            mh = np.nanmean(block, axis=1)  # NaN where cnt==0
        out[f"cohom_count_{p}"] = _bin_count(cnt, COHOM_COUNT_BINS[p])
        out[f"cohom_mean_h2_{p}"] = _bin_mean(mh, COHOM_MEAN_BINS[p])
    return out


def build(data_project: str, out_path: str) -> pd.DataFrame:
    base = Path(data_project)
    spine = pd.read_parquet(base / "03_spine/output/spine.parquet")
    lhs = pd.read_parquet(base / "01_lhs_structure/output/lhs_structure.parquet")
    vars_ = pd.read_parquet(base / "04_variables/output/variables.parquet")
    magma = pd.read_parquet(base / "08_magma_signature/output/magma_signature.parquet")
    cohom = pd.read_parquet(base / "11_cohomology/output/cohomology.parquet").set_index("eq_id")

    n = len(spine)
    assert len(lhs) == n and len(vars_) == n and len(magma) == n and len(cohom) == n, (
        f"row-count mismatch: spine={len(spine)} lhs={len(lhs)} "
        f"vars={len(vars_)} magma={len(magma)} cohom={len(cohom)}"
    )
    # All five are indexed by eq_id; verify alignment.
    for df, name in [(lhs, "lhs"), (vars_, "vars"), (magma, "magma"), (cohom, "cohom")]:
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

    profiles = np.stack(cohom["cohom_profile"].values).astype(np.float64)  # (N, 87)
    assert profiles.shape == (n, COHOM_OFFSETS[-1]), (
        f"unexpected cohom profile shape {profiles.shape}, want ({n}, {COHOM_OFFSETS[-1]})"
    )
    for col_name, arr in cohom_summaries(profiles).items():
        out[col_name] = arr

    # Sanity: no NaN, no out-of-range labels.
    assert out.notna().all().all(), "NaN in invariants output"
    assert out["spine_type"].between(0, len(SPINE_LABELS) - 1).all()
    assert out["lhs_structure"].between(0, len(LHS_LABELS) - 1).all()
    assert out["num_variables"].between(0, 5).all()
    for bit in MAGMA_BITS:
        col = out[f"magma_bit_{bit}"]
        assert col.isin([0, 1]).all(), f"non-binary in magma_bit_{bit}"
    for p in COHOM_PRIMES:
        max_count = len(COHOM_COUNT_BINS[p]) - 1
        max_mean = len(COHOM_MEAN_BINS[p]) - 1
        assert out[f"cohom_count_{p}"].between(0, max_count).all(), f"out-of-range cohom_count_{p}"
        assert out[f"cohom_mean_h2_{p}"].between(0, max_mean).all(), f"out-of-range cohom_mean_h2_{p}"

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
