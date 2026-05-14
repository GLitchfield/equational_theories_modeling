"""Pre-bake per-equation cohomology profiles into a 174-dim numpy array.

Output: ``data/processed/cohom_profiles_174d.npy`` of shape ``[N_eq, 174]``
indexed by ``eq_id - 1`` (1-based corpus IDs → 0-based array rows).

Each row is the concatenation of two 87-dim vectors per the
dual-encoder plan's "mask + values" decision:

  [0:87]    binary mask  — 1.0 if the equation satisfies the linear
                            magma at cell (a, b, p), else 0.0
  [87:174]  values       — dim H² at the cell, normalized by p²
                            (so values land in [0, 1)) where mask=1,
                            else 0.0

Cell ordering matches the source cohomology profile: 4 cells for p=2,
9 for p=3, 25 for p=5, 49 for p=7 (cumulative offsets 0, 4, 13, 38, 87).

Source parquet: ``/home/gavin/git/equational_theories_data/invariants/
11_cohomology/output/cohomology.parquet``.

Reproducible: re-run any time the upstream cohomology data changes;
the output is fully derivable from the source parquet.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COHOM_PARQUET = "/home/gavin/git/equational_theories_data/invariants/11_cohomology/output/cohomology.parquet"
_OUT_PATH = _REPO_ROOT / "data/processed/cohom_profiles_174d.npy"

PRIMES = [2, 3, 5, 7]
BLOCK_SIZES = [4, 9, 25, 49]  # = [p**2 for p in PRIMES]
OFFSETS = np.cumsum([0] + BLOCK_SIZES)  # [0, 4, 13, 38, 87]
N_CELLS = int(OFFSETS[-1])
assert N_CELLS == 87

# Each cell's prime, for value normalization.
CELL_PRIMES = np.concatenate([
    np.full(size, p, dtype=np.int32)
    for p, size in zip(PRIMES, BLOCK_SIZES)
])
assert len(CELL_PRIMES) == N_CELLS


def build(cohom_parquet: str, out_path: Path) -> np.ndarray:
    df = pd.read_parquet(cohom_parquet).set_index("eq_id")
    eq_ids = df.index.to_numpy()
    n = len(eq_ids)

    # Sanity: eq IDs should be a contiguous 1..N range so we can save by row
    # position and recover eq_id later via row_index + 1.
    expected = np.arange(1, n + 1)
    if not np.array_equal(eq_ids, expected):
        raise ValueError(
            f"expected eq_id to be contiguous 1..{n}; got first 5 = "
            f"{eq_ids[:5].tolist()}, last 5 = {eq_ids[-5:].tolist()}"
        )

    profiles = np.stack(df["cohom_profile"].values).astype(np.float64)
    assert profiles.shape == (n, N_CELLS), (
        f"unexpected source shape {profiles.shape}, want ({n}, {N_CELLS})"
    )

    mask = (~np.isnan(profiles)).astype(np.float32)              # [N, 87]
    # Where mask=1, divide by p². Where mask=0, keep value at 0.0.
    p_squared = (CELL_PRIMES.astype(np.float32) ** 2)[None, :]    # [1, 87]
    values = np.where(mask > 0, profiles / p_squared, 0.0).astype(np.float32)

    out = np.concatenate([mask, values], axis=1).astype(np.float32)
    assert out.shape == (n, 2 * N_CELLS)

    # Sanity: mask half is binary; values half lies in [0, 1] where mask=1
    # (dim H² is bounded by p² for the linear magma over F_p so the upper
    # bound is closed at 1.0 after dividing by p²).
    assert np.all((out[:, :N_CELLS] == 0) | (out[:, :N_CELLS] == 1))
    vmax_active = out[:, N_CELLS:][out[:, :N_CELLS] > 0].max() if (out[:, :N_CELLS] > 0).any() else 0.0
    if vmax_active > 1.0 + 1e-6:
        raise ValueError(
            f"values half max ({vmax_active:.4f}) exceeded 1.0 — a cell "
            f"has dim H² > p², which violates the linear-magma cohom bound; "
            f"check the source data."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    return out


def report(out: np.ndarray) -> None:
    n = out.shape[0]
    mask = out[:, :N_CELLS]
    values = out[:, N_CELLS:]
    print(f"shape: {out.shape}, dtype: {out.dtype}")
    print(f"mask half: positive rate = {float(mask.mean()):.4f} "
          f"(satisfying-cell fraction across all (eq, cell) pairs)")
    # Per-prime breakdown
    for p, lo, hi in zip(PRIMES, OFFSETS[:-1], OFFSETS[1:]):
        prime_mask = mask[:, lo:hi]
        prime_vals = values[:, lo:hi]
        sat_frac = float(prime_mask.mean())
        active_vals = prime_vals[prime_mask > 0]
        v_range = (float(active_vals.min()), float(active_vals.max())) if active_vals.size else (0.0, 0.0)
        print(f"  p={p}: satisfying-cell fraction={sat_frac:.4f}, "
              f"values-where-active range=[{v_range[0]:.4f}, {v_range[1]:.4f}]")

    # All-null equations
    eq_satisfies_any = mask.any(axis=1)
    print(f"\nall-null equations (no satisfying cell anywhere): "
          f"{int((~eq_satisfies_any).sum())} of {n} = "
          f"{float((~eq_satisfies_any).mean()):.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=_COHOM_PARQUET,
                        help="source cohomology.parquet path")
    parser.add_argument("--output", default=str(_OUT_PATH),
                        help="output .npy path")
    args = parser.parse_args(argv)

    out = build(args.source, Path(args.output))
    print(f"\nwrote {args.output}")
    report(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
