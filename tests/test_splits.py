"""Tests for src/data/splits.py."""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.splits import random_pair_holdout  # noqa: E402


def test_holdout_size():
    candidates = np.arange(10_000, dtype=np.int32)
    holdout = random_pair_holdout(candidates, holdout_ratio=0.1, seed=42)
    assert len(holdout) == 1_000


def test_holdout_indices_are_subset_of_candidates():
    candidates = np.arange(100, 1100, dtype=np.int32)  # offset to test relative selection
    holdout = random_pair_holdout(candidates, holdout_ratio=0.2, seed=7)
    assert holdout.min() >= 100
    assert holdout.max() < 1100
    assert set(holdout.tolist()).issubset(set(candidates.tolist()))


def test_holdout_no_duplicates():
    candidates = np.arange(50_000, dtype=np.int32)
    holdout = random_pair_holdout(candidates, holdout_ratio=0.1, seed=42)
    assert len(set(holdout.tolist())) == len(holdout)


def test_holdout_sorted():
    candidates = np.arange(50_000, dtype=np.int32)
    holdout = random_pair_holdout(candidates, holdout_ratio=0.1, seed=42)
    assert (holdout[1:] >= holdout[:-1]).all()


def test_holdout_reproducible():
    candidates = np.arange(100_000, dtype=np.int32)
    h1 = random_pair_holdout(candidates, holdout_ratio=0.1, seed=42)
    h2 = random_pair_holdout(candidates, holdout_ratio=0.1, seed=42)
    assert np.array_equal(h1, h2)


def test_holdout_seed_changes_result():
    candidates = np.arange(100_000, dtype=np.int32)
    h1 = random_pair_holdout(candidates, holdout_ratio=0.1, seed=42)
    h2 = random_pair_holdout(candidates, holdout_ratio=0.1, seed=43)
    assert not np.array_equal(h1, h2)


def test_holdout_dtype_is_int32():
    candidates = np.arange(10_000, dtype=np.int64)  # different dtype on input
    holdout = random_pair_holdout(candidates, holdout_ratio=0.1, seed=42)
    assert holdout.dtype == np.int32


def test_holdout_invalid_ratio():
    candidates = np.arange(100, dtype=np.int32)
    for bad in [0.0, 1.0, -0.1, 1.5]:
        try:
            random_pair_holdout(candidates, holdout_ratio=bad, seed=42)
        except ValueError:
            continue
        raise AssertionError(f'expected ValueError for ratio={bad}')


def test_holdout_works_with_non_contiguous_candidates():
    """Mimic the real use case: candidates are the indices of labeled rows
    in a larger pair table (some indices skipped because of unknowns)."""
    rng = np.random.default_rng(0)
    full = np.arange(20_000, dtype=np.int32)
    # Skip ~5% to simulate unknowns
    keep = rng.random(20_000) > 0.05
    candidates = full[keep]
    holdout = random_pair_holdout(candidates, holdout_ratio=0.1, seed=42)
    # Holdout indices must all be from the candidate set, not the skipped rows.
    assert set(holdout.tolist()).issubset(set(candidates.tolist()))


# --- driver -----------------------------------------------------------------

if __name__ == '__main__':
    import inspect
    import traceback

    test_funcs = [
        v for k, v in sorted(globals().items())
        if k.startswith('test_') and inspect.isfunction(v)
    ]
    fails = []
    for fn in test_funcs:
        try:
            fn()
            print(f'PASS  {fn.__name__}')
        except Exception as e:
            fails.append(fn.__name__)
            print(f'FAIL  {fn.__name__}: {type(e).__name__}: {e}')
            traceback.print_exc()
    print()
    print(f'{len(test_funcs) - len(fails)}/{len(test_funcs)} passed')
    sys.exit(1 if fails else 0)
