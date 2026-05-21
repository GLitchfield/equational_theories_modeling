"""Holdout split construction for the implication pair table.

This module provides the pure index-sampling logic. CLI driver lives in
``embeddings/scripts/build_splits.py``. Splits are saved under ``data/splits/`` and
committed for reproducibility.

Split levels (engineering plan §"Holdout Strategy"):
  L1 — random pair holdout (this module: ``random_pair_holdout``)
  L2 — equation holdout (deferred)
  L3 — equivalence-class holdout (deferred)
"""

from __future__ import annotations

import numpy as np


def random_pair_holdout(
    candidate_indices: np.ndarray,
    holdout_ratio: float,
    seed: int,
) -> np.ndarray:
    """Sample a random fraction of ``candidate_indices`` for holdout.

    Args:
        candidate_indices: 1-D integer array of pair-table row indices that
            are eligible for the holdout. Typically this is the indices of
            labeled (non-unknown) rows in ``implication_pairs.parquet``.
        holdout_ratio: Fraction in (0, 1) to sample for holdout.
        seed: RNG seed for reproducibility (numpy ``default_rng``).

    Returns:
        Sorted 1-D ``int32`` array of held-out row indices, drawn uniformly
        without replacement from ``candidate_indices``. The training set is
        the complement (``candidate_indices`` minus this set).
    """
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError(f'holdout_ratio must be in (0, 1); got {holdout_ratio}')
    if candidate_indices.ndim != 1:
        raise ValueError(f'candidate_indices must be 1-D; got shape {candidate_indices.shape}')

    n = len(candidate_indices)
    n_holdout = int(round(n * holdout_ratio))
    if n_holdout == 0 or n_holdout == n:
        raise ValueError(
            f'holdout_ratio={holdout_ratio} on {n} candidates produces empty/full holdout'
        )

    rng = np.random.default_rng(seed)
    chosen_positions = rng.choice(n, size=n_holdout, replace=False)
    holdout = candidate_indices[chosen_positions]
    holdout = np.sort(holdout).astype(np.int32, copy=False)
    return holdout
