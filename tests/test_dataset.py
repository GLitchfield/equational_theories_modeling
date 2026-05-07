"""Tests for src/data/dataset.py."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch_geometric.data import Batch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.dataset import (  # noqa: E402
    ImplicationPairsDataset,
    collate_pairs,
    derive_train_indices,
    make_loaders,
)
from src.data.parser import NODE_FEATURE_DIM, equation_to_pyg  # noqa: E402


def _synthetic():
    """Tiny synthetic dataset: 4 equations, 8 pairs (one is unknown)."""
    graphs = {
        1: equation_to_pyg('x = x'),
        2: equation_to_pyg('x = y'),
        3: equation_to_pyg('x = x ◇ y'),
        4: equation_to_pyg('x ◇ x = x'),
    }
    eq1_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
    eq2_ids = np.array([2, 3, 1, 4, 4, 1, 2, 3], dtype=np.int32)
    values = np.array([1, 0, 1, 0, 2, 1, 0, 0], dtype=np.int8)  # row 4 is unknown
    indices = np.array([0, 1, 2, 3, 5, 6, 7], dtype=np.int64)  # row 4 excluded
    return graphs, eq1_ids, eq2_ids, values, indices


# --- dataset / collate ------------------------------------------------------

def test_dataset_len():
    graphs, e1, e2, v, idx = _synthetic()
    ds = ImplicationPairsDataset(e1, e2, v, graphs, idx)
    assert len(ds) == 7


def test_dataset_getitem_returns_correct_graphs_and_label():
    graphs, e1, e2, v, idx = _synthetic()
    ds = ImplicationPairsDataset(e1, e2, v, graphs, idx)
    g1, g2, label = ds[0]
    row = int(idx[0])
    assert g1 is graphs[int(e1[row])]
    assert g2 is graphs[int(e2[row])]
    assert label == int(v[row])


def test_dataset_no_unknowns_in_exposed_indices():
    """Every sample yielded should have label in {0, 1}; unknown rows should
    have been excluded by the caller (typically via derive_train_indices).
    """
    graphs, e1, e2, v, idx = _synthetic()
    ds = ImplicationPairsDataset(e1, e2, v, graphs, idx)
    for i in range(len(ds)):
        _, _, label = ds[i]
        assert label in (0, 1)


def test_collate_yields_two_batches_and_float_labels():
    graphs, e1, e2, v, idx = _synthetic()
    ds = ImplicationPairsDataset(e1, e2, v, graphs, idx)
    items = [ds[i] for i in range(3)]
    eq1_batch, eq2_batch, labels = collate_pairs(items)
    assert isinstance(eq1_batch, Batch)
    assert isinstance(eq2_batch, Batch)
    assert eq1_batch.num_graphs == 3
    assert eq2_batch.num_graphs == 3
    assert eq1_batch.x.shape[1] == NODE_FEATURE_DIM
    assert eq2_batch.x.shape[1] == NODE_FEATURE_DIM
    assert labels.shape == (3,)
    assert labels.dtype == torch.float


# --- index derivation -------------------------------------------------------

def test_derive_train_indices_excludes_unknowns_and_holdout():
    values = np.array([0, 1, 2, 0, 1, 1, 2, 0], dtype=np.int8)
    holdout = np.array([0, 4], dtype=np.int64)
    train = derive_train_indices(values, holdout)
    # All labeled (value != 2) and not in holdout: rows 1, 3, 5, 7
    assert list(train) == [1, 3, 5, 7]


def test_derive_train_indices_empty_holdout():
    values = np.array([0, 1, 2, 0, 1], dtype=np.int8)
    holdout = np.array([], dtype=np.int64)
    train = derive_train_indices(values, holdout)
    # Just exclude unknowns
    assert list(train) == [0, 1, 3, 4]


def test_derive_train_indices_dtype():
    values = np.array([0, 1], dtype=np.int8)
    holdout = np.array([], dtype=np.int64)
    train = derive_train_indices(values, holdout)
    assert train.dtype == np.int64


# --- end-to-end smoke against real artifacts (optional) ---------------------

def test_make_loaders_smoke():
    """Smoke test with the real cache + pair table + L1 split. Skipped if
    any artifact is missing."""
    pair_path = os.path.join(_REPO_ROOT, 'data/processed/implication_pairs.parquet')
    cache_path = os.path.join(_REPO_ROOT, 'data/processed/equation_graphs.pt')
    split_path = os.path.join(_REPO_ROOT, 'data/splits/level1_holdout_seed42.parquet')
    if not all(os.path.exists(p) for p in [pair_path, cache_path, split_path]):
        return

    train_loader, val_loader, train_ds, val_ds = make_loaders(
        pair_path, cache_path, split_path,
        batch_size=64, num_workers=0, pin_memory=False,
    )
    # Sizes match expectations (10% held out of 22,027,880 labeled).
    assert len(val_ds) == 2_202_788
    assert len(train_ds) == 22_027_880 - 2_202_788

    eq1_batch, eq2_batch, labels = next(iter(train_loader))
    assert eq1_batch.num_graphs == 64
    assert eq2_batch.num_graphs == 64
    assert labels.shape == (64,)
    # No unknowns in either side.
    assert ((labels == 0) | (labels == 1)).all()
    # Feature dimensions match the parser.
    assert eq1_batch.x.shape[1] == NODE_FEATURE_DIM
    assert eq2_batch.x.shape[1] == NODE_FEATURE_DIM

    eq1_batch, eq2_batch, labels = next(iter(val_loader))
    assert eq1_batch.num_graphs == 64
    assert ((labels == 0) | (labels == 1)).all()


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
