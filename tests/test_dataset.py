"""Tests for src/data/dataset.py."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.dataset import (  # noqa: E402
    AUX_HEAD_NAMES,
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


# --- aux labels path --------------------------------------------------------

def _aux_for_synthetic(n_eq: int = 4) -> np.ndarray:
    """Synthetic [n_eq, len(AUX_HEAD_NAMES)] aux-label table.

    Indexed by ``eq_id - 1``. Column k of row e is ``(e * 7 + k) % 6``
    so values land in 0-5 (compatible with the largest head's 6 classes;
    binary heads will receive 0 or 1 values too because the modulo
    cycles through them)."""
    return np.stack([
        [(e * 7 + k) % 6 for k in range(len(AUX_HEAD_NAMES))]
        for e in range(n_eq)
    ]).astype(np.int64)


def test_dataset_with_aux_labels_returns_5_tuple():
    graphs, e1, e2, v, idx = _synthetic()
    aux = _aux_for_synthetic()
    ds = ImplicationPairsDataset(e1, e2, v, graphs, idx, aux_labels=aux)
    item = ds[0]
    assert len(item) == 5, f"expected 5-tuple with aux, got {len(item)}-tuple"
    g1, g2, label, eq1_aux, eq2_aux = item
    row = int(idx[0])
    eq1_id = int(e1[row]); eq2_id = int(e2[row])
    np.testing.assert_array_equal(eq1_aux, aux[eq1_id - 1])
    np.testing.assert_array_equal(eq2_aux, aux[eq2_id - 1])


def test_dataset_without_aux_labels_returns_3_tuple_unchanged():
    """Backward compatibility: aux_labels=None preserves the existing 3-tuple."""
    graphs, e1, e2, v, idx = _synthetic()
    ds = ImplicationPairsDataset(e1, e2, v, graphs, idx)
    item = ds[0]
    assert len(item) == 3


def test_dataset_aux_labels_wrong_shape_raises():
    graphs, e1, e2, v, idx = _synthetic()
    with pytest.raises(ValueError, match="aux_labels must be"):
        ImplicationPairsDataset(
            e1, e2, v, graphs, idx,
            aux_labels=np.zeros((4, 3), dtype=np.int64),  # wrong width
        )


def test_collate_with_aux_yields_5_tuple_with_per_head_dicts():
    graphs, e1, e2, v, idx = _synthetic()
    aux = _aux_for_synthetic()
    ds = ImplicationPairsDataset(e1, e2, v, graphs, idx, aux_labels=aux)
    items = [ds[i] for i in range(3)]
    out = collate_pairs(items)
    assert len(out) == 5
    eq1_batch, eq2_batch, labels, eq1_aux, eq2_aux = out
    assert isinstance(eq1_batch, Batch)
    assert eq1_batch.num_graphs == 3 and eq2_batch.num_graphs == 3
    assert labels.shape == (3,)
    assert set(eq1_aux.keys()) == set(AUX_HEAD_NAMES)
    assert set(eq2_aux.keys()) == set(AUX_HEAD_NAMES)
    for name in AUX_HEAD_NAMES:
        assert eq1_aux[name].shape == (3,)
        assert eq1_aux[name].dtype == torch.int64
        assert eq2_aux[name].shape == (3,)
        assert eq2_aux[name].dtype == torch.int64


def test_collate_aux_values_match_per_equation_lookup():
    """Each batch element's aux values come from the right eq_id row."""
    graphs, e1, e2, v, idx = _synthetic()
    aux = _aux_for_synthetic()
    ds = ImplicationPairsDataset(e1, e2, v, graphs, idx, aux_labels=aux)
    items = [ds[i] for i in range(3)]
    _, _, _, eq1_aux, eq2_aux = collate_pairs(items)
    for k, item in enumerate(items):
        _, _, _, item_eq1_aux, item_eq2_aux = item
        for j, name in enumerate(AUX_HEAD_NAMES):
            assert int(eq1_aux[name][k]) == int(item_eq1_aux[j])
            assert int(eq2_aux[name][k]) == int(item_eq2_aux[j])


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
    # Sizes after the 2026-05-09 cutover to the ETP CSV as ground truth:
    # all 22,028,942 off-diagonal pairs are labeled (no Vampire timeouts);
    # 10% holdout at seed=42.
    assert len(val_ds) == 2_202_894
    assert len(train_ds) == 22_028_942 - 2_202_894

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


def test_make_loaders_val_is_shuffled_unbiased():
    """The val loader's first 200 batches must be representative of the
    full holdout's TRUE/FALSE distribution. Regression test for the bias
    where the unshuffled holdout (sorted by pair-table row, correlated
    with equation complexity) made the first-N-batches slice non-random.
    """
    pair_path = os.path.join(_REPO_ROOT, 'data/processed/implication_pairs.parquet')
    cache_path = os.path.join(_REPO_ROOT, 'data/processed/equation_graphs.pt')
    split_path = os.path.join(_REPO_ROOT, 'data/splits/level1_holdout_seed42.parquet')
    if not all(os.path.exists(p) for p in [pair_path, cache_path, split_path]):
        return

    _, val_loader, _, val_ds = make_loaders(
        pair_path, cache_path, split_path,
        batch_size=2048, num_workers=0, pin_memory=False,
    )

    # Full val TRUE/FALSE rate.
    full_labels = val_ds.values[val_ds.indices]
    full_true_rate = float((full_labels == 1).mean())

    # Capped slice: first 200 batches' labels.
    cap_n = 200 * 2048
    cap_labels = val_ds.values[val_ds.indices[:cap_n]]
    cap_true_rate = float((cap_labels == 1).mean())

    # Pre-fix the gap was ~5.6 pp (0.371 full vs 0.427 capped). After
    # shuffling, both rates should match within ~0.5 pp at this cap size.
    assert abs(cap_true_rate - full_true_rate) < 0.005, (
        f'capped slice TRUE rate {cap_true_rate:.4f} differs from full '
        f'{full_true_rate:.4f} — val shuffle is broken'
    )


def test_make_loaders_no_holdout_uses_all_pairs_and_subsamples_val():
    """``holdout_path=None`` mode: train_ds = every labeled row;
    val_ds = a deterministic random subsample of train_ds for monitoring.
    This is the failure-analysis training path."""
    pair_path = os.path.join(_REPO_ROOT, 'data/processed/implication_pairs.parquet')
    cache_path = os.path.join(_REPO_ROOT, 'data/processed/equation_graphs.pt')
    if not all(os.path.exists(p) for p in [pair_path, cache_path]):
        return
    _, _, train_ds, val_ds = make_loaders(
        pair_path, cache_path, holdout_path=None,
        batch_size=2048, num_workers=0, pin_memory=False,
        no_holdout_val_size=10_000,
    )
    # All labeled pairs in training (the post-2026-05-09 data has zero
    # value=2 unknowns, so this equals the full off-diagonal corpus).
    assert len(train_ds) == 22_028_942
    # Val is a fixed random subsample of train.
    assert len(val_ds) == 10_000
    val_set = set(int(i) for i in val_ds.indices.tolist())
    train_set = set(int(i) for i in train_ds.indices.tolist())
    assert val_set.issubset(train_set), 'no-holdout val must be drawn from train'


def test_make_loaders_no_holdout_val_subsample_is_deterministic():
    pair_path = os.path.join(_REPO_ROOT, 'data/processed/implication_pairs.parquet')
    cache_path = os.path.join(_REPO_ROOT, 'data/processed/equation_graphs.pt')
    if not all(os.path.exists(p) for p in [pair_path, cache_path]):
        return
    _, _, _, val_ds_a = make_loaders(
        pair_path, cache_path, holdout_path=None,
        batch_size=2048, num_workers=0, pin_memory=False,
        val_shuffle_seed=42, no_holdout_val_size=5_000,
    )
    _, _, _, val_ds_b = make_loaders(
        pair_path, cache_path, holdout_path=None,
        batch_size=2048, num_workers=0, pin_memory=False,
        val_shuffle_seed=42, no_holdout_val_size=5_000,
    )
    assert np.array_equal(val_ds_a.indices, val_ds_b.indices)


def test_make_loaders_val_shuffle_is_deterministic():
    """Same seed → same val ordering → epoch-to-epoch and run-to-run
    capped scores remain comparable."""
    pair_path = os.path.join(_REPO_ROOT, 'data/processed/implication_pairs.parquet')
    cache_path = os.path.join(_REPO_ROOT, 'data/processed/equation_graphs.pt')
    split_path = os.path.join(_REPO_ROOT, 'data/splits/level1_holdout_seed42.parquet')
    if not all(os.path.exists(p) for p in [pair_path, cache_path, split_path]):
        return

    _, _, _, val_ds_a = make_loaders(
        pair_path, cache_path, split_path,
        batch_size=2048, num_workers=0, pin_memory=False,
        val_shuffle_seed=42,
    )
    _, _, _, val_ds_b = make_loaders(
        pair_path, cache_path, split_path,
        batch_size=2048, num_workers=0, pin_memory=False,
        val_shuffle_seed=42,
    )
    assert np.array_equal(val_ds_a.indices, val_ds_b.indices)


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
