"""Dataset and DataLoader for paired-graph implication training.

Each sample is ``(eq1_graph, eq2_graph, label)`` where the graphs come from
the parsed-tree cache and the label is in {0, 1} (FALSE / TRUE). Unknown
pairs (``value == 2``) are excluded upstream by the holdout-construction
logic and never appear here.

Custom ``collate_pairs`` batches each side independently with PyG's
``Batch.from_data_list``, yielding ``(eq1_batch: Batch, eq2_batch: Batch,
labels: FloatTensor)`` per minibatch.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from src.data.parser import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


class ImplicationPairsDataset(Dataset):
    """Yields ``(eq1_graph, eq2_graph, label)`` for the implication-pair task.

    Args:
        eq1_ids, eq2_ids, values: 1-D numpy arrays indexed by pair-table row.
            ``eq1_ids[r]`` and ``eq2_ids[r]`` are 1-indexed equation IDs;
            ``values[r]`` is in {0, 1, 2} (FALSE / TRUE / unknown).
        graphs: dict mapping equation ID → PyG Data with the parsed tree.
        indices: 1-D int array of pair-table row indices to expose. Item
            ``self[i]`` corresponds to row ``indices[i]``.
    """

    def __init__(
        self,
        eq1_ids: np.ndarray,
        eq2_ids: np.ndarray,
        values: np.ndarray,
        graphs: dict[int, Data],
        indices: np.ndarray,
    ) -> None:
        self.eq1_ids = eq1_ids
        self.eq2_ids = eq2_ids
        self.values = values
        self.graphs = graphs
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> tuple[Data, Data, int]:
        row = int(self.indices[i])
        eq1_id = int(self.eq1_ids[row])
        eq2_id = int(self.eq2_ids[row])
        return self.graphs[eq1_id], self.graphs[eq2_id], int(self.values[row])


def collate_pairs(
    batch: list[tuple[Data, Data, int]],
) -> tuple[Batch, Batch, torch.Tensor]:
    """Collate a list of ``(eq1_graph, eq2_graph, label)`` into batches.

    Each side is batched independently via ``Batch.from_data_list``.
    Labels are returned as ``float32`` because downstream losses
    (contrastive, BCE) expect float labels.
    """
    eq1_list = [item[0] for item in batch]
    eq2_list = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float)
    return Batch.from_data_list(eq1_list), Batch.from_data_list(eq2_list), labels


# --- loader helpers ---------------------------------------------------------

def load_pair_arrays(
    parquet_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the pair table parquet into three columnar numpy arrays."""
    import pyarrow.parquet as pq  # local import keeps this module light

    table = pq.read_table(parquet_path)
    return (
        table['eq1_id'].to_numpy(),
        table['eq2_id'].to_numpy(),
        table['value'].to_numpy(),
    )


def load_graph_cache(cache_path: str) -> dict[int, Data]:
    """Load the equation-graph cache, validating encoding constants.

    Raises ``RuntimeError`` if the cache's stored ``node_feature_dim`` /
    ``edge_feature_dim`` do not match the parser's current constants —
    a stale cache must be rebuilt rather than silently used.
    """
    cache = torch.load(cache_path, weights_only=False)
    meta = cache.get('meta', {}) if isinstance(cache, dict) else {}
    if meta.get('node_feature_dim') != NODE_FEATURE_DIM:
        raise RuntimeError(
            f'stale cache at {cache_path}: '
            f"node_feature_dim={meta.get('node_feature_dim')} vs parser's {NODE_FEATURE_DIM}. "
            f'Rebuild via scripts/parse_equations.py.'
        )
    if meta.get('edge_feature_dim') != EDGE_FEATURE_DIM:
        raise RuntimeError(
            f'stale cache at {cache_path}: '
            f"edge_feature_dim={meta.get('edge_feature_dim')} vs parser's {EDGE_FEATURE_DIM}. "
            f'Rebuild via scripts/parse_equations.py.'
        )
    return cache['graphs']


def load_holdout_indices(parquet_path: str) -> np.ndarray:
    """Load holdout pair-table row indices from a split parquet."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    return table['pair_idx'].to_numpy().astype(np.int64, copy=False)


def derive_train_indices(values: np.ndarray, holdout_indices: np.ndarray) -> np.ndarray:
    """Compute training indices: every labeled row not in the holdout.

    Returns a sorted int64 array.
    """
    holdout_mask = np.zeros(len(values), dtype=bool)
    holdout_mask[holdout_indices] = True
    train_mask = (values != 2) & ~holdout_mask
    return np.where(train_mask)[0].astype(np.int64, copy=False)


def make_loaders(
    pair_table_path: str,
    graph_cache_path: str,
    holdout_path: str,
    batch_size: int = 2048,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    val_shuffle_seed: int = 0,
) -> tuple[DataLoader, DataLoader, ImplicationPairsDataset, ImplicationPairsDataset]:
    """Build training and validation DataLoaders from the cache + pair table + holdout.

    The val dataset's indices are deterministically permuted with
    ``val_shuffle_seed``. Without this, ``holdout_idx`` is sorted ascending
    by pair-table row, which correlates with equation ID and thus with
    equation complexity — so an evaluator using ``max_batches=N`` on the
    val_loader (i.e. taking the first N batches) reads a biased slice
    rather than a representative sample. Permuting once means the capped
    slice is an unbiased random sample, deterministic across epochs and
    runs.

    Returns:
        ``(train_loader, val_loader, train_dataset, val_dataset)``.
    """
    eq1_ids, eq2_ids, values = load_pair_arrays(pair_table_path)
    graphs = load_graph_cache(graph_cache_path)
    holdout_idx = load_holdout_indices(holdout_path)
    train_idx = derive_train_indices(values, holdout_idx)

    rng = np.random.default_rng(val_shuffle_seed)
    val_idx = holdout_idx.copy()
    rng.shuffle(val_idx)

    train_ds = ImplicationPairsDataset(eq1_ids, eq2_ids, values, graphs, train_idx)
    val_ds = ImplicationPairsDataset(eq1_ids, eq2_ids, values, graphs, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_pairs,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_pairs,
        drop_last=False,
    )
    return train_loader, val_loader, train_ds, val_ds
