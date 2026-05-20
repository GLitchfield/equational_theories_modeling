"""Dataset and DataLoader for paired-graph implication training.

Each sample is ``(eq1_graph, eq2_graph, label)`` where the graphs come from
the parsed-tree cache and the label is in {0, 1} (FALSE / TRUE). Unknown
pairs (``value == 2``) are excluded upstream by the holdout-construction
logic and never appear here.

Custom ``collate_pairs`` batches each side independently with PyG's
``Batch.from_data_list``, yielding ``(eq1_batch: Batch, eq2_batch: Batch,
labels: FloatTensor)`` per minibatch.

When auxiliary labels are loaded, items are 5-tuples
``(eq1_graph, eq2_graph, label, eq1_aux, eq2_aux)`` where each
``*_aux`` is a 1-D int64 numpy array of length
``len(AUX_HEAD_NAMES)``, one entry per auxiliary head. Collation
returns a 5-tuple with the aux fields as ``dict[head_name, LongTensor]``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from lib.data.parser import EDGE_FEATURE_DIM, NODE_FEATURE_DIM

# Order MUST match HEAD_SPECS in src/model/aux_heads.py — collate
# emits dicts keyed by these names so the train loop can hand them
# straight to ``auxiliary_loss``.
AUX_HEAD_NAMES = (
    "spine_type",
    "lhs_structure",
    "num_variables",
    "magma_bit_0",
    "magma_bit_1",
    "magma_bit_2",
    "magma_bit_3",
    "magma_bit_4",
    "magma_bit_6",
    "magma_bit_7",
    "cohom_count_2",
    "cohom_count_3",
    "cohom_count_5",
    "cohom_count_7",
    "cohom_mean_h2_2",
    "cohom_mean_h2_3",
    "cohom_mean_h2_5",
    "cohom_mean_h2_7",
)


class ImplicationPairsDataset(Dataset):
    """Yields ``(eq1_graph, eq2_graph, label)`` for the implication-pair task.

    With ``aux_labels`` provided, items are 5-tuples
    ``(eq1_graph, eq2_graph, label, eq1_aux_row, eq2_aux_row)`` where
    each aux row is a 1-D int64 numpy array of length
    ``len(AUX_HEAD_NAMES)``.

    Args:
        eq1_ids, eq2_ids, values: 1-D numpy arrays indexed by pair-table row.
            ``eq1_ids[r]`` and ``eq2_ids[r]`` are 1-indexed equation IDs;
            ``values[r]`` is in {0, 1, 2} (FALSE / TRUE / unknown).
        graphs: dict mapping equation ID → PyG Data with the parsed tree.
        indices: 1-D int array of pair-table row indices to expose. Item
            ``self[i]`` corresponds to row ``indices[i]``.
        aux_labels: optional ``[N_eq, len(AUX_HEAD_NAMES)]`` int64 array
            indexed by ``eq_id - 1`` (eq IDs are 1-based per the corpus).
            Column order must match ``AUX_HEAD_NAMES``. ``None`` disables
            the aux path and items remain 3-tuples (backward compatible).
    """

    def __init__(
        self,
        eq1_ids: np.ndarray,
        eq2_ids: np.ndarray,
        values: np.ndarray,
        graphs: dict[int, Data],
        indices: np.ndarray,
        aux_labels: np.ndarray | None = None,
    ) -> None:
        self.eq1_ids = eq1_ids
        self.eq2_ids = eq2_ids
        self.values = values
        self.graphs = graphs
        self.indices = np.asarray(indices, dtype=np.int64)
        if aux_labels is not None:
            if aux_labels.ndim != 2 or aux_labels.shape[1] != len(AUX_HEAD_NAMES):
                raise ValueError(
                    f"aux_labels must be [N_eq, {len(AUX_HEAD_NAMES)}]; "
                    f"got shape {aux_labels.shape}"
                )
            aux_labels = np.ascontiguousarray(aux_labels, dtype=np.int64)
        self.aux_labels = aux_labels

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        row = int(self.indices[i])
        eq1_id = int(self.eq1_ids[row])
        eq2_id = int(self.eq2_ids[row])
        graph1 = self.graphs[eq1_id]
        graph2 = self.graphs[eq2_id]
        label = int(self.values[row])
        if self.aux_labels is None:
            return graph1, graph2, label
        return (
            graph1, graph2, label,
            self.aux_labels[eq1_id - 1],
            self.aux_labels[eq2_id - 1],
        )


def collate_pairs(batch):
    """Collate a list of dataset items into batches.

    Backward-compatible: 3-tuple items produce
    ``(eq1_batch, eq2_batch, labels)``; 5-tuple items (aux-active) produce
    ``(eq1_batch, eq2_batch, labels, eq1_aux, eq2_aux)`` where each
    ``*_aux`` is ``dict[head_name, LongTensor[batch]]`` keyed by
    ``AUX_HEAD_NAMES``.
    """
    eq1_list = [item[0] for item in batch]
    eq2_list = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float)
    eq1_batch = Batch.from_data_list(eq1_list)
    eq2_batch = Batch.from_data_list(eq2_list)
    if len(batch[0]) == 3:
        return eq1_batch, eq2_batch, labels
    # Aux-active: stack the per-equation aux rows into per-head tensors.
    eq1_aux_rows = np.stack([item[3] for item in batch])  # (B, n_heads)
    eq2_aux_rows = np.stack([item[4] for item in batch])
    eq1_aux = {
        name: torch.from_numpy(eq1_aux_rows[:, j].astype(np.int64))
        for j, name in enumerate(AUX_HEAD_NAMES)
    }
    eq2_aux = {
        name: torch.from_numpy(eq2_aux_rows[:, j].astype(np.int64))
        for j, name in enumerate(AUX_HEAD_NAMES)
    }
    return eq1_batch, eq2_batch, labels, eq1_aux, eq2_aux


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


def load_aux_labels(parquet_path: str) -> np.ndarray:
    """Load the per-equation auxiliary labels parquet into an int64 array.

    Built by ``scripts/build_equation_invariants.py``. The returned
    array has shape ``[N_eq, len(AUX_HEAD_NAMES)]`` indexed by
    ``eq_id - 1`` (eq IDs in the corpus are 1-based, the array is 0-based)
    with columns in ``AUX_HEAD_NAMES`` order.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    missing = [c for c in AUX_HEAD_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"aux labels parquet at {parquet_path} missing columns: {missing}"
        )
    # Index is eq_id (1-based); confirm the rows are aligned with eq_ids 1..N
    # so the row-position lookup (`aux_labels[eq_id - 1]`) is valid.
    expected_index = np.arange(1, len(df) + 1)
    if not np.array_equal(df.index.to_numpy(), expected_index):
        raise ValueError(
            f"aux labels index at {parquet_path} is not contiguous 1..N "
            f"(eq IDs); got first few = {df.index[:5].tolist()}"
        )
    return df[list(AUX_HEAD_NAMES)].to_numpy(dtype=np.int64)


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
    holdout_path: str | None,
    batch_size: int = 2048,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    val_shuffle_seed: int = 0,
    no_holdout_val_size: int = 1_000_000,
    aux_labels_path: str | None = None,
) -> tuple[DataLoader, DataLoader, ImplicationPairsDataset, ImplicationPairsDataset]:
    """Build training and validation DataLoaders from the cache + pair table.

    With ``holdout_path`` set (the standard L1/L2/L3 path), train_ds gets
    every labeled row not in the holdout and val_ds gets the holdout. The
    val dataset's indices are deterministically permuted with
    ``val_shuffle_seed`` so that capped-eval slices are unbiased.

    With ``holdout_path=None`` (the failure-analysis "no-holdout" mode),
    every labeled row goes into train_ds and val_ds is a deterministic
    random subsample of train (size ``no_holdout_val_size``) used purely
    for per-epoch threshold calibration and bal_acc monitoring. The
    monitor accuracy is in-sample so it overestimates generalization, but
    the alternative — no per-epoch eval at all — gives no signal about
    whether training is progressing. The plan accepts this bias.

    Returns:
        ``(train_loader, val_loader, train_dataset, val_dataset)``.
    """
    eq1_ids, eq2_ids, values = load_pair_arrays(pair_table_path)
    graphs = load_graph_cache(graph_cache_path)
    aux_labels = load_aux_labels(aux_labels_path) if aux_labels_path else None

    rng = np.random.default_rng(val_shuffle_seed)
    if holdout_path is None:
        # No-holdout: train on every labeled row; monitor on a fixed
        # random subsample of train.
        train_idx = derive_train_indices(values, np.array([], dtype=np.int64))
        size = min(no_holdout_val_size, len(train_idx))
        val_idx = rng.choice(train_idx, size=size, replace=False)
    else:
        holdout_idx = load_holdout_indices(holdout_path)
        train_idx = derive_train_indices(values, holdout_idx)
        val_idx = holdout_idx.copy()
        rng.shuffle(val_idx)

    train_ds = ImplicationPairsDataset(
        eq1_ids, eq2_ids, values, graphs, train_idx, aux_labels=aux_labels,
    )
    val_ds = ImplicationPairsDataset(
        eq1_ids, eq2_ids, values, graphs, val_idx, aux_labels=aux_labels,
    )

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
