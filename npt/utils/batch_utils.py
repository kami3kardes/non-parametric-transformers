import warnings
from collections import OrderedDict, defaultdict

import numpy as np
from typing import List, Tuple, Optional
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import column_or_1d

import torch
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

collate_with_pre_batching_err_msg_format = (
    "collate_with_pre_batched_map: "
    "batch must be a list with one map element; found {}")


def collate_with_pre_batching(batch):
    r"""
    Collate function used by our PyTorch dataloader (in both distributed and
    serial settings).

    We avoid adding a batch dimension, as for NPT we have pre-batched data,
    where each element of the dataset is a map.

    :arg batch: List[Dict] (not as general as the default collate fn)
    """
    if len(batch) > 1:
        raise NotImplementedError

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, container_abcs.Mapping):
        return elem  # Just return the dict, as there will only be one in NPT

    raise TypeError(collate_with_pre_batching_err_msg_format.format(elem_type))


# TODO: batching over features?

class StratifiedIndexSampler:
    """
    Minimal StratifiedIndexSampler matching the API used by batch_dataset.
    Purpose: given y indicators aligned to a row_index_order, produce a
    permuted ordering and a list of batch_sizes (by splitting into n_splits).
    This is a simple implementation sufficient for batch ordering / tests.
    """
    def __init__(self, y, n_splits=1, shuffle=True, random_state=None, label_col=None, train_indices=None):
        self.y = np.asarray(y)
        self.n_splits = max(1, int(n_splits))
        self.shuffle = bool(shuffle)
        self.random_state = None if random_state is None else int(random_state)
        self.rng = np.random.RandomState(self.random_state)
        # label_col / train_indices are accepted for compat but not required for this simple impl
        self.label_col = label_col
        self.train_indices = train_indices

    def get_stratified_test_array(self, row_index_order) -> Tuple[np.ndarray, List[int]]:
        row_index_order = np.asarray(row_index_order)
        # Align y to row_index_order if necessary
        if len(self.y) != len(row_index_order):
            if len(self.y) > len(row_index_order) and row_index_order.max() < len(self.y):
                y_aligned = self.y[row_index_order]
            else:
                raise ValueError("StratifiedIndexSampler: y length mismatch with row_index_order")
        else:
            y_aligned = self.y

        # Group indices by label
        labels = np.unique(y_aligned)
        groups = [row_index_order[y_aligned == lbl].copy() for lbl in labels]

        # Shuffle within groups
        if self.shuffle:
            for g in groups:
                self.rng.shuffle(g)

        # Interleave groups round-robin to get class-balanced ordering
        max_len = max((g.size for g in groups), default=0)
        interleaved = []
        for i in range(max_len):
            for g in groups:
                if i < g.size:
                    interleaved.append(g[i])
        if len(interleaved) == 0:
            return row_index_order.copy(), [len(row_index_order)]

        concatenated = np.asarray(interleaved)
        # Split into n_splits roughly equal chunks
        chunks = np.array_split(concatenated, self.n_splits)
        chunks = [c for c in chunks if c.size > 0]
        new_order = np.concatenate(chunks) if len(chunks) > 0 else np.array([], dtype=int)
        batch_sizes = [int(c.size) for c in chunks]
        return new_order, batch_sizes


class ClusteredIndexSampler:
    """
    ClusteredIndexSampler groups indices by cluster id and returns an
    ordering + batch_sizes. Already implemented above; kept for completeness.
    """
    def __init__(self, y, n_splits: int = 1, shuffle: bool = True, random_state: Optional[int] = None):
        self.y = np.asarray(y)
        self.n_splits = max(1, int(n_splits))
        self.shuffle = bool(shuffle)
        self.random_state = None if random_state is None else int(random_state)
        self.rng = np.random.RandomState(self.random_state)

    def get_stratified_test_array(self, row_index_order) -> Tuple[np.ndarray, List[int]]:
        row_index_order = np.asarray(row_index_order)
        if len(self.y) != len(row_index_order):
            if len(self.y) > len(row_index_order) and row_index_order.max() < len(self.y):
                y_aligned = self.y[row_index_order]
            else:
                raise ValueError("ClusteredIndexSampler: y length mismatch with row_index_order")
        else:
            y_aligned = self.y

        unique_clusters = np.unique(y_aligned)
        grouped = []
        for cid in unique_clusters:
            mask = (y_aligned == cid)
            ids = row_index_order[mask].copy()
            if self.shuffle and ids.size > 0:
                self.rng.shuffle(ids)
            grouped.append(ids)

        if len(grouped) == 0:
            return row_index_order.copy(), [len(row_index_order)]

        concatenated = np.concatenate(grouped)
        chunks = np.array_split(concatenated, self.n_splits)
        chunks = [c for c in chunks if c.size > 0]
        if self.shuffle and len(chunks) > 1:
            self.rng.shuffle(chunks)
        new_order = np.concatenate(chunks) if len(chunks) > 0 else np.array([], dtype=int)
        batch_sizes = [int(c.size) for c in chunks]
        return new_order, batch_sizes


class PrototypeIndexSampler:
    """
    PrototypeIndexSampler arranges batches so that each prototype (an actual
    sample index) is grouped with its nearest neighbors. The sampler returns
    an ordering and batch sizes similar to other samplers.

    Expected input for y (first arg) is a dict with keys:
        'prototype_indices': array-like of prototype row indices (into full data)
        'neighbors': list of arrays, each containing neighbor indices for the
                     corresponding prototype (may include the prototype itself)
    """
    def __init__(self, y, n_splits: int = 1, shuffle: bool = True, random_state: Optional[int] = None):
        # Normalize inputs
        if isinstance(y, dict):
            self.prototype_indices = np.asarray(y.get('prototype_indices', []), dtype=np.int64)
            self.neighbors = [np.asarray(n, dtype=np.int64) for n in y.get('neighbors', [])]
        else:
            # backward compat: if y is array-like, treat as prototype indices with no neighbors
            self.prototype_indices = np.asarray(y, dtype=np.int64)
            self.neighbors = [np.array([int(p)]) for p in self.prototype_indices]

        self.n_splits = max(1, int(n_splits))
        self.shuffle = bool(shuffle)
        self.random_state = None if random_state is None else int(random_state)
        self.rng = np.random.RandomState(self.random_state)

    def get_stratified_test_array(self, row_index_order) -> Tuple[np.ndarray, List[int]]:
        # Build blocks: for each prototype, take prototype followed by its neighbors
        blocks = []
        for proto_idx, neigh in zip(self.prototype_indices, self.neighbors):
            block = np.asarray([proto_idx] + [int(x) for x in neigh if int(x) != int(proto_idx)])
            blocks.append(block)

        if len(blocks) == 0:
            return row_index_order.copy(), [len(row_index_order)]

        # Optionally shuffle prototype order
        if self.shuffle:
            self.rng.shuffle(blocks)

        concatenated = np.concatenate(blocks)

        # Ensure we only include indices that exist in row_index_order
        mask = np.isin(concatenated, row_index_order)
        concatenated = concatenated[mask]

        # Append any remaining row_index_order elements that weren't covered
        remaining = np.setdiff1d(row_index_order, concatenated, assume_unique=False)
        if remaining.size > 0:
            if self.shuffle:
                self.rng.shuffle(remaining)
            concatenated = np.concatenate([concatenated, remaining])

        # Split into n_splits chunks
        chunks = np.array_split(concatenated, self.n_splits)
        chunks = [c for c in chunks if c.size > 0]
        new_order = np.concatenate(chunks) if len(chunks) > 0 else np.array([], dtype=int)
        batch_sizes = [int(c.size) for c in chunks]
        return new_order, batch_sizes


# Export what other modules expect to import
__all__ = ['StratifiedIndexSampler', 'ClusteredIndexSampler', 'collate_with_pre_batching']
