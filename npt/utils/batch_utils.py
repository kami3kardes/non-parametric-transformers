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

class LearnedPrototypeIndexSampler:
    """
    Sampler that builds prototype+neighbor blocks using a trainable
    `LearnedPrototypes` instance (or any callable returning prototype vectors).
    """
    def __init__(self, dataset_features, prototypes_getter,
                 k_neighbors: int = 8, n_splits: int = 1,
                 shuffle: bool = True, random_state: Optional[int] = None,
                 backend: str = 'sklearn', 
                 cache_nn_index: bool = True):
        self.dataset_features = np.asarray(dataset_features)
        if self.dataset_features.ndim != 2:
            raise ValueError('dataset_features must be 2D array (N, D)')
        self.prototypes_getter = prototypes_getter
        self.k_neighbors = int(k_neighbors)
        self.n_splits = max(1, int(n_splits))
        self.shuffle = bool(shuffle)
        self.random_state = None if random_state is None else int(random_state)
        self.rng = np.random.RandomState(self.random_state)
        self.backend = backend
        self.cache_nn_index = cache_nn_index

        # Internal state
        self.prototype_indices = np.array([], dtype=np.int64)
        self.neighbors = []
        self.last_update_epoch = None
        self.last_update_step = None
        self.update_count = 0
        
        # Cached NN index
        self._nn_index = None
        self._last_dataset_hash = None

    def _get_prototypes_array(self):
        """Extract prototypes as numpy array from getter."""
        p = self.prototypes_getter
        if callable(p):
            arr = p()
        else:
            arr = getattr(p, 'prototypes', None)
            if arr is None:
                raise ValueError('prototypes_getter must be callable or have `.prototypes`')
        
        if isinstance(arr, np.ndarray):
            out = arr
        elif isinstance(arr, torch.Tensor):
            out = arr.detach().cpu().numpy()
        else:
            out = np.asarray(arr)
        
        if out.ndim != 2:
            raise ValueError('prototypes must be 2D (P, D)')
        return out.astype(np.float32)

    def update(self, epoch=None, step=None):
        """Recompute nearest dataset rows for each prototype.
        
        Args:
            epoch: Current epoch (for tracking)
            step: Current step (for tracking)
            
        Returns:
            info: Dict with update statistics
        """
        import time
        start = time.time()
        
        prototypes = self._get_prototypes_array()
        P, Dp = prototypes.shape
        N, D = self.dataset_features.shape
        
        if Dp != D:
            raise ValueError(f'Prototype dim {Dp} != dataset feature dim {D}')
        
        if P == 0:
            warnings.warn("No prototypes found! Sampler will return empty blocks.")
            self.prototype_indices = np.array([], dtype=np.int64)
            self.neighbors = []
            return {'n_prototypes': 0, 'warning': 'empty_prototypes'}

        # Build or reuse NN index
        if self.cache_nn_index:
            dataset_hash = hash(self.dataset_features.tobytes())
            if self._nn_index is None or self._last_dataset_hash != dataset_hash:
                try:
                    from sklearn.neighbors import NearestNeighbors
                except ImportError:
                    raise RuntimeError('sklearn required for LearnedPrototypeIndexSampler')
                
                self._nn_index = NearestNeighbors(
                    n_neighbors=min(self.k_neighbors, N), 
                    algorithm='auto'
                )
                self._nn_index.fit(self.dataset_features)
                self._last_dataset_hash = dataset_hash
            nn = self._nn_index
        else:
            try:
                from sklearn.neighbors import NearestNeighbors
            except ImportError:
                raise RuntimeError('sklearn required for LearnedPrototypeIndexSampler')
            nn = NearestNeighbors(n_neighbors=min(self.k_neighbors, N), algorithm='auto')
            nn.fit(self.dataset_features)

        # Find nearest neighbors
        distances, indices = nn.kneighbors(prototypes, return_distance=True)
        
        # Store results
        self.prototype_indices = indices[:, 0].astype(np.int64)
        self.neighbors = [inds.astype(np.int64) for inds in indices]
        
        # Update tracking
        self.last_update_epoch = epoch
        self.last_update_step = step
        self.update_count += 1
        
        elapsed = time.time() - start
        
        # Compute prototype diversity
        proto_diversity = self._compute_prototype_diversity(prototypes)
        
        return {
            'prototype_indices': self.prototype_indices,
            'neighbors': self.neighbors,
            'update_count': self.update_count,
            'elapsed_time': elapsed,
            'prototype_diversity': proto_diversity,
            'n_prototypes': P,
            'epoch': epoch,
            'step': step
        }

    def _compute_prototype_diversity(self, prototypes):
        """Compute average pairwise distance between prototypes."""
        if len(prototypes) <= 1:
            return 0.0
        try:
            from scipy.spatial.distance import pdist
            return float(np.mean(pdist(prototypes)))
        except:
            # Fallback if scipy not available
            return 0.0

    def get_stratified_test_array(self, row_index_order) -> Tuple[np.ndarray, List[int]]:
        """Build batches from prototype blocks."""
        
        # Warn if never updated
        if self.update_count == 0:
            warnings.warn(
                "LearnedPrototypeIndexSampler.update() has never been called! "
                "Blocks may be empty or stale. Call update() before get_stratified_test_array().",
                UserWarning
            )
        
        row_index_order = np.asarray(row_index_order)
        blocks = []
        
        for proto_idx, neigh in zip(self.prototype_indices, self.neighbors):
            # Build block: prototype + neighbors (excluding duplicate)
            block = np.asarray([int(proto_idx)] + [int(x) for x in neigh if int(x) != int(proto_idx)])
            blocks.append(block)

        if len(blocks) == 0:
            return row_index_order.copy(), [len(row_index_order)]

        # Shuffle prototype order
        if self.shuffle:
            self.rng.shuffle(blocks)

        concatenated = np.concatenate(blocks)
        
        # Only include indices that exist in row_index_order
        mask = np.isin(concatenated, row_index_order)
        concatenated = concatenated[mask]
        
        # Append remaining indices not covered by prototypes
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
__all__ = [
    'StratifiedIndexSampler', 'ClusteredIndexSampler', 'PrototypeIndexSampler',
    'LearnedPrototypeIndexSampler', 'collate_with_pre_batching'
]
