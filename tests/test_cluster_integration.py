import sys
from pathlib import Path
import numpy as np
import torch
import numpy as np

# Ensure repo root on path so imports work when running this test directly
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_precompute_module():
    import importlib.util
    precompute_path = REPO_ROOT / "scripts" / "precompute_clusters.py"
    spec = importlib.util.spec_from_file_location("precompute_clusters", str(precompute_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_synthetic_data(N=200, n_features=4, n_classes=3):
    # Build per-column data_arrs: each column is (N,2) where [:,0] = value, [:,1] = mask(0)
    data_arrs = []
    rng = np.random.RandomState(0)
    for _ in range(n_features):
        vals = rng.randn(N).astype(np.float32)
        col = np.stack([vals, np.zeros_like(vals)], axis=-1)  # value + zero mask-column
        data_arrs.append(col)
    # label column: one-hot (N, n_classes) plus zero mask column -> shape (N, n_classes+1)
    labels = rng.randint(0, n_classes, size=N)
    onehot = np.eye(n_classes, dtype=np.float32)[labels]
    label_col = np.concatenate([onehot, np.zeros((N, 1), dtype=np.float32)], axis=1)
    data_arrs.append(label_col)

    # missing matrix: no missing
    D = len(data_arrs)
    missing_matrix = np.zeros((N, D), dtype=bool)

    # new_train_val_test_indices: simple split
    n_train = int(0.7 * N)
    n_val = int(0.15 * N)
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, N)
    new_train_val_test_indices = [train_idx, val_idx, test_idx]

    # Per-mode mask matrices: assume last column is target -> mask target presence for each split
    train_mask = np.zeros((N, D), dtype=bool)
    val_mask = np.zeros((N, D), dtype=bool)
    test_mask = np.zeros((N, D), dtype=bool)
    train_mask[train_idx, -1] = True
    val_mask[val_idx, -1] = True
    test_mask[test_idx, -1] = True

    # Convert tensors to torch where appropriate (some code expects torch tensors)
    data_dict = {
        'data_arrs': [torch.tensor(a) for a in data_arrs],
        'missing_matrix': torch.tensor(missing_matrix),
        'new_train_val_test_indices': [np.asarray(i) for i in new_train_val_test_indices],
        'train_mask_matrix': torch.tensor(train_mask),
        'val_mask_matrix': torch.tensor(val_mask),
        'test_mask_matrix': torch.tensor(test_mask),
    }

    metadata = {
        'N': N,
        'D': D,
        'cat_target_cols': [D - 1],  # last column is categorical target
        'num_target_cols': [],
        'cat_features': list(range(D - 1)),  # first D-1 are features
    }

    return data_dict, metadata, labels


def make_minimal_config():
    class C:
        pass
    c = C()
    c.exp_batch_size = 32
    c.data_set = 'tabular'
    c.data_set_on_cuda = False
    c.exp_batch_mode_balancing = False
    c.model_is_semi_supervised = False
    c.exp_batch_cluster_sampling = True
    c.exp_batch_cluster_mix_prob = 1.0
    c.exp_batch_cluster_per_cv = False
    c.np_seed = 42
    c.exp_batch_class_balancing = False
    c.model_label_bert_mask_prob = {'train': 1.0, 'val': 1.0, 'test': 1.0}
    c.model_augmentation_bert_mask_prob = {'train': 0.0, 'val': 0.0, 'test': 0.0}
    # dtype used by mask processing / tensors
    c.data_dtype = 'float32'
    c.verbose = False
    # minimal placeholders used by NPTBatchDataset if referenced
    c.exp_scheduler = 'none'
    return c


def test_cluster_integration():
    precompute = load_precompute_module()
    data_dict, metadata, labels = make_synthetic_data(N=200, n_features=4, n_classes=3)

    # Run clustering preprocessing on the full dataset (no per-cv)
    cfg = type("Cfg", (), {
        'cluster_method': 'kmeans',
        'n_clusters': 5,
        'random_state': 0,
        'use_pca': False,
        'pca_components': 10,
        'standardize': True,
        'impute_strategy': 'mean',
        'compute_metrics': False,
        'min_cluster_size': 4,
        'min_samples': 2
    })()

    updated = precompute.compute_clusters_for_npt(data_dict, metadata, cfg)
    assert 'cluster_assignments' in updated, "precompute did not write cluster_assignments"
    ca = np.asarray(updated['cluster_assignments'])
    assert ca.shape[0] == metadata['N']
    assert ca.dtype in (np.int32, np.int64, np.int_), "cluster_assignments dtype incorrect"

    # Now import NPTBatchDataset and instantiate
    from npt.batch_dataset import NPTBatchDataset
    c = make_minimal_config()

    # NPTBatchDataset constructor signature may vary; try typical args
    try:
        ds = NPTBatchDataset(updated, c, curr_cv_split=0, metadata=metadata, device='cpu', sigmas=None)
    except TypeError:
        # fallback: fewer args
        ds = NPTBatchDataset(updated, c, metadata=metadata, device='cpu')

    # Switch to train mode and generate batch order
    ds.set_mode('train', epoch=0)

    # Basic assertions
    assert hasattr(ds, 'data_arrs'), "NPTBatchDataset did not populate data_arrs"
    # data_arrs should have rows == number of mode indices (train rows), not full N
    first_col = ds.data_arrs[0]
    n_train = int(0.7 * metadata['N'])
    assert first_col.shape[0] == n_train, f"data_arrs rows mismatch: {first_col.shape[0]} != expected train rows {n_train}"

    # If sampler created variable batch_sizes, ensure they sum to N (or number of mode rows)
    if ds.batch_sizes is not None:
        total = sum(ds.batch_sizes)
        # Sampler batch_sizes should sum to number of mode rows (train rows here)
        assert total == n_train, f"batch sizes sum {total} != expected train rows {n_train}"

    # Verify masked_tensors exist and length matches D
    assert ds.masked_tensors is not None, "masked_tensors not created"
    assert len(ds.masked_tensors) == metadata['D'], f"masked_tensors length mismatch {len(ds.masked_tensors)} != {metadata['D']}"

    print("Cluster integration smoke test passed.")
    return True


if __name__ == "__main__":
    ok = test_cluster_integration()
    if ok:
        print("SUCCESS")
    else:
        print("FAILED")