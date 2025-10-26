import sys
from pathlib import Path
import numpy as np
import pickle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def load_precompute():
    import importlib.util
    path = REPO_ROOT / "scripts" / "precompute_clusters.py"
    spec = importlib.util.spec_from_file_location("precompute_clusters", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def make_small_data():
    N = 60
    n_features = 3
    data_arrs = []
    rng = np.random.RandomState(0)
    for _ in range(n_features):
        vals = rng.randn(N).astype(np.float32)
        col = np.stack([vals, np.zeros_like(vals)], axis=-1)
        data_arrs.append(col)
    # label column
    labels = rng.randint(0, 2, size=N)
    onehot = np.eye(2, dtype=np.float32)[labels]
    label_col = np.concatenate([onehot, np.zeros((N,1), dtype=np.float32)], axis=1)
    data_arrs.append(label_col)
    D = len(data_arrs)
    missing = np.zeros((N, D), dtype=bool)
    # splits: 3 folds (train/val/test triple per fold)
    idx = np.arange(N)
    fold_size = N // 3
    splits = []
    for i in range(3):
        start = i * fold_size
        splits.append(idx[start:start+fold_size])
    data_dict = {
        'data_arrs': [a for a in data_arrs],
        'missing_matrix': missing,
        'new_train_val_test_indices': [np.asarray(s) for s in splits],
        'train_mask_matrix': np.zeros((N,D), dtype=bool),
        'val_mask_matrix': np.zeros((N,D), dtype=bool),
        'test_mask_matrix': np.zeros((N,D), dtype=bool),
    }
    metadata = {'N': N, 'D': D, 'cat_target_cols':[D-1], 'num_target_cols':[], 'cat_features': list(range(D-1))}
    return data_dict, metadata

def test_per_cv_precompute_and_map(tmp_path):
    mod = load_precompute()
    data_dict, metadata = make_small_data()
    cfg = type("Cfg", (), {
        'cluster_method':'kmeans', 'n_clusters':4, 'random_state':0,
        'use_pca': False, 'pca_components': 2, 'standardize': True,
        'impute_strategy': 'mean', 'compute_metrics': False,
        'min_cluster_size': 2, 'min_samples': 1
    })()
    # call per-cv path: the script function expects args-like; we call compute_clusters_for_npt on each train slice manually
    # but we want to test that compute_clusters_for_npt handles small arrays and that mapping back to full length works.
    # Simulate the per-cv loop in script
    assignments_per_cv = []
    for split_idx, train_idx in enumerate(data_dict['new_train_val_test_indices']):
        temp = {}
        temp['data_arrs'] = [col[train_idx] if hasattr(col, 'shape') else col for col in data_dict['data_arrs']]
        temp['missing_matrix'] = data_dict['missing_matrix'][train_idx]
        temp_meta = dict(metadata); temp_meta['N'] = len(train_idx)
        assigns = mod.compute_clusters_for_npt(temp, temp_meta, cfg)['cluster_assignments']
        full = -1 * np.ones(metadata['N'], dtype=np.int32)
        full[train_idx] = assigns
        assignments_per_cv.append(full)
    # Basic checks
    assert len(assignments_per_cv) == len(data_dict['new_train_val_test_indices'])
    for a in assignments_per_cv:
        assert a.shape[0] == metadata['N']
        assert a.dtype == np.int32

    # Save and reload pickled data_dict (simulate script save)
    data_dict['cluster_assignments_per_cv'] = assignments_per_cv
    p = tmp_path / "test_data.pkl"
    with open(p, 'wb') as f:
        pickle.dump(data_dict, f)
    with open(p, 'rb') as f:
        loaded = pickle.load(f)
    assert 'cluster_assignments_per_cv' in loaded
    print("per-cv precompute mapping test passed")

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_per_cv_precompute_and_map(Path(tmpdir))