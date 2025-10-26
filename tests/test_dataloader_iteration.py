import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.test_cluster_integration import make_synthetic_data, make_minimal_config, load_precompute_module

def test_dataloader_iteration_smoke():
    precompute = load_precompute_module()
    data_dict, metadata, _ = make_synthetic_data(N=120, n_features=4, n_classes=3)
    cfg = type("Cfg", (), {
        'cluster_method': 'kmeans', 'n_clusters': 4, 'random_state': 0,
        'use_pca': False, 'pca_components': 2, 'standardize': True,
        'impute_strategy': 'mean', 'compute_metrics': False,
        'min_cluster_size': 2, 'min_samples': 1
    })()
    data_dict = precompute.compute_clusters_for_npt(data_dict, metadata, cfg)

    c = make_minimal_config()
    from npt.batch_dataset import NPTBatchDataset
    from torch.utils.data import DataLoader

    ds = NPTBatchDataset(data_dict, c, curr_cv_split=0, metadata=metadata, device='cpu', sigmas=None)
    ds.set_mode('train', epoch=0)

    # collate_with_pre_batching expects a function in utils; use existing collate
    from npt.utils.batch_utils import collate_with_pre_batching

    loader = DataLoader(ds, batch_size=1, collate_fn=collate_with_pre_batching, num_workers=0)
    # iterate few batches
    n = 0
    for batch in loader:
        # basic sanity checks: batch is a dict (or tuple depending on collate)
        assert batch is not None
        n += 1
        if n >= 5:
            break
    print("DataLoader iteration smoke test passed")

if __name__ == "__main__":
    test_dataloader_iteration_smoke()