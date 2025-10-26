import numpy as np
import argparse
import pickle
from pathlib import Path


def compute_clusters_for_npt(data_dict, metadata, config):
    """
    Compute cluster assignments offline before training.

    Args:
        data_dict: Dictionary containing data_arrs and other data
        metadata: Dictionary with feature information
        config: Configuration object with clustering parameters

    Returns:
        data_dict: Updated with cluster_assignments
    """
    print("\n" + "="*70)
    print("COMPUTING CLUSTER ASSIGNMENTS FOR NPT")
    print("="*70)

    # Extract features (excluding targets)
    target_cols = set(
        metadata.get('cat_target_cols', []) +
        metadata.get('num_target_cols', [])
    )

    print(f"\nDataset info:")
    print(f"  Total features: {len(data_dict['data_arrs'])}")
    print(f"  Target columns: {target_cols}")
    print(f"  Non-target features: {len(data_dict['data_arrs']) - len(target_cols)}")
    print(f"  Total samples: {metadata['N']}")

    # Get data matrix (handle missing values)
    feature_data = []
    feature_indices = []

    for col_idx, col in enumerate(data_dict['data_arrs']):
        if col_idx not in target_cols:
            # Assume [:, 0] is data, [:, 1] is mask indicator
            if getattr(col, 'ndim', 1) == 2 and col.shape[1] >= 1:
                vals = col[:, 0]
            else:
                vals = col
            # move to numpy if torch tensor
            if hasattr(vals, 'cpu'):
                vals = vals.cpu().numpy()
            feature_data.append(vals)
            feature_indices.append(col_idx)

    if len(feature_data) == 0:
        raise ValueError("No feature columns found for clustering!")

    X = np.column_stack(feature_data)
    print(f"\nFeature matrix shape: {X.shape}")

    # Handle missing values
    missing_matrix = data_dict.get('missing_matrix')
    if missing_matrix is not None:
        missing_cols = []
        for col_idx in feature_indices:
            if hasattr(missing_matrix, 'cpu'):
                missing_cols.append(missing_matrix[:, col_idx].cpu().numpy())
            else:
                missing_cols.append(missing_matrix[:, col_idx])
        missing_feature_matrix = np.column_stack(missing_cols)
        n_missing = np.sum(missing_feature_matrix)
        missing_rate = n_missing / missing_feature_matrix.size
        print(f"Missing values: {n_missing} ({missing_rate:.2%})")
        X = np.where(missing_feature_matrix, np.nan, X)

    # Impute missing values
    print("\nImputing missing values...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy=config.impute_strategy)
    X_imputed = imputer.fit_transform(X)
    print(f"Imputation complete. Strategy: {config.impute_strategy}")

    # Apply dimensionality reduction if needed
    if getattr(config, 'use_pca', False) and X_imputed.shape[1] > getattr(config, 'pca_components', 50):
        print(f"\nApplying PCA (n_components={config.pca_components})...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=config.pca_components, random_state=config.random_state)
        X_reduced = pca.fit_transform(X_imputed)
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"PCA complete. Explained variance: {explained_var:.2%}")
        # Save pca for later assignment if desired
        data_dict.setdefault('cluster_metadata', {})['pca_components'] = config.pca_components
    else:
        X_reduced = X_imputed
        print("\nSkipping PCA (use_pca=False or too few features)")

    # Standardize features
    if getattr(config, 'standardize', True):
        print("\nStandardizing features...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        data_dict.setdefault('cluster_metadata', {})['standardized'] = True
        data_dict['cluster_metadata']['scaler_mean'] = scaler.mean_.tolist()
        data_dict['cluster_metadata']['scaler_var'] = scaler.var_.tolist()
    else:
        X_scaled = X_reduced
        data_dict.setdefault('cluster_metadata', {})['standardized'] = False

    print(f"Final feature matrix shape: {X_scaled.shape}")

    # Perform clustering
    print(f"\nClustering with {config.cluster_method} (n_clusters={config.n_clusters})...")

    if config.cluster_method == 'kmeans':
        from sklearn.cluster import KMeans
        clusterer = KMeans(
            n_clusters=config.n_clusters,
            random_state=config.random_state,
            n_init=10,
            max_iter=300
        )
        cluster_assignments = clusterer.fit_predict(X_scaled)
        centers = getattr(clusterer, 'cluster_centers_', None)
    elif config.cluster_method == 'gmm':
        from sklearn.mixture import GaussianMixture
        clusterer = GaussianMixture(
            n_components=config.n_clusters,
            random_state=config.random_state,
            max_iter=100
        )
        cluster_assignments = clusterer.fit_predict(X_scaled)
        centers = getattr(clusterer, 'means_', None)
    elif config.cluster_method == 'hdbscan':
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples
        )
        cluster_assignments = clusterer.fit_predict(X_scaled)
        centers = None
    elif config.cluster_method == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        clusterer = AgglomerativeClustering(n_clusters=config.n_clusters)
        cluster_assignments = clusterer.fit_predict(X_scaled)
        centers = None
    else:
        raise ValueError(f"Unknown clustering method: {config.cluster_method}")

    # Handle noise points for HDBSCAN (cluster -1)
    if config.cluster_method == 'hdbscan':
        n_noise = np.sum(cluster_assignments == -1)
        if n_noise > 0:
            print(f"Warning: {n_noise} noise points detected. Assigning to nearest cluster...")
            from sklearn.metrics import pairwise_distances
            if centers is None and hasattr(clusterer, 'cluster_centers_'):
                centers = clusterer.cluster_centers_
            if centers is not None:
                noise_mask = cluster_assignments == -1
                distances = pairwise_distances(X_scaled[noise_mask], centers)
                nearest_clusters = np.argmin(distances, axis=1)
                cluster_assignments[noise_mask] = nearest_clusters

    cluster_assignments = np.asarray(cluster_assignments, dtype=np.int32)

    # Analyze cluster distribution
    unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
    n_clusters_actual = len(unique_clusters)
    print(f"\nClustering complete!")
    print(f"  Number of clusters: {n_clusters_actual}")
    print(f"  Cluster sizes (min/mean/max): {counts.min()}/{counts.mean():.1f}/{counts.max()}")
    print(f"  Cluster size std: {counts.std():.1f}")

    min_viable_size = getattr(config, 'min_viable_size', 4)
    small_clusters = np.sum(counts < min_viable_size)
    if small_clusters > 0:
        print(f"  WARNING: {small_clusters} clusters have <{min_viable_size} samples")

    # Detailed cluster distribution
    print(f"\nCluster distribution:")
    cluster_dist = dict(zip(unique_clusters.tolist(), counts.tolist()))
    for cid in sorted(cluster_dist.keys()):
        count = cluster_dist[cid]
        pct = 100 * count / len(cluster_assignments)
        print(f"  Cluster {cid:3d}: {count:6d} samples ({pct:5.2f}%)")

    # Compute clustering quality metrics
    if getattr(config, 'compute_metrics', False):
        print("\nComputing clustering quality metrics...")
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        try:
            silhouette = silhouette_score(X_scaled, cluster_assignments)
            print(f"  Silhouette score: {silhouette:.3f} (range: -1 to 1, higher is better)")
        except Exception as e:
            print(f"  Could not compute silhouette score: {e}")
        try:
            calinski = calinski_harabasz_score(X_scaled, cluster_assignments)
            print(f"  Calinski-Harabasz score: {calinski:.3f} (higher is better)")
        except Exception as e:
            print(f"  Could not compute Calinski-Harabasz score: {e}")

    # Save cluster assignments and metadata
    data_dict['cluster_assignments'] = cluster_assignments
    if centers is not None:
        data_dict['cluster_centers'] = centers
    data_dict['cluster_metadata'] = {
        'method': config.cluster_method,
        'n_requested': config.n_clusters,
        'n_clusters': n_clusters_actual,
        'cluster_sizes': cluster_dist,
        'feature_indices': feature_indices,
        'used_pca': getattr(config, 'use_pca', False),
        'pca_components': getattr(config, 'pca_components', None),
        'standardized': getattr(config, 'standardize', True),
        'random_state': config.random_state
    }

    print("\n" + "="*70)
    print("CLUSTER ASSIGNMENTS SAVED TO data_dict['cluster_assignments']")
    print("="*70 + "\n")
    return data_dict


def compute_prototypes_for_npt(data_dict, metadata, config):
    """
    Compute prototype (medoid) indices and their nearest neighbors.

    Output written into data_dict['prototypes'] as a dict with keys:
        'prototype_indices': np.array of indices
        'neighbors': list of np.arrays with neighbor indices (each the same len)
        'by': 'cluster' or 'class'
        'k': number of neighbors
    """
    print("\n" + "="*40)
    print("COMPUTING PROTOTYPES FOR NPT")
    print("="*40)

    # Extract features (excluding targets) - reuse logic from compute_clusters_for_npt
    target_cols = set(
        metadata.get('cat_target_cols', []) +
        metadata.get('num_target_cols', [])
    )

    feature_data = []
    feature_indices = []
    for col_idx, col in enumerate(data_dict['data_arrs']):
        if col_idx not in target_cols:
            if getattr(col, 'ndim', 1) == 2 and col.shape[1] >= 1:
                vals = col[:, 0]
            else:
                vals = col
            if hasattr(vals, 'cpu'):
                vals = vals.cpu().numpy()
            feature_data.append(vals)
            feature_indices.append(col_idx)

    if len(feature_data) == 0:
        raise ValueError("No feature columns found for prototype computation!")

    X = np.column_stack(feature_data)

    # Handle missing values similarly
    missing_matrix = data_dict.get('missing_matrix')
    if missing_matrix is not None:
        missing_cols = []
        for col_idx in feature_indices:
            if hasattr(missing_matrix, 'cpu'):
                missing_cols.append(missing_matrix[:, col_idx].cpu().numpy())
            else:
                missing_cols.append(missing_matrix[:, col_idx])
        missing_feature_matrix = np.column_stack(missing_cols)
        X = np.where(missing_feature_matrix, np.nan, X)

    # Impute
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy=getattr(config, 'impute_strategy', 'mean'))
    X_imputed = imputer.fit_transform(X)

    # Standardize if requested
    if getattr(config, 'standardize', True):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        X_scaled = X_imputed

    # Determine grouping: 'cluster' or 'class'
    prototype_by = getattr(config, 'prototype_by', 'cluster')
    n_prototypes_per_group = int(getattr(config, 'n_prototypes_per_group', 1))
    k_neighbors = int(getattr(config, 'prototype_neighbors', 5))

    groups = {}
    if prototype_by == 'cluster':
        if 'cluster_assignments' not in data_dict:
            # Compute clusters first with reasonable defaults
            print("No cluster_assignments found; computing clusters first (kmeans)...")
            cfg = type('TmpCfg', (), {})()
            cfg.cluster_method = getattr(config, 'cluster_method', 'kmeans')
            cfg.n_clusters = getattr(config, 'n_clusters', 10)
            cfg.random_state = getattr(config, 'random_state', 0)
            cfg.min_cluster_size = getattr(config, 'min_cluster_size', 4)
            cfg.min_samples = getattr(config, 'min_samples', 1)
            compute_clusters_for_npt(data_dict, metadata, cfg)

        assigns = np.asarray(data_dict['cluster_assignments'])
        unique = np.unique(assigns)
        for uid in unique:
            groups[int(uid)] = np.where(assigns == uid)[0]
    elif prototype_by == 'class':
        # Use categorical target first if available
        cat_targets = metadata.get('cat_target_cols', [])
        if len(cat_targets) == 0:
            raise ValueError('prototype_by == class requires categorical target columns')
        target_col = data_dict['data_arrs'][cat_targets[0]]
        # target_col may be one-hot; get argmax
        if hasattr(target_col, 'shape') and target_col.shape[1] > 1:
            labels = np.argmax(target_col[:, :-1], axis=1)
        else:
            labels = target_col[:, 0]
        labels = np.asarray(labels)
        unique = np.unique(labels)
        for uid in unique:
            groups[int(uid)] = np.where(labels == uid)[0]
    else:
        raise ValueError('Unknown prototype_by: ' + str(prototype_by))

    prototype_indices = []
    neighbors = []

    # For each group, pick medoids (actual samples minimizing sum distances)
    for gid, ids in groups.items():
        if len(ids) == 0:
            continue

        # Warning for small groups
        if len(ids) < n_prototypes_per_group:
            print(f"\u26A0 Warning: Group {gid} has only {len(ids)} samples but n_prototypes_per_group={n_prototypes_per_group}")
            print(f"           Will select all {len(ids)} samples as prototypes for this group")

        if len(ids) < k_neighbors:
            print(f"\u26A0 Warning: Group {gid} has only {len(ids)} samples but k_neighbors={k_neighbors}")
            print(f"           Neighbors will be limited to {len(ids)} for prototypes in this group")

        A = X_scaled[ids]
        # compute pairwise squared distances
        dif = A[:, None, :] - A[None, :, :]
        d2 = np.sum(dif * dif, axis=2)
        sumd = d2.sum(axis=1)
        # pick the index(s) of smallest sumd
        order = np.argsort(sumd)
        chosen = order[:min(n_prototypes_per_group, len(ids))]

        for ch in chosen:
            proto_idx = int(ids[int(ch)])
            prototype_indices.append(proto_idx)

            # CRITICAL FIX: Find neighbors within the same group
            proto_vec = X_scaled[proto_idx][None, :]
            group_vecs = X_scaled[ids]
            group_d2 = np.sum((group_vecs - proto_vec) ** 2, axis=1)

            # Get k nearest neighbors within the group (handle small groups)
            k_actual = min(k_neighbors, len(ids))
            local_nn_idx = np.argsort(group_d2)[:k_actual]
            # Map local indices back to global indices
            nn_idx = ids[local_nn_idx]
            neighbors.append(nn_idx)

    data_dict['prototypes'] = {
        'prototype_indices': np.asarray(prototype_indices, dtype=np.int32),
        'neighbors': [np.asarray(n, dtype=np.int32) for n in neighbors],
        'by': prototype_by,
        'k': k_neighbors
    }

    # Enhanced logging with statistics
    neighbor_counts = [len(n) for n in neighbors] if len(neighbors) > 0 else [0]
    group_sizes = [len(ids) for ids in groups.values()] if len(groups) > 0 else [0]
    print(f"\n\u2714 Computed {len(prototype_indices)} prototypes (by={prototype_by})")
    print(f"  Total groups: {len(groups)}")
    print(f"  Neighbors per prototype: min={min(neighbor_counts)}, max={max(neighbor_counts)}, mean={np.mean(neighbor_counts):.1f}")
    print(f"  Group sizes: min={min(group_sizes)}, max={max(group_sizes)}")

    return data_dict


def main():
    parser = argparse.ArgumentParser(
        description='Precompute cluster assignments for NPT dataset'
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data_dict pickle file')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to metadata pickle file (if separate)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for updated data_dict (default: overwrite input)')
    parser.add_argument('--output_dir', type=str, default='./cluster_outputs',
                        help='Directory for visualization and reports')
    parser.add_argument('--cluster_method', type=str, default='kmeans',
                        choices=['kmeans', 'gmm', 'hdbscan', 'agglomerative'],
                        help='Clustering algorithm')
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters (not used for HDBSCAN)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--min_cluster_size', type=int, default=10,
                        help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Minimum samples for HDBSCAN')
    parser.add_argument('--use_pca', action='store_true',
                        help='Apply PCA before clustering')
    parser.add_argument('--pca_components', type=int, default=50,
                        help='Number of PCA components')
    parser.add_argument('--standardize', action='store_true', default=True,
                        help='Standardize features before clustering')
    parser.add_argument('--impute_strategy', type=str, default='mean',
                        choices=['mean', 'median', 'most_frequent'],
                        help='Strategy for imputing missing values')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute clustering quality metrics')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate cluster visualization')
    parser.add_argument('--viz_method', type=str, default='pca',
                        choices=['pca', 'tsne'],
                        help='Visualization method')
    parser.add_argument('--per_cv', action='store_true',
                        help='Compute clusters per CV training indices (will save per-fold assignments)')
    parser.add_argument('--min_viable_size', type=int, default=4,
                        help='Minimum viable cluster size warning threshold')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        data_dict = pickle.load(f)

    if args.metadata_path:
        print(f"Loading metadata from: {args.metadata_path}")
        with open(args.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    else:
        metadata = data_dict.get('metadata', {})
        if not metadata:
            raise ValueError("No metadata found. Specify --metadata_path or include in data_dict")

    class Cfg:
        pass
    cfg = Cfg()
    cfg.cluster_method = args.cluster_method
    cfg.n_clusters = args.n_clusters
    cfg.random_state = args.random_state
    cfg.min_cluster_size = args.min_cluster_size
    cfg.min_samples = args.min_samples
    cfg.use_pca = args.use_pca
    cfg.pca_components = args.pca_components
    cfg.standardize = args.standardize
    cfg.impute_strategy = args.impute_strategy
    cfg.compute_metrics = args.compute_metrics
    cfg.viz_method = args.viz_method
    cfg.output_dir = args.output_dir
    cfg.min_viable_size = args.min_viable_size

    if args.per_cv:
        # Expect data_dict['new_train_val_test_indices'] available
        if 'new_train_val_test_indices' not in data_dict:
            raise ValueError("Per-CV clustering requested but data_dict lacks new_train_val_test_indices")
        # Compute clusters using only training rows for each CV split
        all_assignments = []
        for split_idx, indices in enumerate(data_dict['new_train_val_test_indices']):
            print(f"\nComputing clusters for CV split {split_idx}")
            # Build a temporary data_dict containing only training rows for clustering
            temp = {}
            # Slice data_arrs for rows in indices (handle torch tensors / numpy arrays)
            temp['data_arrs'] = [
                (col[indices] if hasattr(col, 'shape') else col)
                for col in data_dict['data_arrs']
            ]
            # Slice missing_matrix if present
            if 'missing_matrix' in data_dict:
                mm = data_dict['missing_matrix']
                temp['missing_matrix'] = (mm[indices] if hasattr(mm, 'shape') else mm)
            # Slice mask matrices if present (train/val/test masks)
            for m_key in ['train_mask_matrix', 'val_mask_matrix', 'test_mask_matrix']:
                if m_key in data_dict:
                    mat = data_dict[m_key]
                    temp[m_key] = (mat[indices] if hasattr(mat, 'shape') else mat)
            # Provide minimal metadata for compute_clusters_for_npt
            temp_metadata = dict(metadata)
            temp_metadata['N'] = len(indices)
            # Run clustering on the per-split (training) subset
            assigns = compute_clusters_for_npt(temp, temp_metadata, cfg)['cluster_assignments']
            # Map assignments back to full dataset: default -1, then fill for indices
            full_assign = -1 * np.ones(metadata['N'], dtype=np.int32)
            full_assign[indices] = assigns
            all_assignments.append(full_assign)
        # Save per-split assignments
        data_dict['cluster_assignments_per_cv'] = all_assignments
    else:
        data_dict = compute_clusters_for_npt(data_dict, metadata, args)

    if args.visualize and 'cluster_assignments' in data_dict:
        target_cols = set(
            metadata.get('cat_target_cols', []) +
            metadata.get('num_target_cols', [])
        )
        feature_data = []
        for col_idx, col in enumerate(data_dict['data_arrs']):
            if col_idx not in target_cols:
                if getattr(col, 'ndim', 1) == 2 and col.shape[1] >= 1:
                    feature_data.append(col[:, 0])
                else:
                    feature_data.append(col)
        X = np.column_stack([d.cpu().numpy() if hasattr(d, 'cpu') else d for d in feature_data])
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            print("\nGenerating cluster visualization...")
            reducer = PCA(n_components=2, random_state=args.random_state)
            X_2d = reducer.fit_transform(X)
            plt.figure(figsize=(8,6))
            scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=data_dict['cluster_assignments'], cmap='tab20', s=10, alpha=0.6)
            plt.colorbar(scatter, label='Cluster ID')
            out = Path(args.output_dir) / f'clusters_{args.cluster_method}.png'
            plt.savefig(out, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {out}")
            plt.close()
        except Exception as e:
            print(f"Visualization failed: {e}")

    # Save updated data_dict
    output_path = args.output_path if args.output_path else args.data_path
    print(f"\nSaving updated data_dict to: {output_path}")
    # Ensure cluster_assignments dtype
    if 'cluster_assignments' in data_dict:
        data_dict['cluster_assignments'] = np.asarray(data_dict['cluster_assignments'], dtype=np.int32)
    if 'cluster_assignments_per_cv' in data_dict:
        data_dict['cluster_assignments_per_cv'] = [np.asarray(a, dtype=np.int32) for a in data_dict['cluster_assignments_per_cv']]
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)

    print("\n✓ Clustering preprocessing complete!")
    print(f"  Use --exp_batch_cluster_sampling True during training to enable cluster batching")


if __name__ == '__main__':
    main()