#!/usr/bin/env python3
"""Extract Splane embeddings for DLPFC sample1.

Requires SPACEL package (Splane module).

Usage:
    python extract_splane_emb.py
"""

import os
import numpy as np
import scanpy as sc
import torch
import warnings
warnings.filterwarnings('ignore')

from SPACEL import Splane

# Fix sklearn compatibility: KMeans 'full' -> 'lloyd' in newer sklearn
# Must patch both the utils module AND base_model's local reference
import SPACEL.Splane.utils as _splane_utils
import SPACEL.Splane.base_model as _splane_bm
_orig_clustering = _splane_utils.clustering
def _patched_clustering(Cluster, feature):
    if hasattr(Cluster, 'algorithm') and Cluster.algorithm == 'full':
        Cluster.algorithm = 'lloyd'
    return _orig_clustering(Cluster, feature)
_splane_utils.clustering = _patched_clustering
_splane_bm.clustering = _patched_clustering


def main():
    data_dir = "./Data/DLPFC_sample1/original_data"
    sample_ids = ["151507", "151508", "151509", "151510"]
    label_key = "original_domain"

    # Load slices
    adata_list = []
    for sid in sample_ids:
        path = os.path.join(data_dir, f"{sid}.h5ad")
        adata = sc.read_h5ad(path)
        # Preprocess
        if 'counts' not in adata.layers:
            adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

        # Add cell type composition from domain labels
        Splane.utils.add_cell_type_composition(adata, celltype_anno=adata.obs[label_key])

        # Need a slice identifier
        adata.obs['slice_id'] = sid

        print(f"  {sid}: {adata.shape[0]} cells, {adata.obs[label_key].nunique()} domains")
        adata_list.append(adata)

    # Get number of clusters from label
    n_clusters = adata_list[0].obs[label_key].nunique()
    print(f"\nn_clusters: {n_clusters}")

    # Init Splane model
    print("\nInitializing Splane...")
    splane_model = Splane.init_model(
        adata_list,
        n_clusters=n_clusters,
        k=2,
        n_neighbors=25,
        gnn_dropout=0.8,      # 0.8 for Visium-like grid data
        latent_dim=16,
        hidden_dims=64,
        use_gpu=False,  # RTX 5080 not supported by old torch in spacel env
        lr=3e-3,
        seed=42,
    )

    # Train
    print("\nTraining Splane...")
    splane_model.train(
        d_l=0.5,            # 0.5 for Visium-like data
        max_epochs=300,
        early_stop_epochs=10,
    )

    # Identify spatial domains (saves to adata.obs['spatial_domain'])
    splane_model.identify_spatial_domain()

    # Extract embeddings manually from trained model
    print("\nExtracting embeddings...")
    splane_model.model_g.load_state_dict(torch.load(splane_model.best_path))
    splane_model.model_g.eval()

    with torch.no_grad():
        encoded, decoded = splane_model.model_g(
            splane_model.graph[0],
            splane_model.graph[1:]
        )

    embeddings = encoded.cpu().detach().numpy()
    print(f"Total embeddings shape: {embeddings.shape}")

    # Split back to per-slice and save
    result = {}
    loc_index = 0
    for i, (adata, sid) in enumerate(zip(adata_list, sample_ids)):
        n_cells = adata.shape[0]
        emb_slice = embeddings[loc_index : loc_index + n_cells]
        loc_index += n_cells
        result[f"emb_{sid}"] = emb_slice
        result[f"domain_{sid}"] = adata.obs.get('spatial_domain', adata.obs[label_key]).values
        print(f"  {sid}: embedding {emb_slice.shape}, norm range [{np.linalg.norm(emb_slice, axis=1).min():.4f}, {np.linalg.norm(emb_slice, axis=1).max():.4f}]")

    # Save
    out_path = "./Data/DLPFC_sample1/splane_embeddings.npz"
    np.savez(out_path, **result)
    print(f"\nSaved to {out_path}")
    print(f"Keys: {list(result.keys())}")


if __name__ == "__main__":
    main()
