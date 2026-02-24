#!/usr/bin/env python3
"""Run Seurat CCA integration on MouseEmbryo data and evaluate ARI/NMI."""
import sys
import warnings
import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from benchmark_insta import run_seurat_embedding, compute_ari_nmi

label_key = "cellbin_SpatialDomain"

# Load raw slices
embryo_raw = []
for si in [1, 2]:
    path = f"./Data/MouseEmbyro/slices{si}.h5ad"
    a = sc.read_h5ad(path)
    print(f"Loaded MouseEmbryo slice{si}: {a.shape}")
    embryo_raw.append(a)

# Count clusters
all_labels = set()
for s in embryo_raw:
    all_labels.update(np.asarray(s.obs[label_key]))
n_clusters = len([l for l in all_labels if str(l).strip() not in ("", "nan")])
print(f"n_clusters = {n_clusters}")

# Run Seurat
print("\nRunning Seurat (CCA integration) ...")
sr_embs = run_seurat_embedding([s.copy() for s in embryo_raw], label_key=label_key)

# Evaluate
sample_ids = ["embryo_s1", "embryo_s2"]
for clust_method in ["mclust", "leiden_spatial"]:
    for i, emb in enumerate(sr_embs):
        labels = np.asarray(embryo_raw[i].obs[label_key])
        coords = embryo_raw[i].obsm["spatial"]
        refine_r = None
        if clust_method == "mclust":
            from sklearn.neighbors import NearestNeighbors
            _nn = NearestNeighbors(n_neighbors=2).fit(coords)
            _d, _ = _nn.kneighbors(coords)
            refine_r = float(np.median(_d[:, 1])) * 3.0
        ari, nmi = compute_ari_nmi(emb, labels, n_clusters,
                                   method=clust_method,
                                   spatial_coords=coords,
                                   refine_radius=refine_r)
        suffix = f"_{clust_method}" if clust_method != "mclust" else ""
        print(f"  Seurat{suffix} | {sample_ids[i]} | ARI={ari:.4f}  NMI={nmi:.4f}")
