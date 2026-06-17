"""Paper metrics for INST-Align experiments.

The public experiments report:

1. OT Accuracy: optimal-transport weighted label agreement.
2. NN Accuracy: bidirectional nearest-neighbour label agreement.
3. Chamfer distance: symmetric geometric point-cloud distance.
4. ARI and NMI: computed in ``run_embedding.py`` and ``run_ablation_insta.py``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import ot
import pandas as pd
import scipy.spatial.distance
from numpy.typing import NDArray
from scipy.spatial import cKDTree


def mapping_accuracy_ot(
    labels1: NDArray,
    labels2: NDArray,
    coords1: NDArray,
    coords2_aligned: NDArray,
    max_samples: int = 10000,
    seed: int = 42,
) -> float:
    """Optimal-transport weighted label accuracy."""
    l1_all = np.asarray(labels1)
    l2_all = np.asarray(labels2)
    c1_all = np.asarray(coords1)
    c2_all = np.asarray(coords2_aligned)
    n1, n2 = len(l1_all), len(l2_all)

    rng = np.random.default_rng(seed)
    if n1 > max_samples or n2 > max_samples:
        idx1 = rng.choice(n1, min(n1, max_samples), replace=False)
        idx2 = rng.choice(n2, min(n2, max_samples), replace=False)
        c1, l1 = c1_all[idx1], l1_all[idx1]
        c2, l2 = c2_all[idx2], l2_all[idx2]
    else:
        c1, l1, c2, l2 = c1_all, l1_all, c2_all, l2_all

    label_match = (l1[:, None] == l2[None, :]).astype(np.float32)
    spatial_dist = scipy.spatial.distance.cdist(c1, c2).astype(np.float32)
    a = np.ones(len(l1), dtype=np.float32) / len(l1)
    b = np.ones(len(l2), dtype=np.float32) / len(l2)
    pi_spatial = ot.emd(a, b, spatial_dist)
    return float(np.sum(pi_spatial * label_match))


def mapping_accuracy_nn_bidi(
    labels1: NDArray,
    labels2: NDArray,
    coords1: NDArray,
    coords2_aligned: NDArray,
) -> float:
    """Bidirectional nearest-neighbour label accuracy."""
    l1 = np.asarray(labels1)
    l2 = np.asarray(labels2)
    c1 = np.asarray(coords1, dtype=np.float64)
    c2 = np.asarray(coords2_aligned, dtype=np.float64)

    tree2 = cKDTree(c2)
    _, idx_fwd = tree2.query(c1, k=1)
    match_fwd = float(np.mean(l1 == l2[idx_fwd]))

    tree1 = cKDTree(c1)
    _, idx_bwd = tree1.query(c2, k=1)
    match_bwd = float(np.mean(l2 == l1[idx_bwd]))

    return (match_fwd + match_bwd) / 2.0


def chamfer_distance(
    coords1: NDArray,
    coords2_aligned: NDArray,
) -> float:
    """Symmetric Chamfer distance between two point clouds."""
    c1 = np.asarray(coords1, dtype=np.float64)
    c2 = np.asarray(coords2_aligned, dtype=np.float64)

    tree1 = cKDTree(c1)
    tree2 = cKDTree(c2)

    d12, _ = tree2.query(c1, k=1)
    d21, _ = tree1.query(c2, k=1)
    return float((d12.mean() + d21.mean()) / 2)


def compute_istbench_metrics(
    slices_list: list,
    slice_names: List[str],
    label_key: str = "original_domain",
) -> pd.DataFrame:
    """Compute OT Acc, NN Acc, and Chamfer for consecutive aligned slices."""
    results = []
    for i in range(len(slices_list) - 1):
        s1, s2 = slices_list[i], slices_list[i + 1]
        c1 = s1.obsm.get("spatial_aligned", s1.obsm["spatial"])
        c2 = s2.obsm.get("spatial_aligned", s2.obsm["spatial"])

        if label_key in s1.obs and label_key in s2.obs:
            labels1 = np.asarray(s1.obs[label_key])
            labels2 = np.asarray(s2.obs[label_key])
            acc_ot = mapping_accuracy_ot(labels1, labels2, c1, c2)
            acc_nn = mapping_accuracy_nn_bidi(labels1, labels2, c1, c2)
        else:
            acc_ot = float("nan")
            acc_nn = float("nan")

        results.append({
            "slice_pair": f"{slice_names[i]}-{slice_names[i + 1]}",
            "Accuracy": acc_ot,
            "Accuracy_NN": acc_nn,
            "Chamfer": chamfer_distance(c1, c2),
        })

    return pd.DataFrame(results)
