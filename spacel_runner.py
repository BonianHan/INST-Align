#!/usr/bin/env python3
"""Standalone SPACEL runner — designed to be called from the spacel conda env.

Usage::

    conda run -n spacel python spacel_runner.py \\
        --slice1 /tmp/spacel_s1.h5ad \\
        --slice2 /tmp/spacel_s2.h5ad \\
        --label_key original_domain \\
        --output /tmp/spacel_result.npz

Outputs an .npz file with keys:
    coords1, coords2  — aligned coordinates (N1,2) and (N2,2)
    elapsed            — wall-clock time in seconds
"""

import argparse
import time

import anndata as ad
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Run SPACEL alignment")
    parser.add_argument("--slice1", required=True, help="Path to slice1 h5ad")
    parser.add_argument("--slice2", required=True, help="Path to slice2 h5ad")
    parser.add_argument("--label_key", default="original_domain")
    parser.add_argument("--output", required=True, help="Path to output .npz")
    args = parser.parse_args()

    from SPACEL import Scube

    s1 = ad.read_h5ad(args.slice1)
    s2 = ad.read_h5ad(args.slice2)

    start = time.time()
    Scube.align([s1, s2], cluster_key=args.label_key, n_neighbors=15, n_threads=10, p=2)
    elapsed = time.time() - start

    coords1 = s1.obsm["spatial_aligned"]
    coords2 = s2.obsm["spatial_aligned"]
    if hasattr(coords1, "to_numpy"):
        coords1 = coords1.to_numpy()
    if hasattr(coords2, "to_numpy"):
        coords2 = coords2.to_numpy()

    np.savez(args.output, coords1=coords1, coords2=coords2, elapsed=np.array(elapsed))
    print(f"SPACEL done: {elapsed:.2f}s, saved to {args.output}")


if __name__ == "__main__":
    main()
