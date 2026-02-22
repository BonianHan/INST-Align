#!/usr/bin/env bash
# =============================================================================
# INR-Align Full Benchmark Script
# =============================================================================
# Runs all alignment methods on all datasets and computes three metrics:
#   - OT Accuracy   (PASTE-style transport-plan weighted label match)
#   - NN Accuracy   (bidirectional nearest-neighbour, iSTBench-style)
#   - Ratio         (many-to-one collapse measure, lower=better)
#
# Methods compared:
#   1. No-align       -- raw coordinates, no alignment
#   2. PASTE          -- Fused Gromov-Wasserstein (alpha=0.1)
#   3. STalign        -- LDDMM diffeomorphic registration
#   4. Spateo         -- morpho_align rigid + nonrigid
#   5. Ours           -- adaptive_icp + INR deformation (rigid / spatial)
#
# Usage:
#   bash run_benchmark.sh                              # Full benchmark
#   bash run_benchmark.sh --no_paste                   # Skip PASTE
#   bash run_benchmark.sh --no_spateo                  # Skip Spateo
#   bash run_benchmark.sh --no_stalign                  # Skip STalign
#   bash run_benchmark.sh --quick                      # Ours only (skip all baselines)
#
# Results saved to:
#   benchmark_results.csv   -- per-pair (OT Acc, NN Acc, Ratio)
#   benchmark_summary.csv   -- per-dataset mean +/- std for all 3 metrics
#   benchmark_results.png   -- box plot
# =============================================================================

set -euo pipefail

# --- Conda environment ---
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spateo

cd "$(dirname "$0")"

# --- Parse convenience flags ---
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --quick)
            EXTRA_ARGS+=(--no_paste --no_spateo --no_stalign)
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

# --- Datasets ---
DATASETS=(
    DLPFC
    STARMap
    BaristaSeq
    MERFISH_Brain_S3
    MERFISH_Brain_S8
    MERFISH_Brain_S12
)

# =============================================================================
# Current best configuration (2026-02-08)
#
# Key settings:
#   - ExprField enabled (canonical expression field + batch correction)
#   - lam_canonical = 0.005 (ramp from 0 -> full weight during warmup)
#   - DeformationNet: 6-layer MLP, 128 hidden, 6 PE freqs
#   - Matcher: softmax (sinkhorn_iters=0), adaptive tau [0.01, 0.5]
#   - Training: 200 epochs, batch 2500 (fixed), topK=64
#   - Per-dataset overrides (in run_test_acc.py DATASET_OVERRIDES):
#       DLPFC:    lam_jacobian=0.1,   warmup=0.3
#       STARMap:  lam_jacobian=0.01,  warmup=0.4
#       BaristaSeq: lam_jacobian=0.01, warmup=0.4
#       MERFISH:  lam_jacobian=0.001, warmup=0.4
#   - ICP: icp_only mode (no rotation search)
# =============================================================================

python -u run_test_acc.py \
    --datasets "${DATASETS[@]}" \
    --use_expr_field \
    \
    --model_hidden 128 \
    --model_layers 6 \
    --model_n_freqs 6 \
    --model_max_freq_log2 5 \
    \
    --matcher_tau_init 0.1 \
    --matcher_tau_min 0.01 \
    --matcher_tau_max 0.5 \
    --matcher_lambda_feat 1.0 \
    --matcher_ema_decay 0.9 \
    \
    --train_epochs 200 \
    --train_batch_size 2500 \
    --train_topk 64 \
    --train_lr 0.001 \
    --train_weight_rev 1.0 \
    --train_scheduler_patience 30 \
    --train_scheduler_factor 0.5 \
    --train_scheduler_min_lr 1e-6 \
    \
    --icp_mode icp_only \
    --icp_angle_step 15 \
    --icp_icp_max_iter 100 \
    \
    --expr_hidden 256 \
    --expr_n_freqs 6 \
    --expr_max_freq_log2 5 \
    --expr_batch_emb_dim 16 \
    --expr_latent_dim 32 \
    --expr_norm_method per_gene \
    --expr_n_hvg 200 \
    \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Benchmark complete ==="
echo "Results: benchmark_results.csv, benchmark_summary.csv, benchmark_results.png"
