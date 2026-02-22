"""inr_align: Spatial transcriptomics alignment via implicit neural representations.

Quick start::

    from inr_align import PipelineConfig, run
    aligned, metrics = run(PipelineConfig(dataset="STARMap"))

Benchmark::

    from inr_align.benchmark import benchmark_all, print_summary
"""

from inr_align.config import (
    DLPFC_SAMPLE_GROUPS,
    SLICE_ORDER,
    ICPConfig,
    JointConfig,
    MatcherConfig,
    ModelConfig,
    PipelineConfig,
    TrainConfig,
    add_pipeline_args,
    config_from_args,
    print_config,
)
from inr_align.model import (
    DeformationNet,
    ExprEncoder,
    ExprINR,
    ExprDecoder,
    WindowedPositionalEncoding,
    UnifiedCostMatcher,
    adaptive_icp,
    build_joint_models,
    build_knn_graph,
)
from inr_align.loss import (
    compute_P_matrix,
    dice_loss,
    jacobian_reg,
    matching_loss_joint,
    recon_loss,
    recon_loss_from_emb,
)
from inr_align.engine import TrainResult, apply_model, apply_model_with_inr, train
from inr_align.metrics import (
    calculate_clc,
    chamfer_distance,
    compute_istbench_metrics,
    compute_lisi,
    compute_silhouette,
    coords_to_pi,
    evaluate_alignment,
    mapping_accuracy_nn,
    mapping_accuracy_ot,
    mapping_accuracy_paste,
    sparse_P_to_dense_pi,
)
from inr_align.utils import (
    denormalize_coordinates,
    detect_grid_spacing,
    normalize_coordinates,
)
from inr_align.run import (
    align_pair, run, save_inr_results, train_dataset,
)
from inr_align.benchmark import benchmark_all, benchmark_dataset, print_summary, plot_comparison

__version__ = "0.4.0"

__all__ = [
    # Config
    "PipelineConfig",
    "ModelConfig",
    "MatcherConfig",
    "TrainConfig",
    "ICPConfig",
    "JointConfig",
    "SLICE_ORDER",
    "DLPFC_SAMPLE_GROUPS",
    "add_pipeline_args",
    "config_from_args",
    "print_config",
    # Model
    "DeformationNet",
    "ExprEncoder",
    "ExprINR",
    "ExprDecoder",
    "WindowedPositionalEncoding",
    "UnifiedCostMatcher",
    "adaptive_icp",
    "build_joint_models",
    "build_knn_graph",
    # Loss
    "compute_P_matrix",
    "dice_loss",
    "jacobian_reg",
    "matching_loss_joint",
    "recon_loss",
    "recon_loss_from_emb",
    # Engine
    "train",
    "apply_model",
    "apply_model_with_inr",
    "TrainResult",
    # Metrics
    "mapping_accuracy_nn",
    "mapping_accuracy_ot",
    "mapping_accuracy_paste",
    "calculate_clc",
    "evaluate_alignment",
    "coords_to_pi",
    "sparse_P_to_dense_pi",
    "compute_istbench_metrics",
    "compute_lisi",
    "compute_silhouette",
    "chamfer_distance",
    # Utils
    "normalize_coordinates",
    "denormalize_coordinates",
    "detect_grid_spacing",
    # Pipeline
    "run",
    "align_pair",
    "save_inr_results",
    "train_dataset",
    # Benchmark
    "benchmark_all",
    "benchmark_dataset",
    "print_summary",
    "plot_comparison",
]
