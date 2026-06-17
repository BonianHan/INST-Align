"""insta: Spatial transcriptomics alignment via implicit neural representations.

Quick start::

    from insta import PipelineConfig, run
    aligned, metrics = run(PipelineConfig(dataset="STARMap"))

Spatial alignment experiment::

    from insta.spatial_alignment import run_spatial_alignment_all, print_summary
"""

from insta.config import (
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
from insta.model import (
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
from insta.loss import (
    compute_P_matrix,
    dice_loss,
    jacobian_reg,
    matching_loss_joint,
    recon_loss,
    recon_loss_from_emb,
)
from insta.trainer import TrainResult, apply_model, apply_model_with_inr, train
from insta.metrics import (
    chamfer_distance,
    compute_istbench_metrics,
    mapping_accuracy_nn_bidi,
    mapping_accuracy_ot,
)
from insta.utils import (
    coords_to_pi,
    denormalize_coordinates,
    detect_grid_spacing,
    mapping_accuracy_paste,
    normalize_coordinates,
    sparse_P_to_dense_pi,
)
from insta.pipeline import (
    align_pair, run, save_inr_results, train_dataset,
)
from insta.spatial_alignment import (
    plot_comparison,
    print_summary,
    run_spatial_alignment_all,
    run_spatial_alignment_dataset,
)

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
    "mapping_accuracy_nn_bidi",
    "mapping_accuracy_ot",
    "compute_istbench_metrics",
    "chamfer_distance",
    # Utils
    "normalize_coordinates",
    "denormalize_coordinates",
    "detect_grid_spacing",
    "mapping_accuracy_paste",
    "coords_to_pi",
    "sparse_P_to_dense_pi",
    # Pipeline
    "run",
    "align_pair",
    "save_inr_results",
    "train_dataset",
    # Spatial alignment experiment
    "run_spatial_alignment_all",
    "run_spatial_alignment_dataset",
    "print_summary",
    "plot_comparison",
]
