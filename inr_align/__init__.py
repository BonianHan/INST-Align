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
    ExprFieldConfig,
    ExpressionINRConfig,
    ICPConfig,
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
    ExprField,
    ExpressionINR,
    NerfiesPositionalEncoding,
    UnifiedCostMatcher,
    adaptive_icp,
    normalize_expression,
    pretrain_expression_inr,
    pretrain_expr_field,
)
from inr_align.loss import (
    canonical_consistency_loss,
    compute_P_matrix,
    expression_reconstruction_loss,
    jacobian_reg,
)
from inr_align.train import TrainResult, apply_model, train
from inr_align.metrics import (
    calculate_clc,
    compute_istbench_metrics,
    coords_to_pi,
    evaluate_alignment,
    mapping_accuracy_nn,
    mapping_accuracy_ot,
    mapping_accuracy_paste,
    sparse_P_to_dense_pi,
)
from inr_align.run import (
    align_pair, run, save_inr_results, train_dataset,
    PretrainedExprField, PretrainedExprINR, pretrain_ref_expression_inr,
    pretrain_expr_field_pipeline,
)
from inr_align.benchmark import benchmark_all, benchmark_dataset, print_summary, plot_comparison

__version__ = "0.1.0"

__all__ = [
    # Config
    "PipelineConfig",
    "ModelConfig",
    "MatcherConfig",
    "TrainConfig",
    "ICPConfig",
    "ExprFieldConfig",
    "ExpressionINRConfig",
    "SLICE_ORDER",
    "DLPFC_SAMPLE_GROUPS",
    "add_pipeline_args",
    "config_from_args",
    "print_config",
    # Model
    "DeformationNet",
    "ExprField",
    "ExpressionINR",
    "NerfiesPositionalEncoding",
    "UnifiedCostMatcher",
    "adaptive_icp",
    "normalize_expression",
    "pretrain_expression_inr",
    "pretrain_expr_field",
    # Loss
    "compute_P_matrix",
    "jacobian_reg",
    "expression_reconstruction_loss",
    "canonical_consistency_loss",
    # Train
    "train",
    "apply_model",
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
    # Pipeline
    "run",
    "align_pair",
    "save_inr_results",
    "train_dataset",
    "PretrainedExprField",
    "PretrainedExprINR",
    "pretrain_ref_expression_inr",
    "pretrain_expr_field_pipeline",
    # Benchmark
    "benchmark_all",
    "benchmark_dataset",
    "print_summary",
    "plot_comparison",
]
