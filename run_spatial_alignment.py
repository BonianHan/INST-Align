#!/usr/bin/env python
"""Run the paper spatial alignment experiment.

This entry point corresponds to Table 1 in the paper. It can run INST-Align,
the baseline methods, or both while saving separate result tables.
"""

from __future__ import annotations

import argparse

from insta.config import add_pipeline_args, config_from_args, print_config
from run_baseline import main as run_baselines
from run_insta import main as run_insta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spatial alignment experiment")
    parser.add_argument(
        "--methods",
        choices=["all", "insta", "baseline"],
        default="all",
        help="Which part of the spatial alignment experiment to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run. Default: paper spatial alignment datasets.",
    )
    parser.add_argument(
        "--output",
        default="spatial_alignment",
        help="Output file prefix.",
    )
    parser.add_argument("--no_paste", action="store_true", help="Skip PASTE")
    parser.add_argument("--no_paste2", action="store_true", help="Skip PASTE2")
    parser.add_argument("--no_gpsa", action="store_true", help="Skip GPSA")
    parser.add_argument("--no_spateo", action="store_true", help="Skip Spateo")
    parser.add_argument("--no_stalign", action="store_true", help="Skip STalign")
    add_pipeline_args(parser)

    args = parser.parse_args()
    cfg = config_from_args(args)
    print_config(cfg)

    if args.methods in ("all", "baseline"):
        run_baselines(
            config=cfg,
            datasets=args.datasets,
            run_paste=not args.no_paste,
            run_paste2=not args.no_paste2,
            run_gpsa=not args.no_gpsa,
            run_spateo=not args.no_spateo,
            run_stalign=not args.no_stalign,
            output_name=f"{args.output}_baseline",
        )

    if args.methods in ("all", "insta"):
        run_insta(
            config=cfg,
            datasets=args.datasets,
            output_name=f"{args.output}_insta",
        )


if __name__ == "__main__":
    main()
