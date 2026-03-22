from __future__ import annotations

import argparse
import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnn.config import load_experiment_config
from qnn.trainer import QNNTrainer, train_qnn_from_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QNN models from processed dataset bundles.")
    parser.add_argument("--config", default="config/model_config.yaml", help="Path to model config YAML.")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["returns", "cov", "both"],
        help="Which target to train.",
    )
    parser.add_argument(
        "--runtime-profile",
        default=None,
        choices=["realism", "balanced", "fast_exact", None],
        help="Optional runtime profile override.",
    )
    parser.add_argument("--run-tag", default=None, help="Optional explicit run tag.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = ["returns", "cov"] if args.mode == "both" else [args.mode]
    for mode in modes:
        overrides = {}
        if args.runtime_profile is not None:
            overrides["runtime"] = {"profile": args.runtime_profile}
        experiment = load_experiment_config(mode=mode, config_path=args.config, overrides=overrides)
        result = QNNTrainer(experiment).train(no_plots=args.no_plots, run_tag=args.run_tag)
        print(
            f"[{mode}] run_tag={result.artifacts.run_tag} "
            f"rmse={result.summary['test_metrics']['rmse']:.6f} "
            f"mae={result.summary['test_metrics']['mae']:.6f}"
        )


if __name__ == "__main__":
    main()
