from __future__ import annotations

import argparse
import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnn.study import QNNStudyRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna tuning for QNN experiments.")
    parser.add_argument("--config", default="config/model_config.yaml", help="Path to model config YAML.")
    parser.add_argument(
        "--mode",
        default="returns",
        choices=["returns", "cov"],
        help="Target mode to optimize.",
    )
    parser.add_argument("--study-name", default=None, help="Optional study name override.")
    parser.add_argument("--n-trials", type=int, default=None, help="Optional trial count override.")
    parser.add_argument(
        "--runtime-profile",
        default=None,
        choices=["realism", "balanced", "fast_exact", None],
        help="Optional runtime profile override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = QNNStudyRunner(config_path=args.config).run(
        mode=args.mode,
        study_name=args.study_name,
        n_trials=args.n_trials,
        runtime_profile=args.runtime_profile,
    )
    print(f"Study '{study.study_name}' best value: {study.best_value:.6f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
