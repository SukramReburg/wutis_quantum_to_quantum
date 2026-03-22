from __future__ import annotations

import argparse
import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data_report import MarketDataInvestigator
from analysis.model_report import ModelPerformanceInvestigator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all available analysis reports.")
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    args = parser.parse_args()

    data_location = MarketDataInvestigator(
        model_config_path=args.config,
        data_config_path=args.data_config,
    ).save_report()
    print(f"[data] plots: {data_location.plots_dir}")

    investigator = ModelPerformanceInvestigator(
        model_config_path=args.config,
        data_config_path=args.data_config,
    )
    for mode in ["returns", "cov"]:
        try:
            plots_dir, _ = investigator.save_report(mode)
            print(f"[{mode}] plots: {plots_dir}")
        except FileNotFoundError as exc:
            print(f"[{mode}] skipped: {exc}")


if __name__ == "__main__":
    main()
