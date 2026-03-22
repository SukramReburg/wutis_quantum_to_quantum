from __future__ import annotations

import argparse
import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.common import AnalysisPathManager
from qnn.config import load_project_config
from optimizer.engine import BacktestEngine
from optimizer.reporting import BacktestReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the optimizer backtest using QNN prediction bundles.")
    parser.add_argument("--config", default="config/model_config.yaml", help="Path to model config YAML.")
    parser.add_argument("--returns-pred", default=None, help="Optional returns prediction artifact path.")
    parser.add_argument("--cov-pred", default=None, help="Optional covariance prediction artifact path.")
    parser.add_argument("--no-plots", action="store_true", help="Disable backtest plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = BacktestEngine(model_config_path=args.config)
    project_config = load_project_config(args.config, base_dir=engine.base_dir)
    result = engine.run(returns_path=args.returns_pred, cov_path=args.cov_pred)

    print("=== QNN Portfolio ===")
    for key, value in result.summary.items():
        print(f"{key:>15}: {value:.4f}")
    print("\n=== Equal-Weight Benchmark ===")
    for key, value in result.benchmark_summary.items():
        print(f"{key:>15}: {value:.4f}")

    if not args.no_plots:
        path_manager = AnalysisPathManager(
            engine.base_dir,
            plots_root=project_config["paths"]["plots"],
            reports_root=project_config["paths"].get("reports", "analysis/reports"),
        )
        location = path_manager.optimizer()
        outputs = BacktestReporter(
            plots_dir=location.plots_dir,
            reports_dir=location.reports_dir,
        ).save(result)
        print(f"\nSaved backtest plots to {location.plots_dir}")
        print(f"Saved backtest reports to {location.reports_dir}")
        for key, value in outputs.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
