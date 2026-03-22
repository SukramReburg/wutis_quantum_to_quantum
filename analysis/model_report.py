from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer.bundle import PredictionBundle
from optimizer.reconstruct_cov import rebuild_covariance
from data.common import resolve_path
from qnn.plotting import QNNPlotter
from analysis.common import AnalysisPathManager, analysis_roots, load_analysis_configs, save_json


class ModelPerformanceInvestigator:
    """Loads prediction bundles and trainer metrics to build analysis-focused model reports."""

    def __init__(
        self,
        base_dir: str | None = None,
        model_config_path: str = "config/model_config.yaml",
        data_config_path: str = "config/data_config.yaml",
    ):
        project_dir, model_config, _ = load_analysis_configs(
            base_dir=base_dir,
            model_config_path=model_config_path,
            data_config_path=data_config_path,
        )
        self.base_dir = project_dir
        self.model_config = model_config
        plots_root, reports_root = analysis_roots(model_config, {})
        self.path_manager = AnalysisPathManager(
            project_dir,
            plots_root=plots_root,
            reports_root=reports_root,
        )
        self.plotter = QNNPlotter(dpi=180, rolling_window=5, max_assets=6)

    def _latest_result_root(self, mode: str) -> str:
        results_root = resolve_path(self.base_dir, self.model_config["paths"]["results"])
        return os.path.join(results_root, "latest", mode)

    def _load_metrics(self, mode: str) -> dict[str, Any]:
        metrics_path = os.path.join(self._latest_result_root(mode), "metrics.npz")
        if not os.path.exists(metrics_path):
            return {}
        data = np.load(metrics_path, allow_pickle=True)
        return {key: data[key] for key in data.files if key != "meta"}

    def _load_summary(self, mode: str) -> dict[str, Any]:
        summary_path = os.path.join(self._latest_result_root(mode), "summary.json")
        if not os.path.exists(summary_path):
            return {}
        with open(summary_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def default_prediction_path(self, mode: str) -> str:
        latest_root = self._latest_result_root(mode)
        prediction_path = os.path.join(latest_root, "predictions.npz")
        if os.path.exists(prediction_path):
            return prediction_path
        raise FileNotFoundError(f"No latest prediction artifact found for mode '{mode}'.")

    @staticmethod
    def _save(path: str, dpi: int = 180) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()
        return path

    def save_report(self, mode: str, prediction_path: str | None = None) -> tuple[str, str]:
        bundle = PredictionBundle.load(
            prediction_path or self.default_prediction_path(mode),
            kind="returns" if mode == "returns" else "cov",
        )
        metrics = self._load_metrics(mode)
        summary = self._load_summary(mode)
        location = self.path_manager.model(mode)

        prediction_payload = {
            "Y_pred_test": bundle.y_pred,
            "Y_true_test": bundle.y_true,
            "Y_pred_std_test": bundle.y_pred_std if bundle.y_pred_std is not None else np.zeros_like(bundle.y_pred),
            "asset_symbols": np.asarray(bundle.asset_symbols, dtype=str),
            "sample_dates": np.asarray(bundle.sample_dates, dtype=str),
        }
        if metrics:
            self.plotter.save_all(
                plots_dir=location.plots_dir,
                mode=mode,
                metrics=metrics,
                predictions=prediction_payload,
                summary=summary or {"n_assets": len(bundle.asset_symbols)},
            )

        if mode == "returns":
            dates = bundle.sample_index if len(bundle.sample_index) else pd.RangeIndex(bundle.y_pred.shape[0])
            top_assets = bundle.asset_symbols[: min(4, len(bundle.asset_symbols))]
            fig, axes = plt.subplots(len(top_assets), 1, figsize=(12, 3 * len(top_assets)), sharex=True)
            if len(top_assets) == 1:
                axes = [axes]
            for axis, asset in zip(axes, top_assets):
                idx = bundle.asset_symbols.index(asset)
                axis.plot(dates, bundle.y_true[:, idx], label="Actual")
                axis.plot(dates, bundle.y_pred[:, idx], label="Predicted", linestyle="--")
                axis.set_title(f"{asset} Return Prediction")
                axis.grid(True, alpha=0.3)
                axis.legend()
            self._save(os.path.join(location.plots_dir, "returns_timeseries_panel.png"))

            cumulative_true = np.exp(np.cumsum(bundle.y_true, axis=0)) - 1.0
            cumulative_pred = np.exp(np.cumsum(bundle.y_pred, axis=0)) - 1.0
            plt.figure(figsize=(12, 6))
            for asset in top_assets:
                idx = bundle.asset_symbols.index(asset)
                plt.plot(dates, cumulative_true[:, idx], label=f"{asset} actual")
                plt.plot(dates, cumulative_pred[:, idx], linestyle="--", label=f"{asset} pred")
            plt.title("Cumulative Return Comparison")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.grid(True, alpha=0.3)
            plt.legend(ncol=2)
            self._save(os.path.join(location.plots_dir, "returns_cumulative_comparison.png"))
        else:
            n_assets = len(bundle.asset_symbols)
            if bundle.y_true.shape[0] > 0:
                true_cov = rebuild_covariance(bundle.y_true[0], n_assets)
                pred_cov = rebuild_covariance(bundle.y_pred[0], n_assets)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].imshow(true_cov, cmap="coolwarm")
                axes[0].set_title("Actual Covariance (sample 0)")
                axes[1].imshow(pred_cov, cmap="coolwarm")
                axes[1].set_title("Predicted Covariance (sample 0)")
                self._save(os.path.join(location.plots_dir, "covariance_sample_heatmaps.png"))

                diff = []
                for idx in range(bundle.y_true.shape[0]):
                    diff.append(
                        np.linalg.norm(
                            rebuild_covariance(bundle.y_true[idx], n_assets)
                            - rebuild_covariance(bundle.y_pred[idx], n_assets),
                            ord="fro",
                        )
                    )
                plt.figure(figsize=(10, 4.5))
                plt.plot(bundle.sample_index if len(bundle.sample_index) else np.arange(len(diff)), diff, marker="o")
                plt.title("Covariance Frobenius Error Over Time")
                plt.xlabel("Sample")
                plt.ylabel("Frobenius Error")
                plt.grid(True, alpha=0.3)
                self._save(os.path.join(location.plots_dir, "covariance_frobenius_over_time.png"))

        report_summary = {
            "mode": mode,
            "prediction_path": bundle.path,
            "runtime_profile": bundle.runtime_profile,
            "loss_name": bundle.loss_name,
            "best_epoch": bundle.best_epoch,
            "n_samples": int(bundle.y_pred.shape[0]),
            "n_outputs": int(bundle.y_pred.shape[1]),
            "asset_symbols": bundle.asset_symbols,
        }
        if summary:
            report_summary["training_summary"] = summary
        save_json(os.path.join(location.reports_dir, "summary.json"), report_summary)
        return location.plots_dir, location.reports_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model-performance plots from latest prediction artifacts.")
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    parser.add_argument("--mode", default="both", choices=["returns", "cov", "both"])
    args = parser.parse_args()
    investigator = ModelPerformanceInvestigator(
        model_config_path=args.config,
        data_config_path=args.data_config,
    )
    modes = ["returns", "cov"] if args.mode == "both" else [args.mode]
    for mode in modes:
        plots_dir, reports_dir = investigator.save_report(mode)
        print(f"[{mode}] plots: {plots_dir}")
        print(f"[{mode}] reports: {reports_dir}")


if __name__ == "__main__":
    main()
