from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.common import resolve_path
from analysis.common import (
    AnalysisOutputLocation,
    AnalysisPathManager,
    analysis_roots,
    load_analysis_configs,
    save_json,
)


class MarketDataInvestigator:
    """Loads merged market data and produces stage-level quality/EDA plots."""

    def __init__(
        self,
        base_dir: str | None = None,
        model_config_path: str = "config/model_config.yaml",
        data_config_path: str = "config/data_config.yaml",
    ):
        project_dir, model_config, data_config = load_analysis_configs(
            base_dir=base_dir,
            model_config_path=model_config_path,
            data_config_path=data_config_path,
        )
        self.base_dir = project_dir
        self.model_config = model_config
        self.data_config = data_config
        plots_root, reports_root = analysis_roots(model_config, data_config)
        self.path_manager = AnalysisPathManager(
            project_dir,
            plots_root=plots_root,
            reports_root=reports_root,
        )

    def load_merged_data(self) -> pd.DataFrame:
        raw_dir = resolve_path(self.base_dir, self.data_config["paths"]["raw"])
        merged_path = os.path.join(raw_dir, "merged_data.csv")
        return pd.read_csv(merged_path, index_col="timestamp", parse_dates=True)

    @staticmethod
    def _save_figure(path: str, dpi: int = 180) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()
        return path

    @staticmethod
    def _log_return_columns(frame: pd.DataFrame) -> list[str]:
        return [column for column in frame.columns if column.endswith("_log")]

    def build_summary(self, merged_df: pd.DataFrame) -> dict[str, Any]:
        log_cols = self._log_return_columns(merged_df)
        summary = {
            "n_rows": int(len(merged_df)),
            "n_columns": int(len(merged_df.columns)),
            "n_assets": int(len(log_cols)),
            "assets": [column.replace("_log", "") for column in log_cols],
            "date_start": str(merged_df.index.min()) if len(merged_df) else None,
            "date_end": str(merged_df.index.max()) if len(merged_df) else None,
            "missing_ratio": float(merged_df.isna().mean().mean()),
        }
        return summary

    def save_report(self, run_tag: str | None = None) -> AnalysisOutputLocation:
        merged_df = self.load_merged_data()
        location = self.path_manager.data(run_tag=run_tag)
        log_cols = self._log_return_columns(merged_df)
        if not log_cols:
            raise ValueError("Merged market data does not contain any '*_log' columns.")

        returns = merged_df[log_cols].dropna()
        returns.columns = [column.replace("_log", "") for column in returns.columns]
        top_assets = returns.var().sort_values(ascending=False).head(min(6, returns.shape[1])).index

        plt.figure(figsize=(12, 10))
        sns.heatmap(returns.cov(), cmap="coolwarm", center=0.0)
        plt.title("Log-Return Covariance Heatmap")
        self._save_figure(os.path.join(location.plots_dir, "covariance_heatmap.png"))

        plt.figure(figsize=(12, 10))
        sns.heatmap(returns.corr(), cmap="coolwarm", center=0.0, vmin=-1.0, vmax=1.0)
        plt.title("Log-Return Correlation Heatmap")
        self._save_figure(os.path.join(location.plots_dir, "correlation_heatmap.png"))

        plt.figure(figsize=(10, 5))
        plt.hist(returns.values.reshape(-1), bins=80, alpha=0.85)
        plt.title("Distribution of Log Returns")
        plt.xlabel("Log Return")
        plt.ylabel("Count")
        self._save_figure(os.path.join(location.plots_dir, "returns_distribution.png"))

        plt.figure(figsize=(12, 6))
        for asset in top_assets:
            plt.plot((np.exp(returns[asset].cumsum()) - 1.0), label=asset)
        plt.title("Cumulative Log-Return Paths")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._save_figure(os.path.join(location.plots_dir, "cumulative_paths.png"))

        rolling_vol = returns[top_assets].rolling(21).std() * np.sqrt(252)
        plt.figure(figsize=(12, 6))
        for asset in top_assets:
            plt.plot(rolling_vol.index, rolling_vol[asset], label=asset)
        plt.title("21-Day Rolling Volatility")
        plt.xlabel("Date")
        plt.ylabel("Annualized Volatility")
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._save_figure(os.path.join(location.plots_dir, "rolling_volatility.png"))

        available_counts = merged_df.notna().sum().sort_values(ascending=False)
        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(available_counts)), available_counts.values)
        plt.xticks(np.arange(len(available_counts)), available_counts.index, rotation=90)
        plt.title("Non-Null Observations Per Feature")
        plt.ylabel("Count")
        self._save_figure(os.path.join(location.plots_dir, "feature_availability.png"))

        summary = self.build_summary(merged_df)
        summary["top_assets_by_variance"] = list(top_assets)
        save_json(os.path.join(location.reports_dir, "summary.json"), summary)
        return location


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exploratory plots for merged market data.")
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    args = parser.parse_args()
    investigator = MarketDataInvestigator(
        model_config_path=args.config,
        data_config_path=args.data_config,
    )
    location = investigator.save_report()
    print(f"Saved data investigation plots to {location.plots_dir}")


if __name__ == "__main__":
    main()
