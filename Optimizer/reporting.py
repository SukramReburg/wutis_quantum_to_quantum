from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .KPIs import drawdown, rolling_sharpe, rolling_volatility
from .engine import BacktestResult


class BacktestReporter:
    """Writes optimizer diagnostics into separate plot and report folders."""

    def __init__(self, plots_dir: str, reports_dir: str, dpi: int = 180):
        self.plots_dir = plots_dir
        self.reports_dir = reports_dir
        self.dpi = dpi
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

    def _save(self, filename: str) -> str:
        path = os.path.join(self.plots_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return path

    def _summary_payload(self, result: BacktestResult) -> dict[str, object]:
        return {
            "portfolio": result.summary,
            "benchmark": result.benchmark_summary,
            "diagnostics_mean": result.diagnostics.mean(numeric_only=True).to_dict(),
            "n_rebalance_periods": int(len(result.weights)),
            "n_daily_observations": int(len(result.portfolio_simple_returns)),
        }

    def save(self, result: BacktestResult) -> dict[str, str]:
        with open(os.path.join(self.reports_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(self._summary_payload(result), handle, indent=2, sort_keys=True)
        result.weights.to_csv(os.path.join(self.reports_dir, "weights.csv"))
        result.diagnostics.to_csv(os.path.join(self.reports_dir, "diagnostics.csv"))

        outputs = {}
        outputs["summary"] = os.path.join(self.reports_dir, "summary.json")

        qnn_equity = (1.0 + result.portfolio_simple_returns).cumprod()
        benchmark_equity = (1.0 + result.benchmark_simple_returns).cumprod()

        plt.figure(figsize=(10, 5))
        plt.plot(qnn_equity, label="QNN Portfolio")
        plt.plot(benchmark_equity, label="Equal Weight", linestyle="--")
        plt.title("Cumulative Performance")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        outputs["equity"] = self._save("equity_curve.png")

        plt.figure(figsize=(10, 5))
        plt.plot(drawdown(result.portfolio_log_returns.values), label="QNN Drawdown")
        plt.plot(drawdown(result.benchmark_log_returns.values), label="Benchmark Drawdown", linestyle="--")
        plt.title("Drawdown Comparison")
        plt.xlabel("Observation")
        plt.ylabel("Drawdown")
        plt.grid(True, alpha=0.3)
        plt.legend()
        outputs["drawdown"] = self._save("drawdown.png")

        top_assets = result.weights.mean().sort_values(ascending=False).head(5).index
        plt.figure(figsize=(10, 5))
        for asset in top_assets:
            plt.plot(result.weights.index, result.weights[asset], label=asset)
        plt.title("Top Asset Weights")
        plt.xlabel("Rebalance Date")
        plt.ylabel("Weight")
        plt.grid(True, alpha=0.3)
        plt.legend()
        outputs["top_weights"] = self._save("top_asset_weights.png")

        plt.figure(figsize=(10, 5))
        plt.imshow(result.weights.T, aspect="auto", cmap="viridis")
        plt.yticks(np.arange(len(result.weights.columns)), result.weights.columns)
        plt.xticks(np.arange(len(result.weights.index)), result.weights.index.strftime("%Y-%m-%d"), rotation=90)
        plt.colorbar(label="Weight")
        plt.title("Weight Heatmap")
        outputs["weight_heatmap"] = self._save("weight_heatmap.png")

        rolling_sharpe_qnn = rolling_sharpe(result.portfolio_log_returns)
        rolling_vol_qnn = rolling_volatility(result.portfolio_log_returns)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(rolling_sharpe_qnn.index, rolling_sharpe_qnn.values)
        axes[0].set_title("Rolling Sharpe")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(rolling_vol_qnn.index, rolling_vol_qnn.values)
        axes[1].set_title("Rolling Volatility")
        axes[1].grid(True, alpha=0.3)
        outputs["rolling_risk"] = self._save("rolling_risk_metrics.png")

        if not result.diagnostics.empty:
            plt.figure(figsize=(10, 4.5))
            plt.plot(result.diagnostics.index, result.diagnostics["turnover"], marker="o")
            plt.title("Turnover By Rebalance")
            plt.xlabel("Rebalance Date")
            plt.ylabel("Turnover")
            plt.grid(True, alpha=0.3)
            outputs["turnover"] = self._save("turnover.png")

            plt.figure(figsize=(10, 4.5))
            plt.scatter(
                result.diagnostics["predicted_vol"],
                result.diagnostics["realized_vol"],
                c=result.diagnostics["turnover"],
                cmap="viridis",
            )
            plt.xlabel("Predicted Volatility")
            plt.ylabel("Realized Volatility")
            plt.title("Predicted vs Realized Volatility")
            plt.grid(True, alpha=0.3)
            outputs["risk_scatter"] = self._save("predicted_vs_realized_volatility.png")

            plt.figure(figsize=(10, 4.5))
            plt.plot(
                result.diagnostics.index,
                result.diagnostics["covariance_frobenius_error"],
                marker="o",
            )
            plt.title("Covariance Reconstruction Error")
            plt.xlabel("Rebalance Date")
            plt.ylabel("Frobenius Error")
            plt.grid(True, alpha=0.3)
            outputs["covariance_error"] = self._save("covariance_error.png")

        return outputs
