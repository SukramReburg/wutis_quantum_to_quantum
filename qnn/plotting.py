from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from Optimizer.reconstruct_cov import make_psd, rebuild_covariance


def _ensure_matplotlib():
    if plt is None:
        raise RuntimeError("matplotlib is required to save QNN plots.")


def _save_current(path: str, dpi: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values
    result = np.empty_like(values, dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        result[idx] = np.mean(values[start : idx + 1])
    return result


class QNNPlotter:
    def __init__(self, dpi: int = 180, rolling_window: int = 5, max_assets: int = 6):
        self.dpi = dpi
        self.rolling_window = rolling_window
        self.max_assets = max_assets

    def save_all(
        self,
        plots_dir: str,
        mode: str,
        metrics: dict[str, Any],
        predictions: dict[str, Any],
        summary: dict[str, Any],
        circuit_plotter=None,
    ) -> list[str]:
        _ensure_matplotlib()
        saved: list[str] = []
        os.makedirs(plots_dir, exist_ok=True)

        saved.extend(
            [
                self._plot_learning_curves(plots_dir, metrics),
                self._plot_error_curves(plots_dir, metrics),
                self._plot_gradients(plots_dir, metrics),
                self._plot_updates(plots_dir, metrics),
                self._plot_qnn_outputs(plots_dir, metrics),
                self._plot_theta_heatmap(plots_dir, metrics),
                self._plot_residual_histogram(plots_dir, predictions),
                self._plot_prediction_scatter(plots_dir, predictions),
                self._plot_rolling_error(plots_dir, predictions),
                self._plot_uncertainty(plots_dir, predictions),
            ]
        )

        if mode == "returns":
            saved.extend(
                [
                    self._plot_per_asset_errors(plots_dir, predictions),
                    self._plot_sign_accuracy(plots_dir, predictions),
                ]
            )
        if mode == "cov":
            saved.append(self._plot_covariance_diagnostics(plots_dir, predictions, summary))

        return [path for path in saved if path is not None]

    def _plot_learning_curves(self, plots_dir: str, metrics: dict[str, Any]) -> str:
        train = np.asarray(metrics.get("train_loss_per_epoch", []), dtype=float)
        val = np.asarray(metrics.get("val_loss_per_epoch", []), dtype=float)
        smooth = np.asarray(metrics.get("smoothed_val_loss_per_epoch", []), dtype=float)
        plt.figure(figsize=(8, 4.5))
        plt.plot(train, label="Train objective")
        plt.plot(val, label="Validation objective")
        if smooth.size:
            plt.plot(smooth, label="Smoothed validation", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curves")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = os.path.join(plots_dir, "learning_curves.png")
        _save_current(path, self.dpi)
        return path

    def _plot_error_curves(self, plots_dir: str, metrics: dict[str, Any]) -> str:
        plt.figure(figsize=(8, 4.5))
        for key, label in [
            ("train_rmse_per_epoch", "Train RMSE"),
            ("val_rmse_per_epoch", "Validation RMSE"),
            ("train_mae_per_epoch", "Train MAE"),
            ("val_mae_per_epoch", "Validation MAE"),
        ]:
            values = np.asarray(metrics.get(key, []), dtype=float)
            if values.size:
                plt.plot(values, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title("Error Curves")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = os.path.join(plots_dir, "error_curves.png")
        _save_current(path, self.dpi)
        return path

    def _plot_gradients(self, plots_dir: str, metrics: dict[str, Any]) -> str:
        plt.figure(figsize=(8, 4.5))
        for key, label in [
            ("quantum_grad_norm_per_epoch", "Quantum grad"),
            ("classical_grad_norm_per_epoch", "Classical grad"),
        ]:
            values = np.asarray(metrics.get(key, []), dtype=float)
            if values.size:
                plt.plot(values, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("L2 norm")
        plt.title("Gradient Norms")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = os.path.join(plots_dir, "gradient_norms.png")
        _save_current(path, self.dpi)
        return path

    def _plot_updates(self, plots_dir: str, metrics: dict[str, Any]) -> str:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for key, label in [
            ("quantum_update_norm_per_epoch", "Quantum update"),
            ("classical_update_norm_per_epoch", "Classical update"),
        ]:
            values = np.asarray(metrics.get(key, []), dtype=float)
            if values.size:
                axes[0].plot(values, label=label)
        ratio = np.asarray(metrics.get("update_balance_ratio_per_epoch", []), dtype=float)
        axes[0].set_ylabel("L2 norm")
        axes[0].set_title("Update Norms")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        if ratio.size:
            axes[1].plot(ratio, color="#55A868", label="Quantum/Classical")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Ratio")
        axes[1].set_title("Update Balance")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        path = os.path.join(plots_dir, "update_dynamics.png")
        _save_current(path, self.dpi)
        return path

    def _plot_qnn_outputs(self, plots_dir: str, metrics: dict[str, Any]) -> str:
        plt.figure(figsize=(8, 4.5))
        for key, label in [
            ("qnn_output_mean_per_epoch", "Mean"),
            ("qnn_output_var_per_epoch", "Variance"),
            ("qnn_output_min_per_epoch", "Min"),
            ("qnn_output_max_per_epoch", "Max"),
        ]:
            values = np.asarray(metrics.get(key, []), dtype=float)
            if values.size:
                plt.plot(values, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Raw Quantum Output Statistics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = os.path.join(plots_dir, "qnn_output_statistics.png")
        _save_current(path, self.dpi)
        return path

    def _plot_theta_heatmap(self, plots_dir: str, metrics: dict[str, Any]) -> str | None:
        theta = np.asarray(metrics.get("theta_trajectory", []), dtype=float)
        if theta.ndim != 2 or theta.size == 0:
            return None
        plt.figure(figsize=(9, 5))
        image = plt.imshow(theta, aspect="auto", cmap="viridis")
        plt.colorbar(image, label="Parameter value")
        plt.xlabel("Parameter index")
        plt.ylabel("Epoch")
        plt.title("Quantum Parameter Trajectory")
        path = os.path.join(plots_dir, "theta_heatmap.png")
        _save_current(path, self.dpi)
        return path

    def _plot_residual_histogram(self, plots_dir: str, predictions: dict[str, Any]) -> str:
        residuals = np.asarray(predictions["Y_pred_test"]) - np.asarray(predictions["Y_true_test"])
        plt.figure(figsize=(8, 4.5))
        plt.hist(residuals.reshape(-1), bins=40, alpha=0.8)
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.title("Residual Distribution")
        path = os.path.join(plots_dir, "residual_histogram.png")
        _save_current(path, self.dpi)
        return path

    def _plot_prediction_scatter(self, plots_dir: str, predictions: dict[str, Any]) -> str:
        y_true = np.asarray(predictions["Y_true_test"]).reshape(-1)
        y_pred = np.asarray(predictions["Y_pred_test"]).reshape(-1)
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(y_true, y_pred, s=8, alpha=0.5)
        low = min(float(np.min(y_true)), float(np.min(y_pred)))
        high = max(float(np.max(y_true)), float(np.max(y_pred)))
        plt.plot([low, high], [low, high], color="black", linestyle="--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Predicted vs True")
        plt.grid(True, alpha=0.3)
        path = os.path.join(plots_dir, "prediction_scatter.png")
        _save_current(path, self.dpi)
        return path

    def _plot_rolling_error(self, plots_dir: str, predictions: dict[str, Any]) -> str:
        residuals = np.asarray(predictions["Y_pred_test"]) - np.asarray(predictions["Y_true_test"])
        per_sample_rmse = np.sqrt(np.mean(residuals**2, axis=1))
        rolling = _rolling(per_sample_rmse, self.rolling_window)
        plt.figure(figsize=(8, 4.5))
        plt.plot(per_sample_rmse, label="Per-sample RMSE", alpha=0.5)
        plt.plot(rolling, label=f"Rolling {self.rolling_window}", linewidth=2)
        plt.xlabel("Sample")
        plt.ylabel("RMSE")
        plt.title("Rolling Prediction Error")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = os.path.join(plots_dir, "rolling_error.png")
        _save_current(path, self.dpi)
        return path

    def _plot_uncertainty(self, plots_dir: str, predictions: dict[str, Any]) -> str | None:
        if "Y_pred_std_test" not in predictions:
            return None
        std = np.asarray(predictions["Y_pred_std_test"], dtype=float)
        if std.size == 0:
            return None
        mean_std = std.mean(axis=1)
        plt.figure(figsize=(8, 4.5))
        plt.plot(mean_std, label="Mean predictive std")
        plt.xlabel("Sample")
        plt.ylabel("Std")
        plt.title("Prediction Uncertainty")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = os.path.join(plots_dir, "prediction_uncertainty.png")
        _save_current(path, self.dpi)
        return path

    def _plot_per_asset_errors(self, plots_dir: str, predictions: dict[str, Any]) -> str:
        y_true = np.asarray(predictions["Y_true_test"], dtype=float)
        y_pred = np.asarray(predictions["Y_pred_test"], dtype=float)
        labels = np.asarray(predictions.get("asset_symbols", []), dtype=str)
        if labels.size != y_true.shape[1]:
            labels = np.asarray([f"dim_{idx}" for idx in range(y_true.shape[1])], dtype=str)
        mae = np.mean(np.abs(y_pred - y_true), axis=0)
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
        order = np.argsort(rmse)[::-1][: self.max_assets]
        x = np.arange(len(order))
        plt.figure(figsize=(10, 4.5))
        plt.bar(x - 0.18, mae[order], width=0.36, label="MAE")
        plt.bar(x + 0.18, rmse[order], width=0.36, label="RMSE")
        plt.xticks(x, labels[order], rotation=45, ha="right")
        plt.ylabel("Error")
        plt.title("Per-Asset Error")
        plt.legend()
        path = os.path.join(plots_dir, "per_asset_error.png")
        _save_current(path, self.dpi)
        return path

    def _plot_sign_accuracy(self, plots_dir: str, predictions: dict[str, Any]) -> str:
        y_true = np.asarray(predictions["Y_true_test"], dtype=float)
        y_pred = np.asarray(predictions["Y_pred_test"], dtype=float)
        accuracy = np.mean(np.sign(y_true) == np.sign(y_pred), axis=0)
        labels = np.asarray(predictions.get("asset_symbols", []), dtype=str)
        if labels.size != accuracy.size:
            labels = np.asarray([f"dim_{idx}" for idx in range(accuracy.size)], dtype=str)
        order = np.argsort(accuracy)[::-1][: self.max_assets]
        plt.figure(figsize=(10, 4.5))
        plt.bar(np.arange(len(order)), accuracy[order])
        plt.xticks(np.arange(len(order)), labels[order], rotation=45, ha="right")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.0)
        plt.title("Return Sign Accuracy")
        path = os.path.join(plots_dir, "sign_accuracy.png")
        _save_current(path, self.dpi)
        return path

    def _plot_covariance_diagnostics(
        self,
        plots_dir: str,
        predictions: dict[str, Any],
        summary: dict[str, Any],
    ) -> str:
        y_true = np.asarray(predictions["Y_true_test"], dtype=float)
        y_pred = np.asarray(predictions["Y_pred_test"], dtype=float)
        n_assets = int(summary.get("n_assets", 0))
        frob_errors = []
        min_eigs = []
        for idx in range(y_true.shape[0]):
            cov_true = rebuild_covariance(y_true[idx], n_assets)
            cov_pred = make_psd(rebuild_covariance(y_pred[idx], n_assets))
            frob_errors.append(float(np.linalg.norm(cov_pred - cov_true, ord="fro")))
            min_eigs.append(float(np.min(np.linalg.eigvalsh(cov_pred))))
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(frob_errors, label="Frobenius error")
        axes[0].set_ylabel("Error")
        axes[0].set_title("Covariance Reconstruction Error")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].plot(min_eigs, label="Min eigenvalue", color="#C44E52")
        axes[1].axhline(0.0, color="black", linestyle="--")
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("Eigenvalue")
        axes[1].set_title("Predicted PSD Diagnostics")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        path = os.path.join(plots_dir, "covariance_diagnostics.png")
        _save_current(path, self.dpi)
        return path
