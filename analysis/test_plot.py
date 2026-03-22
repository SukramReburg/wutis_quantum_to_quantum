from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

from data.common import npz_string_list, resolve_path


def _default_prediction_path(base_dir: str, model_config: dict) -> str:
    results_dir = resolve_path(base_dir, model_config["paths"]["results"])
    candidates = [
        os.path.join(results_dir, "latest", "returns", "predictions.npz"),
        os.path.join(results_dir, "qnn_returns_angles_hybrid_rxrz_predictions.npz"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("No returns prediction artifact found.")


def plot_returns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    asset_idx: int = 0,
    asset_name: str | None = None,
    use_cumulative: bool = True,
    show: bool = False,
):
    assert y_true.shape == y_pred.shape, "Y_true and Y_pred shapes must match"
    n_assets = y_true.shape[1]
    if not (0 <= asset_idx < n_assets):
        raise ValueError(f"asset_idx must be in [0, {n_assets - 1}]")

    asset_name = asset_name or f"Asset {asset_idx}"
    series_true = y_true[:, asset_idx]
    series_pred = y_pred[:, asset_idx]

    with open("config/data_config.yaml", "r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(base_dir, data_config["paths"]["plots"], "assets")
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(series_true))
    if use_cumulative:
        plt.plot(x, np.exp(np.cumsum(series_true)) - 1.0, label="Actual cumulative return")
        plt.plot(x, np.exp(np.cumsum(series_pred)) - 1.0, label="Predicted cumulative return", linestyle="--")
        plt.ylabel("Cumulative return")
        filename = f"cumulative_returns_asset_{asset_idx}.png"
        plt.title(f"Cumulative returns path - {asset_name}")
    else:
        plt.plot(x, series_true, label="Actual log-return")
        plt.plot(x, series_pred, label="Predicted log-return", linestyle="--")
        plt.ylabel("Log-return")
        filename = f"log_returns_asset_{asset_idx}.png"
        plt.title(f"Daily log-returns - {asset_name}")

    plt.xlabel("Test step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=180)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    with open("config/model_config.yaml", "r", encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prediction_path = _default_prediction_path(base_dir, model_config)
    data = np.load(prediction_path)
    y_true = data["Y_true_test"]
    y_pred = data["Y_pred_test"]
    assets = npz_string_list(data, "asset_symbols")
    if not assets:
        with open("config/data_config.yaml", "r", encoding="utf-8") as handle:
            data_config = yaml.safe_load(handle)
        assets = sorted(data_config["assets"])

    for idx, asset in enumerate(assets):
        plot_returns(y_true, y_pred, asset_idx=idx, asset_name=asset, use_cumulative=True)
        plot_returns(y_true, y_pred, asset_idx=idx, asset_name=asset, use_cumulative=False)
