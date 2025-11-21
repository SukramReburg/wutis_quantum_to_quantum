import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

def plot_returns(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    asset_idx: int = 0,
    asset_name: str | None = None,
    use_cumulative: bool = True,
    show: bool = False
):
    """
    Plot actual vs predicted returns (or cumulative returns) for one asset.

    Y_true, Y_pred: arrays of shape (T, n_assets)
    asset_idx: index of the asset column to plot
    use_cumulative: if True -> plot cumulative return path;
                    if False -> plot raw daily log-returns.
    """
    assert Y_true.shape == Y_pred.shape, "Y_true and Y_pred shapes must match"
    T, n_assets = Y_true.shape
    if not (0 <= asset_idx < n_assets):
        raise ValueError(f"asset_idx must be in [0, {n_assets-1}]")

    if asset_name is None:
        asset_name = f"Asset {asset_idx}"

    # extract series for this asset
    r_true = Y_true[:, asset_idx]
    r_pred = Y_pred[:, asset_idx]

    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(base_dir, paths['plots'], 'assets')
    os.makedirs(plots_dir, exist_ok=True)
    if use_cumulative:
        # for log-returns, cumulative return = exp(sum r) - 1
        cum_true = np.exp(np.cumsum(r_true)) - 1.0
        cum_pred = np.exp(np.cumsum(r_pred)) - 1.0

        x = np.arange(len(cum_true))

        plt.figure(figsize=(10, 5))
        plt.plot(x, cum_true, label="Actual cumulative return")
        plt.plot(x, cum_pred, label="Predicted cumulative return", linestyle="--")
        plt.xlabel("Test step")
        plt.ylabel("Cumulative return")
        plt.title(f"Cumulative returns path – {asset_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(plots_dir, f"cumulative_returns_asset_{asset_idx}.png")
        plt.savefig(path)
        if show:
            plt.show()
    else:
        # raw daily log-returns
        x = np.arange(len(r_true))

        plt.figure(figsize=(10, 5))
        plt.plot(x, r_true, label="Actual log-return")
        plt.plot(x, r_pred, label="Predicted log-return", linestyle="--")
        plt.xlabel("Test step")
        plt.ylabel("Log-return")
        plt.title(f"Daily log-returns - {asset_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(plots_dir, f"log_returns_asset_{asset_idx}.png")
        plt.savefig(path)
        if show:
            plt.show()

if __name__ == "__main__":
    # path to your predictions file from the QNN script
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_dir, paths['results'])
    pred_path = os.path.join(results_path, "qnn_torch_returns_angles_hybrid_predictions.npz")

    data = np.load(pred_path)
    Y_true = data["Y_true_test"]   # shape: (T, n_assets)
    Y_pred = data["Y_pred_test"]   # shape: (T, n_assets)

    print("Y_true shape:", Y_true.shape)
    print("Y_pred shape:", Y_pred.shape)

    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    assets = config['assets']

    for asset_idx, asset_name in enumerate(assets):
        plot_returns(Y_true, Y_pred, asset_idx=asset_idx, asset_name=asset_name, use_cumulative=True)       
        plot_returns(Y_true, Y_pred, asset_idx=asset_idx, asset_name=asset_name, use_cumulative=False)