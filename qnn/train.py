"""
PyTorch + Qiskit QNN training for:
- next-period covariance (mode="cov")
- next-period returns (mode="returns")
"""
import os
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import yaml

from encode import transform_features
from model import (build_estimator_qnn,
                       build_rxrz_qnn,
                       HybridQNNModel,
                       visualise_circuit)
# ======================================================================
# CONFIG / PATHS
# ======================================================================

def load_paths_from_config(config_path: str = "config/data_config.yaml"):
    """
    Loads base dir and processed dataset path from your config.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    paths = config["paths"]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, paths["processed"])
    return config, base_dir, dataset_path

# ======================================================================
# TRAINING LOOP (PYTORCH)
# ======================================================================

def train_qnn(
    X_train_raw: np.ndarray,
    Y_train: np.ndarray,
    X_test_raw: np.ndarray,
    Y_test: np.ndarray,
    n_qubits: int,
    n_layers: int,
    feature_mode: str = "angles",       # "angles" or "pca"
    use_dense_head: bool = True,        # True=hybrid, False=pure multi-output QNN
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    circuit_png_path: Optional[str] = None,
    pred_save_path: Optional[str] = None,
    circuit_type: str = "zz_feature",   # "zz_feature" or "rxrz"
    entanglement: str = "ring",  # for rxrz QNN
    learning_curve_png_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
):
    """
    Train a QNN with PyTorch and optionally save artifacts.
    """
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure shapes/types
    X_train_raw = np.asarray(X_train_raw, dtype=np.float32)
    X_test_raw = np.asarray(X_test_raw, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    Y_test = np.asarray(Y_test, dtype=np.float32)

    if Y_train.ndim != 2 or Y_test.ndim != 2:
        raise ValueError("Y_train and Y_test must be 2D (n_samples, n_outputs).")

    n_outputs = Y_train.shape[1]

    # Feature preprocessing
    X_train, X_test, _ = transform_features(
        X_train_raw, X_test_raw, n_qubits, feature_mode=feature_mode
    )

    # Decide how many outputs QNN itself should have
    if use_dense_head:
        n_qnn_outputs = 1
    else:
        n_qnn_outputs = n_outputs

    # Build QNN and circuit
    if circuit_type == "zz_feature":
        qnn, qc = build_estimator_qnn(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_qnn_outputs=n_qnn_outputs,
        )
    elif circuit_type == "rxrz":
        qnn, qc = build_rxrz_qnn(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_qnn_outputs=n_qnn_outputs,
            entanglement="ring",
        )
    else:
        raise ValueError("circuit_type must be 'zz_feature' or 'rxrz'.")

    visualise_circuit(qc, circuit_png_path)

    # Build PyTorch model
    model = HybridQNNModel(
        qnn=qnn,
        n_outputs=n_outputs,
        n_qnn_outputs=n_qnn_outputs,
        use_dense_head=use_dense_head,
    ).to(device)

    # Convert data to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation / test MSE
        model.eval()
        with torch.no_grad():
            preds_test = model(X_test_t)
            test_loss = loss_fn(preds_test, Y_test_t).item()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f"Epoch {epoch:03d} | train MSE={train_loss:.6e} | test MSE={test_loss:.6e}"
        )

    # Final predictions
    model.eval()
    with torch.no_grad():
        Y_pred_test = model(X_test_t).cpu().numpy()
    final_mse = float(np.mean((Y_pred_test - Y_test) ** 2))
    print(f"\nFinal test MSE: {final_mse:.6e}")
    print(f"Y_pred_test shape={Y_pred_test.shape}, Y_test shape={Y_test.shape}")

    # Save predictions
    if pred_save_path is not None:
        directory = os.path.dirname(pred_save_path)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(
            pred_save_path,
            Y_pred_test=Y_pred_test,
            Y_true_test=Y_test,
        )
        print(f"Saved predictions to {pred_save_path}")

    # Plot learning curves (train/test MSE)
    if learning_curve_png_path is not None:
        directory = os.path.dirname(learning_curve_png_path)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, n_epochs + 1), train_losses, label="Train MSE")
        plt.plot(range(1, n_epochs + 1), test_losses, label="Test MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("QNN Learning Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(learning_curve_png_path, dpi=200)
        plt.close()
        print(f"Saved learning curves to {learning_curve_png_path}")

    # Save model state_dict for reuse
    if model_save_path is not None:
        directory = os.path.dirname(model_save_path)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved PyTorch model to {model_save_path}")

    return {
        "model": model,
        "qnn": qnn,
        "Y_pred_test": Y_pred_test,
        "train_losses": np.array(train_losses, dtype=np.float32),
        "test_losses": np.array(test_losses, dtype=np.float32),
        "final_mse": final_mse,
    }


# ======================================================================
# TOP-LEVEL WRAPPER: LOAD NPZ AND TRAIN
# ======================================================================

def train_qnn_from_npz(
    config_path: str,
    mode: str,                      # 'returns' or 'cov'
    n_qubits: int,
    n_layers: int = 2,
    feature_mode: str = "angles",   # 'angles' or 'pca'
    use_dense_head: bool = True,    # True=hybrid, False=pure
    npz_name: str = "qnn_datasets.npz",
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    circuit_png_name: Optional[str] = None,
    pred_npz_name: Optional[str] = None,
    circuit_type: str = "zz_feature",         # 'zz_feature' or 'rxrz'
    entanglement: str = "ring",               # 'ring' or 'linear' (for rxrz)
    learning_curve_png_name: Optional[str] = None,
    model_name: Optional[str] = None,
    save_artifacts: bool = True,             # False = no png/npz/pth (for Optuna)
):
    """
    Loads your saved qnn_datasets.npz and trains a PyTorch-based QNN model
    for either 'returns' or 'cov'.
    """
    _, base_dir, dataset_path = load_paths_from_config(config_path)
    npz_path = os.path.join(dataset_path, npz_name)
    data = np.load(npz_path)

    if mode == "returns":
        X_train_raw = data["X_train_ret"]
        X_test_raw = data["X_test_ret"]
        Y_train = data["Y_train_ret"]
        Y_test = data["Y_test_ret"]
    elif mode == "cov":
        X_train_raw = data["X_train_cov"]
        X_test_raw = data["X_test_cov"]
        Y_train = data["Y_train_cov"]
        Y_test = data["Y_test_cov"]
    else:
        raise ValueError("mode must be 'returns' or 'cov'.")

    print(
        f"[{mode}] training PyTorch QNN, "
        f"X_train shape={X_train_raw.shape}, Y_train shape={Y_train.shape}, "
        f"feature_mode={feature_mode}, use_dense_head={use_dense_head}, "
        f"circuit_type={circuit_type}"
    )

    # results path from model_config
    model_config_path = "config/model_config.yaml"
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    paths = model_config["paths"]
    results_path = os.path.join(base_dir, paths["results"])
    plots_path = os.path.join(base_dir, paths["plots"])
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    if save_artifacts:
        tag = f"{mode}_{feature_mode}_{'hybrid' if use_dense_head else 'pure'}_{circuit_type}"
        if circuit_png_name is None:
            circuit_png_name = f"qnn_{tag}_circuit.png"
        if pred_npz_name is None:
            pred_npz_name = f"qnn_{tag}_predictions.npz"
        if learning_curve_png_name is None:
            learning_curve_png_name = f"qnn_{tag}_learning_curves.png"
        if model_name is None:
            model_name = f"qnn_{tag}_model.pth"

        circuit_png_path = os.path.join(plots_path, circuit_png_name)
        pred_save_path = os.path.join(results_path, pred_npz_name)
        learning_curve_png_path = os.path.join(plots_path, learning_curve_png_name)
        model_save_path = os.path.join(results_path, model_name)
    else:
        circuit_png_path = None
        pred_save_path = None
        learning_curve_png_path = None
        model_save_path = None

    result = train_qnn(
        X_train_raw=X_train_raw,
        Y_train=Y_train,
        X_test_raw=X_test_raw,
        Y_test=Y_test,
        n_qubits=n_qubits,
        n_layers=n_layers,
        feature_mode=feature_mode,
        use_dense_head=use_dense_head,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        circuit_png_path=circuit_png_path,
        pred_save_path=pred_save_path,
        entanglement=entanglement,
        circuit_type=circuit_type,
        learning_curve_png_path=learning_curve_png_path,
        model_save_path=model_save_path,
    )

    return result


# ======================================================================
# MAIN 
# ======================================================================

if __name__ == "__main__":

    print("\n===== Training RETURNS QNN =====")
    train_qnn_from_npz(
        config_path="config/data_config.yaml",
        mode="returns",
        n_qubits=6,
        n_layers=6,
        feature_mode="angles",
        use_dense_head=True,
        n_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        circuit_type="rxrz",
        save_artifacts=True,
    )

    print("\n===== Training COV QNN =====")
    train_qnn_from_npz(
        config_path="config/data_config.yaml",
        mode="cov",
        n_qubits=7,
        n_layers=6,
        feature_mode="pca",
        use_dense_head=True,
        n_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        circuit_type="rxrz",
        save_artifacts=True,
    )

