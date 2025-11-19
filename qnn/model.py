# qnn_qiskit_scalar.py (or extend your existing qnn_qiskit.py)

import os
import yaml
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.visualization import circuit_drawer

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.optimizers import L_BFGS_B


# ---------- utilities from before ----------

def load_paths_from_config(config_path: str = 'config/data_config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, paths['processed'])
    return config, base_dir, dataset_path


def compress_features_to_angles(X: np.ndarray, n_qubits: int) -> np.ndarray:
    n_samples, n_feat = X.shape
    angles = np.zeros((n_samples, n_qubits), dtype=np.float32)
    for i in range(n_samples):
        chunks = np.array_split(X[i], n_qubits)
        angles[i] = np.array([chunk.mean() for chunk in chunks], dtype=np.float32)
    return angles


# ---------- SCALAR QNN BUILDERS ----------

def build_scalar_qnn_circuit(n_qubits: int, n_layers: int):
    """
    Same ansatz as before, but we will use a *single* observable
    (e.g. Z on first qubit) so the output is scalar.
    """
    x_params = ParameterVector("x", n_qubits)
    theta_params = ParameterVector("θ", n_layers * n_qubits * 2)

    qc = QuantumCircuit(n_qubits, name="ScalarQNN")

    # feature encoding
    for q in range(n_qubits):
        qc.rx(x_params[q], q)

    # variational layers
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.rz(theta_params[idx], q)
            idx += 1
            qc.rx(theta_params[idx], q)
            idx += 1

        # ring entanglement
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        if n_qubits > 1:
            qc.cx(n_qubits - 1, 0)

    return qc, list(x_params), list(theta_params)


def build_scalar_estimator_qnn(n_qubits: int, n_layers: int):
    qc, x_params, theta_params = build_scalar_qnn_circuit(n_qubits, n_layers)

    # SINGLE observable → scalar output in [-1, 1]
    pauli_str = ["I"] * n_qubits
    pauli_str[0] = "Z"             # measure Z on first qubit
    observable = SparsePauliOp.from_list([("".join(pauli_str), 1.0)])

    estimator = Estimator()

    qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        observables=observable,    # <--- single op, scalar output
        input_params=x_params,
        weight_params=theta_params,
    )
    print(f"Scalar QNN num_inputs = {qnn.num_inputs}, "
          f"num_weights = {qnn.num_weights}, "
          f"output_shape = {qnn.output_shape}")
    return qnn, qc


def visualise_circuit(qc: QuantumCircuit, save_path: str | None = None):
    print("\nQuantum circuit:\n")
    print(qc)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        circuit_drawer(qc, output="mpl", filename=save_path)
        print(f"\nCircuit diagram saved to: {save_path}\n")


def train_scalar_qnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int,
    n_layers: int,
    circuit_png_path: str | None = None,
    pred_save_path: str | None = None,
):
    """
    Train a scalar-output QNN regressor.

    y_train, y_test must be 1D arrays: shape (n_samples,).
    """
    # sanity: make sure targets are 1D
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    qnn, qc = build_scalar_estimator_qnn(n_qubits, n_layers)
    visualise_circuit(qc, circuit_png_path)

    optimizer = L_BFGS_B(maxiter=50)

    regressor = NeuralNetworkRegressor(
        neural_network=qnn,
        loss="squared_error",
        optimizer=optimizer,
    )

    print("\nTraining scalar QNN regressor...")
    regressor.fit(X_train, y_train)

    y_pred_test = regressor.predict(X_test)
    y_pred_test = np.asarray(y_pred_test).reshape(-1)

    mse = np.mean((y_pred_test - y_test) ** 2)
    print(f"\nTest MSE (scalar): {mse:.6e}")

    if pred_save_path is not None:
        os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
        np.savez_compressed(
            pred_save_path,
            y_pred_test=y_pred_test,
            y_true_test=y_test,
        )
        print(f"Saved predictions to {pred_save_path}")

    return regressor, y_pred_test


def train_scalar_qnn_from_npz(
    config_path: str,
    mode: str,              # 'returns' or 'cov'
    target_index: int,      # which column to train on
    n_qubits: int = 2,
    n_layers: int = 2,
    npz_name: str = "qnn_datasets.npz",
    circuit_png_name: str | None = None,
    pred_npz_name: str | None = None,
):
    """
    Train a scalar QNN on one target column (asset or covariance component).
    """

    config, base_dir, dataset_path = load_paths_from_config(config_path)
    npz_path = os.path.join(dataset_path, npz_name)
    data = np.load(npz_path)

    if mode == "returns":
        X_train_raw = data["X_train_ret"]
        X_test_raw  = data["X_test_ret"]
        Y_train_all = data["Y_train_ret"]
        Y_test_all  = data["Y_test_ret"]
    elif mode == "cov":
        X_train_raw = data["X_train_cov"]
        X_test_raw  = data["X_test_cov"]
        Y_train_all = data["Y_train_cov"]
        Y_test_all  = data["Y_test_cov"]
    else:
        raise ValueError("mode must be 'returns' or 'cov'.")

    y_train = Y_train_all[:, target_index]
    y_test  = Y_test_all[:, target_index]

    print(f"[{mode}] training scalar QNN on target index {target_index}, "
          f"X_train shape = {X_train_raw.shape}, y_train shape = {y_train.shape}")

    # compress features -> angles for chosen number of qubits
    X_train = compress_features_to_angles(X_train_raw, n_qubits)
    X_test  = compress_features_to_angles(X_test_raw,  n_qubits)

    if circuit_png_name is None:
        circuit_png_name = f"qnn_{mode}_target{target_index}_circuit.png"
    if pred_npz_name is None:
        pred_npz_name = f"qnn_{mode}_target{target_index}_predictions.npz"
    
    config_path = 'config/model_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, paths['results'])

    circuit_png_path = os.path.join(path, circuit_png_name)
    pred_save_path   = os.path.join(path, pred_npz_name)
    os.makedirs(path, exist_ok=True)


    regressor, y_pred_test = train_scalar_qnn(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_png_path=circuit_png_path,
        pred_save_path=pred_save_path,
    )

    return regressor, y_pred_test


if __name__ == "__main__":
    # Example: returns for assets 0..3
    for idx in range(4):
        print(f"\n===== Training RETURNS QNN for asset index {idx} =====")
        train_scalar_qnn_from_npz(
            config_path="config/data_config.yaml",
            mode="returns",
            target_index=idx,
            n_qubits=4,   # can be 1, 2, 3... (2 is a nice start)
            n_layers=4,
        )

    # Example: do the same for covariances later
    # for idx in range(4):
    #     print(f\"\n===== Training COV QNN for component index {idx} =====\")
    #     train_scalar_qnn_from_npz(
    #         config_path=\"config/data_config.yaml\",
    #         mode=\"cov\",
    #         target_index=idx,
    #         n_qubits=2,
    #         n_layers=2,
    #     )
