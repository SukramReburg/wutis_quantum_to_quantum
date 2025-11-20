import os
import yaml
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.visualization import circuit_drawer

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import L_BFGS_B

class MultiOutputQNNRegressor:
    """Tiny wrapper to have a .predict(X) API like sklearn."""
    def __init__(self, qnn: EstimatorQNN, theta_opt: np.ndarray):
        self.qnn = qnn
        self.theta_opt = np.array(theta_opt, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.asarray(self.qnn.forward(X, self.theta_opt))


def train_multi_qnn(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    n_qubits: int,
    n_layers: int,
    circuit_png_path: str | None = None,
    pred_save_path: str | None = None,
):
    """
    Train a multi-output QNN regressor using manual optimization.

    Y_train, Y_test must be 2D arrays: shape (n_samples, n_outputs).
    Loss: mean squared error over all outputs.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    Y_test  = np.asarray(Y_test, dtype=float)

    if Y_train.ndim != 2:
        raise ValueError(f"Y_train must be 2D (n_samples, n_outputs). Got shape {Y_train.shape}.")
    if Y_test.ndim != 2:
        raise ValueError(f"Y_test must be 2D (n_samples, n_outputs). Got shape {Y_test.shape}.")

    n_outputs = Y_train.shape[1]

    qnn, qc = build_multi_output_estimator_qnn(n_qubits, n_layers, n_outputs)
    visualise_circuit(qc, circuit_png_path)

    # random-ish small initial point
    rng = np.random.default_rng(42)
    initial_point = 0.1 * (rng.random(qnn.num_weights) - 0.5)

    optimizer = L_BFGS_B(maxiter=50)

    # ---- objective and gradient for L-BFGS-B ----
    def objective(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        y_pred = np.asarray(qnn.forward(X_train, theta))   # (N, n_outputs)
        loss = np.mean((y_pred - Y_train) ** 2)
        return float(loss)

    def gradient(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)

        # forward pass
        y_pred = np.asarray(qnn.forward(X_train, theta))   # (N, n_outputs)
        n_samples = Y_train.shape[0]

        # dL/dy for MSE: L = mean((y_pred - y_true)^2)
        dout = (2.0 / n_samples) * (y_pred - Y_train)      # (N, n_outputs)

        # backward: gives Jacobian of outputs wrt weights
        # w_jac shape: (N, n_outputs, num_weights)
        _, w_jac = qnn.backward(X_train, theta)

        # contract dL/dy with dy/dθ over (sample, output) axes
        # result shape: (num_weights,)
        grad_theta = np.tensordot(dout, w_jac, axes=([0, 1], [0, 1]))

        return np.asarray(grad_theta, dtype=float)

    print("\nTraining multi-output QNN regressor (manual)...")

    result = optimizer.minimize(
        fun=objective,
        x0=initial_point,
        jac=gradient,
    )

    theta_opt = result.x
    print(
        f"Optimization finished. Final loss = {result.fun:.6e}, "
        f"n_iters = {result.nit}, status = {result.message}"
    )

    # evaluate on test set
    Y_pred_test = np.asarray(qnn.forward(X_test, theta_opt))
    mse = np.mean((Y_pred_test - Y_test) ** 2)
    print(f"\nTest MSE (multi-output): {mse:.6e}")
    print(f"Y_pred_test shape = {Y_pred_test.shape}, Y_test shape = {Y_test.shape}")

    if pred_save_path is not None:
        os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
        np.savez_compressed(
            pred_save_path,
            Y_pred_test=Y_pred_test,
            Y_true_test=Y_test,
        )
        print(f"Saved predictions to {pred_save_path}")

    regressor = MultiOutputQNNRegressor(qnn, theta_opt)
    return regressor, Y_pred_test

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


# ---------- CIRCUIT / QNN BUILDERS ----------

def build_qnn_circuit(n_qubits: int, n_layers: int):
    """
    Same ansatz as before, but we will now allow multiple observables
    (one per output). Circuit itself stays the same.
    """
    x_params = ParameterVector("x", n_qubits)
    theta_params = ParameterVector("θ", n_layers * n_qubits * 2)

    qc = QuantumCircuit(n_qubits, name="MultiOutputQNN")

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


def build_multi_output_estimator_qnn(n_qubits: int, n_layers: int, n_outputs: int):
    """
    Build a QNN whose output is a vector of length n_outputs, by using
    different Z/I Pauli strings as observables.

    We can have up to (2^n_qubits - 1) distinct non-identity Z/I strings.
    Requirement: n_outputs <= 2^n_qubits - 1.
    """

    max_outputs = 2 ** n_qubits - 1
    if n_outputs > max_outputs:
        raise ValueError(
            f"Cannot build {n_outputs} distinct Z/I observables with {n_qubits} qubits. "
            f"Maximum is {max_outputs}. Increase n_qubits."
        )

    qc, x_params, theta_params = build_qnn_circuit(n_qubits, n_layers)

    observables = []
    # we skip 0 (all I) and start from 1
    for out_idx in range(n_outputs):
        code = out_idx + 1  # 1..n_outputs
        bits = np.binary_repr(code, width=n_qubits)
        pauli_list = ['Z' if b == '1' else 'I' for b in bits]
        pauli_str = "".join(pauli_list)
        observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))

    estimator = Estimator()

    qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        observables=observables,
        input_params=x_params,
        weight_params=theta_params,
    )

    print(
        f"Multi-output QNN num_inputs = {qnn.num_inputs}, "
        f"num_weights = {qnn.num_weights}, "
        f"output_shape = {qnn.output_shape}"
    )

    return qnn, qc


def visualise_circuit(qc: QuantumCircuit, save_path: str | None = None):
    print("\nQuantum circuit:\n")
    print(qc)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        circuit_drawer(qc, output="mpl", filename=save_path)
        print(f"\nCircuit diagram saved to: {save_path}\n")


def train_multi_qnn_from_npz(
    config_path: str,
    mode: str,              # 'returns' or 'cov'
    n_qubits: int,
    n_layers: int = 2,
    npz_name: str = "qnn_datasets.npz",
    circuit_png_name: str | None = None,
    pred_npz_name: str | None = None,
):
    data_config, base_dir, dataset_path = load_paths_from_config(config_path)
    npz_path = os.path.join(dataset_path, npz_name)
    data = np.load(npz_path)

    if mode == "returns":
        X_train_raw = data["X_train_ret"]
        X_test_raw  = data["X_test_ret"]
        Y_train_all = data["Y_train_ret"]   # (1014, 12)
        Y_test_all  = data["Y_test_ret"]
    elif mode == "cov":
        X_train_raw = data["X_train_cov"]
        X_test_raw  = data["X_test_cov"]
        Y_train_all = data["Y_train_cov"]   # (1014, 78)
        Y_test_all  = data["Y_test_cov"]
    else:
        raise ValueError("mode must be 'returns' or 'cov'.")

    print(
        f"[{mode}] training multi-output QNN, "
        f"X_train shape = {X_train_raw.shape}, Y_train shape = {Y_train_all.shape}"
    )

    # compress features -> angles
    X_train = compress_features_to_angles(X_train_raw, n_qubits)
    X_test  = compress_features_to_angles(X_test_raw,  n_qubits)

    # paths from model_config
    model_config_path = 'config/model_config.yaml'
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    paths = model_config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_dir, paths['results'])
    os.makedirs(results_path, exist_ok=True)

    if circuit_png_name is None:
        circuit_png_name = f"qnn_{mode}_multi_output_circuit.png"
    if pred_npz_name is None:
        pred_npz_name = f"qnn_{mode}_multi_output_predictions.npz"

    circuit_png_path = os.path.join(results_path, circuit_png_name)
    pred_save_path   = os.path.join(results_path, pred_npz_name)

    regressor, Y_pred_test = train_multi_qnn(
        X_train=X_train,
        Y_train=Y_train_all,
        X_test=X_test,
        Y_test=Y_test_all,
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_png_path=circuit_png_path,
        pred_save_path=pred_save_path,
    )

    return regressor, Y_pred_test


if __name__ == "__main__":
    # RETURNS: 12 outputs → need n_qubits >= 4
    print("\n===== Training RETURNS QNN for ALL assets =====")
    train_multi_qnn_from_npz(
        config_path="config/data_config.yaml",
        mode="returns",
        n_qubits=4,
        n_layers=4,
    )

    # COVARIANCES: 78 outputs → need n_qubits >= 7
    print("\n===== Training COV QNN for ALL cov components =====")
    train_multi_qnn_from_npz(
        config_path="config/data_config.yaml",
        mode="cov",
        n_qubits=7,  # 2^7 - 1 = 127 >= 78
        n_layers=4,
    )

