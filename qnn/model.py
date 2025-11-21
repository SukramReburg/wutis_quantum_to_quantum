"""
Features:
- feature_mode: "angles" (chunk average) or "pca"
- use_dense_head: True  -> hybrid (QNN + MLP head)
                   False -> pure multi-output QNN

Circuit types:
- circuit_type = "zz_feature"  -> ZZFeatureMap + RealAmplitudes
- circuit_type = "rxrz"        -> custom Rx/Rz data-reuploading circuit

Artifacts per training run (if enabled):
- circuit PNG
- learning-curve PNG (train/test MSE)
- predictions .npz
- model .pth (PyTorch state_dict)
"""

import os
import numpy as np
from typing import Tuple, Optional

import torch
import torch.nn as nn

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.visualization import circuit_drawer

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


# ======================================================================
# QNN BUILDERS
# ======================================================================

def build_estimator_qnn(
    n_qubits: int,
    n_layers: int,
    n_qnn_outputs: int,
) -> Tuple[EstimatorQNN, "QuantumCircuit"]:
    """
    Build an EstimatorQNN using ZZFeatureMap + RealAmplitudes.

    n_qnn_outputs:
      - 1       -> scalar QNN
      - >1      -> multi-output QNN with n_qnn_outputs observables
    """
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=n_layers,
        entanglement="circular",
    )
    qc = feature_map.compose(ansatz)

    # Observables:
    if n_qnn_outputs == 1:
        observables = SparsePauliOp.from_list([("Z" * n_qubits, 1.0)])
    else:
        max_outputs = 2**n_qubits - 1
        if n_qnn_outputs > max_outputs:
            raise ValueError(
                f"Cannot have {n_qnn_outputs} outputs with {n_qubits} qubits. "
                f"Maximum is {max_outputs}. Increase n_qubits or reduce outputs."
            )
        observables = []
        for k in range(1, n_qnn_outputs + 1):
            bits = np.binary_repr(k, width=n_qubits)
            pauli_str = "".join("Z" if b == "1" else "I" for b in bits)
            observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )

    print(
        f"ZZFeature QNN built: num_inputs={qnn.num_inputs}, "
        f"num_weights={qnn.num_weights}, output_shape={qnn.output_shape}"
    )

    return qnn, qc


def build_rxrz_qnn(
    n_qubits: int,
    n_layers: int,
    n_qnn_outputs: int,
    entanglement: str = "ring",  # "ring" or "linear"
) -> Tuple[EstimatorQNN, "QuantumCircuit"]:
    """
    Data-reuploading QNN (pretty Rx/Rz circuit):

    For each layer ℓ and qubit q:
        Rx(x[q])  ->  Rz(θ[idx])  ->  Rx(θ[idx+1])
    then an entangling pattern (CNOT ring or linear).

    Inputs:  x[0..n_qubits-1]
    Weights: θ[0..(2 * n_qubits * n_layers - 1)]
    """
    x = ParameterVector("x", n_qubits)                 # data / input params
    theta = ParameterVector("θ", 2 * n_qubits * n_layers)  # trainable weights

    qc = QuantumCircuit(n_qubits)
    t_idx = 0

    for _layer in range(n_layers):
        # single-qubit rotations
        for q in range(n_qubits):
            qc.rx(x[q], q)             # data encoding
            qc.rz(theta[t_idx], q)     # weight 1
            t_idx += 1
            qc.rx(theta[t_idx], q)     # weight 2
            t_idx += 1

        # entanglement
        if entanglement == "linear":
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
        elif entanglement == "ring":
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(n_qubits - 1, 0)
        else:
            raise ValueError("entanglement must be 'linear' or 'ring'")

    # Observables
    if n_qnn_outputs == 1:
        observables = SparsePauliOp.from_list([("Z" * n_qubits, 1.0)])
    else:
        max_outputs = 2**n_qubits - 1
        if n_qnn_outputs > max_outputs:
            raise ValueError(
                f"Cannot have {n_qnn_outputs} outputs with {n_qubits} qubits. "
                f"Maximum is {max_outputs}."
            )
        observables = []
        for k in range(1, n_qnn_outputs + 1):
            bits = np.binary_repr(k, width=n_qubits)
            pauli_str = "".join("Z" if b == "1" else "I" for b in bits)
            observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=list(x),
        weight_params=list(theta),
        input_gradients=True,
    )

    print(
        f"RxRz QNN built: num_inputs={qnn.num_inputs}, "
        f"num_weights={qnn.num_weights}, output_shape={qnn.output_shape}"
    )

    return qnn, qc


def visualise_circuit(qc, save_path: Optional[str] = None):
    """
    Print and optionally save the quantum circuit.
    """
    print("\nQuantum circuit:\n")
    print(qc)
    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        circuit_drawer(qc, output="mpl", filename=save_path)
        print(f"\nCircuit diagram saved to: {save_path}\n")


# ======================================================================
# PYTORCH MODEL (QNN + OPTIONAL DENSE HEAD)
# ======================================================================

class HybridQNNModel(nn.Module):
    """
    Wraps a Qiskit EstimatorQNN as a PyTorch module via TorchConnector,
    and optionally adds a classical dense head.

    If use_dense_head=True:
        QNN outputs dimension n_qnn_outputs, then:
          quantum -> Linear(n_qnn_outputs, 32) -> ReLU -> Linear(32, n_outputs)

    If use_dense_head=False:
        QNN outputs dimension n_qnn_outputs=n_outputs, model is pure quantum.
    """

    def __init__(
        self,
        qnn: EstimatorQNN,
        n_outputs: int,
        n_qnn_outputs: int,
        use_dense_head: bool,
    ):
        super().__init__()

        # QNN initial weights (use NumPy RNG)
        rng = np.random.default_rng(42)
        initial_weights = 0.1 * (2.0 * rng.random(qnn.num_weights) - 1.0)

        # QNN as PyTorch module
        self.quantum = TorchConnector(qnn, initial_weights=initial_weights)
        self.use_dense_head = use_dense_head
        self.n_qnn_outputs = n_qnn_outputs
        self.n_outputs = n_outputs

        if self.use_dense_head:
            self.head = nn.Sequential(
                nn.Linear(n_qnn_outputs, 32),
                nn.ReLU(),
                nn.Linear(32, n_outputs),
            )
        else:
            self.head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.quantum(x)  # shape: (batch_size, n_qnn_outputs)
        if self.use_dense_head:
            return self.head(q_out)
        else:
            return q_out
