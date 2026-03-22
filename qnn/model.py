from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

try:
    from .config import ModelConfig, QNNExperimentConfig
    from .runtime import OptionalDependencyError, QuantumRuntimeContext, _import_qiskit_runtime
except ImportError:
    from qnn.config import ModelConfig, QNNExperimentConfig
    from qnn.runtime import OptionalDependencyError, QuantumRuntimeContext, _import_qiskit_runtime


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for QNN training but is not installed.")


def _import_qiskit_building():
    try:
        from qiskit.circuit import ParameterVector, QuantumCircuit
        from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.visualization import circuit_drawer
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError("Qiskit is required for QNN circuit construction.") from exc

    _, EstimatorQNN, _, _ = _import_qiskit_runtime()
    return ParameterVector, QuantumCircuit, RealAmplitudes, ZZFeatureMap, SparsePauliOp, circuit_drawer, EstimatorQNN


@dataclass
class CircuitSpec:
    circuit_type: str
    entanglement: str
    n_qubits: int
    n_layers: int
    n_qnn_outputs: int
    dense_hidden_dims: list[int]
    use_dense_head: bool
    initial_weight_scale: float


@dataclass
class QNNBuildResult:
    qnn: Any
    circuit: Any
    model: Any
    n_qnn_outputs: int
    runtime_metadata: dict[str, Any]


def _filter_kwargs(callable_obj, payload: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in payload.items() if key in parameters and value is not None}


def entangling_edges(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    if entanglement == "linear":
        return [(qubit, qubit + 1) for qubit in range(n_qubits - 1)]
    if entanglement == "ring":
        if n_qubits == 1:
            return []
        return [(qubit, qubit + 1) for qubit in range(n_qubits - 1)] + [(n_qubits - 1, 0)]
    raise ValueError("entanglement must be 'linear' or 'ring'.")


def _observable_list(sparse_pauli_op, n_qubits: int, n_outputs: int):
    if n_outputs == 1:
        return sparse_pauli_op.from_list([("Z" * n_qubits, 1.0)])
    max_outputs = 2**n_qubits - 1
    if n_outputs > max_outputs:
        raise ValueError(
            f"Cannot have {n_outputs} outputs with {n_qubits} qubits. Maximum is {max_outputs}."
        )
    observables = []
    for k in range(1, n_outputs + 1):
        bits = np.binary_repr(k, width=n_qubits)
        pauli = "".join("Z" if bit == "1" else "I" for bit in bits)
        observables.append(sparse_pauli_op.from_list([(pauli, 1.0)]))
    return observables


class HybridQNNModel(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(
        self,
        qnn: Any,
        n_outputs: int,
        n_qnn_outputs: int,
        use_dense_head: bool,
        dense_hidden_dims: list[int],
        initial_weight_scale: float,
        seed: int,
    ):
        _require_torch()
        super().__init__()
        from qiskit_machine_learning.connectors import TorchConnector

        rng = np.random.default_rng(seed)
        initial_weights = initial_weight_scale * (2.0 * rng.random(qnn.num_weights) - 1.0)
        self.quantum = TorchConnector(qnn, initial_weights=initial_weights)
        self.use_dense_head = use_dense_head
        self.n_outputs = n_outputs
        self.n_qnn_outputs = n_qnn_outputs

        if use_dense_head:
            hidden_dims = list(dense_hidden_dims) or [32]
            layers: list[Any] = []
            in_dim = n_qnn_outputs
            for hidden in hidden_dims:
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.ReLU())
                in_dim = hidden
            layers.append(nn.Linear(in_dim, n_outputs))
            self.head = nn.Sequential(*layers)
        else:
            self.head = None

    def forward(self, x):
        q_out = self.quantum(x)
        if self.head is None:
            return q_out
        return self.head(q_out)


class QNNBuilder:
    @staticmethod
    def create_spec(model: ModelConfig, n_outputs: int) -> CircuitSpec:
        n_qnn_outputs = 1 if model.use_dense_head else n_outputs
        return CircuitSpec(
            circuit_type=model.circuit_type,
            entanglement=model.entanglement,
            n_qubits=model.n_qubits,
            n_layers=model.n_layers,
            n_qnn_outputs=n_qnn_outputs,
            dense_hidden_dims=list(model.dense_hidden_dims),
            use_dense_head=model.use_dense_head,
            initial_weight_scale=model.initial_weight_scale,
        )

    def build(
        self,
        experiment: QNNExperimentConfig,
        n_outputs: int,
        runtime_context: QuantumRuntimeContext,
    ) -> QNNBuildResult:
        spec = self.create_spec(experiment.model, n_outputs)
        (
            ParameterVector,
            QuantumCircuit,
            RealAmplitudes,
            ZZFeatureMap,
            SparsePauliOp,
            _,
            EstimatorQNN,
        ) = _import_qiskit_building()

        if spec.circuit_type == "zz_feature":
            feature_map = ZZFeatureMap(feature_dimension=spec.n_qubits, reps=1)
            ansatz = RealAmplitudes(
                num_qubits=spec.n_qubits,
                reps=spec.n_layers,
                entanglement=spec.entanglement,
            )
            circuit = feature_map.compose(ansatz)
            input_params = feature_map.parameters
            weight_params = ansatz.parameters
        elif spec.circuit_type == "rxrz":
            x = ParameterVector("x", spec.n_qubits)
            theta = ParameterVector("theta", 2 * spec.n_qubits * spec.n_layers)
            circuit = QuantumCircuit(spec.n_qubits)
            index = 0
            for _layer in range(spec.n_layers):
                for qubit in range(spec.n_qubits):
                    circuit.rx(x[qubit], qubit)
                    circuit.rz(theta[index], qubit)
                    index += 1
                    circuit.rx(theta[index], qubit)
                    index += 1
                for control, target in entangling_edges(spec.n_qubits, spec.entanglement):
                    circuit.cx(control, target)
            input_params = list(x)
            weight_params = list(theta)
        else:
            raise ValueError("circuit_type must be 'zz_feature' or 'rxrz'.")

        observables = _observable_list(SparsePauliOp, spec.n_qubits, spec.n_qnn_outputs)
        qnn_kwargs = {
            "circuit": circuit,
            "estimator": runtime_context.estimator,
            "observables": observables,
            "input_params": input_params,
            "weight_params": weight_params,
            "gradient": runtime_context.gradient,
            "input_gradients": experiment.runtime.input_gradients,
            "default_precision": experiment.runtime.default_precision,
        }
        qnn = EstimatorQNN(**_filter_kwargs(EstimatorQNN.__init__, qnn_kwargs))
        model = HybridQNNModel(
            qnn=qnn,
            n_outputs=n_outputs,
            n_qnn_outputs=spec.n_qnn_outputs,
            use_dense_head=spec.use_dense_head,
            dense_hidden_dims=spec.dense_hidden_dims,
            initial_weight_scale=spec.initial_weight_scale,
            seed=experiment.model.seed,
        )
        return QNNBuildResult(
            qnn=qnn,
            circuit=circuit,
            model=model,
            n_qnn_outputs=spec.n_qnn_outputs,
            runtime_metadata=runtime_context.metadata.__dict__,
        )


def build_estimator_qnn(
    n_qubits: int,
    n_layers: int,
    n_qnn_outputs: int,
    runtime_context: QuantumRuntimeContext | None = None,
    entanglement: str = "circular",
):
    if runtime_context is None:
        raise ValueError("runtime_context is required in the refactored QNN builder.")
    model_config = ModelConfig(
        circuit_type="zz_feature",
        entanglement=entanglement,
        n_qubits=n_qubits,
        n_layers=n_layers,
        use_dense_head=n_qnn_outputs == 1,
    )
    experiment = QNNExperimentConfig(mode="returns", base_dir=".", config_path="", model=model_config)
    result = QNNBuilder().build(experiment, n_outputs=n_qnn_outputs, runtime_context=runtime_context)
    return result.qnn, result.circuit


def build_rxrz_qnn(
    n_qubits: int,
    n_layers: int,
    n_qnn_outputs: int,
    entanglement: str = "ring",
    runtime_context: QuantumRuntimeContext | None = None,
):
    if runtime_context is None:
        raise ValueError("runtime_context is required in the refactored QNN builder.")
    model_config = ModelConfig(
        circuit_type="rxrz",
        entanglement=entanglement,
        n_qubits=n_qubits,
        n_layers=n_layers,
        use_dense_head=n_qnn_outputs == 1,
    )
    experiment = QNNExperimentConfig(mode="returns", base_dir=".", config_path="", model=model_config)
    result = QNNBuilder().build(experiment, n_outputs=n_qnn_outputs, runtime_context=runtime_context)
    return result.qnn, result.circuit


def visualise_circuit(qc, save_path: Optional[str] = None):
    if save_path is None:
        return
    _, _, _, _, _, circuit_drawer, _ = _import_qiskit_building()
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    circuit_drawer(qc, output="mpl", filename=save_path)
