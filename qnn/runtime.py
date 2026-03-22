from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any

from .config import NoiseConfig, RuntimeConfig


class OptionalDependencyError(RuntimeError):
    pass


def _import_qiskit_runtime():
    try:
        from qiskit_aer.primitives import Estimator as AerEstimator
        from qiskit_machine_learning.neural_networks import EstimatorQNN
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "Qiskit Aer and qiskit-machine-learning are required for QNN execution."
        ) from exc

    try:
        from qiskit.primitives import StatevectorEstimator
    except Exception:
        StatevectorEstimator = None

    try:
        from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
    except Exception:
        ParamShiftEstimatorGradient = None

    return AerEstimator, EstimatorQNN, StatevectorEstimator, ParamShiftEstimatorGradient


def _filter_kwargs(callable_obj, payload: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in payload.items() if key in parameters and value is not None}


@dataclass
class RuntimeMetadata:
    profile: str
    primitive_name: str
    backend_options: dict[str, Any]
    run_options: dict[str, Any]
    gradient: str
    device: str
    noise: dict[str, Any]


@dataclass
class QuantumRuntimeContext:
    estimator: Any
    gradient: Any
    metadata: RuntimeMetadata


class NoiseModelFactory:
    @staticmethod
    def describe(config: NoiseConfig) -> dict[str, Any]:
        payload = asdict(config)
        payload["enabled"] = bool(config.enabled and config.family != "none")
        return payload

    @staticmethod
    def build(config: NoiseConfig):
        if not config.enabled or config.family == "none":
            return None

        try:
            from qiskit_aer.noise import (
                NoiseModel,
                ReadoutError,
                amplitude_damping_error,
                depolarizing_error,
                phase_damping_error,
                thermal_relaxation_error,
            )
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "Qiskit Aer noise models are required when explicit noise is enabled."
            ) from exc

        noise_model = NoiseModel()
        family = config.family
        one_qubit_gates = ["rx", "ry", "rz", "sx", "x"]
        two_qubit_gates = ["cx", "cz"]

        if family == "depolarizing":
            one_qubit_error = depolarizing_error(config.single_qubit_prob, 1)
            two_qubit_error = depolarizing_error(config.two_qubit_prob, 2)
            for gate in one_qubit_gates:
                noise_model.add_all_qubit_quantum_error(one_qubit_error, gate)
            for gate in two_qubit_gates:
                noise_model.add_all_qubit_quantum_error(two_qubit_error, gate)
            return noise_model

        if family == "readout":
            probability = min(max(config.readout_prob, 0.0), 0.5)
            readout = ReadoutError(
                [[1.0 - probability, probability], [probability, 1.0 - probability]]
            )
            noise_model.add_all_qubit_readout_error(readout)
            return noise_model

        if family == "thermal_relaxation":
            one_qubit_error = thermal_relaxation_error(
                config.t1,
                config.t2,
                config.gate_time_1q,
                config.excited_population,
            )
            two_qubit_error = thermal_relaxation_error(
                config.t1,
                config.t2,
                config.gate_time_2q,
                config.excited_population,
            ).tensor(
                thermal_relaxation_error(
                    config.t1,
                    config.t2,
                    config.gate_time_2q,
                    config.excited_population,
                )
            )
            for gate in one_qubit_gates:
                noise_model.add_all_qubit_quantum_error(one_qubit_error, gate)
            for gate in two_qubit_gates:
                noise_model.add_all_qubit_quantum_error(two_qubit_error, gate)
            return noise_model

        if family == "amplitude_damping":
            error = amplitude_damping_error(config.single_qubit_prob)
            for gate in one_qubit_gates:
                noise_model.add_all_qubit_quantum_error(error, gate)
            for gate in two_qubit_gates:
                noise_model.add_all_qubit_quantum_error(error.tensor(error), gate)
            return noise_model

        if family == "phase_damping":
            error = phase_damping_error(config.single_qubit_prob)
            for gate in one_qubit_gates:
                noise_model.add_all_qubit_quantum_error(error, gate)
            for gate in two_qubit_gates:
                noise_model.add_all_qubit_quantum_error(error.tensor(error), gate)
            return noise_model

        if family == "backend_calibrated":
            raise ValueError(
                "backend_calibrated noise requires a live backend object and is not supported via YAML only."
            )

        raise ValueError(f"Unsupported noise family '{family}'.")


class QuantumRuntimeFactory:
    @staticmethod
    def resolve_device(runtime: RuntimeConfig, available_devices: list[str] | None = None) -> str:
        if runtime.device and runtime.device.lower() != "auto":
            return runtime.device.upper()
        if available_devices and runtime.use_gpu_if_available and "GPU" in available_devices:
            return "GPU"
        return "CPU"

    @staticmethod
    def resolve_backend_options(runtime: RuntimeConfig, noise_model=None) -> dict[str, Any]:
        backend_options: dict[str, Any] = {
            "method": runtime.method,
            "device": runtime.device,
            "precision": runtime.precision,
            "max_parallel_threads": runtime.max_parallel_threads,
            "max_parallel_experiments": runtime.max_parallel_experiments,
            "max_parallel_shots": runtime.max_parallel_shots,
            "max_job_size": runtime.max_job_size,
            "max_shot_size": runtime.max_shot_size,
        }
        if runtime.target_gpus:
            backend_options["target_gpus"] = list(runtime.target_gpus)
        if noise_model is not None:
            backend_options["noise_model"] = noise_model
        return {key: value for key, value in backend_options.items() if value is not None}

    @classmethod
    def create(cls, runtime: RuntimeConfig, noise: NoiseConfig) -> QuantumRuntimeContext:
        AerEstimator, _, StatevectorEstimator, ParamShiftEstimatorGradient = _import_qiskit_runtime()

        noise_model = NoiseModelFactory.build(noise)
        available_devices = None
        if runtime.device == "auto":
            try:
                from qiskit_aer import AerSimulator

                available_devices = list(AerSimulator().available_devices())
            except Exception:
                available_devices = None

        resolved_device = cls.resolve_device(runtime, available_devices=available_devices)
        backend_options = cls.resolve_backend_options(
            RuntimeConfig(
                **{
                    **asdict(runtime),
                    "device": resolved_device,
                }
            ),
            noise_model=noise_model,
        )

        profile = runtime.profile
        gradient = None
        gradient_name = "auto"
        if runtime.gradient == "param_shift" and ParamShiftEstimatorGradient is not None:
            gradient = ParamShiftEstimatorGradient()
            gradient_name = "param_shift"
        elif runtime.gradient not in {"auto", "default", "param_shift"}:
            raise ValueError(f"Unsupported gradient mode '{runtime.gradient}'.")

        if profile == "fast_exact" and StatevectorEstimator is not None:
            estimator = StatevectorEstimator()
            primitive_name = "StatevectorEstimator"
            run_options: dict[str, Any] = {"shots": None}
        else:
            shots = runtime.shots
            if profile == "fast_exact":
                shots = 0
            run_options = {"shots": shots or None, "seed": runtime.seed}
            estimator_kwargs = {
                "backend_options": backend_options,
                "run_options": {k: v for k, v in run_options.items() if v is not None},
                "approximation": shots in {0, None},
                "skip_transpilation": runtime.skip_transpilation,
            }
            estimator = AerEstimator(**_filter_kwargs(AerEstimator.__init__, estimator_kwargs))
            primitive_name = "AerEstimator"

        metadata = RuntimeMetadata(
            profile=profile,
            primitive_name=primitive_name,
            backend_options={k: v for k, v in backend_options.items() if k != "noise_model"},
            run_options=run_options,
            gradient=gradient_name,
            device=resolved_device,
            noise=NoiseModelFactory.describe(noise),
        )
        return QuantumRuntimeContext(estimator=estimator, gradient=gradient, metadata=metadata)
