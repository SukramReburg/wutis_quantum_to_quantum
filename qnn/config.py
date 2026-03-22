from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from data.common import (
    DAILY_DATASET_FILENAME,
    WEEKLY_DATASET_FILENAME,
    load_yaml_config,
    project_base_dir,
    resolve_path,
)


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def set_nested(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current = mapping
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def get_nested(mapping: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


@dataclass
class PathConfig:
    models: str = "qnn/models"
    results: str = "qnn/results"
    plots: str = "qnn/plots"
    optuna: str = "qnn/results/optuna"
    backtests: str = "Optimizer/backtest_results"


@dataclass
class RuntimeConfig:
    profile: str = "realism"
    primitive: str = "auto"
    shots: int = 2048
    validation_shots: int = 4096
    evaluation_repeats: int = 3
    method: str = "automatic"
    device: str = "auto"
    precision: str = "double"
    seed: int = 42
    gradient: str = "auto"
    input_gradients: bool = True
    default_precision: float | None = 0.015625
    skip_transpilation: bool = False
    max_parallel_threads: int = 0
    max_parallel_experiments: int = 0
    max_parallel_shots: int = 0
    max_job_size: int | None = None
    max_shot_size: int | None = None
    target_gpus: list[int] = field(default_factory=list)
    use_gpu_if_available: bool = True


@dataclass
class NoiseConfig:
    enabled: bool = False
    family: str = "none"
    single_qubit_prob: float = 0.001
    two_qubit_prob: float = 0.01
    readout_prob: float = 0.01
    t1: float = 100_000.0
    t2: float = 80_000.0
    gate_time_1q: float = 50.0
    gate_time_2q: float = 300.0
    excited_population: float = 0.0
    backend_name: str | None = None


@dataclass
class ResourceConfig:
    prefer_cuda: bool = True
    torch_threads: int = 0
    dataloader_workers: int = 0
    pin_memory: bool = True
    optuna_jobs: int = 1


@dataclass
class ModelConfig:
    feature_mode: str = "angles"
    circuit_type: str = "rxrz"
    entanglement: str = "ring"
    n_qubits: int = 6
    n_layers: int = 4
    use_dense_head: bool = True
    dense_hidden_dims: list[int] = field(default_factory=lambda: [32])
    initial_weight_scale: float = 0.1
    seed: int = 42


@dataclass
class TrainingConfig:
    data_config_path: str = "config/data_config.yaml"
    dataset_frequency: str = "weekly"
    npz_name: str | None = None
    batch_size: int = 32
    n_epochs: int = 30
    optimizer_name: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0
    normalize_targets: bool = True
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    validation_smoothing_alpha: float = 0.3
    min_epochs_before_stop: int = 5
    objective_metric: str = "rmse"
    save_trial_artifacts: bool = False


@dataclass
class LossConfig:
    name: str = "huber"
    delta: float = 0.05


@dataclass
class SchedulerConfig:
    name: str = "plateau"
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 1e-5
    t_max: int = 20


@dataclass
class MetricsConfig:
    sensitivity_eps: float = 0.001
    sensitivity_samples: int = 32
    save_prediction_samples: bool = False
    save_metrics: bool = True
    save_summary: bool = True
    save_model: bool = True
    save_predictions: bool = True
    latest_dirname: str = "latest"
    runs_dirname: str = "runs"


@dataclass
class PlotConfig:
    enabled: bool = True
    include_circuit: bool = True
    max_assets: int = 6
    rolling_window: int = 5
    dpi: int = 180


@dataclass
class TuningConfig:
    enabled: bool = False
    n_trials: int = 20
    n_jobs: int = 1
    timeout_seconds: int | None = None
    study_name: str | None = None
    direction: str = "minimize"
    metric: str = "rmse"
    storage: str | None = None
    search_space: dict[str, Any] = field(default_factory=dict)
    mode_search_space: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    objective: str = "mean_variance"
    solver: str = "auto"
    risk_aversion: float = 1.0
    return_weight: float = 1.0
    target_return: float | None = None
    weight_max: float = 0.35
    long_only: bool = True
    l2_reg: float = 1e-3
    turnover_penalty: float = 0.05
    max_iter: int = 500
    step_size: float = 0.05
    benchmark: str = "equal_weight"


@dataclass
class QNNExperimentConfig:
    mode: str
    base_dir: str
    config_path: str
    paths: PathConfig = field(default_factory=PathConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def resolved_config_path(self) -> str:
        return resolve_path(self.base_dir, self.config_path)

    def resolved_data_config_path(self) -> str:
        return resolve_path(self.base_dir, self.training.data_config_path)

    def resolved_dataset_filename(self) -> str:
        if self.training.npz_name:
            return self.training.npz_name
        if self.training.dataset_frequency == "daily":
            return DAILY_DATASET_FILENAME
        return WEEKLY_DATASET_FILENAME

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_filename"] = self.resolved_dataset_filename()
        return payload


def default_model_config_dict() -> dict[str, Any]:
    defaults = QNNExperimentConfig(
        mode="returns",
        base_dir=".",
        config_path="config/model_config.yaml",
    ).to_dict()
    defaults.pop("dataset_filename", None)
    defaults["experiments"] = {
        "returns": {
            "model": {
                "feature_mode": "angles",
                "circuit_type": "rxrz",
                "entanglement": "ring",
                "n_qubits": 6,
                "n_layers": 4,
                "use_dense_head": True,
            }
        },
        "cov": {
            "model": {
                "feature_mode": "pca",
                "circuit_type": "rxrz",
                "entanglement": "ring",
                "n_qubits": 7,
                "n_layers": 4,
                "use_dense_head": True,
            }
        },
    }
    return defaults


def load_project_config(
    config_path: str = "config/model_config.yaml",
    base_dir: str | None = None,
) -> dict[str, Any]:
    project_dir = project_base_dir(__file__, base_dir)
    merged = default_model_config_dict()
    loaded = load_yaml_config(config_path, project_dir)
    return deep_merge(merged, loaded)


def _build_dataclass(cls, data: dict[str, Any]):
    field_names = cls.__dataclass_fields__.keys()
    return cls(**{name: deepcopy(data.get(name)) for name in field_names if name in data})


def load_experiment_config(
    mode: str,
    config_path: str = "config/model_config.yaml",
    base_dir: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> QNNExperimentConfig:
    if mode not in {"returns", "cov"}:
        raise ValueError("mode must be 'returns' or 'cov'.")

    project_dir = project_base_dir(__file__, base_dir)
    merged = load_project_config(config_path=config_path, base_dir=project_dir)
    experiment_overrides = merged.get("experiments", {}).get(mode, {})
    merged = deep_merge(merged, experiment_overrides)
    if overrides:
        merged = deep_merge(merged, overrides)

    config = QNNExperimentConfig(
        mode=mode,
        base_dir=project_dir,
        config_path=config_path,
        paths=_build_dataclass(PathConfig, merged.get("paths", {})),
        runtime=_build_dataclass(RuntimeConfig, merged.get("runtime", {})),
        noise=_build_dataclass(NoiseConfig, merged.get("noise", {})),
        resources=_build_dataclass(ResourceConfig, merged.get("resources", {})),
        model=_build_dataclass(ModelConfig, merged.get("model", {})),
        training=_build_dataclass(TrainingConfig, merged.get("training", {})),
        loss=_build_dataclass(LossConfig, merged.get("loss", {})),
        scheduler=_build_dataclass(SchedulerConfig, merged.get("scheduler", {})),
        metrics=_build_dataclass(MetricsConfig, merged.get("metrics", {})),
        plots=_build_dataclass(PlotConfig, merged.get("plots", {})),
        tuning=_build_dataclass(TuningConfig, merged.get("tuning", {})),
        optimizer=_build_dataclass(OptimizerConfig, merged.get("optimizer", {})),
    )
    if not config.training.npz_name:
        config.training.npz_name = config.resolved_dataset_filename()
    if config.runtime.profile == "realism" and config.runtime.evaluation_repeats < 2:
        config.runtime.evaluation_repeats = 2
    if config.training.objective_metric == "loss":
        config.training.objective_metric = config.loss.name
    return config
