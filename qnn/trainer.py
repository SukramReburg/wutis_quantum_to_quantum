from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

from data.common import npz_scalar, npz_string_list, resolve_path

try:
    from .artifacts import ArtifactManager, RunArtifacts
    from .config import QNNExperimentConfig, load_experiment_config
    from .encode import FeatureEncoder
    from .losses import LossFactory, LossSpec, regression_metrics
    from .metrics import QNNMetricsCollector
    from .model import QNNBuilder, visualise_circuit
    from .plotting import QNNPlotter
    from .runtime import QuantumRuntimeFactory
except ImportError:
    from qnn.artifacts import ArtifactManager, RunArtifacts
    from qnn.config import QNNExperimentConfig, load_experiment_config
    from qnn.encode import FeatureEncoder
    from qnn.losses import LossFactory, LossSpec, regression_metrics
    from qnn.metrics import QNNMetricsCollector
    from qnn.model import QNNBuilder, visualise_circuit
    from qnn.plotting import QNNPlotter
    from qnn.runtime import QuantumRuntimeFactory

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:
    torch = None
    DataLoader = None
    TensorDataset = None


def _require_torch() -> None:
    if torch is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required for QNN training but is not installed.")


@dataclass
class DatasetSplit:
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    Y_train: np.ndarray
    Y_test: np.ndarray
    asset_symbols: list[str]
    sample_dates: list[str]
    target_frequency: str
    dataset_path: str


@dataclass
class TrainingResult:
    model: Any
    qnn: Any
    artifacts: RunArtifacts
    summary: dict[str, Any]
    metrics: dict[str, Any]
    predictions: dict[str, Any]


class QNNTrainer:
    def __init__(
        self,
        experiment: QNNExperimentConfig,
        builder: QNNBuilder | None = None,
        runtime_factory: QuantumRuntimeFactory | None = None,
        artifact_manager: ArtifactManager | None = None,
        plotter: QNNPlotter | None = None,
    ):
        self.experiment = experiment
        self.builder = builder or QNNBuilder()
        self.runtime_factory = runtime_factory or QuantumRuntimeFactory()
        self.artifact_manager = artifact_manager or ArtifactManager(experiment)
        self.plotter = plotter or QNNPlotter(
            dpi=experiment.plots.dpi,
            rolling_window=experiment.plots.rolling_window,
            max_assets=experiment.plots.max_assets,
        )

    @staticmethod
    def _fmt_metric(value: float | int | None, digits: int = 6) -> str:
        if value is None:
            return "n/a"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if np.isnan(numeric) or np.isinf(numeric):
            return str(numeric)
        return f"{numeric:.{digits}f}"

    def _print_run_header(
        self,
        artifacts: RunArtifacts,
        dataset: DatasetSplit,
        device,
        loss_name: str,
    ) -> None:
        print(
            (
                f"[{self.experiment.mode}] run_tag={artifacts.run_tag} "
                f"device={device.type} runtime={self.experiment.runtime.profile} "
                f"loss={loss_name} epochs={self.experiment.training.n_epochs} "
                f"train_samples={len(dataset.X_train_raw)} test_samples={len(dataset.X_test_raw)} "
                f"outputs={dataset.Y_train.shape[1]}"
            ),
            flush=True,
        )

    def _print_epoch_progress(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        current_lr: float,
        smoothed_val: float,
        monitor_name: str,
        monitor_value: float,
        best_monitor: float,
        best_epoch: int,
        epochs_without_improvement: int,
        improved: bool,
    ) -> None:
        marker = "*" if improved else ""
        print(
            (
                f"[{self.experiment.mode}] "
                f"epoch={epoch:03d}/{self.experiment.training.n_epochs} "
                f"train_loss={self._fmt_metric(train_metrics['loss'])} "
                f"val_loss={self._fmt_metric(val_metrics['loss'])} "
                f"train_rmse={self._fmt_metric(train_metrics['rmse'])} "
                f"val_rmse={self._fmt_metric(val_metrics['rmse'])} "
                f"val_mae={self._fmt_metric(val_metrics['mae'])} "
                f"lr={self._fmt_metric(current_lr, digits=8)} "
                f"smoothed_val={self._fmt_metric(smoothed_val)} "
                f"{monitor_name}={self._fmt_metric(monitor_value)} "
                f"best={self._fmt_metric(best_monitor)} "
                f"best_epoch={best_epoch} "
                f"patience={epochs_without_improvement}/{self.experiment.training.early_stopping_patience} "
                f"uncertainty={self._fmt_metric(val_metrics.get('uncertainty'))}"
                f"{marker}"
            ),
            flush=True,
        )

    def _set_seeds(self) -> None:
        _require_torch()
        seed = self.experiment.runtime.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _configure_resources(self) -> torch.device:
        _require_torch()
        self._set_seeds()
        if self.experiment.resources.torch_threads > 0:
            torch.set_num_threads(self.experiment.resources.torch_threads)
        prefer_cuda = self.experiment.resources.prefer_cuda and torch.cuda.is_available()
        return torch.device("cuda" if prefer_cuda else "cpu")

    def load_dataset(self) -> DatasetSplit:
        data_config_path = self.experiment.resolved_data_config_path()
        with open(data_config_path, "r", encoding="utf-8") as handle:
            data_config = yaml.safe_load(handle)
        processed_dir = resolve_path(self.experiment.base_dir, data_config["paths"]["processed"])
        dataset_path = os.path.join(processed_dir, self.experiment.resolved_dataset_filename())
        bundle = np.load(dataset_path)

        if self.experiment.mode == "returns":
            x_train_key = "X_train_ret"
            x_test_key = "X_test_ret"
            y_train_key = "Y_train_ret"
            y_test_key = "Y_test_ret"
            sample_key = "sample_dates_ret"
        else:
            x_train_key = "X_train_cov"
            x_test_key = "X_test_cov"
            y_train_key = "Y_train_cov"
            y_test_key = "Y_test_cov"
            sample_key = "sample_dates_cov"

        asset_symbols = npz_string_list(bundle, "asset_symbols")
        sample_dates = npz_string_list(bundle, sample_key) or npz_string_list(bundle, "sample_dates")
        if sample_dates and len(sample_dates) != len(bundle[y_train_key]) + len(bundle[y_test_key]):
            raise ValueError(f"Sample dates in '{sample_key}' do not match the train/test split.")

        target_frequency = npz_scalar(bundle, "target_frequency", self.experiment.training.dataset_frequency)
        return DatasetSplit(
            X_train_raw=np.asarray(bundle[x_train_key], dtype=np.float32),
            X_test_raw=np.asarray(bundle[x_test_key], dtype=np.float32),
            Y_train=np.asarray(bundle[y_train_key], dtype=np.float32),
            Y_test=np.asarray(bundle[y_test_key], dtype=np.float32),
            asset_symbols=asset_symbols,
            sample_dates=sample_dates[-len(bundle[y_test_key]) :] if sample_dates else [],
            target_frequency=str(target_frequency),
            dataset_path=dataset_path,
        )

    def _build_optimizer(self, model) -> Any:
        _require_torch()
        name = self.experiment.training.optimizer_name.lower()
        kwargs = {
            "lr": self.experiment.training.learning_rate,
            "weight_decay": self.experiment.training.weight_decay,
        }
        if name == "adam":
            return torch.optim.Adam(model.parameters(), **kwargs)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), **kwargs)
        if name == "sgd":
            return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
        raise ValueError(f"Unsupported optimizer '{name}'.")

    def _build_scheduler(self, optimizer) -> Any | None:
        _require_torch()
        name = self.experiment.scheduler.name.lower()
        if name == "none":
            return None
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.experiment.scheduler.factor,
                patience=self.experiment.scheduler.patience,
                min_lr=self.experiment.scheduler.min_lr,
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.experiment.scheduler.t_max,
                eta_min=self.experiment.scheduler.min_lr,
            )
        raise ValueError(f"Unsupported scheduler '{name}'.")

    def _evaluate(
        self,
        model,
        x_tensor,
        y_tensor,
        loss_fn,
        target_scaler: StandardScaler | None,
        repeats: int,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray | None]:
        _require_torch()
        was_training = model.training
        model.eval()
        predictions = []
        with torch.no_grad():
            for _ in range(max(1, repeats)):
                predictions.append(model(x_tensor))
        stack = torch.stack(predictions, dim=0)
        mean_pred = stack.mean(dim=0)
        std_pred = stack.std(dim=0, unbiased=False) if stack.shape[0] > 1 else torch.zeros_like(mean_pred)
        loss_value = float(loss_fn(mean_pred, y_tensor).item())
        model.train(was_training)

        y_true_np = y_tensor.detach().cpu().numpy()
        y_pred_np = mean_pred.detach().cpu().numpy()
        y_std_np = std_pred.detach().cpu().numpy()
        if target_scaler is not None:
            y_true_np = target_scaler.inverse_transform(y_true_np)
            y_pred_np = target_scaler.inverse_transform(y_pred_np)
            y_std_np = y_std_np * target_scaler.scale_
        metrics = regression_metrics(y_true_np, y_pred_np, delta=self.experiment.loss.delta)
        metrics["loss"] = loss_value
        metrics["uncertainty"] = float(np.mean(y_std_np))
        return metrics, y_true_np, y_pred_np, y_std_np

    def _summary_payload(
        self,
        artifacts: RunArtifacts,
        dataset: DatasetSplit,
        runtime_metadata: dict[str, Any],
        metrics: QNNMetricsCollector,
        test_metrics: dict[str, float],
        best_epoch: int,
        stopped_epoch: int,
    ) -> dict[str, Any]:
        return {
            "run_tag": artifacts.run_tag,
            "mode": self.experiment.mode,
            "dataset_path": dataset.dataset_path,
            "dataset_frequency": dataset.target_frequency,
            "asset_symbols": list(dataset.asset_symbols),
            "n_assets": len(dataset.asset_symbols),
            "n_outputs": int(dataset.Y_train.shape[1]),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "loss_name": self.experiment.loss.name,
            "runtime_profile": self.experiment.runtime.profile,
            "runtime": runtime_metadata,
            "config": self.experiment.to_dict(),
            "test_metrics": test_metrics,
            "latest_learning_rate": float(metrics.learning_rate_per_epoch[-1]) if metrics.learning_rate_per_epoch else None,
            "sensitivity_score": float(metrics.sensitivity_scores[-1]) if metrics.sensitivity_scores else None,
        }

    def train(
        self,
        no_plots: bool = False,
        run_tag: str | None = None,
        save_artifacts: bool = True,
    ) -> TrainingResult:
        _require_torch()
        device = self._configure_resources()
        dataset = self.load_dataset()

        if dataset.Y_train.ndim != 2 or dataset.Y_test.ndim != 2:
            raise ValueError("Targets must be 2D arrays of shape (samples, outputs).")

        encoder = FeatureEncoder(
            n_qubits=self.experiment.model.n_qubits,
            feature_mode=self.experiment.model.feature_mode,
        )
        x_train = encoder.fit_transform(dataset.X_train_raw)
        x_test = encoder.transform(dataset.X_test_raw)

        target_scaler = StandardScaler() if self.experiment.training.normalize_targets else None
        y_train = dataset.Y_train.copy()
        y_test = dataset.Y_test.copy()
        if target_scaler is not None:
            y_train = target_scaler.fit_transform(y_train).astype(np.float32)
            y_test = target_scaler.transform(y_test).astype(np.float32)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

        runtime_context = self.runtime_factory.create(self.experiment.runtime, self.experiment.noise)
        build_result = self.builder.build(
            experiment=self.experiment,
            n_outputs=dataset.Y_train.shape[1],
            runtime_context=runtime_context,
        )
        model = build_result.model.to(device)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        loss_fn, loss_name = LossFactory.create(LossSpec(name=self.experiment.loss.name, delta=self.experiment.loss.delta))

        train_loader = DataLoader(
            TensorDataset(x_train_tensor, y_train_tensor),
            batch_size=self.experiment.training.batch_size,
            shuffle=True,
            num_workers=self.experiment.resources.dataloader_workers,
            pin_memory=self.experiment.resources.pin_memory and device.type == "cuda",
        )

        artifacts = self.artifact_manager.prepare(self.experiment.mode, run_tag=run_tag)
        if self.experiment.plots.enabled and not no_plots and self.experiment.plots.include_circuit:
            visualise_circuit(build_result.circuit, os.path.join(artifacts.archive.plots_dir, "circuit.png"))
            visualise_circuit(build_result.circuit, os.path.join(artifacts.latest.plots_dir, "circuit.png"))

        metrics = QNNMetricsCollector()
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_monitor = float("inf")
        smoothed_val = None
        epochs_without_improvement = 0
        self._print_run_header(artifacts=artifacts, dataset=dataset, device=device, loss_name=loss_name)

        for epoch in range(1, self.experiment.training.n_epochs + 1):
            model.train()
            metrics.start_epoch()
            train_preds = []
            train_targets = []
            total_loss = 0.0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                metrics.log_after_backward(model)
                with torch.no_grad():
                    metrics.log_qnn_output(model.quantum(xb))
                if self.experiment.training.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.experiment.training.gradient_clip_norm
                    )
                optimizer.step()
                total_loss += float(loss.item()) * xb.size(0)
                train_preds.append(preds.detach().cpu().numpy())
                train_targets.append(yb.detach().cpu().numpy())

            train_pred_np = np.concatenate(train_preds, axis=0)
            train_true_np = np.concatenate(train_targets, axis=0)
            if target_scaler is not None:
                train_pred_eval = target_scaler.inverse_transform(train_pred_np)
                train_true_eval = target_scaler.inverse_transform(train_true_np)
            else:
                train_pred_eval = train_pred_np
                train_true_eval = train_true_np
            train_metrics = regression_metrics(
                train_true_eval,
                train_pred_eval,
                delta=self.experiment.loss.delta,
            )
            train_metrics["loss"] = total_loss / len(train_loader.dataset)

            val_metrics, y_true_eval, y_pred_eval, y_std_eval = self._evaluate(
                model=model,
                x_tensor=x_test_tensor,
                y_tensor=y_test_tensor,
                loss_fn=loss_fn,
                target_scaler=target_scaler,
                repeats=self.experiment.runtime.evaluation_repeats,
            )

            if smoothed_val is None:
                smoothed_val = val_metrics["loss"]
            else:
                alpha = self.experiment.training.validation_smoothing_alpha
                smoothed_val = alpha * val_metrics["loss"] + (1.0 - alpha) * smoothed_val

            current_lr = float(optimizer.param_groups[0]["lr"])
            metrics.end_epoch(
                model=model,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr,
                smoothed_val_loss=float(smoothed_val),
                validation_uncertainty=float(val_metrics["uncertainty"]),
            )

            if scheduler is not None:
                if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    scheduler.step(smoothed_val)
                else:
                    scheduler.step()

            monitor_name = self.experiment.training.objective_metric
            if monitor_name == "loss":
                monitor_value = float(smoothed_val)
            else:
                monitor_value = float(val_metrics[monitor_name])
            improved = monitor_value < (best_monitor - self.experiment.training.early_stopping_min_delta)
            if improved:
                best_monitor = monitor_value
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self._print_epoch_progress(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                current_lr=current_lr,
                smoothed_val=float(smoothed_val),
                monitor_name=monitor_name,
                monitor_value=monitor_value,
                best_monitor=best_monitor,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                improved=improved,
            )

            if (
                epoch >= self.experiment.training.min_epochs_before_stop
                and epochs_without_improvement >= self.experiment.training.early_stopping_patience
            ):
                print(
                    (
                        f"[{self.experiment.mode}] early_stopping "
                        f"stopped_epoch={epoch} best_epoch={best_epoch} "
                        f"best_{monitor_name}={self._fmt_metric(best_monitor)}"
                    ),
                    flush=True,
                )
                break

        stopped_epoch = len(metrics.train_loss_per_epoch)
        model.load_state_dict(best_state)
        final_metrics, y_true_test, y_pred_test, y_pred_std = self._evaluate(
            model=model,
            x_tensor=x_test_tensor,
            y_tensor=y_test_tensor,
            loss_fn=loss_fn,
            target_scaler=target_scaler,
            repeats=max(1, self.experiment.runtime.evaluation_repeats),
        )
        metrics.run_sensitivity_test(
            model,
            x_test_tensor,
            eps=self.experiment.metrics.sensitivity_eps,
            n_samples=self.experiment.metrics.sensitivity_samples,
        )

        prediction_payload: dict[str, Any] = {
            "Y_pred_test": y_pred_test.astype(np.float32),
            "Y_true_test": y_true_test.astype(np.float32),
            "Y_pred_std_test": y_pred_std.astype(np.float32) if y_pred_std is not None else np.zeros_like(y_pred_test),
            "asset_symbols": np.asarray(dataset.asset_symbols, dtype=str),
            "sample_dates": np.asarray(dataset.sample_dates, dtype=str),
            "sample_dates_test": np.asarray(dataset.sample_dates, dtype=str),
            "target_frequency": np.asarray(dataset.target_frequency, dtype=str),
            "runtime_profile": np.asarray(self.experiment.runtime.profile, dtype=str),
            "loss_name": np.asarray(loss_name, dtype=str),
            "best_epoch": np.asarray(best_epoch, dtype=np.int32),
            "mode": np.asarray(self.experiment.mode, dtype=str),
        }
        if self.experiment.metrics.save_prediction_samples:
            prediction_payload["feature_metadata"] = np.asarray(
                json.dumps(encoder.metadata()),
                dtype=str,
            )

        metrics_payload = metrics.as_dict()
        metrics_payload["meta"] = np.asarray(
            [
                json.dumps(
                    {
                        "run_tag": artifacts.run_tag,
                        "mode": self.experiment.mode,
                        "loss_name": loss_name,
                        "runtime_profile": self.experiment.runtime.profile,
                    }
                )
            ],
            dtype=object,
        )

        summary = self._summary_payload(
            artifacts=artifacts,
            dataset=dataset,
            runtime_metadata=build_result.runtime_metadata,
            metrics=metrics,
            test_metrics=final_metrics,
            best_epoch=best_epoch,
            stopped_epoch=stopped_epoch,
        )
        if save_artifacts and self.experiment.metrics.save_predictions:
            self.artifact_manager.save_predictions(artifacts, prediction_payload)
        if save_artifacts and self.experiment.metrics.save_metrics:
            self.artifact_manager.save_metrics(artifacts, metrics_payload)
        if save_artifacts and self.experiment.metrics.save_summary:
            self.artifact_manager.save_summary(artifacts, summary)
        if save_artifacts and self.experiment.metrics.save_model:
            self.artifact_manager.save_model_state(artifacts, model.state_dict())
        if save_artifacts and self.experiment.plots.enabled and not no_plots:
            self.plotter.save_all(
                plots_dir=artifacts.archive.plots_dir,
                mode=self.experiment.mode,
                metrics=metrics.as_dict(),
                predictions=prediction_payload,
                summary=summary,
            )
            self.plotter.save_all(
                plots_dir=artifacts.latest.plots_dir,
                mode=self.experiment.mode,
                metrics=metrics.as_dict(),
                predictions=prediction_payload,
                summary=summary,
            )

        print(
            (
                f"[{self.experiment.mode}] completed "
                f"best_epoch={best_epoch} stopped_epoch={stopped_epoch} "
                f"test_rmse={self._fmt_metric(final_metrics.get('rmse'))} "
                f"test_mae={self._fmt_metric(final_metrics.get('mae'))} "
                f"artifacts={artifacts.latest.root}"
            ),
            flush=True,
        )

        return TrainingResult(
            model=model,
            qnn=build_result.qnn,
            artifacts=artifacts,
            summary=summary,
            metrics=metrics.as_dict(),
            predictions=prediction_payload,
        )


def train_qnn_from_npz(
    config_path: str,
    mode: str,
    n_qubits: int,
    n_layers: int = 2,
    feature_mode: str = "angles",
    use_dense_head: bool = True,
    npz_name: Optional[str] = None,
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    circuit_type: str = "zz_feature",
    entanglement: str = "ring",
    save_artifacts: bool = True,
    runtime_profile: Optional[str] = None,
    no_plots: bool = False,
) -> dict[str, Any]:
    overrides = {
        "model": {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "feature_mode": feature_mode,
            "use_dense_head": use_dense_head,
            "circuit_type": circuit_type,
            "entanglement": entanglement,
        },
        "training": {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    }
    if npz_name is not None:
        overrides["training"]["npz_name"] = npz_name
    if runtime_profile is not None:
        overrides["runtime"] = {"profile": runtime_profile}
    experiment = load_experiment_config(mode=mode, config_path=config_path, overrides=overrides)
    result = QNNTrainer(experiment).train(save_artifacts=save_artifacts, no_plots=no_plots)
    return {
        "model": result.model,
        "qnn": result.qnn,
        "Y_pred_test": result.predictions["Y_pred_test"],
        "Y_true_test": result.predictions["Y_true_test"],
        "metrics": result.metrics,
        "summary": result.summary,
        "final_mse": result.summary["test_metrics"]["mse"],
    }
