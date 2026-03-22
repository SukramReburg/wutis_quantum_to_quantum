from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for QNN metric collection.")


@dataclass
class QNNMetricsCollector:
    train_loss_per_epoch: list[float] = field(default_factory=list)
    val_loss_per_epoch: list[float] = field(default_factory=list)
    smoothed_val_loss_per_epoch: list[float] = field(default_factory=list)
    train_mse_per_epoch: list[float] = field(default_factory=list)
    val_mse_per_epoch: list[float] = field(default_factory=list)
    train_rmse_per_epoch: list[float] = field(default_factory=list)
    val_rmse_per_epoch: list[float] = field(default_factory=list)
    train_mae_per_epoch: list[float] = field(default_factory=list)
    val_mae_per_epoch: list[float] = field(default_factory=list)
    val_sign_accuracy_per_epoch: list[float] = field(default_factory=list)
    learning_rate_per_epoch: list[float] = field(default_factory=list)
    validation_uncertainty_per_epoch: list[float] = field(default_factory=list)
    quantum_grad_norm_per_epoch: list[float] = field(default_factory=list)
    classical_grad_norm_per_epoch: list[float] = field(default_factory=list)
    quantum_update_norm_per_epoch: list[float] = field(default_factory=list)
    classical_update_norm_per_epoch: list[float] = field(default_factory=list)
    update_balance_ratio_per_epoch: list[float] = field(default_factory=list)
    qnn_output_mean_per_epoch: list[float] = field(default_factory=list)
    qnn_output_var_per_epoch: list[float] = field(default_factory=list)
    qnn_output_min_per_epoch: list[float] = field(default_factory=list)
    qnn_output_max_per_epoch: list[float] = field(default_factory=list)
    theta_trajectory: list[list[float]] = field(default_factory=list)
    sensitivity_scores: list[float] = field(default_factory=list)

    _quantum_grad_sum: float = 0.0
    _classical_grad_sum: float = 0.0
    _n_steps: int = 0
    _classical_n_steps: int = 0
    _qnn_out_sum: float = 0.0
    _qnn_out_sq_sum: float = 0.0
    _qnn_out_min: float = float("inf")
    _qnn_out_max: float = float("-inf")
    _qnn_out_count: int = 0
    _prev_quantum_params: Any = None
    _prev_classical_params: Any = None

    def start_epoch(self) -> None:
        self._quantum_grad_sum = 0.0
        self._classical_grad_sum = 0.0
        self._n_steps = 0
        self._classical_n_steps = 0
        self._qnn_out_sum = 0.0
        self._qnn_out_sq_sum = 0.0
        self._qnn_out_min = float("inf")
        self._qnn_out_max = float("-inf")
        self._qnn_out_count = 0

    def _grad_norm(self, parameters) -> float:
        _require_torch()
        total_sq = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float().cpu()
            total_sq += float(torch.sum(grad * grad).item())
        return float(total_sq**0.5) if total_sq > 0.0 else 0.0

    def log_after_backward(self, model) -> None:
        quantum_grad = self._grad_norm(model.quantum.parameters())
        self._quantum_grad_sum += quantum_grad
        self._n_steps += 1

        if getattr(model, "head", None) is not None:
            classical_grad = self._grad_norm(model.head.parameters())
            self._classical_grad_sum += classical_grad
            self._classical_n_steps += 1

    def log_qnn_output(self, q_out) -> None:
        _require_torch()
        values = q_out.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        self._qnn_out_sum += float(values.sum().item())
        self._qnn_out_sq_sum += float((values * values).sum().item())
        self._qnn_out_min = min(self._qnn_out_min, float(values.min().item()))
        self._qnn_out_max = max(self._qnn_out_max, float(values.max().item()))
        self._qnn_out_count += int(values.numel())

    def _flatten(self, parameters):
        _require_torch()
        chunks = [parameter.detach().reshape(-1).cpu() for parameter in parameters]
        if not chunks:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    def _log_parameter_updates(self, model) -> None:
        _require_torch()
        quantum_vector = self._flatten(model.quantum.parameters())
        if self._prev_quantum_params is None or quantum_vector.numel() == 0:
            quantum_update = 0.0
        else:
            quantum_update = float(torch.norm(quantum_vector - self._prev_quantum_params, p=2).item())
        self._prev_quantum_params = quantum_vector

        if getattr(model, "head", None) is not None:
            classical_vector = self._flatten(model.head.parameters())
        else:
            classical_vector = torch.empty(0, dtype=torch.float32)

        if self._prev_classical_params is None or classical_vector.numel() == 0:
            classical_update = 0.0
        else:
            classical_update = float(
                torch.norm(classical_vector - self._prev_classical_params, p=2).item()
            )
        self._prev_classical_params = classical_vector

        self.quantum_update_norm_per_epoch.append(quantum_update)
        self.classical_update_norm_per_epoch.append(classical_update)
        self.theta_trajectory.append(quantum_vector.tolist() if quantum_vector.numel() else [])
        if classical_update > 0.0:
            self.update_balance_ratio_per_epoch.append(quantum_update / classical_update)
        else:
            self.update_balance_ratio_per_epoch.append(float("nan"))

    def end_epoch(
        self,
        model,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        learning_rate: float,
        smoothed_val_loss: float,
        validation_uncertainty: float,
    ) -> None:
        quantum_grad = self._quantum_grad_sum / self._n_steps if self._n_steps else 0.0
        classical_grad = (
            self._classical_grad_sum / self._classical_n_steps if self._classical_n_steps else float("nan")
        )
        self.quantum_grad_norm_per_epoch.append(quantum_grad)
        self.classical_grad_norm_per_epoch.append(classical_grad)

        if self._qnn_out_count:
            mean = self._qnn_out_sum / self._qnn_out_count
            mean_sq = self._qnn_out_sq_sum / self._qnn_out_count
            variance = max(0.0, mean_sq - mean * mean)
            q_min = self._qnn_out_min
            q_max = self._qnn_out_max
        else:
            mean = variance = q_min = q_max = 0.0

        self.qnn_output_mean_per_epoch.append(mean)
        self.qnn_output_var_per_epoch.append(variance)
        self.qnn_output_min_per_epoch.append(q_min)
        self.qnn_output_max_per_epoch.append(q_max)

        self.train_loss_per_epoch.append(float(train_metrics["loss"]))
        self.val_loss_per_epoch.append(float(val_metrics["loss"]))
        self.smoothed_val_loss_per_epoch.append(float(smoothed_val_loss))
        self.train_mse_per_epoch.append(float(train_metrics["mse"]))
        self.val_mse_per_epoch.append(float(val_metrics["mse"]))
        self.train_rmse_per_epoch.append(float(train_metrics["rmse"]))
        self.val_rmse_per_epoch.append(float(val_metrics["rmse"]))
        self.train_mae_per_epoch.append(float(train_metrics["mae"]))
        self.val_mae_per_epoch.append(float(val_metrics["mae"]))
        self.val_sign_accuracy_per_epoch.append(float(val_metrics.get("sign_accuracy", float("nan"))))
        self.learning_rate_per_epoch.append(float(learning_rate))
        self.validation_uncertainty_per_epoch.append(float(validation_uncertainty))
        self._log_parameter_updates(model)

    def run_sensitivity_test(self, model, x, eps: float = 1e-3, n_samples: int = 32) -> float:
        _require_torch()
        if x.size(0) == 0:
            self.sensitivity_scores.append(0.0)
            return 0.0
        sample_count = min(n_samples, x.size(0))
        indices = torch.randperm(x.size(0), device=x.device)[:sample_count]
        x_ref = x[indices]
        noise = eps * torch.randn_like(x_ref)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            y = model(x_ref)
            y_pert = model(x_ref + noise)
        model.train(was_training)
        score = float(torch.norm(y_pert - y, dim=-1).mean().item())
        self.sensitivity_scores.append(score)
        return score

    def as_dict(self) -> dict[str, list[float] | list[list[float]]]:
        return {
            "train_loss_per_epoch": list(self.train_loss_per_epoch),
            "val_loss_per_epoch": list(self.val_loss_per_epoch),
            "smoothed_val_loss_per_epoch": list(self.smoothed_val_loss_per_epoch),
            "train_mse_per_epoch": list(self.train_mse_per_epoch),
            "val_mse_per_epoch": list(self.val_mse_per_epoch),
            "train_rmse_per_epoch": list(self.train_rmse_per_epoch),
            "val_rmse_per_epoch": list(self.val_rmse_per_epoch),
            "train_mae_per_epoch": list(self.train_mae_per_epoch),
            "val_mae_per_epoch": list(self.val_mae_per_epoch),
            "val_sign_accuracy_per_epoch": list(self.val_sign_accuracy_per_epoch),
            "learning_rate_per_epoch": list(self.learning_rate_per_epoch),
            "validation_uncertainty_per_epoch": list(self.validation_uncertainty_per_epoch),
            "quantum_grad_norm_per_epoch": list(self.quantum_grad_norm_per_epoch),
            "classical_grad_norm_per_epoch": list(self.classical_grad_norm_per_epoch),
            "quantum_update_norm_per_epoch": list(self.quantum_update_norm_per_epoch),
            "classical_update_norm_per_epoch": list(self.classical_update_norm_per_epoch),
            "update_balance_ratio_per_epoch": list(self.update_balance_ratio_per_epoch),
            "qnn_output_mean_per_epoch": list(self.qnn_output_mean_per_epoch),
            "qnn_output_var_per_epoch": list(self.qnn_output_var_per_epoch),
            "qnn_output_min_per_epoch": list(self.qnn_output_min_per_epoch),
            "qnn_output_max_per_epoch": list(self.qnn_output_max_per_epoch),
            "theta_trajectory": list(self.theta_trajectory),
            "sensitivity_scores": list(self.sensitivity_scores),
        }
