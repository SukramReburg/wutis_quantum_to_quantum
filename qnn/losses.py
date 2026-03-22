from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

_LOSS_BASE = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for QNN training but is not installed.")


class LogCoshLoss(_LOSS_BASE):  # type: ignore[misc]
    def __init__(self):
        _require_torch()
        super().__init__()

    def forward(self, prediction, target):
        delta = prediction - target
        return torch.mean(delta + torch.nn.functional.softplus(-2.0 * delta) - math.log(2.0))


@dataclass
class LossSpec:
    name: str
    delta: float = 0.05


class LossFactory:
    @staticmethod
    def create(spec: LossSpec | Any):
        _require_torch()
        name = getattr(spec, "name", str(spec)).lower()
        delta = float(getattr(spec, "delta", 0.05))
        if name == "mse":
            return nn.MSELoss(), "mse"
        if name == "mae":
            return nn.L1Loss(), "mae"
        if name in {"huber", "smooth_l1"}:
            return nn.SmoothL1Loss(beta=delta), "huber"
        if name == "log_cosh":
            return LogCoshLoss(), "log_cosh"
        raise ValueError(f"Unsupported loss '{name}'.")


def huber_loss_numpy(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 0.05) -> np.ndarray:
    err = np.asarray(y_pred) - np.asarray(y_true)
    abs_err = np.abs(err)
    quadratic = np.minimum(abs_err, delta)
    linear = abs_err - quadratic
    return 0.5 * quadratic**2 + delta * linear


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 0.05,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residuals = y_pred - y_true
    mse = float(np.mean(residuals**2))
    mae = float(np.mean(np.abs(residuals)))
    huber = float(np.mean(huber_loss_numpy(y_true, y_pred, delta=delta)))
    sign_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "huber": huber,
        "sign_accuracy": sign_accuracy,
    }
