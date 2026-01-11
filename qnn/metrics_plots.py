from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

A4_SIZE = (8.27, 11.69)


def _apply_a4_layout(fig: plt.Figure, *, left: float = 0.1, right: float = 0.95,
                     top: float = 0.9, bottom: float = 0.08, hspace: Optional[float] = None) -> None:
    fig.set_size_inches(*A4_SIZE, forward=True)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    if hspace is not None:
        fig.subplots_adjust(hspace=hspace)


def _as_array(metrics: Dict[str, object], key: str) -> Optional[np.ndarray]:
    if key not in metrics:
        return None
    return np.asarray(metrics[key])


def _has_finite(values: Sequence[np.ndarray]) -> bool:
    for v in values:
        if v is None:
            continue
        if np.any(np.isfinite(v)):
            return True
    return False


def _maybe_log_scale(ax: plt.Axes, values: Sequence[np.ndarray]) -> None:
    vals = [v[np.isfinite(v)] for v in values if v is not None and np.any(np.isfinite(v))]
    if not vals:
        return
    vmin = min(float(np.min(v)) for v in vals)
    vmax = max(float(np.max(v)) for v in vals)
    if vmin > 0 and vmax / vmin >= 1e3:
        ax.set_yscale("log")


def plot_loss_curves(metrics: Dict[str, object]) -> Optional[plt.Figure]:
    train = _as_array(metrics, "train_loss_per_epoch")
    val = _as_array(metrics, "val_loss_per_epoch")
    if train is None and val is None:
        return None

    fig, ax = plt.subplots()
    if train is not None:
        ax.plot(train, label="Train")
    if val is not None:
        ax.plot(val, label="Validation")
    ax.set_title("Loss Curves (MSE per Epoch)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _apply_a4_layout(fig)
    return fig


def plot_grad_norms(metrics: Dict[str, object]) -> Optional[plt.Figure]:
    q_grad = _as_array(metrics, "quantum_grad_norm_per_epoch")
    c_grad = _as_array(metrics, "classical_grad_norm_per_epoch")
    if not _has_finite([q_grad, c_grad]):
        return None

    fig, ax = plt.subplots()
    if q_grad is not None:
        ax.plot(q_grad, label="Quantum grad norm")
    if c_grad is not None:
        ax.plot(c_grad, label="Classical grad norm")
    ax.set_title("Gradient Norms per Epoch (L2)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 norm")
    _maybe_log_scale(ax, [q_grad, c_grad])
    ax.legend()
    ax.grid(True, alpha=0.3)
    _apply_a4_layout(fig)
    return fig


def plot_update_norms(metrics: Dict[str, object]) -> Optional[plt.Figure]:
    q_upd = _as_array(metrics, "quantum_update_norm_per_epoch")
    c_upd = _as_array(metrics, "classical_update_norm_per_epoch")
    if not _has_finite([q_upd, c_upd]):
        return None

    fig, ax = plt.subplots()
    if q_upd is not None:
        ax.plot(q_upd, label="Quantum update norm")
    if c_upd is not None:
        ax.plot(c_upd, label="Classical update norm")
    ax.set_title("Parameter Update Norms per Epoch (L2)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 norm")
    _maybe_log_scale(ax, [q_upd, c_upd])
    ax.legend()
    ax.grid(True, alpha=0.3)
    _apply_a4_layout(fig)
    return fig


def plot_update_ratio(metrics: Dict[str, object]) -> Optional[plt.Figure]:
    ratio = _as_array(metrics, "update_balance_ratio_per_epoch")
    if ratio is None or not np.any(np.isfinite(ratio)):
        return None

    fig, ax = plt.subplots()
    ax.plot(ratio, label="||d_theta|| / ||d_phi||")
    ax.set_title("Update Ratio per Epoch (||d_theta|| / ||d_phi||)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _apply_a4_layout(fig)
    return fig


def plot_qnn_output_variance(metrics: Dict[str, object]) -> Optional[plt.Figure]:
    var = _as_array(metrics, "qnn_output_var_per_epoch")
    if var is None or not np.any(np.isfinite(var)):
        return None

    fig, ax = plt.subplots()
    ax.plot(var, label="Variance")
    ax.set_title("QNN Output Variance per Epoch (raw output before head)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _apply_a4_layout(fig)
    return fig


def plot_theta_heatmap(metrics: Dict[str, object]) -> Optional[plt.Figure]:
    theta = _as_array(metrics, "theta_trajectory")
    if theta is None or theta.ndim != 2:
        return None

    fig, ax = plt.subplots()
    im = ax.imshow(theta, aspect="auto", cmap="viridis")
    ax.set_title("Quantum Parameter Trajectory Heatmap (theta per Epoch)")
    ax.set_xlabel("Parameter index")
    ax.set_ylabel("Epoch")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Parameter value")
    _apply_a4_layout(fig, right=0.88)
    return fig


def plot_sensitivity_score(metrics: Dict[str, object]) -> Optional[plt.Figure]:
    scores = _as_array(metrics, "sensitivity_scores")
    if scores is None or not np.any(np.isfinite(scores)):
        return None

    if scores.size == 1:
        return None

    fig, ax = plt.subplots()
    ax.plot(scores, marker="o", label="Sensitivity")
    ax.set_title("Prediction Smoothness (Mean L2 diff per run)")
    ax.set_xlabel("Run")
    ax.set_ylabel("Mean L2 diff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _apply_a4_layout(fig)
    return fig


def plot_optimization_dynamics(metrics: Dict[str, object]) -> Optional[plt.Figure]:
    q_grad = _as_array(metrics, "quantum_grad_norm_per_epoch")
    c_grad = _as_array(metrics, "classical_grad_norm_per_epoch")
    q_upd = _as_array(metrics, "quantum_update_norm_per_epoch")
    c_upd = _as_array(metrics, "classical_update_norm_per_epoch")
    ratio = _as_array(metrics, "update_balance_ratio_per_epoch")

    if not _has_finite([q_grad, c_grad, q_upd, c_upd, ratio]):
        return None

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax0, ax1 = axes

    if _has_finite([q_grad, c_grad]):
        if q_grad is not None:
            ax0.plot(q_grad, label="Quantum grad norm")
        if c_grad is not None:
            ax0.plot(c_grad, label="Classical grad norm")
        ax0.set_title("Gradient Norms per Epoch (L2)")
        ax0.set_ylabel("L2 norm")
        _maybe_log_scale(ax0, [q_grad, c_grad])
        ax0.legend()
        ax0.grid(True, alpha=0.3)
    else:
        ax0.set_visible(False)

    if _has_finite([q_upd, c_upd, ratio]):
        if q_upd is not None:
            ax1.plot(q_upd, label="Quantum update norm")
        if c_upd is not None:
            ax1.plot(c_upd, label="Classical update norm")
        ax1.set_title("Update Norms and Ratio per Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("L2 norm")
        _maybe_log_scale(ax1, [q_upd, c_upd])
        ax1.grid(True, alpha=0.3)

        if ratio is not None and np.any(np.isfinite(ratio)):
            ax1b = ax1.twinx()
            ax1b.plot(ratio, color="#55A868", linestyle="--", label="Update ratio")
            ax1b.set_ylabel("Ratio (d_theta / d_phi)")

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1b.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper right")
        else:
            ax1.legend()
    else:
        ax1.set_visible(False)

    _apply_a4_layout(fig, hspace=0.35)
    return fig
