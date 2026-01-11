from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
import math
import torch
from model import HybridQNNModel


@dataclass
class QNNEvaluationLogger:
    """Collects diagnostic metrics during training.

    General Useage --> just plug into your training loop:

    >>> metrics = QNNEvaluationLogger()
    >>> for epoch in range(n_epochs):
    ...     metrics.start_epoch()
    ...     for xb, yb in train_loader:
    ...         optimizer.zero_grad()
    ...         preds = model(xb)
    ...         loss = loss_fn(preds, yb)
    ...         loss.backward()
    ...         metrics.log_after_backward(model)
    ...         optimizer.step()
    >>>     metrics.end_epoch()
    """

    quantum_grad_norm_per_epoch: List[float] = field(default_factory=list)
    classical_grad_norm_per_epoch: List[float] = field(default_factory=list)

    # Quantum output statistics (before dense head)
    qnn_output_mean_per_epoch: List[float] = field(default_factory=list)
    qnn_output_var_per_epoch: List[float] = field(default_factory=list)
    qnn_output_min_per_epoch: List[float] = field(default_factory=list)
    qnn_output_max_per_epoch: List[float] = field(default_factory=list)

    # Quantum parameter trajectory (for heatmaps)
    theta_trajectory: List[list[float]] = field(default_factory=list)

    # Joint metrics
    train_loss_per_epoch: List[float] = field(default_factory=list)
    val_loss_per_epoch: List[float] = field(default_factory=list)
    quantum_update_norm_per_epoch: List[float] = field(default_factory=list)
    classical_update_norm_per_epoch: List[float] = field(default_factory=list)
    update_balance_ratio_per_epoch: List[float] = field(default_factory=list)
    sensitivity_scores: List[float] = field(default_factory=list)

    # internal accumulators for one epoch
    _quantum_grad_sum: float = 0.0
    _n_steps: int = 0
    _classical_grad_sum: float = 0.0
    _classical_n_steps: int = 0

    # accumulators for QNN output statistics
    _qnn_out_sum: float = 0.0
    _qnn_out_sq_sum: float = 0.0
    _qnn_out_min: float = float("inf")
    _qnn_out_max: float = float("-inf")
    _qnn_out_count: int = 0

    # previous epoch parameters (flattened) for update norms
    _prev_quantum_params: torch.Tensor | None = None
    _prev_classical_params: torch.Tensor | None = None

    def start_epoch(self) -> None:
        """Reset epoch accumulators.

        Call once at the beginning of every epoch before processing batches.
        """
        self._quantum_grad_sum = 0.0
        self._n_steps = 0
        self._classical_grad_sum = 0.0
        self._classical_n_steps = 0

        # reset QNN output stats accumulators
        self._qnn_out_sum = 0.0
        self._qnn_out_sq_sum = 0.0
        self._qnn_out_min = float("inf")
        self._qnn_out_max = float("-inf")
        self._qnn_out_count = 0

    @torch.no_grad()
    def log_qnn_output(self, q_out: torch.Tensor) -> None:
        """Accumulate statistics of the raw QNN output for this epoch.

        ``q_out`` is expected to be the tensor returned by ``model.quantum(x)``
        before the classical dense head. We track mean/variance and global
        min/max across all batches in the epoch.
        """
        if q_out is None:
            return

        vals = q_out.detach().float().reshape(-1)
        if vals.numel() == 0:
            return

        self._qnn_out_sum += float(vals.sum().item())
        self._qnn_out_sq_sum += float((vals * vals).sum().item())

        vmin = float(vals.min().item())
        vmax = float(vals.max().item())
        if vmin < self._qnn_out_min:
            self._qnn_out_min = vmin
        if vmax > self._qnn_out_max:
            self._qnn_out_max = vmax

        self._qnn_out_count += int(vals.numel())

    @torch.no_grad()
    def _compute_quantum_grad_norm(self, model: HybridQNNModel) -> float:
        """Return L2 norm of all quantum-parameter gradients.

        Assumes ``loss.backward()`` has already been called.
        If some gradients are ``None`` (e.g. due to no contribution
        in the current batch), they are skipped.
        """
        total_sq = 0.0
        for p in model.quantum.parameters():
            if p.grad is None:
                continue
            # move to CPU to avoid device issues
            g = p.grad.detach().float().cpu()
            total_sq += float(torch.sum(g * g).item())

        if total_sq == 0.0:
            return 0.0
        return float(total_sq ** 0.5)

    @torch.no_grad()
    def _compute_classical_grad_norm(self, model: HybridQNNModel) -> float:
        """Return L2 norm of dense-head gradients (if present)."""
        if not getattr(model, "use_dense_head", False) or getattr(model, "head", None) is None:
            return float("nan")

        total_sq = 0.0
        for p in model.head.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float().cpu()
            total_sq += float(torch.sum(g * g).item())

        if total_sq == 0.0:
            return 0.0
        return float(total_sq ** 0.5)

    def log_after_backward(self, model: HybridQNNModel) -> None:
        """Log quantum gradient norm for one optimisation step.

        Call this *after* ``loss.backward()`` and *before* ``optimizer.step()``.
        """
        grad_norm = self._compute_quantum_grad_norm(model)
        self._quantum_grad_sum += grad_norm
        self._n_steps += 1

        classical_grad_norm = self._compute_classical_grad_norm(model)
        if not math.isnan(classical_grad_norm):
            self._classical_grad_sum += classical_grad_norm
            self._classical_n_steps += 1

    def end_epoch(self, model: HybridQNNModel, train_loss: float, val_loss: float) -> None:
        """Finalize current epoch.

        - Stores mean quantum gradient norm for the epoch.
        - Computes per-epoch parameter update norms for quantum and classical
          parts and the update-balance ratio.
        - Logs train/validation loss.
        """
        if self._n_steps == 0:
            mean_norm = 0.0
        else:
            mean_norm = self._quantum_grad_sum / float(self._n_steps)
        self.quantum_grad_norm_per_epoch.append(mean_norm)

        if self._classical_n_steps == 0:
            classical_mean_norm = float("nan")
        else:
            classical_mean_norm = self._classical_grad_sum / float(self._classical_n_steps)
        self.classical_grad_norm_per_epoch.append(classical_mean_norm)

        # finalize QNN output statistics for this epoch
        if self._qnn_out_count > 0:
            mean = self._qnn_out_sum / float(self._qnn_out_count)
            mean_sq = self._qnn_out_sq_sum / float(self._qnn_out_count)
            var = max(0.0, mean_sq - mean * mean)
            out_min = self._qnn_out_min
            out_max = self._qnn_out_max
        else:
            mean = 0.0
            var = 0.0
            out_min = 0.0
            out_max = 0.0

        self.qnn_output_mean_per_epoch.append(mean)
        self.qnn_output_var_per_epoch.append(var)
        self.qnn_output_min_per_epoch.append(out_min)
        self.qnn_output_max_per_epoch.append(out_max)

        # log losses
        self.train_loss_per_epoch.append(float(train_loss))
        self.val_loss_per_epoch.append(float(val_loss))

        # log parameter update norms + balance ratio
        self._log_parameter_updates(model)

    @torch.no_grad()
    def _flatten_params(self, params_iter) -> torch.Tensor:
        """Flatten a parameter iterator into a single 1D CPU tensor.

        If there are no parameters (e.g. no dense head), returns an empty
        tensor of shape (0,).
        """
        params = [p.detach().reshape(-1).cpu() for p in params_iter]
        if not params:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(params, dim=0)

    @torch.no_grad()
    def _log_parameter_updates(self, model: HybridQNNModel) -> None:
        """Compute and store per-epoch parameter update norms.

        Uses the difference between the current flattened parameter vectors
        and those stored from the previous epoch.
        """
        # quantum params
        q_vec = self._flatten_params(model.quantum.parameters())
        if self._prev_quantum_params is None or q_vec.numel() == 0:
            dq_norm = 0.0
        else:
            dq = q_vec - self._prev_quantum_params
            dq_norm = float(torch.norm(dq, p=2).item())
        self._prev_quantum_params = q_vec

        # classical (dense head) params, if present
        if getattr(model, "use_dense_head", False) and getattr(model, "head", None) is not None:
            c_vec = self._flatten_params(model.head.parameters())
        else:
            c_vec = torch.empty(0, dtype=torch.float32)

        if self._prev_classical_params is None or c_vec.numel() == 0:
            dc_norm = 0.0
        else:
            dc = c_vec - self._prev_classical_params
            dc_norm = float(torch.norm(dc, p=2).item())
        self._prev_classical_params = c_vec

        self.quantum_update_norm_per_epoch.append(dq_norm)
        self.classical_update_norm_per_epoch.append(dc_norm)

        # store full quantum parameter vector for trajectory visualization
        if q_vec.numel() > 0:
            self.theta_trajectory.append(q_vec.tolist())
        else:
            self.theta_trajectory.append([])

        # ratio ||Δθ|| / ||Δϕ||, careful with zero denominator
        if dc_norm > 0.0:
            ratio = dq_norm / dc_norm
        else:
            ratio = float("nan")
        self.update_balance_ratio_per_epoch.append(ratio)

    @torch.no_grad()
    def run_sensitivity_test(self,model: HybridQNNModel, X: torch.Tensor, eps: float = 1e-3, n_samples: int = 32) -> float:
        """Run prediction smoothness / sensitivity test once per training run.

        Picks up to ``n_samples`` from ``X``, adds small Gaussian noise ``eps``,
        and measures mean L2 distance between predictions.
        The resulting scalar is stored in ``sensitivity_scores``.
        """
        if X.size(0) == 0:
            score = 0.0
            self.sensitivity_scores.append(score)
            return score

        n = min(n_samples, X.size(0))
        idx = torch.randperm(X.size(0), device=X.device)[:n]
        x_ref = X[idx]
        noise = eps * torch.randn_like(x_ref)

        was_training = model.training
        model.eval()
        y = model(x_ref)
        y_pert = model(x_ref + noise)
        model.train(was_training)
        diff = torch.norm(y_pert - y, dim=-1)
        score = float(diff.mean().item())

        self.sensitivity_scores.append(score)
        return score

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def as_dict(self) -> Dict[str, List[float]]:
        """Return all tracked metrics as a plain dict.

        This is convenient for logging, saving to npz, etc.
        """
        return {
            "quantum_grad_norm_per_epoch": list(self.quantum_grad_norm_per_epoch),
            "classical_grad_norm_per_epoch": list(self.classical_grad_norm_per_epoch),
            "qnn_output_mean_per_epoch": list(self.qnn_output_mean_per_epoch),
            "qnn_output_var_per_epoch": list(self.qnn_output_var_per_epoch),
            "qnn_output_min_per_epoch": list(self.qnn_output_min_per_epoch),
            "qnn_output_max_per_epoch": list(self.qnn_output_max_per_epoch),
            "theta_trajectory": list(self.theta_trajectory),
            "train_loss_per_epoch": list(self.train_loss_per_epoch),
            "val_loss_per_epoch": list(self.val_loss_per_epoch),
            "quantum_update_norm_per_epoch": list(self.quantum_update_norm_per_epoch),
            "classical_update_norm_per_epoch": list(self.classical_update_norm_per_epoch),
            "update_balance_ratio_per_epoch": list(self.update_balance_ratio_per_epoch),
            "sensitivity_scores": list(self.sensitivity_scores),
        }
