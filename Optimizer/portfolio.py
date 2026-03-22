from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize


@dataclass
class PortfolioOptimizerConfig:
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


class PortfolioOptimizer:
    def __init__(self, config: PortfolioOptimizerConfig):
        self.config = config

    def _objective(self, weights, mu, cov, previous_weights):
        objective = 0.5 * self.config.risk_aversion * float(weights @ cov @ weights)
        if self.config.objective == "mean_variance":
            objective -= self.config.return_weight * float(mu @ weights)
        objective += self.config.l2_reg * float(weights @ weights)
        if previous_weights is not None:
            diff = weights - previous_weights
            objective += self.config.turnover_penalty * float(diff @ diff)
        return objective

    def _gradient(self, weights, mu, cov, previous_weights):
        gradient = self.config.risk_aversion * (cov @ weights)
        if self.config.objective == "mean_variance":
            gradient -= self.config.return_weight * mu
        gradient += 2.0 * self.config.l2_reg * weights
        if previous_weights is not None:
            gradient += 2.0 * self.config.turnover_penalty * (weights - previous_weights)
        return gradient

    def optimize(
        self,
        mu: np.ndarray,
        cov_mat: np.ndarray,
        previous_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mu = np.asarray(mu, dtype=float).reshape(-1)
        cov = np.asarray(cov_mat, dtype=float)
        if cov.shape[0] != cov.shape[1] or cov.shape[0] != mu.size:
            raise ValueError("mu and cov_mat dimensions do not align.")

        n_assets = mu.size
        initial = (
            np.asarray(previous_weights, dtype=float).reshape(-1)
            if previous_weights is not None
            else np.ones(n_assets) / n_assets
        )
        initial = initial / initial.sum()

        bounds = None
        if self.config.long_only:
            bounds = [(0.0, self.config.weight_max)] * n_assets
        elif self.config.weight_max is not None:
            bounds = [(-self.config.weight_max, self.config.weight_max)] * n_assets

        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
        if self.config.target_return is not None and self.config.objective == "mean_variance":
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights: float(mu @ weights) - self.config.target_return,
                }
            )

        result = minimize(
            fun=lambda weights: self._objective(weights, mu, cov, previous_weights),
            x0=initial,
            jac=lambda weights: self._gradient(weights, mu, cov, previous_weights),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.config.max_iter, "ftol": 1e-9},
        )
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        weights = np.asarray(result.x, dtype=float)
        weights = weights / weights.sum()
        return weights
