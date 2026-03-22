from __future__ import annotations

import numpy as np

from .portfolio import PortfolioOptimizer, PortfolioOptimizerConfig


def optimize_weights(mu, cov_mat, u=None, previous_weights=None):
    config = PortfolioOptimizerConfig(target_return=u)
    return PortfolioOptimizer(config).optimize(
        np.asarray(mu, dtype=float),
        np.asarray(cov_mat, dtype=float),
        previous_weights=np.asarray(previous_weights, dtype=float) if previous_weights is not None else None,
    )
