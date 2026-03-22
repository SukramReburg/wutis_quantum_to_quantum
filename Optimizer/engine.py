from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data.common import resolve_path

from .KPIs import performance_summary
from .bundle import PredictionBundle
from .portfolio import PortfolioOptimizer, PortfolioOptimizerConfig
from .reconstruct_cov import make_psd, rebuild_covariance


@dataclass
class BacktestResult:
    weights: pd.DataFrame
    portfolio_simple_returns: pd.Series
    benchmark_simple_returns: pd.Series
    portfolio_log_returns: pd.Series
    benchmark_log_returns: pd.Series
    predicted_returns: pd.DataFrame
    predicted_covariances: dict[pd.Timestamp, np.ndarray]
    realized_covariances: dict[pd.Timestamp, np.ndarray]
    diagnostics: pd.DataFrame
    summary: dict[str, Any]
    benchmark_summary: dict[str, Any]


class BacktestEngine:
    def __init__(self, model_config_path: str = "config/model_config.yaml", base_dir: str | None = None):
        self.base_dir = os.path.abspath(base_dir or os.getcwd())
        self.model_config_path = resolve_path(self.base_dir, model_config_path)
        with open(self.model_config_path, "r", encoding="utf-8") as handle:
            self.model_config = yaml.safe_load(handle)
        data_config_path = self.model_config.get("training", {}).get("data_config_path", "config/data_config.yaml")
        with open(resolve_path(self.base_dir, data_config_path), "r", encoding="utf-8") as handle:
            self.data_config = yaml.safe_load(handle)

    def default_prediction_path(self, mode: str) -> str:
        results_root = resolve_path(self.base_dir, self.model_config["paths"]["results"])
        candidates = [
            os.path.join(results_root, "latest", mode, "predictions.npz"),
            os.path.join(results_root, f"{mode}_latest_predictions.npz"),
        ]
        if mode == "returns":
            candidates.extend(
                [
                    os.path.join(results_root, "qnn_returns_angles_hybrid_rxrz_predictions.npz"),
                    os.path.join(self.base_dir, "Optimizer", "prediction data", "qnn_returns_angles_hybrid_rxrz_predictions.npz"),
                ]
            )
        else:
            candidates.extend(
                [
                    os.path.join(results_root, "qnn_cov_pca_hybrid_rxrz_predictions.npz"),
                    os.path.join(self.base_dir, "Optimizer", "prediction data", "qnn_cov_pca_hybrid_rxrz_predictions.npz"),
                ]
            )
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"No default prediction artifact found for mode '{mode}'.")

    def load_market_data(self) -> pd.DataFrame:
        raw_dir = resolve_path(self.base_dir, self.data_config["paths"]["raw"])
        merged_path = os.path.join(raw_dir, "merged_data.csv")
        return pd.read_csv(merged_path, index_col="timestamp", parse_dates=True)

    @staticmethod
    def build_holding_windows(
        prediction_dates: pd.DatetimeIndex,
        daily_index: pd.DatetimeIndex,
    ) -> list[tuple[pd.Timestamp, pd.DatetimeIndex]]:
        windows = []
        for idx, rebalance_date in enumerate(prediction_dates):
            if idx < len(prediction_dates) - 1:
                next_date = prediction_dates[idx + 1]
                mask = (daily_index > rebalance_date) & (daily_index <= next_date)
            else:
                mask = daily_index > rebalance_date
            windows.append((rebalance_date, daily_index[mask]))
        return windows

    def run(
        self,
        returns_path: str | None = None,
        cov_path: str | None = None,
    ) -> BacktestResult:
        returns_bundle = PredictionBundle.load(
            returns_path or self.default_prediction_path("returns"),
            kind="returns",
            expected_frequency="weekly",
        )
        cov_bundle = PredictionBundle.load(
            cov_path or self.default_prediction_path("cov"),
            kind="cov",
            expected_frequency="weekly",
        )
        if returns_bundle.asset_symbols != cov_bundle.asset_symbols:
            raise ValueError("Returns and covariance prediction artifacts use different asset orders.")
        if returns_bundle.sample_dates and cov_bundle.sample_dates and returns_bundle.sample_dates != cov_bundle.sample_dates:
            raise ValueError("Returns and covariance prediction artifacts use different sample dates.")

        prediction_dates = returns_bundle.sample_index if len(returns_bundle.sample_index) else cov_bundle.sample_index
        market_data = self.load_market_data()
        ret_cols = [column for column in market_data.columns if column.endswith("_log")]
        log_ret_daily = market_data[ret_cols].dropna()
        log_ret_daily.columns = [column.replace("_log", "") for column in ret_cols]
        log_ret_daily = log_ret_daily[returns_bundle.asset_symbols]
        simple_ret_daily = np.expm1(log_ret_daily)

        predicted_returns = pd.DataFrame(
            np.expm1(returns_bundle.y_pred),
            index=prediction_dates,
            columns=returns_bundle.asset_symbols,
        )
        predicted_covariances = {
            prediction_dates[idx]: make_psd(rebuild_covariance(cov_bundle.y_pred[idx], len(returns_bundle.asset_symbols)))
            for idx in range(cov_bundle.y_pred.shape[0])
        }
        realized_covariances: dict[pd.Timestamp, np.ndarray] = {}
        windows = self.build_holding_windows(prediction_dates, simple_ret_daily.index)

        optimizer_config = PortfolioOptimizerConfig(**self.model_config.get("optimizer", {}))
        optimizer = PortfolioOptimizer(optimizer_config)

        previous_weights = None
        weights_history: dict[pd.Timestamp, np.ndarray] = {}
        portfolio_returns = []
        benchmark_returns = []
        portfolio_dates = []
        benchmark_dates = []
        diagnostic_rows = []

        equal_weights = np.ones(len(returns_bundle.asset_symbols)) / len(returns_bundle.asset_symbols)

        for rebalance_date, holding_index in windows:
            if rebalance_date not in predicted_covariances:
                continue
            mu = predicted_returns.loc[rebalance_date].values
            sigma = predicted_covariances[rebalance_date]
            weights = optimizer.optimize(mu, sigma, previous_weights=previous_weights)
            weights_history[rebalance_date] = weights

            window_log = log_ret_daily.loc[holding_index]
            window_simple = simple_ret_daily.loc[holding_index]
            if len(window_log) >= 2:
                realized_covariances[rebalance_date] = window_log.cov().values
            realized_cov = realized_covariances.get(rebalance_date)
            realized_return = float(window_log.sum().mean()) if len(window_log) else np.nan

            for day, row in window_simple.iterrows():
                portfolio_returns.append(float(np.dot(weights, row.values)))
                benchmark_returns.append(float(np.dot(equal_weights, row.values)))
                portfolio_dates.append(day)
                benchmark_dates.append(day)

            turnover = (
                float(np.abs(weights - previous_weights).sum())
                if previous_weights is not None
                else 0.0
            )
            predicted_vol = float(np.sqrt(np.maximum(weights @ sigma @ weights, 0.0)))
            realized_vol = (
                float(np.sqrt(np.maximum(weights @ realized_cov @ weights, 0.0)))
                if realized_cov is not None
                else np.nan
            )
            cov_error = (
                float(np.linalg.norm(realized_cov - sigma, ord="fro"))
                if realized_cov is not None
                else np.nan
            )
            diagnostic_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "predicted_return": float(np.dot(weights, mu)),
                    "predicted_vol": predicted_vol,
                    "realized_vol": realized_vol,
                    "turnover": turnover,
                    "covariance_frobenius_error": cov_error,
                    "realized_average_log_return": realized_return,
                }
            )
            previous_weights = weights

        portfolio_simple_returns = pd.Series(
            portfolio_returns,
            index=pd.DatetimeIndex(portfolio_dates),
            name="portfolio_simple_return",
        ).sort_index()
        benchmark_simple_returns = pd.Series(
            benchmark_returns,
            index=pd.DatetimeIndex(benchmark_dates),
            name="benchmark_simple_return",
        ).sort_index()

        common_index = portfolio_simple_returns.index.intersection(benchmark_simple_returns.index)
        portfolio_simple_returns = portfolio_simple_returns.loc[common_index]
        benchmark_simple_returns = benchmark_simple_returns.loc[common_index]

        portfolio_log_returns = np.log1p(portfolio_simple_returns)
        benchmark_log_returns = np.log1p(benchmark_simple_returns)
        weights_df = pd.DataFrame.from_dict(
            weights_history,
            orient="index",
            columns=returns_bundle.asset_symbols,
        ).sort_index()
        diagnostics = pd.DataFrame(diagnostic_rows).set_index("rebalance_date").sort_index()

        return BacktestResult(
            weights=weights_df,
            portfolio_simple_returns=portfolio_simple_returns,
            benchmark_simple_returns=benchmark_simple_returns,
            portfolio_log_returns=portfolio_log_returns,
            benchmark_log_returns=benchmark_log_returns,
            predicted_returns=predicted_returns,
            predicted_covariances=predicted_covariances,
            realized_covariances=realized_covariances,
            diagnostics=diagnostics,
            summary=performance_summary(portfolio_log_returns.values),
            benchmark_summary=performance_summary(benchmark_log_returns.values),
        )
