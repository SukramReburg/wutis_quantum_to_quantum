from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from Optimizer.bundle import PredictionBundle
from Optimizer.engine import BacktestEngine
from Optimizer.portfolio import PortfolioOptimizer, PortfolioOptimizerConfig
from qnn.config import NoiseConfig, RuntimeConfig, load_experiment_config
from qnn.losses import regression_metrics
from qnn.metrics import QNNMetricsCollector
from qnn.model import QNNBuilder, entangling_edges
from qnn.runtime import NoiseModelFactory, QuantumRuntimeFactory
from qnn.study import QNNStudyRunner


class QNNOptimizerStackTests(unittest.TestCase):
    def test_config_loads_mode_overrides_and_search_spaces(self):
        returns_cfg = load_experiment_config(mode="returns", config_path="config/model_config.yaml")
        cov_cfg = load_experiment_config(mode="cov", config_path="config/model_config.yaml")

        self.assertEqual(returns_cfg.runtime.profile, "realism")
        self.assertEqual(returns_cfg.model.feature_mode, "angles")
        self.assertEqual(cov_cfg.model.feature_mode, "pca")
        self.assertEqual(cov_cfg.model.n_qubits, 7)

        study_runner = QNNStudyRunner(config_path="config/model_config.yaml")
        cov_space = study_runner._search_space("cov")
        self.assertIn("runtime.profile", cov_space)
        self.assertIn("loss.delta", cov_space)

    def test_runtime_and_noise_selection_are_config_driven(self):
        runtime = RuntimeConfig(device="auto", use_gpu_if_available=True)
        self.assertEqual(
            QuantumRuntimeFactory.resolve_device(runtime, available_devices=["CPU", "GPU"]),
            "GPU",
        )

        no_noise = NoiseModelFactory.describe(NoiseConfig(enabled=False, family="none"))
        depolarizing = NoiseModelFactory.describe(NoiseConfig(enabled=True, family="depolarizing"))
        self.assertFalse(no_noise["enabled"])
        self.assertTrue(depolarizing["enabled"])
        self.assertEqual(depolarizing["family"], "depolarizing")

    def test_losses_metrics_and_metric_serialization(self):
        y_true = np.array([[0.1, -0.2], [0.3, 0.2]], dtype=np.float32)
        y_pred = np.array([[0.0, -0.1], [0.4, 0.1]], dtype=np.float32)
        metrics = regression_metrics(y_true, y_pred, delta=0.05)
        self.assertAlmostEqual(metrics["mse"], 0.01, places=6)
        self.assertAlmostEqual(metrics["mae"], 0.1, places=6)
        self.assertGreaterEqual(metrics["sign_accuracy"], 0.5)

        collector = QNNMetricsCollector()
        collector.train_loss_per_epoch.extend([0.9, 0.5])
        collector.val_loss_per_epoch.extend([1.0, 0.6])
        collector.theta_trajectory.extend([[0.1, 0.2], [0.15, 0.18]])
        payload = collector.as_dict()
        self.assertEqual(payload["train_loss_per_epoch"], [0.9, 0.5])
        self.assertEqual(len(payload["theta_trajectory"]), 2)

    def test_entanglement_spec_is_preserved(self):
        cfg = load_experiment_config(mode="returns", config_path="config/model_config.yaml")
        cfg.model.entanglement = "linear"
        spec = QNNBuilder.create_spec(cfg.model, n_outputs=4)
        self.assertEqual(spec.entanglement, "linear")
        self.assertEqual(entangling_edges(4, "linear"), [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(entangling_edges(4, "ring"), [(0, 1), (1, 2), (2, 3), (3, 0)])

    def test_optimizer_uses_mu_constraints_and_turnover(self):
        mu = np.array([0.2, 0.05], dtype=float)
        cov = np.array([[0.05, 0.0], [0.0, 0.05]], dtype=float)

        mean_var = PortfolioOptimizer(
            PortfolioOptimizerConfig(objective="mean_variance", weight_max=0.8, turnover_penalty=0.0)
        )
        weights = mean_var.optimize(mu, cov)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertLessEqual(float(weights.max()), 0.8 + 1e-6)
        self.assertGreater(weights[0], weights[1])

        min_var = PortfolioOptimizer(
            PortfolioOptimizerConfig(objective="min_variance", weight_max=0.8, turnover_penalty=0.0)
        )
        min_var_weights = min_var.optimize(mu, cov)
        self.assertAlmostEqual(min_var_weights[0], min_var_weights[1], places=3)

        sticky = PortfolioOptimizer(
            PortfolioOptimizerConfig(
                objective="mean_variance",
                weight_max=0.8,
                turnover_penalty=10.0,
            )
        )
        previous = np.array([0.7, 0.3], dtype=float)
        sticky_weights = sticky.optimize(mu, cov, previous_weights=previous)
        self.assertLess(np.linalg.norm(sticky_weights - previous), np.linalg.norm(weights - previous))

    def test_holding_windows_are_half_open_without_double_count(self):
        prediction_dates = pd.DatetimeIndex(["2024-01-05", "2024-01-12"])
        daily_index = pd.DatetimeIndex(
            ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-12", "2024-01-15"]
        )
        windows = BacktestEngine.build_holding_windows(prediction_dates, daily_index)
        first_window = list(windows[0][1].strftime("%Y-%m-%d"))
        second_window = list(windows[1][1].strftime("%Y-%m-%d"))
        self.assertEqual(first_window, ["2024-01-08", "2024-01-09", "2024-01-12"])
        self.assertEqual(second_window, ["2024-01-15"])

    def test_prediction_bundle_and_backtest_alignment_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "config").mkdir(parents=True)
            (base / "data" / "raw").mkdir(parents=True)
            (base / "qnn" / "results" / "latest" / "returns").mkdir(parents=True)
            (base / "qnn" / "results" / "latest" / "cov").mkdir(parents=True)

            with open(base / "config" / "data_config.yaml", "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "paths": {"raw": "data/raw", "processed": "data/processed", "plots": "analysis/plots"},
                    },
                    handle,
                    sort_keys=False,
                )
            with open(base / "config" / "model_config.yaml", "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "paths": {
                            "results": "qnn/results",
                            "plots": "qnn/plots",
                            "models": "qnn/models",
                            "optuna": "qnn/results/optuna",
                            "backtests": "Optimizer/backtest_results",
                        },
                        "training": {"data_config_path": "config/data_config.yaml"},
                        "optimizer": {
                            "objective": "mean_variance",
                            "weight_max": 0.8,
                            "turnover_penalty": 0.0,
                            "risk_aversion": 1.0,
                            "return_weight": 1.0,
                            "long_only": True,
                            "l2_reg": 0.0,
                            "solver": "auto",
                            "max_iter": 200,
                            "step_size": 0.05,
                            "benchmark": "equal_weight",
                        },
                    },
                    handle,
                    sort_keys=False,
                )

            dates = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
            merged = pd.DataFrame(
                {
                    "AAA_log": np.linspace(0.01, 0.02, len(dates)),
                    "BBB_log": np.linspace(-0.01, 0.015, len(dates)),
                },
                index=dates,
            )
            merged.index.name = "timestamp"
            merged.to_csv(base / "data" / "raw" / "merged_data.csv")

            sample_dates = np.asarray(["2024-01-05T00:00:00+00:00", "2024-01-12T00:00:00+00:00"], dtype=str)
            returns_path = base / "qnn" / "results" / "latest" / "returns" / "predictions.npz"
            cov_path = base / "qnn" / "results" / "latest" / "cov" / "predictions.npz"
            np.savez_compressed(
                returns_path,
                Y_pred_test=np.asarray([[0.02, 0.01], [0.01, 0.015]], dtype=np.float32),
                Y_true_test=np.asarray([[0.015, 0.005], [0.012, 0.013]], dtype=np.float32),
                asset_symbols=np.asarray(["AAA", "BBB"], dtype=str),
                sample_dates=sample_dates,
                target_frequency=np.asarray("weekly", dtype=str),
            )
            np.savez_compressed(
                cov_path,
                Y_pred_test=np.asarray([[0.10, 0.01, 0.08], [0.11, 0.015, 0.09]], dtype=np.float32),
                Y_true_test=np.asarray([[0.09, 0.02, 0.07], [0.10, 0.01, 0.08]], dtype=np.float32),
                asset_symbols=np.asarray(["AAA", "BBB"], dtype=str),
                sample_dates=sample_dates,
                target_frequency=np.asarray("weekly", dtype=str),
            )

            bundle = PredictionBundle.load(str(returns_path), kind="returns", expected_frequency="weekly")
            self.assertEqual(bundle.asset_symbols, ["AAA", "BBB"])
            self.assertEqual(len(bundle.sample_dates), 2)

            engine = BacktestEngine(
                model_config_path=str(base / "config" / "model_config.yaml"),
                base_dir=str(base),
            )
            result = engine.run()
            self.assertGreater(len(result.weights), 0)
            self.assertEqual(list(result.weights.columns), ["AAA", "BBB"])

            np.savez_compressed(
                cov_path,
                Y_pred_test=np.asarray([[0.10, 0.01, 0.08], [0.11, 0.015, 0.09]], dtype=np.float32),
                Y_true_test=np.asarray([[0.09, 0.02, 0.07], [0.10, 0.01, 0.08]], dtype=np.float32),
                asset_symbols=np.asarray(["BBB", "AAA"], dtype=str),
                sample_dates=sample_dates,
                target_frequency=np.asarray("weekly", dtype=str),
            )
            with self.assertRaises(ValueError):
                engine.run()


if __name__ == "__main__":
    unittest.main()
