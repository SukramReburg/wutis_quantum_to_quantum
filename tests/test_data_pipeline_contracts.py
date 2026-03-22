from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.common import (
    DAILY_DATASET_FILENAME,
    DAILY_SCALER_FILENAMES,
    WEEKLY_DATASET_FILENAME,
    WEEKLY_SCALER_FILENAMES,
)
from data.datasets import prepare_qnn_cov_dataset, prepare_qnn_ret_dataset, save_dataset_bundle_from_config
from data.datasets_weekly import (
    prepare_weekly_qnn_cov_dataset,
    prepare_weekly_qnn_ret_dataset,
    save_dataset_bundle_from_config as save_weekly_dataset_bundle,
)
from data.fetch import compute_fetch_end_datetime, fetch_and_save_data
from data.preprocess import preprocess_and_save_data


def make_market_frame(symbol: str, dates, start_price: float = 100.0) -> pd.DataFrame:
    closes = start_price + np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(dates),
            "timestamp": dates,
            "open": closes - 0.5,
            "high": closes + 0.5,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.linspace(1_000, 2_000, len(dates)),
            "trade_count": np.arange(len(dates)) + 1,
            "vwap": closes - 0.25,
        }
    )


def make_feature_merged_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=12, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "AAA_signal": np.linspace(0.1, 1.2, len(dates)),
            "AAA_log": np.linspace(0.01, 0.12, len(dates)),
            "BBB_signal": np.linspace(1.1, 2.2, len(dates)),
            "BBB_log": np.linspace(0.02, 0.13, len(dates)),
        },
        index=dates,
    )


def make_weekly_feature_merged_df() -> pd.DataFrame:
    dates = pd.DatetimeIndex(
        [
            "2024-01-05 00:00:00+00:00",  # partial first week, should be dropped
            "2024-01-08 00:00:00+00:00",
            "2024-01-09 00:00:00+00:00",
            "2024-01-10 00:00:00+00:00",
            "2024-01-11 00:00:00+00:00",
            "2024-01-12 00:00:00+00:00",
            "2024-01-15 00:00:00+00:00",
            "2024-01-16 00:00:00+00:00",
            "2024-01-17 00:00:00+00:00",
            "2024-01-18 00:00:00+00:00",
            "2024-01-19 00:00:00+00:00",
            "2024-01-22 00:00:00+00:00",
            "2024-01-23 00:00:00+00:00",
            "2024-01-24 00:00:00+00:00",
            "2024-01-25 00:00:00+00:00",
            "2024-01-26 00:00:00+00:00",
            "2024-01-29 00:00:00+00:00",
            "2024-01-30 00:00:00+00:00",
            "2024-01-31 00:00:00+00:00",
            "2024-02-01 00:00:00+00:00",
            "2024-02-02 00:00:00+00:00",
            "2024-02-05 00:00:00+00:00",
            "2024-02-06 00:00:00+00:00",
            "2024-02-07 00:00:00+00:00",
            "2024-02-08 00:00:00+00:00",
            "2024-02-09 00:00:00+00:00",
        ]
    )
    return pd.DataFrame(
        {
            "AAA_signal": np.linspace(0.1, 2.6, len(dates)),
            "AAA_log": np.linspace(0.01, 0.26, len(dates)),
            "BBB_signal": np.linspace(1.1, 3.6, len(dates)),
            "BBB_log": np.linspace(0.02, 0.27, len(dates)),
        },
        index=dates,
    )


class DataPipelineContractTests(unittest.TestCase):
    def make_temp_project(self):
        temp_dir = tempfile.TemporaryDirectory()
        base_dir = Path(temp_dir.name)
        for relative in ["config", "source", "data/raw", "data/processed/scalers"]:
            (base_dir / relative).mkdir(parents=True, exist_ok=True)
        return temp_dir, base_dir

    def write_project_config(self, base_dir: Path, indicators=None):
        config = {
            "assets": ["ZZZ", "AAA"],
            "cov_window": 2,
            "indicators": indicators or [{"name": "log"}, {"name": "volume_change"}],
            "lookback_window": 2,
            "n_assets": 2,
            "paths": {
                "plots": "analysis/plots",
                "processed": "data/processed",
                "raw": "data/raw",
                "scalers": "data/processed/scalers",
            },
            "records_number_threshold": 1,
            "start_year": 2024,
            "train_size": 0.6,
            "use_past_cov_in_features": True,
            "use_past_ret_in_features": True,
        }
        with open(base_dir / "config" / "data_config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        with open(base_dir / "config" / "config.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "alpaca_api": {
                        "api_key": "test-key",
                        "secret_key": "test-secret",
                        "base_url": "https://paper-api.alpaca.markets",
                    }
                },
                f,
                sort_keys=False,
            )

    def write_assets_csv(self, base_dir: Path, tickers):
        pd.DataFrame(
            {
                "ticker": tickers,
                "category": ["Test"] * len(tickers),
                "name": [f"{ticker} Inc." for ticker in tickers],
            }
        ).to_csv(base_dir / "source" / "assets.csv", index=False)

    def write_merged_csv(self, base_dir: Path, frame: pd.DataFrame):
        frame.index.name = "timestamp"
        frame.to_csv(base_dir / "data" / "raw" / "merged_data.csv")

    def test_compute_fetch_end_datetime_handles_month_boundaries(self):
        self.assertEqual(
            compute_fetch_end_datetime(datetime(2026, 3, 1, 12, 0, 0)),
            datetime(2026, 2, 28, 12, 0, 0),
        )
        self.assertEqual(
            compute_fetch_end_datetime(datetime(2026, 1, 1, 9, 30, 0)),
            datetime(2025, 12, 31, 9, 30, 0),
        )

    def test_fetch_then_preprocess_drops_stale_tickers_on_rerun(self):
        temp_dir, base_dir = self.make_temp_project()
        self.addCleanup(temp_dir.cleanup)

        self.write_project_config(base_dir)
        self.write_assets_csv(base_dir, ["ZZZ", "AAA"])
        dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")

        first_run = {
            "ZZZ": make_market_frame("ZZZ", dates, 20.0),
            "AAA": make_market_frame("AAA", dates, 10.0),
        }
        second_run = {
            "ZZZ": pd.DataFrame(),
            "AAA": make_market_frame("AAA", dates, 10.0),
        }

        def fetch_stub_factory(mapping):
            def _fetch_stub(_client, ticker, _start_year, _end_dt):
                return mapping[ticker].copy()

            return _fetch_stub

        fetch_and_save_data(
            base_dir=str(base_dir),
            fetch_func=fetch_stub_factory(first_run),
            client_loader=lambda *_args, **_kwargs: object(),
        )
        preprocess_and_save_data(base_dir=str(base_dir))

        raw_files = sorted(path.name for path in (base_dir / "data" / "raw" / "tickers").glob("*.csv"))
        self.assertEqual(raw_files, ["AAA.csv", "ZZZ.csv"])

        fetch_and_save_data(
            base_dir=str(base_dir),
            fetch_func=fetch_stub_factory(second_run),
            client_loader=lambda *_args, **_kwargs: object(),
        )
        preprocess_and_save_data(base_dir=str(base_dir))

        raw_files = sorted(path.name for path in (base_dir / "data" / "raw" / "tickers").glob("*.csv"))
        self.assertEqual(raw_files, ["AAA.csv"])

        merged = pd.read_csv(base_dir / "data" / "raw" / "merged_data.csv")
        self.assertIn("AAA_log", merged.columns)
        self.assertNotIn("ZZZ_log", merged.columns)

    def test_preprocess_writes_sorted_assets_and_daily_metadata_uses_same_order(self):
        temp_dir, base_dir = self.make_temp_project()
        self.addCleanup(temp_dir.cleanup)

        self.write_project_config(base_dir)
        self.write_assets_csv(base_dir, ["ZZZ", "AAA"])
        dates = pd.date_range("2024-01-02", periods=6, freq="B", tz="UTC")

        def fetch_stub(_client, ticker, _start_year, _end_dt):
            start_price = 10.0 if ticker == "AAA" else 20.0
            return make_market_frame(ticker, dates, start_price)

        fetch_and_save_data(
            base_dir=str(base_dir),
            fetch_func=fetch_stub,
            client_loader=lambda *_args, **_kwargs: object(),
        )
        preprocess_and_save_data(base_dir=str(base_dir))
        save_dataset_bundle_from_config(base_dir=str(base_dir))

        with open(base_dir / "config" / "data_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.assertEqual(config["assets"], ["AAA", "ZZZ"])

        bundle = np.load(base_dir / "data" / "processed" / DAILY_DATASET_FILENAME)
        self.assertEqual(bundle["asset_symbols"].tolist(), ["AAA", "ZZZ"])

    def test_daily_dataset_split_validation_and_insufficient_sample_errors(self):
        merged_df = make_feature_merged_df()

        X_train_ret, X_test_ret, Y_train_ret, Y_test_ret, _, metadata = prepare_qnn_ret_dataset(
            merged_df,
            train_ratio=0.6,
            lookback_window=2,
            use_past_ret_in_features=True,
            return_metadata=True,
        )

        self.assertGreater(X_train_ret.shape[0], 0)
        self.assertGreater(X_test_ret.shape[0], 0)
        self.assertEqual(len(metadata["sample_dates"]), Y_train_ret.shape[0] + Y_test_ret.shape[0])

        with self.assertRaises(ValueError):
            prepare_qnn_ret_dataset(
                merged_df,
                train_ratio=1.0,
                lookback_window=2,
                use_past_ret_in_features=True,
            )

        with self.assertRaises(ValueError):
            prepare_qnn_cov_dataset(
                merged_df,
                train_ratio=0.6,
                cov_window=len(merged_df) + 1,
                use_past_cov_in_features=True,
            )

    def test_weekly_returns_and_covariance_share_calendar_after_partial_first_week(self):
        merged_df = make_weekly_feature_merged_df()

        _, _, Y_train_ret, Y_test_ret, _, ret_metadata = prepare_weekly_qnn_ret_dataset(
            merged_df,
            train_ratio=0.6,
            lookback_weeks=2,
            use_past_ret_in_features=True,
            return_metadata=True,
        )
        _, _, Y_train_cov, Y_test_cov, _, cov_metadata = prepare_weekly_qnn_cov_dataset(
            merged_df,
            train_ratio=0.6,
            cov_lookback_weeks=2,
            use_past_cov_in_features=True,
            return_metadata=True,
        )

        self.assertEqual(
            ret_metadata["eligible_weeks"].tolist(),
            cov_metadata["eligible_weeks"].tolist(),
        )
        self.assertEqual(
            ret_metadata["sample_dates"].tolist(),
            cov_metadata["sample_dates"].tolist(),
        )
        self.assertEqual(
            len(ret_metadata["sample_dates"]),
            Y_train_ret.shape[0] + Y_test_ret.shape[0],
        )
        self.assertEqual(
            len(cov_metadata["sample_dates"]),
            Y_train_cov.shape[0] + Y_test_cov.shape[0],
        )

    def test_daily_and_weekly_artifacts_can_be_written_back_to_back(self):
        temp_dir, base_dir = self.make_temp_project()
        self.addCleanup(temp_dir.cleanup)

        self.write_project_config(base_dir, indicators=[{"name": "log"}, {"name": "volume_change"}])
        self.write_merged_csv(base_dir, make_weekly_feature_merged_df())

        daily_path, _ = save_dataset_bundle_from_config(base_dir=str(base_dir))
        weekly_path, _ = save_weekly_dataset_bundle(base_dir=str(base_dir))

        self.assertTrue(Path(daily_path).exists())
        self.assertTrue(Path(weekly_path).exists())
        self.assertNotEqual(Path(daily_path).read_bytes(), Path(weekly_path).read_bytes())

        scalers_dir = base_dir / "data" / "processed" / "scalers"
        for scaler_name in DAILY_SCALER_FILENAMES.values():
            self.assertTrue((scalers_dir / scaler_name).exists())
        for scaler_name in WEEKLY_SCALER_FILENAMES.values():
            self.assertTrue((scalers_dir / scaler_name).exists())

        daily_bundle = np.load(base_dir / "data" / "processed" / DAILY_DATASET_FILENAME)
        weekly_bundle = np.load(base_dir / "data" / "processed" / WEEKLY_DATASET_FILENAME)
        self.assertEqual(daily_bundle["target_frequency"].item(), "daily")
        self.assertEqual(weekly_bundle["target_frequency"].item(), "weekly")


if __name__ == "__main__":
    unittest.main()
