from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data.common import npz_scalar, npz_string_list


@dataclass
class PredictionBundle:
    path: str
    kind: str
    y_pred: np.ndarray
    y_true: np.ndarray
    asset_symbols: list[str]
    sample_dates: list[str]
    target_frequency: str
    runtime_profile: str | None
    loss_name: str | None
    best_epoch: int | None
    y_pred_std: np.ndarray | None

    @property
    def n_assets(self) -> int:
        return len(self.asset_symbols)

    @property
    def sample_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.to_datetime(self.sample_dates)) if self.sample_dates else pd.DatetimeIndex([])

    @classmethod
    def load(cls, path: str, kind: str, expected_frequency: str | None = None) -> "PredictionBundle":
        data = np.load(path)
        asset_symbols = npz_string_list(data, "asset_symbols")
        sample_dates = npz_string_list(data, "sample_dates") or npz_string_list(data, "sample_dates_test")
        target_frequency = str(npz_scalar(data, "target_frequency", "weekly"))
        if expected_frequency is not None and target_frequency != expected_frequency:
            raise ValueError(
                f"Expected {expected_frequency} predictions in {path}, got {target_frequency}."
            )

        y_pred = np.asarray(data["Y_pred_test"], dtype=np.float32)
        y_true = np.asarray(data["Y_true_test"], dtype=np.float32)
        y_pred_std = (
            np.asarray(data["Y_pred_std_test"], dtype=np.float32)
            if "Y_pred_std_test" in data.files
            else None
        )
        runtime_profile = npz_scalar(data, "runtime_profile")
        loss_name = npz_scalar(data, "loss_name")
        best_epoch = npz_scalar(data, "best_epoch")

        if y_pred.ndim != 2 or y_true.ndim != 2:
            raise ValueError(f"Prediction bundle {path} must store 2D arrays.")
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Prediction bundle {path} has mismatched true/pred shapes.")
        if sample_dates and len(sample_dates) != y_pred.shape[0]:
            raise ValueError(f"Prediction bundle {path} has inconsistent sample_dates length.")
        if kind == "returns" and asset_symbols and len(asset_symbols) != y_pred.shape[1]:
            raise ValueError(f"Returns bundle {path} asset_symbols length does not match outputs.")
        if kind == "cov" and asset_symbols:
            expected_dim = len(asset_symbols) * (len(asset_symbols) + 1) // 2
            if y_pred.shape[1] != expected_dim:
                raise ValueError(
                    f"Covariance bundle {path} expected {expected_dim} outputs for {len(asset_symbols)} assets."
                )

        return cls(
            path=path,
            kind=kind,
            y_pred=y_pred,
            y_true=y_true,
            asset_symbols=asset_symbols,
            sample_dates=sample_dates,
            target_frequency=target_frequency,
            runtime_profile=str(runtime_profile) if runtime_profile is not None else None,
            loss_name=str(loss_name) if loss_name is not None else None,
            best_epoch=int(best_epoch) if best_epoch is not None else None,
            y_pred_std=y_pred_std,
        )
