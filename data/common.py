from __future__ import annotations

import os
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

DAILY_DATASET_FILENAME = "qnn_datasets_daily.npz"
WEEKLY_DATASET_FILENAME = "qnn_datasets_weekly.npz"

DAILY_SCALER_FILENAMES = {
    "cov": "qnn_cov_scaler_daily.joblib",
    "ret": "qnn_ret_scaler_daily.joblib",
}

WEEKLY_SCALER_FILENAMES = {
    "cov": "qnn_cov_scaler_weekly.joblib",
    "ret": "qnn_ret_scaler_weekly.joblib",
}


def project_base_dir(anchor_file: str, base_dir: str | None = None) -> str:
    if base_dir is not None:
        return os.path.abspath(base_dir)
    return os.path.dirname(os.path.dirname(os.path.abspath(anchor_file)))


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def load_yaml_config(path: str, base_dir: str) -> dict[str, Any]:
    resolved = resolve_path(base_dir, path)
    with open(resolved, "r") as f:
        return yaml.safe_load(f)


def dump_yaml_config(path: str, data: dict[str, Any], base_dir: str) -> None:
    resolved = resolve_path(base_dir, path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def extract_asset_symbols(columns: Iterable[str]) -> list[str]:
    symbols = {column.split("_", 1)[0] for column in columns if column.endswith("_log")}
    return sorted(symbols)


def serialise_timestamps(values: Iterable[Any]) -> np.ndarray:
    return np.asarray([pd.Timestamp(value).isoformat() for value in values], dtype=str)


def npz_scalar(data: Any, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def npz_string_list(data: Any, key: str) -> list[str]:
    if key not in data.files:
        return []
    values = np.asarray(data[key]).tolist()
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def split_and_scale(
    X: np.ndarray,
    Y: np.ndarray,
    train_ratio: float,
    scaler_save_path: str | None = None,
):
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same number of samples.")
    if X.shape[0] == 0:
        raise ValueError("Dataset is empty.")
    if X.shape[1] == 0:
        raise ValueError("Feature matrix must contain at least one feature.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be strictly between 0 and 1.")

    n_samples = X.shape[0]
    split_idx = int(n_samples * train_ratio)
    if split_idx <= 0 or split_idx >= n_samples:
        raise ValueError(
            "train_ratio produced an empty train or test split. "
            "Adjust train_ratio or provide more samples."
        )

    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_save_path is not None:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)

    return X_train_scaled, X_test_scaled, Y_train, Y_test, scaler


def dataset_artifact_paths(config: dict[str, Any], base_dir: str, frequency: str) -> dict[str, str]:
    processed_dir = resolve_path(base_dir, config["paths"]["processed"])
    scalers_dir = resolve_path(base_dir, config["paths"]["scalers"])

    if frequency == "daily":
        dataset_name = DAILY_DATASET_FILENAME
        scaler_names = DAILY_SCALER_FILENAMES
    elif frequency == "weekly":
        dataset_name = WEEKLY_DATASET_FILENAME
        scaler_names = WEEKLY_SCALER_FILENAMES
    else:
        raise ValueError(f"Unsupported frequency '{frequency}'.")

    return {
        "dataset": os.path.join(processed_dir, dataset_name),
        "cov_scaler": os.path.join(scalers_dir, scaler_names["cov"]),
        "ret_scaler": os.path.join(scalers_dir, scaler_names["ret"]),
    }


def save_dataset_bundle(save_path: str, payload: dict[str, Any]) -> None:
    serialised: dict[str, Any] = {}
    for key, value in payload.items():
        if key.startswith("sample_dates") or key.startswith("eligible_weeks"):
            serialised[key] = serialise_timestamps(value)
        elif key == "asset_symbols":
            serialised[key] = np.asarray(value, dtype=str)
        elif isinstance(value, str):
            serialised[key] = np.asarray(value, dtype=str)
        else:
            serialised[key] = value

    if (
        "sample_dates_ret" in serialised
        and "sample_dates_cov" in serialised
        and np.array_equal(serialised["sample_dates_ret"], serialised["sample_dates_cov"])
    ):
        serialised["sample_dates"] = serialised["sample_dates_ret"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **serialised)
