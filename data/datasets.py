from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    from .common import (
        dataset_artifact_paths,
        extract_asset_symbols,
        load_yaml_config,
        project_base_dir,
        resolve_path,
        save_dataset_bundle,
        split_and_scale,
    )
except ImportError:
    from common import (
        dataset_artifact_paths,
        extract_asset_symbols,
        load_yaml_config,
        project_base_dir,
        resolve_path,
        save_dataset_bundle,
        split_and_scale,
    )


def prepare_qnn_cov_dataset(
    merged_df: pd.DataFrame,
    train_ratio: float,
    cov_window: int = 20,
    use_past_cov_in_features: bool = True,
    scaler_save_path: str | None = None,
    return_metadata: bool = False,
):
    """
    Prepare a next-step covariance dataset from daily log returns.

    sample_dates in the returned metadata correspond to the target dates of y_t.
    """
    ret_cols_all = [c for c in merged_df.columns if c.endswith("_log")]
    symbols_all = extract_asset_symbols(ret_cols_all)
    if not symbols_all:
        raise ValueError(
            "No *_log columns found in merged_df. "
            "Make sure your LogReturns indicator is applied and named 'log'."
        )

    ret_cols = [f"{sym}_log" for sym in symbols_all]
    log_ret = merged_df[ret_cols].copy().dropna()
    ret_index = log_ret.index
    n_time = len(log_ret)

    if cov_window > n_time:
        raise ValueError("cov_window is larger than number of return observations.")

    indicators = merged_df.loc[ret_index].copy()
    feature_cols = [
        c
        for c in indicators.columns
        if c.split("_", 1)[0] in symbols_all and not c.endswith("_close") and not c.endswith("_log")
    ]
    base_features = indicators[feature_cols]

    cov_list = []
    cov_dates = []
    for end_idx in range(cov_window - 1, n_time):
        window = log_ret.iloc[end_idx - cov_window + 1 : end_idx + 1]
        cov_list.append(window.cov().values)
        cov_dates.append(ret_index[end_idx])

    cov_list = np.stack(cov_list, axis=0)
    cov_dates = pd.DatetimeIndex(cov_dates)

    X_list = []
    Y_list = []
    sample_dates = []

    n_assets = len(symbols_all)
    iu = np.triu_indices(n_assets)

    for i in range(0, len(cov_dates) - 1):
        feature_date = cov_dates[i]
        target_date = cov_dates[i + 1]

        x_base = base_features.loc[feature_date].values
        C_t = cov_list[i]
        C_tp1 = cov_list[i + 1]

        if use_past_cov_in_features:
            x_vec = np.concatenate([x_base, C_t[iu]], axis=0)
        else:
            x_vec = x_base

        X_list.append(x_vec)
        Y_list.append(C_tp1[iu])
        sample_dates.append(target_date)

    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)
    split = split_and_scale(X, Y, train_ratio, scaler_save_path)

    if return_metadata:
        return (*split, {"sample_dates": pd.DatetimeIndex(sample_dates), "asset_symbols": symbols_all})
    return split


def prepare_qnn_ret_dataset(
    merged_df: pd.DataFrame,
    train_ratio: float,
    lookback_window: int = 20,
    use_past_ret_in_features: bool = True,
    scaler_save_path: str | None = None,
    return_metadata: bool = False,
):
    """
    Prepare a next-day returns dataset from daily log returns.

    sample_dates in the returned metadata correspond to the target dates of y_t.
    """
    ret_cols_all = [c for c in merged_df.columns if c.endswith("_log")]
    symbols_all = extract_asset_symbols(ret_cols_all)
    if not symbols_all:
        raise ValueError(
            "No *_log columns found in merged_df. "
            "Make sure your LogReturns indicator is applied and named 'log'."
        )

    ret_cols = [f"{sym}_log" for sym in symbols_all]
    log_ret = merged_df[ret_cols].copy().dropna()
    ret_index = log_ret.index
    n_time = len(log_ret)

    if lookback_window >= n_time:
        raise ValueError("lookback_window is too large compared to number of return observations.")

    indicators = merged_df.loc[ret_index].copy()
    feature_cols = [
        c
        for c in indicators.columns
        if c.split("_", 1)[0] in symbols_all and not c.endswith("_close") and not c.endswith("_log")
    ]
    base_features = indicators[feature_cols]

    X_list = []
    Y_list = []
    sample_dates = []

    for i in range(lookback_window - 1, n_time - 1):
        feature_date = ret_index[i]
        target_date = ret_index[i + 1]

        parts = [base_features.loc[feature_date].values]
        if use_past_ret_in_features:
            past_window = log_ret.iloc[i - lookback_window + 1 : i + 1]
            parts.append(past_window.values.ravel())

        X_list.append(np.concatenate(parts, axis=0))
        Y_list.append(log_ret.loc[target_date].values)
        sample_dates.append(target_date)

    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)
    split = split_and_scale(X, Y, train_ratio, scaler_save_path)

    if return_metadata:
        return (*split, {"sample_dates": pd.DatetimeIndex(sample_dates), "asset_symbols": symbols_all})
    return split


def build_dataset_bundle(
    config_path: str = "config/data_config.yaml",
    base_dir: str | None = None,
    lookback_window: int | None = None,
    cov_window: int | None = None,
    use_past_cov_in_features: bool | None = None,
    use_past_ret_in_features: bool | None = None,
):
    base_dir = project_base_dir(__file__, base_dir)
    config = load_yaml_config(config_path, base_dir)
    paths = config["paths"]

    merged_save_path = os.path.join(resolve_path(base_dir, paths["raw"]), "merged_data.csv")
    merged_df = pd.read_csv(merged_save_path, index_col="timestamp", parse_dates=True)

    train_size = config["train_size"]
    lookback_window = config["lookback_window"] if lookback_window is None else lookback_window
    cov_window = config["cov_window"] if cov_window is None else cov_window
    if use_past_cov_in_features is None:
        use_past_cov_in_features = config["use_past_cov_in_features"]
    if use_past_ret_in_features is None:
        use_past_ret_in_features = config["use_past_ret_in_features"]

    artifacts = dataset_artifact_paths(config, base_dir, "daily")

    (
        X_train_cov,
        X_test_cov,
        Y_train_cov,
        Y_test_cov,
        _,
        cov_metadata,
    ) = prepare_qnn_cov_dataset(
        merged_df,
        train_ratio=train_size,
        cov_window=cov_window,
        use_past_cov_in_features=use_past_cov_in_features,
        scaler_save_path=artifacts["cov_scaler"],
        return_metadata=True,
    )

    (
        X_train_ret,
        X_test_ret,
        Y_train_ret,
        Y_test_ret,
        _,
        ret_metadata,
    ) = prepare_qnn_ret_dataset(
        merged_df,
        train_ratio=train_size,
        lookback_window=lookback_window,
        use_past_ret_in_features=use_past_ret_in_features,
        scaler_save_path=artifacts["ret_scaler"],
        return_metadata=True,
    )

    asset_symbols = ret_metadata["asset_symbols"]
    if asset_symbols != cov_metadata["asset_symbols"]:
        raise ValueError("Returns and covariance datasets resolved different asset orders.")

    return {
        "X_train_cov": X_train_cov,
        "X_test_cov": X_test_cov,
        "Y_train_cov": Y_train_cov,
        "Y_test_cov": Y_test_cov,
        "X_train_ret": X_train_ret,
        "X_test_ret": X_test_ret,
        "Y_train_ret": Y_train_ret,
        "Y_test_ret": Y_test_ret,
        "asset_symbols": asset_symbols,
        "sample_dates_cov": cov_metadata["sample_dates"],
        "sample_dates_ret": ret_metadata["sample_dates"],
        "target_frequency": "daily",
        "lookback": lookback_window,
        "cov_window": cov_window,
        "train_ratio": train_size,
    }


def create_datasets(
    lookback_window: int | None = None,
    cov_window: int | None = None,
    use_past_cov_in_features: bool | None = None,
    use_past_ret_in_features: bool | None = None,
):
    bundle = build_dataset_bundle(
        lookback_window=lookback_window,
        cov_window=cov_window,
        use_past_cov_in_features=use_past_cov_in_features,
        use_past_ret_in_features=use_past_ret_in_features,
    )
    return (
        bundle["X_train_cov"],
        bundle["X_test_cov"],
        bundle["Y_train_cov"],
        bundle["Y_test_cov"],
        bundle["X_train_ret"],
        bundle["X_test_ret"],
        bundle["Y_train_ret"],
        bundle["Y_test_ret"],
    )


def save_dataset_bundle_from_config(
    config_path: str = "config/data_config.yaml",
    base_dir: str | None = None,
):
    base_dir = project_base_dir(__file__, base_dir)
    config = load_yaml_config(config_path, base_dir)
    artifacts = dataset_artifact_paths(config, base_dir, "daily")
    bundle = build_dataset_bundle(config_path=config_path, base_dir=base_dir)
    save_dataset_bundle(artifacts["dataset"], bundle)
    print(f"Daily datasets created and saved successfully to {artifacts['dataset']}.")
    return artifacts["dataset"], bundle


if __name__ == "__main__":
    save_dataset_bundle_from_config()
