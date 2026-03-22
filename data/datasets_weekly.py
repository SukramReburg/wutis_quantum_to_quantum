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


def build_weekly_views(merged_df: pd.DataFrame):
    ret_cols_all = [c for c in merged_df.columns if c.endswith("_log")]
    symbols_all = extract_asset_symbols(ret_cols_all)
    if not symbols_all:
        raise ValueError("No *_log columns found in merged_df.")

    ret_cols = [f"{sym}_log" for sym in symbols_all]
    log_ret_daily = merged_df[ret_cols].copy().dropna()

    eligible_weeks = []
    weekly_returns = []
    weekly_covariances = []
    for week_end, week_df in log_ret_daily.groupby(pd.Grouper(freq="W-FRI")):
        if len(week_df) < 2:
            continue
        eligible_weeks.append(week_end)
        weekly_returns.append(week_df.sum().values)
        weekly_covariances.append(week_df.cov().values)

    if not eligible_weeks:
        raise ValueError("No eligible weekly windows could be computed.")

    eligible_weeks = pd.DatetimeIndex(eligible_weeks)
    weekly_ret = pd.DataFrame(weekly_returns, index=eligible_weeks, columns=symbols_all)
    weekly_cov = np.stack(weekly_covariances, axis=0)
    indicators_weekly = merged_df.resample("W-FRI").last().loc[eligible_weeks]

    return {
        "asset_symbols": symbols_all,
        "eligible_weeks": eligible_weeks,
        "weekly_ret": weekly_ret,
        "weekly_cov": weekly_cov,
        "indicators_weekly": indicators_weekly,
    }


def prepare_weekly_qnn_ret_dataset(
    merged_df: pd.DataFrame,
    train_ratio: float,
    lookback_weeks: int = 4,
    use_past_ret_in_features: bool = True,
    scaler_save_path: str | None = None,
    return_metadata: bool = False,
):
    weekly = build_weekly_views(merged_df)
    weekly_ret = weekly["weekly_ret"]
    indicators_weekly = weekly["indicators_weekly"]
    symbols_all = weekly["asset_symbols"]
    weekly_index = weekly["eligible_weeks"]

    n_weeks = len(weekly_ret)
    if lookback_weeks > n_weeks - 1:
        raise ValueError("Not enough weeks for given lookback_weeks.")

    feature_cols = [
        c
        for c in indicators_weekly.columns
        if c.split("_", 1)[0] in symbols_all and not c.endswith("_close") and not c.endswith("_log")
    ]
    base_features_weekly = indicators_weekly[feature_cols]

    X_list = []
    Y_list = []
    sample_dates = []

    for i in range(lookback_weeks - 1, n_weeks - 1):
        week_t = weekly_index[i]
        week_tp1 = weekly_index[i + 1]

        parts = [base_features_weekly.loc[week_t].values]
        if use_past_ret_in_features:
            past_weekly_ret = weekly_ret.iloc[i - lookback_weeks + 1 : i + 1]
            parts.append(past_weekly_ret.values.ravel())

        X_list.append(np.concatenate(parts, axis=0))
        Y_list.append(weekly_ret.loc[week_tp1].values)
        sample_dates.append(week_tp1)

    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)
    split = split_and_scale(X, Y, train_ratio, scaler_save_path)

    if return_metadata:
        return (
            *split,
            {
                "sample_dates": pd.DatetimeIndex(sample_dates),
                "asset_symbols": symbols_all,
                "eligible_weeks": weekly_index,
            },
        )
    return split


def prepare_weekly_qnn_cov_dataset(
    merged_df: pd.DataFrame,
    train_ratio: float,
    cov_lookback_weeks: int = 4,
    use_past_cov_in_features: bool = True,
    scaler_save_path: str | None = None,
    return_metadata: bool = False,
):
    weekly = build_weekly_views(merged_df)
    weekly_cov = weekly["weekly_cov"]
    indicators_weekly = weekly["indicators_weekly"]
    symbols_all = weekly["asset_symbols"]
    weekly_index = weekly["eligible_weeks"]

    n_weeks = len(weekly_index)
    if cov_lookback_weeks > n_weeks - 1:
        raise ValueError("Not enough weeks for given cov_lookback_weeks.")

    feature_cols = [
        c
        for c in indicators_weekly.columns
        if c.split("_", 1)[0] in symbols_all and not c.endswith("_close") and not c.endswith("_log")
    ]
    base_features_weekly = indicators_weekly[feature_cols]

    iu = np.triu_indices(len(symbols_all))
    X_list = []
    Y_list = []
    sample_dates = []

    for i in range(cov_lookback_weeks - 1, n_weeks - 1):
        week_t = weekly_index[i]
        week_tp1 = weekly_index[i + 1]

        parts = [base_features_weekly.loc[week_t].values]
        if use_past_cov_in_features:
            past_covs = weekly_cov[i - cov_lookback_weeks + 1 : i + 1]
            parts.append(past_covs[:, iu[0], iu[1]].ravel())

        X_list.append(np.concatenate(parts, axis=0))
        Y_list.append(weekly_cov[i + 1][iu])
        sample_dates.append(week_tp1)

    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)
    split = split_and_scale(X, Y, train_ratio, scaler_save_path)

    if return_metadata:
        return (
            *split,
            {
                "sample_dates": pd.DatetimeIndex(sample_dates),
                "asset_symbols": symbols_all,
                "eligible_weeks": weekly_index,
            },
        )
    return split


def build_dataset_bundle(
    config_path: str = "config/data_config.yaml",
    base_dir: str | None = None,
    lookback_weeks: int | None = None,
    cov_lookback_weeks: int | None = None,
    use_past_cov_in_features: bool | None = None,
    use_past_ret_in_features: bool | None = None,
):
    base_dir = project_base_dir(__file__, base_dir)
    config = load_yaml_config(config_path, base_dir)
    paths = config["paths"]

    merged_save_path = os.path.join(resolve_path(base_dir, paths["raw"]), "merged_data.csv")
    merged_df = pd.read_csv(merged_save_path, index_col="timestamp", parse_dates=True)

    train_size = config["train_size"]
    lookback_weeks = config.get("lookback_window", 4) if lookback_weeks is None else lookback_weeks
    cov_lookback_weeks = config.get("cov_window", 4) if cov_lookback_weeks is None else cov_lookback_weeks
    if use_past_cov_in_features is None:
        use_past_cov_in_features = config["use_past_cov_in_features"]
    if use_past_ret_in_features is None:
        use_past_ret_in_features = config["use_past_ret_in_features"]

    artifacts = dataset_artifact_paths(config, base_dir, "weekly")

    (
        X_train_cov,
        X_test_cov,
        Y_train_cov,
        Y_test_cov,
        _,
        cov_metadata,
    ) = prepare_weekly_qnn_cov_dataset(
        merged_df=merged_df,
        train_ratio=train_size,
        cov_lookback_weeks=cov_lookback_weeks,
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
    ) = prepare_weekly_qnn_ret_dataset(
        merged_df=merged_df,
        train_ratio=train_size,
        lookback_weeks=lookback_weeks,
        use_past_ret_in_features=use_past_ret_in_features,
        scaler_save_path=artifacts["ret_scaler"],
        return_metadata=True,
    )

    if ret_metadata["asset_symbols"] != cov_metadata["asset_symbols"]:
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
        "asset_symbols": ret_metadata["asset_symbols"],
        "sample_dates_cov": cov_metadata["sample_dates"],
        "sample_dates_ret": ret_metadata["sample_dates"],
        "eligible_weeks_cov": cov_metadata["eligible_weeks"],
        "eligible_weeks_ret": ret_metadata["eligible_weeks"],
        "target_frequency": "weekly",
        "lookback": lookback_weeks,
        "cov_window": cov_lookback_weeks,
        "train_ratio": train_size,
    }


def create_datasets(
    lookback_weeks: int | None = None,
    cov_lookback_weeks: int | None = None,
    use_past_cov_in_features: bool | None = None,
    use_past_ret_in_features: bool | None = None,
):
    bundle = build_dataset_bundle(
        lookback_weeks=lookback_weeks,
        cov_lookback_weeks=cov_lookback_weeks,
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
    artifacts = dataset_artifact_paths(config, base_dir, "weekly")
    bundle = build_dataset_bundle(config_path=config_path, base_dir=base_dir)
    save_dataset_bundle(artifacts["dataset"], bundle)
    print(f"Weekly datasets created and saved successfully to {artifacts['dataset']}.")
    return artifacts["dataset"], bundle


if __name__ == "__main__":
    save_dataset_bundle_from_config()
