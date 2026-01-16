import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os 
import yaml 

def _split_and_scale(X: np.ndarray,
                     Y: np.ndarray,
                     train_ratio: float,
                     scaler_save_path: str | None = None):
    """Time-based train/test split and scaling of X only."""
    n_samples = X.shape[0]
    split_idx = int(n_samples * train_ratio)

    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_save_path is not None:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)

    return X_train_scaled, X_test_scaled, Y_train, Y_test, scaler



def prepare_weekly_qnn_ret_dataset(
    merged_df: pd.DataFrame,
    train_ratio: float,
    lookback_weeks: int = 4,
    use_past_ret_in_features: bool = True,
    scaler_save_path: str = None,
):
    """
    Prepare X_train, X_test, Y_train, Y_test for a QNN that predicts
    *next-week log-returns* r_{t+1} from information at time t.

    Uses existing log-return indicator columns '<SYMBOL>_log'.

    Logic:
    - Use '<SYMBOL>_log' columns as log returns.
    - For each valid time t:
        X_t = indicators at time t (+ optionally flattened past returns window)
        y_t = next-day returns r_{t+1}
    - Split by time (no shuffling) and scale X.
    """

    # 1) Determine symbols based on *_log columns (daily log returns)
    ret_cols_all = [c for c in merged_df.columns if c.endswith('_log')]
    symbols_all = sorted({c.split('_')[0] for c in ret_cols_all})
    if len(symbols_all) == 0:
        raise ValueError("No *_log columns found in merged_df.")

    #n_assets = len(symbols_all)
    ret_cols = [f"{sym}_log" for sym in symbols_all]
    log_ret_daily = merged_df[ret_cols].copy().dropna()

    # 2) Weekly aggregated returns: sum of daily log returns in each week
    # Week ends on Friday; adjust 'W-FRI' if needed - sample from Monday to Friday
    weekly_ret = log_ret_daily.resample("W-FRI").sum()
    weekly_index = weekly_ret.index
    n_weeks = len(weekly_ret)
    if n_weeks <= lookback_weeks + 1:
        raise ValueError("Not enough weeks for given lookback_weeks.")

    # 3) Weekly indicators: take last daily row of each week
    indicators_weekly = merged_df.resample("W-FRI").last().loc[weekly_index]

    feature_cols = [
        c for c in indicators_weekly.columns
        if (c.split('_')[0] in symbols_all)
           and (not c.endswith('_close'))
           and (not c.endswith('_log'))
    ]
    base_features_weekly = indicators_weekly[feature_cols]

    # 4) Build weekly samples
    X_list, Y_list = [], []

    # we need lookback_weeks in the past and 1 week for the future
    for i in range(lookback_weeks, n_weeks - 1):
        # week t index
        week_t = weekly_index[i]
        week_tp1 = weekly_index[i + 1]

        x_base = base_features_weekly.loc[week_t].values  # indicators at end of week t
        parts = [x_base]

        if use_past_ret_in_features:
            # past lookback_weeks weekly returns up to week t
            past_weekly_ret = weekly_ret.iloc[i - lookback_weeks : i]  # (lookback_weeks, n_assets)
            parts.append(past_weekly_ret.values.ravel())

        x_vec = np.concatenate(parts, axis=0)

        # target: aggregated return of next week
        y_vec = weekly_ret.loc[week_tp1].values  # (n_assets,)
        X_list.append(x_vec)
        Y_list.append(y_vec)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    print(f"Built WEEKLY returns dataset: X shape = {X.shape}, Y shape = {Y.shape}")

    return _split_and_scale(X, Y, train_ratio, scaler_save_path)


def prepare_weekly_qnn_cov_dataset(
    merged_df: pd.DataFrame,
    train_ratio: float,
    cov_lookback_weeks: int = 4,
    use_past_cov_in_features: bool = True,
    scaler_save_path: str = None,
):
    """
    Weekly covariance dataset:
      - One sample per week.
      - X_t: indicators at end of week t (+ optional past weekly covariances).
      - y_t: covariance matrix of DAILY returns within week t+1 (flattened upper triangle).

    Steps:
      1) Use daily '<SYMBOL>_log' columns as daily log returns.
      2) Group daily returns by calendar week (ending Friday) and compute
         for each week: realised covariance of daily returns.
      3) For each week t with enough history:
           - features: indicators at end of week t
                      (+ past cov_lookback_weeks weekly covariances, flattened)
           - target: upper-triangular of next-week covariance (week t+1).
    """

    # 1) Symbols and daily log returns
    ret_cols_all = [c for c in merged_df.columns if c.endswith("_log")]
    symbols_all = sorted({c.split("_")[0] for c in ret_cols_all})
    if len(symbols_all) == 0:
        raise ValueError("No *_log columns found in merged_df.")

    n_assets = len(symbols_all)
    ret_cols = [f"{sym}_log" for sym in symbols_all]
    log_ret_daily = merged_df[ret_cols].copy().dropna()

    # 2) Compute weekly realised covariance of DAILY returns
    # Group by calendar weeks ending on Friday
    weekly_cov_list = []
    week_ends = []

    for week_end, week_df in log_ret_daily.groupby(pd.Grouper(freq="W-FRI")):
        # skip empty groups
        if len(week_df) == 0:
            continue
        # if only one day in the week, cov is degenerate; you can choose to skip or accept
        if len(week_df) < 2:
            continue
        week_ends.append(week_end)
        weekly_cov_list.append(week_df.cov().values)  # (n_assets, n_assets)

    if len(weekly_cov_list) == 0:
        raise ValueError("No weekly covariance matrices could be computed.")

    weekly_cov = np.stack(weekly_cov_list, axis=0)     # shape: (n_weeks, n_assets, n_assets)
    week_ends = pd.DatetimeIndex(week_ends)            # length n_weeks
    n_weeks = weekly_cov.shape[0]

    if n_weeks <= cov_lookback_weeks + 1:
        raise ValueError("Not enough weeks for given cov_lookback_weeks.")

    # 3) Weekly indicators at week-ends (same week definition)
    indicators_weekly = merged_df.resample("W-FRI").last().loc[week_ends]

    feature_cols = [
        c for c in indicators_weekly.columns
        if (c.split("_")[0] in symbols_all)
           and (not c.endswith("_close"))
           and (not c.endswith("_log"))
    ]
    base_features_weekly = indicators_weekly[feature_cols]

    # 4) Build weekly samples
    iu = np.triu_indices(n_assets)  # indices for upper-triangular part
    X_list, Y_list = [], []

    # Need cov_lookback_weeks of history plus 1 future week for target
    for i in range(cov_lookback_weeks, n_weeks - 1):
        week_t = week_ends[i]
        C_tp1 = weekly_cov[i + 1]          # cov of next week (week t+1)

        # Base indicators at end of week t
        x_base = base_features_weekly.loc[week_t].values
        parts = [x_base]

        if use_past_cov_in_features:
            # past cov_lookback_weeks weekly covariances up to week t
            past_covs = weekly_cov[i - cov_lookback_weeks : i]  # (L, n_assets, n_assets)
            past_cov_flat = past_covs[:, iu[0], iu[1]].ravel()
            parts.append(past_cov_flat)

        x_vec = np.concatenate(parts, axis=0)
        y_vec = C_tp1[iu]  # upper-triangular of next-week covariance

        X_list.append(x_vec)
        Y_list.append(y_vec)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    print(f"Built WEEKLY covariance dataset: X shape = {X.shape}, Y shape = {Y.shape}")

    return _split_and_scale(X, Y, train_ratio, scaler_save_path)


def create_datasets(
    lookback_weeks: int | None = None,
    cov_lookback_weeks: int | None = None,
    use_past_cov_in_features: bool | None = None,
    use_past_ret_in_features: bool | None = None,
):
    # Load config
    with open("config/data_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load merged daily data
    merged_save_path = os.path.join(base_dir, paths["raw"], "merged_data.csv")
    merged_df = pd.read_csv(merged_save_path, index_col="timestamp", parse_dates=True)

    train_size = config["train_size"]

    # If kwargs are None, fall back to config values (reuse existing fields as "weeks")
    # e.g. interpret config['lookback_window'] as "lookback_weeks"
    if lookback_weeks is None:
        lookback_weeks = config.get("lookback_window", 4)
    if cov_lookback_weeks is None:
        cov_lookback_weeks = config.get("cov_window", 4)
    if use_past_cov_in_features is None:
        use_past_cov_in_features = config["use_past_cov_in_features"]
    if use_past_ret_in_features is None:
        use_past_ret_in_features = config["use_past_ret_in_features"]

    # ---------- WEEKLY covariance dataset ----------
    scaler_path_cov = os.path.join(base_dir, paths["scalers"], "qnn_cov_scaler.joblib")
    print("Cov scaler will be saved to:", scaler_path_cov)
    (
        X_train_cov,
        X_test_cov,
        Y_train_cov,
        Y_test_cov,
        scaler_cov,
    ) = prepare_weekly_qnn_cov_dataset(
        merged_df=merged_df,
        train_ratio=train_size,
        cov_lookback_weeks=cov_lookback_weeks,
        use_past_cov_in_features=use_past_cov_in_features,
        scaler_save_path=scaler_path_cov,
    )

    # ---------- WEEKLY returns dataset ----------
    scaler_path_ret = os.path.join(base_dir, paths["scalers"], "qnn_ret_scaler.joblib")
    (
        X_train_ret,
        X_test_ret,
        Y_train_ret,
        Y_test_ret,
        scaler_ret,
    ) = prepare_weekly_qnn_ret_dataset(
        merged_df=merged_df,
        train_ratio=train_size,
        lookback_weeks=lookback_weeks,
        use_past_ret_in_features=use_past_ret_in_features,
        scaler_save_path=scaler_path_ret,
    )

    return (
        X_train_cov,
        X_test_cov,
        Y_train_cov,
        Y_test_cov,
        X_train_ret,
        X_test_ret,
        Y_train_ret,
        Y_test_ret,
    )

if __name__ == "__main__":
    (
        X_train_cov,
        X_test_cov,
        Y_train_cov,
        Y_test_cov,
        X_train_ret,
        X_test_ret,
        Y_train_ret,
        Y_test_ret,
    ) = create_datasets()

    with open("config/data_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    paths = config["paths"]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_path = os.path.join(base_dir, paths["processed"])
    os.makedirs(dataset_path, exist_ok=True)
    np.savez_compressed(
        os.path.join(dataset_path, "qnn_datasets.npz"),
        X_train_cov=X_train_cov,
        X_test_cov=X_test_cov,
        Y_train_cov=Y_train_cov,
        Y_test_cov=Y_test_cov,
        X_train_ret=X_train_ret,
        X_test_ret=X_test_ret,
        Y_train_ret=Y_train_ret,
        Y_test_ret=Y_test_ret,
    )
    print("Weekly datasets created and saved successfully.")

