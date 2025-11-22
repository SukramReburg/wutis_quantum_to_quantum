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


def prepare_qnn_cov_dataset(
    merged_df: pd.DataFrame,
    train_ratio: float,
    cov_window: int = 20,
    use_past_cov_in_features: bool = True,
    scaler_save_path: str = None
):
    """
    Prepare X_train, X_test, Y_train, Y_test for a QNN that predicts *next-day*
    rolling covariance matrices C_{t+1} from information at time t.

    Logic:
    - Use existing log-return indicator columns '<SYMBOL>_log'.
    - Compute a time series of rolling covariances C_t using the past `cov_window`
      log-returns up to t (t - cov_window + 1 ... t).
    - Build samples (X_t, y_t):
        X_t = indicators at time t (+ optionally flattened C_t)
        y_t = flattened upper-triangular of C_{t+1}
    - Split by time (no shuffling) and scale X.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged dataframe with indicators and '<SYMBOL>_log' return columns.
    train_ratio : float
        Fraction of samples used for training (from the start).
    cov_window : int
        Rolling window length (in days) to estimate covariances C_t.
    use_past_cov_in_features : bool
        If True, append flattened C_t to the feature vector X_t.
    scaler_save_path : str or None
        Optional path to save fitted StandardScaler.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    Y_train, Y_test : np.ndarray
    scaler : StandardScaler
    """

    # 1) Find symbols that have *_log columns
    ret_cols_all = [c for c in merged_df.columns if c.endswith('_log')]
    symbols_all = sorted({c.split('_')[0] for c in ret_cols_all})

    if len(symbols_all) == 0:
        raise ValueError(
            "No *_log columns found in merged_df. "
            "Make sure your LogReturns indicator is applied and named 'log'."
        )

    print(f"Selected symbols for QNN covariance prediction: {symbols_all}")

    # 2) Build log-returns dataframe directly from indicator columns
    ret_cols = [f"{sym}_log" for sym in symbols_all]
    log_ret = merged_df[ret_cols].copy().dropna()
    ret_index = log_ret.index
    n_time = len(log_ret)

    if cov_window > n_time:
        raise ValueError("cov_window is larger than number of return observations.")

    # 3) Align indicators with log-returns index
    indicators = merged_df.loc[ret_index].copy()

    # Feature columns: all indicators for selected symbols,
    # EXCLUDING *_close and *_log (we handle returns separately)
    feature_cols = [
        c for c in indicators.columns
        if (c.split('_')[0] in symbols_all)
           and (not c.endswith('_close'))
           and (not c.endswith('_log'))
    ]
    base_features = indicators[feature_cols]

    # 4) Compute rolling covariances C_t using past cov_window returns
    cov_list = []
    cov_dates = []

    for end_idx in range(cov_window - 1, n_time):
        # returns from end_idx - cov_window + 1 ... end_idx (inclusive)
        window = log_ret.iloc[end_idx - cov_window + 1 : end_idx + 1]
        cov = window.cov().values  # shape (n_assets, n_assets)
        cov_list.append(cov)
        cov_dates.append(ret_index[end_idx])

    cov_list = np.stack(cov_list, axis=0)   # (T_cov, n_assets, n_assets)
    cov_dates = np.array(cov_dates)         # (T_cov,)

    # 5) Build (X_t, y_t) with next-day target: y_t = C_{t+1}
    X_list = []
    Y_list = []

    n_assets = len(symbols_all)
    iu = np.triu_indices(n_assets)

    for i in range(0, len(cov_dates) - 1):
        date_t = cov_dates[i]

        # base indicators at time t
        x_base = base_features.loc[date_t].values  # (n_base_features,)

        C_t = cov_list[i]
        C_tp1 = cov_list[i + 1]

        y_vec = C_tp1[iu]  # target: next-day cov

        if use_past_cov_in_features:
            x_cov = C_t[iu]
            x_vec = np.concatenate([x_base, x_cov], axis=0)
        else:
            x_vec = x_base

        X_list.append(x_vec)
        Y_list.append(y_vec)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    print(f"Built dataset (one-step ahead cov): X shape = {X.shape}, Y shape = {Y.shape}")

    # 6) Split + scale via helper
    return _split_and_scale(X, Y, train_ratio, scaler_save_path)


def prepare_qnn_ret_dataset(
    merged_df: pd.DataFrame,
    train_ratio: float,
    lookback_window: int = 20,
    use_past_ret_in_features: bool = True,
    scaler_save_path: str = None
):
    """
    Prepare X_train, X_test, Y_train, Y_test for a QNN that predicts
    *next-day log-returns* r_{t+1} from information at time t.

    Uses existing log-return indicator columns '<SYMBOL>_log'.

    Logic:
    - Use '<SYMBOL>_log' columns as log returns.
    - For each valid time t:
        X_t = indicators at time t (+ optionally flattened past returns window)
        y_t = next-day returns r_{t+1}
    - Split by time (no shuffling) and scale X.
    """

    # 1) Determine symbols based on *_log columns
    ret_cols_all = [c for c in merged_df.columns if c.endswith('_log')]
    symbols_all = sorted({c.split('_')[0] for c in ret_cols_all})

    if len(symbols_all) == 0:
        raise ValueError(
            "No *_log columns found in merged_df. "
            "Make sure your LogReturns indicator is applied and named 'log'."
        )

    n_assets = len(symbols_all)
    print(f"Selected symbols for QNN returns prediction: {symbols_all}")

    # 2) Log-returns directly from indicators
    ret_cols = [f"{sym}_log" for sym in symbols_all]
    log_ret = merged_df[ret_cols].copy().dropna()
    ret_index = log_ret.index
    n_time = len(log_ret)

    if lookback_window >= n_time:
        raise ValueError("lookback_window is too large compared to number of return observations.")

    # 3) Align indicators with returns index
    indicators = merged_df.loc[ret_index].copy()

    # Feature columns: indicators for selected symbols, excluding *_close and *_log
    feature_cols = [
        c for c in indicators.columns
        if (c.split('_')[0] in symbols_all)
           and (not c.endswith('_close'))
           and (not c.endswith('_log'))
    ]
    base_features = indicators[feature_cols]

    # 4) Build (X_t, y_t)
    X_list = []
    Y_list = []

    # Start at i = lookback_window - 1 so we have enough past returns,
    # end at n_time - 2 so that r_{i+1} exists.
    for i in range(lookback_window - 1, n_time - 1):
        date_t = ret_index[i]
        date_tp1 = ret_index[i + 1]

        # Base indicators at time t
        x_base = base_features.loc[date_t].values  # (n_base_features,)

        parts = [x_base]

        if use_past_ret_in_features:
            # Past returns window [i - lookback_window + 1, ..., i]
            past_window = log_ret.iloc[i - lookback_window + 1 : i + 1]  # (lookback_window, n_assets)
            parts.append(past_window.values.ravel())

        x_vec = np.concatenate(parts, axis=0)

        # Target: next-day returns r_{t+1}
        y_vec = log_ret.loc[date_tp1].values  # (n_assets,)

        X_list.append(x_vec)
        Y_list.append(y_vec)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    print(f"Built returns dataset: X shape = {X.shape}, Y shape = {Y.shape}")

    # 5) Split + scale via helper
    return _split_and_scale(X, Y, train_ratio, scaler_save_path)



def create_datasets(lookback_window = None, 
                    cov_window = None, 
                    use_past_cov_in_features = None, 
                    use_past_ret_in_features = None):
    
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    merged_save_path = os.path.join(base_dir, paths['raw'], 'merged_data.csv')
    merged_df = pd.read_csv(merged_save_path, index_col='timestamp', parse_dates=True)

    train_size = config['train_size']   # e.g. 0.8
    if lookback_window is None:
        lookback_window = config['lookback_window']
    if cov_window is None:
        cov_window = config['cov_window']
    if use_past_cov_in_features is None:
        use_past_cov_in_features = config['use_past_cov_in_features']
    if use_past_ret_in_features is None:
        use_past_ret_in_features = config['use_past_ret_in_features']
    
    # ---------- Covariance dataset ----------
    scaler_path_cov = os.path.join(base_dir, paths['scalers'], "qnn_cov_scaler.joblib")
    X_train_cov, X_test_cov, Y_train_cov, Y_test_cov, scaler_cov = prepare_qnn_cov_dataset(
        merged_df,
        train_ratio=train_size,
        cov_window=cov_window,
        use_past_cov_in_features=use_past_cov_in_features,
        scaler_save_path=scaler_path_cov
    )

    # ---------- Returns dataset ----------
    scaler_path_ret = os.path.join(base_dir, paths['scalers'], "qnn_ret_scaler.joblib")
    X_train_ret, X_test_ret, Y_train_ret, Y_test_ret, scaler_ret = prepare_qnn_ret_dataset(
        merged_df,
        train_ratio=train_size,
        lookback_window=lookback_window,
        use_past_ret_in_features=use_past_ret_in_features,
        scaler_save_path=scaler_path_ret,
    )


    return (X_train_cov, X_test_cov, Y_train_cov, Y_test_cov,
            X_train_ret, X_test_ret, Y_train_ret, Y_test_ret)

if __name__ == "__main__":
    X_train_cov, X_test_cov, Y_train_cov, Y_test_cov, X_train_ret, X_test_ret, Y_train_ret, Y_test_ret = create_datasets()

    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_path = os.path.join(base_dir, paths['processed'])
    os.makedirs(dataset_path, exist_ok=True)
    np.savez_compressed(
        os.path.join(dataset_path, 'qnn_datasets.npz'),
        X_train_cov=X_train_cov,
        X_test_cov=X_test_cov,
        Y_train_cov=Y_train_cov,
        Y_test_cov=Y_test_cov,
        X_train_ret=X_train_ret,
        X_test_ret=X_test_ret,
        Y_train_ret=Y_train_ret,
        Y_test_ret=Y_test_ret
    )
    print("Datasets created and saved successfully.")