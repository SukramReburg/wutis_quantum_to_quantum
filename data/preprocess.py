from __future__ import annotations

import os

import pandas as pd

try:
    from .common import dump_yaml_config, load_yaml_config, project_base_dir, resolve_path
    from .indicators import indicators_impl
except ImportError:
    from common import dump_yaml_config, load_yaml_config, project_base_dir, resolve_path
    from indicators import indicators_impl


RAW_COLUMNS_TO_DROP = ["high", "low", "volume", "open", "close", "trade_count", "vwap"]


def add_indicators(data: pd.DataFrame, indicator_configs: list[dict]) -> pd.DataFrame:
    """Add configured technical indicators without mutating the caller's dataframe."""
    result = data.copy()
    for indicator_cfg in indicator_configs:
        name = indicator_cfg["name"]
        params = indicator_cfg.get("params", {})
        if name not in indicators_impl:
            print(f"Indicator '{name}' not recognized.")
            continue

        indicator = indicators_impl[name](result, **params, name=name)
        result = indicator.add_indicator()

    result = result.drop(columns=RAW_COLUMNS_TO_DROP, errors="ignore")
    return result


def _extract_symbol(frame: pd.DataFrame) -> str:
    if "symbol" in frame.columns and not frame["symbol"].empty:
        return str(frame["symbol"].iloc[0])
    raise ValueError("Each dataframe must provide a non-empty 'symbol' column or be passed with an explicit symbol.")


def merge_dataframes(df_list, return_symbols: bool = False):
    """Merge per-symbol indicator frames on shared timestamps using a deterministic asset order."""
    if not df_list:
        empty = pd.DataFrame()
        return (empty, []) if return_symbols else empty

    frames_by_symbol: dict[str, pd.DataFrame] = {}
    for item in df_list:
        if isinstance(item, tuple):
            symbol, frame = item
        else:
            frame = item
            symbol = _extract_symbol(frame)
        frames_by_symbol[str(symbol)] = frame.copy()

    symbols = sorted(frames_by_symbol)
    prefixed_frames = []
    for symbol in symbols:
        frame = frames_by_symbol[symbol]
        if frame.index.name != "timestamp":
            if "timestamp" not in frame.columns:
                raise ValueError(f"Frame for {symbol} must have a 'timestamp' index or column.")
            frame = frame.set_index("timestamp")

        frame = frame.sort_index()
        frame = frame.drop(columns=["symbol"], errors="ignore")
        frame.columns = [f"{symbol}_{column}" for column in frame.columns]
        prefixed_frames.append(frame)

    merged_df = pd.concat(prefixed_frames, axis=1, join="inner").sort_index()
    merged_df = merged_df.dropna().copy()
    merged_df.index.name = "timestamp"

    print(f"Merged data shape: {merged_df.shape}")
    return (merged_df, symbols) if return_symbols else merged_df


def preprocess_and_save_data(
    config_path: str = "config/data_config.yaml",
    base_dir: str | None = None,
):
    base_dir = project_base_dir(__file__, base_dir)
    config = load_yaml_config(config_path, base_dir)
    paths = config["paths"]
    indicator_configs = config["indicators"]

    tickers_path = os.path.join(resolve_path(base_dir, paths["raw"]), "tickers")
    if not os.path.isdir(tickers_path):
        raise FileNotFoundError(f"Ticker directory not found: {tickers_path}")

    frames = []
    for file_name in sorted(os.listdir(tickers_path)):
        if not file_name.endswith(".csv"):
            continue

        symbol = os.path.splitext(file_name)[0]
        frame = pd.read_csv(
            os.path.join(tickers_path, file_name),
            index_col="timestamp",
            parse_dates=True,
        )
        frame = add_indicators(frame, indicator_configs)
        frames.append((symbol, frame))

    merged_df, asset_symbols = merge_dataframes(frames, return_symbols=True)
    if merged_df.empty:
        raise ValueError("Merged dataframe is empty after preprocessing.")

    merged_save_path = os.path.join(resolve_path(base_dir, paths["raw"]), "merged_data.csv")
    os.makedirs(os.path.dirname(merged_save_path), exist_ok=True)
    merged_df.to_csv(merged_save_path)

    config["assets"] = asset_symbols
    config["n_assets"] = len(asset_symbols)
    dump_yaml_config(config_path, config, base_dir)

    return merged_df, asset_symbols


if __name__ == "__main__":
    preprocess_and_save_data()
