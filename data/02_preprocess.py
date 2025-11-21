import pandas as pd 
import os 
import yaml
import numpy as np

from indicators import *

from sklearn.preprocessing import StandardScaler
import joblib


def add_indicators(data, indicator_configs):
    """Add technical indicators to the dataframe based on the configuration."""
    for indicator_cfg in indicator_configs:
        name = indicator_cfg['name']
        params = indicator_cfg.get('params', {})
        if name in indicators_impl:
            indicator_class = indicators_impl[name]
            indicator = indicator_class(data, **params, name=name)
            data = indicator.add_indicator()
        else:
            print(f"Indicator '{name}' not recognized.")
    # Drop raw OHLCV 
    data.drop(
        columns=['high', 'low', 'volume', 'open','close','trade_count', 'vwap'],
        inplace=True,
        errors="ignore"
    )
    return data


def merge_dataframes(df_list):
    """Merge a list of dataframes on their index (date) using an inner join."""
    if not df_list:
        return pd.DataFrame()
    
    merged_df = pd.DataFrame()
    for df in df_list:
        # if 'symbol' column exists, use it for prefix
        if 'symbol' in df.columns:
            symbol = df['symbol'].iloc[0]
        else:
            symbol = "UNK"

        # make sure timestamp is a column for the merge
        if df.index.name == 'timestamp':
            df = df.reset_index()
        
        df.columns = [
            f"{symbol}_{col}" if col not in ['timestamp'] else 'timestamp'
            for col in df.columns
        ]

        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='timestamp', how='outer')

    merged_df.sort_values('timestamp', inplace=True)
    merged_df.ffill(inplace=True)
    merged_df.dropna(inplace=True)  

    # Drop any remaining string columns (e.g. symbol)
    string_columns = merged_df.select_dtypes(include=['object']).columns
    merged_df.drop(columns=string_columns, inplace=True)

    # Set timestamp back as index
    merged_df.set_index('timestamp', inplace=True)

    print(f"Merged data shape: {merged_df.shape}")
    return merged_df


if __name__ == "__main__":
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    indicator_configs = config['indicators']
    train_size = config['train_size']   # e.g. 0.8
    n_assets = config.get('n_assets', 10)
    cov_window = config.get('cov_window', 20)
    lookback_window = config.get('lookback_window', 20)
    use_past_cov_in_features = config.get('use_past_cov_in_features', True)
    use_past_ret_in_features = config.get('use_past_ret_in_features', True)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tickers_path = os.path.join(base_dir, paths['raw'], 'tickers')

    df_list = []
    for file in os.listdir(tickers_path):
        if file.endswith('.csv'):
            df = pd.read_csv(
                os.path.join(tickers_path, file),
                index_col='timestamp',
                parse_dates=True
            )
            df = add_indicators(df, indicator_configs)
            df_list.append(df)

    merged_df = merge_dataframes(df_list)
    merged_save_path = os.path.join(base_dir, paths['raw'], 'merged_data.csv')
    os.makedirs(os.path.dirname(merged_save_path), exist_ok=True)
    merged_df.to_csv(merged_save_path)

    