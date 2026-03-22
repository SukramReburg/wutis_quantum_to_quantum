from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

try:
    from .common import dump_yaml_config, load_yaml_config, project_base_dir, resolve_path
except ImportError:
    from common import dump_yaml_config, load_yaml_config, project_base_dir, resolve_path


def compute_fetch_end_datetime(now: datetime | None = None) -> datetime:
    reference = now or datetime.now()
    return reference - timedelta(days=1)


def load_alpaca_client(api_config_path: str, base_dir: str):
    from alpaca.data.historical import StockHistoricalDataClient

    config = load_yaml_config(api_config_path, base_dir)
    api_config = config["alpaca_api"]
    return StockHistoricalDataClient(api_config["api_key"], api_config["secret_key"])


def fetch_alpaca(
    client: Any,
    ticker: str,
    start_year: int,
    end_dt: datetime,
) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    start_dt = datetime(start_year, 1, 1, 0, 0, 0, 0)

    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
        adjustment=Adjustment.ALL,
        volume_adjustment=True,
    )

    try:
        return client.get_stock_bars(request_params).df
    except Exception as exc:
        print(f"Error fetching data for {ticker}: {exc}")
        return pd.DataFrame()


def replace_directory(source_dir: str, target_dir: str) -> None:
    backup_dir = None
    if os.path.isdir(target_dir):
        backup_dir = f"{target_dir}.backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        os.replace(target_dir, backup_dir)

    try:
        os.replace(source_dir, target_dir)
    except Exception:
        if backup_dir is not None and not os.path.exists(target_dir):
            os.replace(backup_dir, target_dir)
        raise
    else:
        if backup_dir is not None and os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)


def fetch_and_save_data(
    config_path: str = "config/data_config.yaml",
    api_config_path: str = "config/config.yaml",
    base_dir: str | None = None,
    fetch_func=fetch_alpaca,
    client_loader=load_alpaca_client,
    now: datetime | None = None,
):
    base_dir = project_base_dir(__file__, base_dir)
    config = load_yaml_config(config_path, base_dir)
    assets_path = resolve_path(base_dir, "source/assets.csv")
    assets_df = pd.read_csv(assets_path)

    start_year = config["start_year"]
    records_number_threshold = config["records_number_threshold"]
    paths = config["paths"]

    raw_dir = resolve_path(base_dir, paths["raw"])
    target_dir = os.path.join(raw_dir, "tickers")
    os.makedirs(raw_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="tickers_run_", dir=raw_dir)

    client = client_loader(api_config_path, base_dir)
    end_dt = compute_fetch_end_datetime(now)

    tickers = [str(ticker) for ticker in assets_df["ticker"].tolist()]
    ticker_success_count = 0
    successful_tickers: list[str] = []

    try:
        for ticker_count, ticker in enumerate(tickers, start=1):
            df = fetch_func(client, ticker, start_year, end_dt)
            if not df.empty and len(df) >= records_number_threshold:
                ticker_success_count += 1
                successful_tickers.append(ticker)
                df.to_csv(os.path.join(temp_dir, f"{ticker}.csv"))
                print(
                    f"Data for {ticker} fetched successfully. "
                    f"Number of records: {len(df)}. ({ticker_count}/{len(tickers)})"
                )
            else:
                print(f"Not enough data fetched for {ticker}. ({ticker_count}/{len(tickers)})")

        if ticker_success_count == 0:
            raise RuntimeError("No tickers met the records threshold; raw ticker directory not replaced.")

        replace_directory(temp_dir, target_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print("Data fetching completed successfully.")
    print(f"Count of tickers fetched: {ticker_success_count}")

    config["n_assets"] = ticker_success_count
    dump_yaml_config(config_path, config, base_dir)

    return successful_tickers


if __name__ == "__main__":
    fetch_and_save_data()
