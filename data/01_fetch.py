import yaml
import pandas as pd
from datetime import datetime
import os 

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def fetch_alpaca(ticker, start_year):

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    api_key = config['alpaca_api']['api_key']
    secret_key = config['alpaca_api']['secret_key']

    start_dt = datetime(start_year,1,1,0,0,0,0)

    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=datetime.now().replace(day=datetime.now().day-1)
    )

    client = StockHistoricalDataClient(api_key, secret_key)

    try:
        df = client.get_stock_bars(request_params).df
    except Exception as e:
        print(f"Error fetching data for {ticker}")
        df = pd.DataFrame()

    return df

def fetch_and_save_data():
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    assets_df = pd.read_csv('source/assets.csv')
    start_year = config['start_year']
    records_number_threshold = config['records_number_threshold']
    
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, paths['raw'], 'tickers')
    os.makedirs(path, exist_ok=True)
    tickers_num = len(assets_df)
    ticker_count = 0
    ticker_success_count = 0
    for ticker in list(assets_df['ticker']):

        ticker_count += 1
        df = fetch_alpaca(ticker,start_year)
        if not df.empty and len(df) >= records_number_threshold:
            ticker_success_count += 1
            df.to_csv(os.path.join(path, f"{ticker}.csv"))
            print(f"Data for {ticker} fetched successfully. Number of records: {len(df)}. ({ticker_count}/{tickers_num})")
            # print(f"Fetching data for {ticker}, and start year {start_year}...")
        else:
            print(f"Not enough data fetched for {ticker}. ({ticker_count}/{tickers_num})")

    print("Data fetching completed successfully.")
    print(f"Count of tickers fetched: {ticker_success_count}")


if __name__ == "__main__":
    fetch_and_save_data()
