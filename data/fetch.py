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

if __name__ == "__main__":

    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    assets_df = pd.read_csv('source/assets.csv')
    start_year = config['start_year']
    
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, paths['raw'])
    os.makedirs(path, exist_ok=True)
    print(assets_df)
    print(list(assets_df['ticker']))
    for ticker in list(assets_df['ticker']):
        df = fetch_alpaca(ticker,start_year)
        if not df.empty:
            df.to_csv(os.path.join(path, f"{ticker}.csv"))
            # print(f"Fetching data for {ticker}, and start year {start_year}...")
        else:
            print(f"No data fetched for {ticker}.")
