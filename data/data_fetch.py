
import pandas as pd
import pytz

from api.fyers_session_management import get_fyers_session


# Set display options to print all columns
pd.set_option('display.max_columns', None)



def fetch_historical_data(symbol, interval, range_from, range_to):
    fyers = get_fyers_session()
    # Prepare request data
    data = {
        "symbol": symbol,
        "resolution": interval,
        "date_format": "0",
        "range_from": range_from,  # Unix timestamp for the start date
        "range_to": range_to,      # Unix timestamp for the end date
        "cont_flag": "1"
    }
    response = fyers.history(data)
    # print(f'raw data for {symbol}, time interval {interval}: {response}')

    # Check for errors in the response
    if response.get('s') != 'ok':
        print(f"Error fetching data for {symbol}: {
              response.get('message', 'Unknown error')}")
        print(f'response: {response}')
        return None

    # Parse response into a DataFrame
    candles = response.get('candles', [])
    df = pd.DataFrame(candles, columns=[
                      "timestamp", "open", "high", "low", "close", "volume"])

    # Convert timestamp to datetime for readability
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    ist = pytz.timezone('Asia/Kolkata')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(ist)
    df.set_index('timestamp', inplace=True)

    # print first row of the data, open, high, low, close, volume in different rows
    # print(f'first row, high: {df.iloc[0]["high"]}, low: {df.iloc[0]["low"]}, close: {
    #       df.iloc[0]["close"]}, volume: {df.iloc[0]["volume"]}')

    # print(f"Data for {symbol} fetched successfully.")
    # print(f'data: {df}')
    return df


def fetch_historical_raw_data(symbol, interval, range_from, range_to):
    fyers = get_fyers_session()
    # Prepare request data
    data = {
        "symbol": symbol,
        "resolution": interval,
        "date_format": "0",
        "range_from": range_from,  # Unix timestamp for the start date
        "range_to": range_to,      # Unix timestamp for the end date
        "cont_flag": "1"
    }
    response = fyers.history(data)
    # print(f'raw data for {symbol}, time interval {interval}: {response}')

    # Check for errors in the response
    if response.get('s') != 'ok':
        print(f"Error fetching data for {symbol}: {
              response.get('message', 'Unknown error')}")
        print(f'response: {response}')
        return None

    return response
    # Parse response into a DataFrame
    # candles = response.get('candles', [])
    # df = pd.DataFrame(candles, columns=[
    #                   "timestamp", "open", "high", "low", "close", "volume"])

    # Convert timestamp to datetime for readability
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    # ist = pytz.timezone('Asia/Kolkata')
    # df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(ist)
    # df.set_index('timestamp', inplace=True)

    # print first row of the data, open, high, low, close, volume in different rows
    # print(f'first row, high: {df.iloc[0]["high"]}, low: {df.iloc[0]["low"]}, close: {
    #       df.iloc[0]["close"]}, volume: {df.iloc[0]["volume"]}')

    # print(f"Data for {symbol} fetched successfully.")
    # print(f'data: {df}')
    return df