
# fetch data and store in a cache
# read data from cache and return based on demand

import os
import json
import datetime
import pandas as pd
import sys

# Add paths for imports
sys.path.append('/Users/bogalaxminarayana/myGit/ai_stock_predictions')

from src.data.data_fetch import fetch_historical_raw_data
from utilities.date_utilities import get_current_epoch_timestamp, get_epoch_timestamp_from_datetime_ist_string
from src.data.symbols_management import filterNifty50Symbols, getAllIntradayNSEApprovedSymbols, getAllIntradayNSEApprovedSymbolsExceptNifty50, getAllIntradayNSEApprovedSymbolsExceptNifty500, getBlueChipsSymbos, getNifty200Symbols, getNifty500Symbols, getNifty500SymbolsExceptNifty50, getNifty50Symbols

from datetime import datetime, timezone, timedelta

oct_weekdays = ['2024-10-01', '2024-10-03', '2024-10-04', '2024-10-07', '2024-10-08', '2024-10-09', '2024-10-10', '2024-10-11', '2024-10-14',
                 '2024-10-15', '2024-10-16', '2024-10-17', '2024-10-18', '2024-10-21', '2024-10-22', '2024-10-23', '2024-10-24', '2024-10-25', '2024-10-28',
                   '2024-10-29', '2024-10-30', '2024-10-31']

nov_weekdays = ['2024-11-04', '2024-11-05', '2024-11-06', '2024-11-07', '2024-11-08', '2024-11-12', '2024-11-13', '2024-11-14',
                '2024-11-18', '2024-11-19', '2024-11-21', '2024-11-22', '2024-11-25', '2024-11-26', '2024-11-27', '2024-11-28', '2024-11-29']

dec_weekdays = ['2024-12-02', '2024-12-03', '2024-12-04', '2024-12-05', '2024-12-06', '2024-12-09', '2024-12-10', '2024-12-11', '2024-12-12', '2024-12-13',
                    '2024-12-16', '2024-12-17', '2024-12-18', '2024-12-19', '2024-12-20', '2024-12-23', '2024-12-24', '2024-12-26', '2024-12-27',
                    '2024-12-30', '2024-12-31']

available_days_2025 = ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10', '2025-01-13', '2025-01-14', '2025-01-15', '2025-01-16', '2025-01-17', '2025-01-20', '2025-01-21', '2025-01-22', '2025-01-23', '2025-01-24', '2025-01-27', '2025-01-28', '2025-01-29', '2025-01-30', '2025-02-03', '2025-02-04', '2025-02-05', '2025-02-06', '2025-02-07', '2025-02-10', '2025-02-11', '2025-02-12', '2025-02-13', '2025-02-14', '2025-02-17', '2025-02-18', '2025-02-19', '2025-02-20', '2025-02-21', '2025-02-24', '2025-02-25', '2025-02-27', '2025-02-28', '2025-03-03', '2025-03-04', '2025-03-05', '2025-03-06', '2025-03-07', '2025-03-10', '2025-03-11'] 

available_days_2023 = ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09", "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13", "2023-01-16", "2023-01-17", "2023-01-18", "2023-01-19", "2023-01-20", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-27", "2023-01-30", "2023-01-31", "2023-02-01", "2023-02-02", "2023-02-03", "2023-02-06", "2023-02-07", "2023-02-08", "2023-02-09", "2023-02-10", "2023-02-13", "2023-02-14", "2023-02-15", "2023-02-16", "2023-02-17", "2023-02-20", "2023-02-21", "2023-02-22", "2023-02-23", "2023-02-24", "2023-02-27", "2023-02-28", "2023-03-01", "2023-03-02", "2023-03-03", "2023-03-06", "2023-03-08", "2023-03-09", "2023-03-10", "2023-03-13", "2023-03-14", "2023-03-15", "2023-03-16", "2023-03-17", "2023-03-20", "2023-03-21", "2023-03-22", "2023-03-23", "2023-03-24", "2023-03-27", "2023-03-28", "2023-03-29", "2023-04-03", "2023-04-05", "2023-04-06", "2023-04-10", "2023-04-11", "2023-04-12", "2023-04-13", "2023-04-17", "2023-04-18", "2023-04-19", "2023-04-20", "2023-04-21", "2023-04-24", "2023-04-25", "2023-04-26", "2023-04-27", "2023-04-28", "2023-05-02", "2023-05-03", "2023-05-04", "2023-05-05", "2023-05-08", "2023-05-09", "2023-05-10", "2023-05-11", "2023-05-12", "2023-05-15", "2023-05-16", "2023-05-17", "2023-05-18", "2023-05-19", "2023-05-22", "2023-05-23", "2023-05-24", "2023-05-25", "2023-05-26", "2023-05-29", "2023-05-30", "2023-05-31", "2023-06-01", "2023-06-02", "2023-06-05", "2023-06-06", "2023-06-07", "2023-06-08", "2023-06-09", "2023-06-12", "2023-06-13", "2023-06-14", "2023-06-15", "2023-06-16", "2023-06-19", "2023-06-20", "2023-06-21", "2023-06-22", "2023-06-23", "2023-06-26", "2023-06-27", "2023-06-28", "2023-07-03", "2023-07-04", "2023-07-05", "2023-07-06", "2023-07-07", "2023-07-10", "2023-07-11", "2023-07-12", "2023-07-13", "2023-07-14", "2023-07-17", "2023-07-18", "2023-07-19", "2023-07-20", "2023-07-21", "2023-07-24", "2023-07-25", "2023-07-26", "2023-07-27", "2023-07-28", "2023-07-31", "2023-08-01", "2023-08-02", "2023-08-03", "2023-08-04", "2023-08-07", "2023-08-08", "2023-08-09", "2023-08-10", "2023-08-11", "2023-08-14", "2023-08-16", "2023-08-17", "2023-08-18", "2023-08-21", "2023-08-22", "2023-08-23", "2023-08-24", "2023-08-25", "2023-08-28", "2023-08-29", "2023-08-30", "2023-08-31", "2023-09-01", "2023-09-04", "2023-09-05", "2023-09-06", "2023-09-07", "2023-09-08", "2023-09-11", "2023-09-12", "2023-09-13", "2023-09-14", "2023-09-15", "2023-09-18", "2023-09-20", "2023-09-21", "2023-09-22", "2023-09-25", "2023-09-26", "2023-09-27", "2023-09-28", "2023-09-29", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06", "2023-10-09", "2023-10-10", "2023-10-11", "2023-10-12", "2023-10-13", "2023-10-16", "2023-10-17", "2023-10-18", "2023-10-19", "2023-10-20", "2023-10-23", "2023-10-25", "2023-10-26", "2023-10-27", "2023-10-30", "2023-10-31", "2023-11-01", "2023-11-02", "2023-11-03", "2023-11-06", "2023-11-07", "2023-11-08", "2023-11-09", "2023-11-10", "2023-11-12", "2023-11-13", "2023-11-15", "2023-11-16", "2023-11-17", "2023-11-20", "2023-11-21", "2023-11-22", "2023-11-23", "2023-11-24", "2023-11-28", "2023-11-29", "2023-11-30", "2023-12-01", "2023-12-04", "2023-12-05", "2023-12-06", "2023-12-07", "2023-12-08", "2023-12-11", "2023-12-12", "2023-12-13", "2023-12-14", "2023-12-15", "2023-12-18", "2023-12-19", "2023-12-20", "2023-12-21", "2023-12-22", "2023-12-26", "2023-12-27", "2023-12-28", "2023-12-29"]

available_days_2024 = [ "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08","2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-20", "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-19", "2024-02-20", "2024-02-21", "2024-02-22", "2024-02-23", "2024-02-26", "2024-02-27", "2024-02-28", "2024-02-29", "2024-03-01", "2024-03-02", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14", "2024-03-15", "2024-03-18", "2024-03-19", "2024-03-20", "2024-03-21", "2024-03-22", "2024-03-26", "2024-03-27", "2024-03-28", "2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04", "2024-04-05", "2024-04-08", "2024-04-09", "2024-04-10", "2024-04-12", "2024-04-15", "2024-04-16", "2024-04-18", "2024-04-19", "2024-04-22", "2024-04-23", "2024-04-24", "2024-04-25", "2024-04-26", "2024-04-29", "2024-04-30", "2024-05-02", "2024-05-03", "2024-05-06", "2024-05-07", "2024-05-08", "2024-05-09", "2024-05-10", "2024-05-13", "2024-05-14", "2024-05-15", "2024-05-16", "2024-05-17", "2024-05-18", "2024-05-21", "2024-05-22", "2024-05-23", "2024-05-24", "2024-05-27", "2024-05-28", "2024-05-29", "2024-05-30", "2024-05-31", "2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07", "2024-06-10", "2024-06-11", "2024-06-12", "2024-06-13", "2024-06-14", "2024-06-18", "2024-06-19", "2024-06-20", "2024-06-21", "2024-06-24", "2024-06-25", "2024-06-26", "2024-06-27", "2024-06-28", "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04", "2024-07-05", "2024-07-08", "2024-07-09", "2024-07-10", "2024-07-11", "2024-07-12", "2024-07-15", "2024-07-16", "2024-07-18", "2024-07-19", "2024-07-22", "2024-07-23", "2024-07-24", "2024-07-25", "2024-07-26", "2024-07-29", "2024-07-30", "2024-07-31", "2024-08-01", "2024-08-02", "2024-08-05", "2024-08-06", "2024-08-07", "2024-08-08", "2024-08-09", "2024-08-12", "2024-08-13", "2024-08-14", "2024-08-16", "2024-08-19", "2024-08-20", "2024-08-21", "2024-08-22", "2024-08-23", "2024-08-26", "2024-08-27", "2024-08-28", "2024-08-29", "2024-08-30", "2024-09-02", "2024-09-03", "2024-09-04", "2024-09-05", "2024-09-06", "2024-09-09", "2024-09-10", "2024-09-11", "2024-09-12", "2024-09-13", "2024-09-16", "2024-09-17", "2024-09-18", "2024-09-19", "2024-09-20", "2024-09-23", "2024-09-24", "2024-09-25", "2024-09-26", "2024-09-27", "2024-10-01", "2024-10-03", "2024-10-04", "2024-10-07", "2024-10-08", "2024-10-09", "2024-10-10", "2024-10-11", "2024-10-14", "2024-10-15", "2024-10-16", "2024-10-17", "2024-10-18", "2024-10-21", "2024-10-22", "2024-10-23", "2024-10-24", "2024-10-25", "2024-10-28", "2024-10-29", "2024-10-30", "2024-10-31", "2024-11-01", "2024-11-04", "2024-11-05", "2024-11-06", "2024-11-07", "2024-11-08", "2024-11-11", "2024-11-12", "2024-11-13", "2024-11-14", "2024-11-18", "2024-11-19", "2024-11-21", "2024-11-22", "2024-11-25", "2024-11-26", "2024-11-27", "2024-11-28", "2024-11-29", "2024-12-02", "2024-12-03", "2024-12-04", "2024-12-05", "2024-12-06", "2024-12-09", "2024-12-10", "2024-12-11", "2024-12-12", "2024-12-13", "2024-12-16", "2024-12-17", "2024-12-18", "2024-12-19", "2024-12-20", "2024-12-23", "2024-12-24", "2024-12-26", "2024-12-27", "2024-12-30"]

# combine 2023, 2024 and 2025 data
available_days_all = available_days_2023 + available_days_2024 + available_days_2025


def get_all_available_days():
    return available_days_all[10:]

def get_available_days_for_2023():
    return available_days_2023[10:]

def get_available_days_for_2024():
    return available_days_2024

def get_available_days_for_2025():
    return available_days_2025


def get_start_end_timestamps(date_str):
    # Parse the input date string
    date = datetime.strptime(date_str, "%Y-%m-%d")

    # Define the start and end times in GMT
    start_time = date.replace(
        hour=3, minute=30, second=0, microsecond=0, tzinfo=timezone.utc)
    end_time = date.replace(hour=10, minute=0, second=0,
                            microsecond=0, tzinfo=timezone.utc)

    # Convert to Unix timestamps
    start_timestamp = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())

    return start_timestamp, end_timestamp


def save_single_day_cache_historical_data(symbol, date_str, interval):
    cache_dir = f"cache/{symbol}/interval_{interval}/"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}day_{date_str}.json"

    # derive start and end date from the day given. ex: 2024-12-01. start_date should be 3.30 am GMT in unix timestamp and end_date should be 10.00 am GMT of the same day.
    start_date, end_date = get_start_end_timestamps(date_str)
    data = fetch_historical_raw_data(symbol, interval, start_date, end_date)

    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to cache for {symbol}, {date_str}, {interval}")

def save_historical_data_2024Jul01_to_2025Jan18(symbol, interval):
    start_date = '2024-07-01'
    start_time = get_epoch_timestamp_from_datetime_ist_string(f'{start_date} 09:00:00')
    end_date = '2025-01-18'
    end_time = get_epoch_timestamp_from_datetime_ist_string(f'{end_date} 09:00:00')

    cache_dir = f"cache_raw_data_all_intraday/{symbol}/interval_{interval}/"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}day_{start_date}_to_{end_date}.json"

    # if cache file already exists, skip
    if os.path.exists(cache_file):
        print(f"Data already exists for {symbol}, {start_date}, {end_date}, {interval}")
        return

    data = fetch_historical_raw_data(symbol, interval, start_time, end_time)

    with open(cache_file, 'w') as f:
        # json.dump(data, f, indent=4)
        json.dump(data, f)
    print(f"Data saved to cache for {symbol}, {start_date}, {end_date}, {interval}")


def save_validation_period_data(symbol, interval):
    """Download 10-minute data for AI validation period: Aug 2023 to Aug 2025"""
    dates = [
        # 2023 Q3-Q4
        {"start_date": "2023-08-01", "end_date": "2023-09-30"},
        {"start_date": "2023-10-01", "end_date": "2023-12-31"},
        # 2024 Q1-Q4
        {"start_date": "2024-01-01", "end_date": "2024-03-31"},
        {"start_date": "2024-04-01", "end_date": "2024-06-30"},
        {"start_date": "2024-07-01", "end_date": "2024-09-30"},
        {"start_date": "2024-10-01", "end_date": "2024-12-31"},
        # 2025 Q1-Q3
        {"start_date": "2025-01-01", "end_date": "2025-03-31"},
        {"start_date": "2025-04-01", "end_date": "2025-06-30"},
        {"start_date": "2025-07-01", "end_date": "2025-08-31"}
    ]

    for date in dates:
        start_date = date["start_date"]
        end_date = date["end_date"]
        start_time = get_epoch_timestamp_from_datetime_ist_string(f'{start_date} 09:00:00')
        end_time = get_epoch_timestamp_from_datetime_ist_string(f'{end_date} 21:00:00')

        cache_dir = f"cache_raw_data_all_intraday/{symbol}/interval_{interval}/"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}day_{start_date}_to_{end_date}.json"

        # if cache file already exists, skip
        if os.path.exists(cache_file):
            print(f"Data already exists for {symbol}, {start_date}, {end_date}, {interval}")
            continue

        data = fetch_historical_raw_data(symbol, interval, start_time, end_time)

        with open(cache_file, 'w') as f:
            json.dump(data, f)
        print(f"Data saved to cache for {symbol}, {start_date}, {end_date}, {interval}")


def save_2025_histotical_data(symbol, interval):
    dates = [
        {"start_date": "2025-03-11", "end_date": "2025-03-11"}
    ]

    for date in dates:
        start_date = date["start_date"]
        end_date = date["end_date"]
        start_time = get_epoch_timestamp_from_datetime_ist_string(f'{start_date} 09:00:00')
        end_time = get_epoch_timestamp_from_datetime_ist_string(f'{end_date} 21:00:00')

        cache_dir = f"cache_raw_data_all_intraday/{symbol}/interval_{interval}/"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}day_{start_date}_to_{end_date}.json"

        # if cache file already exists, skip
        if os.path.exists(cache_file):
            print(f"Data already exists for {symbol}, {start_date}, {end_date}, {interval}")
            continue

        data = fetch_historical_raw_data(symbol, interval, start_time, end_time)

        with open(cache_file, 'w') as f:
            # json.dump(data, f, indent=4)
            json.dump(data, f)
        print(f"Data saved to cache for {symbol}, {start_date}, {end_date}, {interval}")


def save_2024_histotical_data(symbol, interval):
    dates = [
        {"start_date": "2024-01-01", "end_date": "2024-03-31"},
        {"start_date": "2024-04-01", "end_date": "2024-06-30"},
        {"start_date": "2024-07-01", "end_date": "2024-09-30"},
        {"start_date": "2024-10-01", "end_date": "2024-12-31"}
    ]

    for date in dates:
        start_date = date["start_date"]
        end_date = date["end_date"]
        start_time = get_epoch_timestamp_from_datetime_ist_string(f'{start_date} 09:00:00')
        end_time = get_epoch_timestamp_from_datetime_ist_string(f'{end_date} 09:00:00')

        cache_dir = f"cache_raw_data_all_intraday/{symbol}/interval_{interval}/"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}day_{start_date}_to_{end_date}.json"

        # if cache file already exists, skip
        if os.path.exists(cache_file):
            print(f"Data already exists for {symbol}, {start_date}, {end_date}, {interval}")
            continue

        data = fetch_historical_raw_data(symbol, interval, start_time, end_time)

        with open(cache_file, 'w') as f:
            # json.dump(data, f, indent=4)
            json.dump(data, f)
        print(f"Data saved to cache for {symbol}, {start_date}, {end_date}, {interval}")


def save_2023_histotical_data(symbol, interval):
    dates = [
        {"start_date": "2023-01-01", "end_date": "2023-03-31"},
        {"start_date": "2023-04-01", "end_date": "2023-06-30"},
        {"start_date": "2023-07-01", "end_date": "2023-09-30"},
        {"start_date": "2023-10-01", "end_date": "2023-12-31"}
    ]

    for date in dates:
        start_date = date["start_date"]
        end_date = date["end_date"]
        start_time = get_epoch_timestamp_from_datetime_ist_string(f'{start_date} 09:00:00')
        end_time = get_epoch_timestamp_from_datetime_ist_string(f'{end_date} 09:00:00')

        cache_dir = f"cache_raw_data_all_intraday/{symbol}/interval_{interval}/"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}day_{start_date}_to_{end_date}.json"

        # if cache file already exists, skip
        if os.path.exists(cache_file):
            print(f"Data already exists for {symbol}, {start_date}, {end_date}, {interval}")
            continue

        data = fetch_historical_raw_data(symbol, interval, start_time, end_time)

        with open(cache_file, 'w') as f:
            # json.dump(data, f, indent=4)
            json.dump(data, f)
        print(f"Data saved to cache for {symbol}, {start_date}, {end_date}, {interval}")


def save_2022_2021_2020_histotical_data(symbol, interval):
    dates = [       
        {"start_date": "2020-01-01", "end_date": "2020-03-31"},
        {"start_date": "2020-04-01", "end_date": "2020-06-30"},
        {"start_date": "2020-07-01", "end_date": "2020-09-30"},
        {"start_date": "2020-10-01", "end_date": "2020-12-31"},
        {"start_date": "2021-01-01", "end_date": "2021-03-31"},
        {"start_date": "2021-04-01", "end_date": "2021-06-30"},
        {"start_date": "2021-07-01", "end_date": "2021-09-30"},
        {"start_date": "2021-10-01", "end_date": "2021-12-31"},
        {"start_date": "2022-01-01", "end_date": "2022-03-31"},
        {"start_date": "2022-04-01", "end_date": "2022-06-30"},
        {"start_date": "2022-07-01", "end_date": "2022-09-30"},
        {"start_date": "2022-10-01", "end_date": "2022-12-31"}
    ]

    for date in dates:
        start_date = date["start_date"]
        end_date = date["end_date"]
        start_time = get_epoch_timestamp_from_datetime_ist_string(f'{start_date} 09:00:00')
        end_time = get_epoch_timestamp_from_datetime_ist_string(f'{end_date} 09:00:00')

        cache_dir = f"cache_raw_data_all_intraday/{symbol}/interval_{interval}/"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}day_{start_date}_to_{end_date}.json"

        # if cache file already exists, skip
        if os.path.exists(cache_file):
            print(f"Data already exists for {symbol}, {start_date}, {end_date}, {interval}")
            continue

        data = fetch_historical_raw_data(symbol, interval, start_time, end_time)

        with open(cache_file, 'w') as f:
            # json.dump(data, f, indent=4)
            json.dump(data, f)
        print(f"Data saved to cache for {symbol}, {start_date}, {end_date}, {interval}")

# save_2024_histotical_data("NSE:RELIANCE-EQ", "5")
# save_single_day_cache_historical_data("NSE:ASIANPAINT-EQ","2024-12-02", "5S")

def save_dec_5sec_historical_data(days):
    symbols = getBlueChipsSymbos()
    for symbol in symbols:
        for date_str in days:
            save_single_day_cache_historical_data(symbol, date_str, "5S")

    print("---END---")


def save_dec_5min_historical_data(days):
    symbols = getBlueChipsSymbos()
    for symbol in symbols:
        for date_str in days:
            save_single_day_cache_historical_data(symbol, date_str, "5")

    print("---END---")

def save_dec_10min_historical_data(days):
    symbols = getBlueChipsSymbos()
    for symbol in symbols:
        for date_str in days:
            save_single_day_cache_historical_data(symbol, date_str, "10")

    print("---END---")


def save_dec_15min_historical_data(days):
    symbols = getBlueChipsSymbos()
    for symbol in symbols:
        for date_str in days:
            save_single_day_cache_historical_data(symbol, date_str, "15")

    print("---END---")


def save_all_data():
    # symbols nifty50 - bluchips
    # days - oct_weekdays, nov_weekdays, dec_weekdays, jan_weekdays
    # resolutions = ["5S", "5", "10", "15"]
    # resolutions = ["1"]
    resolutions = ["1", "5", "10", "15"]
    symbols = getNifty500Symbols()
    # symbols = getBlueChipsSymbos()
    # days = oct_weekdays + nov_weekdays + dec_weekdays + jan_weekdays
    # print(f"Total symbols: {len(symbols)}")
    # print(f"Total days: {len(days)}")
    # print(f"Total resolutions: {len(resolutions)}")
    
    print("===== Saving data for save_2025_histotical_data")
    for symbol in symbols:
        for resolution in resolutions:
            save_2025_histotical_data(symbol, resolution)
    print("===== Data saved for save_2025_histotical_data")
    
    
# save_all_data()

def get_cached_data(symbol, date_str, interval):
    cache_dir = f"cache2/{symbol}/interval_{interval}/"
    cache_file = f"{cache_dir}day_{date_str}.json"

    with open(cache_file, 'r') as f:
        data = json.load(f)
    return data

# print(get_cached_data("NSE:RELIANCE-EQ","2024-12-02", "5S"))


def get_cached_data_in_dataframe(symbol, interval, date_str):
    data = get_cached_data(symbol, date_str, interval)
    if "candles" in data:
        df = pd.DataFrame(data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime_uk"] = pd.to_datetime(df["timestamp"], unit="s")  # Convert to datetime
        return df
    else:
        raise ValueError("Failed to fetch data. Check your API or inputs. response: ", data)
    
def get_end_of_day_price(symbol, date_str):
    data = get_cached_data_in_dataframe(symbol, "1", date_str)
    return data.iloc[-1]["close"], data.iloc[-1]["timestamp"]

def get_close_price_for_given_timestamp(symbol, date_str, timestamp):
    data = get_cached_data_in_dataframe(symbol, "1", date_str)
    closePrice = data[data["timestamp"] == timestamp]["close"].values[0]
    return closePrice

def get_1min_candle_data_for_given_timestamp(symbol, date_str, timestamp):
    data = get_cached_data_in_dataframe(symbol, "1", date_str)
    candle = data[data["timestamp"] == timestamp]
    # // convert candle data to flat data with open, high, low, close, volume
    flat_data = {"open": candle["open"].values[0], "high": candle["high"].values[0], "low": candle["low"].values[0], "close": candle["close"].values[0], "volume": candle["volume"].values[0]}
    return flat_data

# print(get_end_of_day_price("NSE:RELIANCE-EQ", "2024-12-05"))
# print(get_1min_candle_data_for_given_timestamp("NSE:ATGL-EQ", "2024-05-31", 1717145880))

def calculate_n_days_date(date_str, n):
    """
    Calculate n days before or after a given date in the `dec_weekdays` list.

    Args:
        date_str (str): Target date as a string in 'YYYY-MM-DD' format.
        n (int): Number of days to go forward (if positive) or backward (if negative).

    Returns:
        str: The date n days from the target date.

    Raises:
        ValueError: If the date is not in the list or the calculated index is out of range.
    """
    if date_str not in available_days_all:
        print(f"The date {date_str} is not in the list of weekdays.")
        raise ValueError(f"The date {date_str} is not in the list of weekdays.")

    index = available_days_all.index(date_str)
    target_index = index + n

    if target_index < 0 or target_index >= len(available_days_all):
        raise ValueError(f"Cannot calculate {n} days from {date_str}, as it exceeds the available range.")

    return available_days_all[target_index]

# print(calculate_n_days_date("2024-12-12", 13))

def get_cached_data_in_dataframe_for_date_range(symbol, interval, start_date_str, n_days):
    """
    Fetch cached data for a given symbol and interval over a range of dates.

    Args:
        symbol (str): Stock symbol.
        interval (str): Data interval (e.g., '5min', '10min').
        start_date_str (str): Start date as a string in 'YYYY-MM-DD' format.
        n_days (int): Number of days to fetch data. Positive for future days, negative for past days.

    Returns:
        DataFrame: Combined DataFrame with data for the specified date range.

    Raises:
        ValueError: If an error occurs during date range calculation or data fetching.
    """
    date_list = []
    step = 1 if n_days > 0 else -1

    # Generate the date range using calculate_n_days_date
    for i in range(abs(n_days) + 1):  # Include the start_date_str itself
        date_str = calculate_n_days_date(start_date_str, i * step)
        date_list.append(date_str)

    dataframes = []
    for date_str in date_list:
        data = get_cached_data_in_dataframe(symbol, interval, date_str)
        if data is not None and not data.empty:
            dataframes.append(data)
        else:
            print(f"No data found for {date_str}.")

    if not dataframes:
        raise ValueError("No data available for the specified date range.")

    # Combine the dataframes. sort by timestamp small to large
    combined_df = pd.concat(dataframes).sort_values("timestamp").reset_index(drop=True)
    return combined_df
    

# data = get_cached_data_in_dataframe_for_date_range("NSE:RELIANCE-EQ", "10", "2024-08-12", -5)
# print(data)

def print_available_file_name():
    available_days = []
    for root, dirs, files in os.walk("cache2/NSE:ADANIENT-EQ/interval_5S/"):
        for file in files:
            # print the file name only if it's a json file
            # if file.endswith(".json"):
            #     print(file)

            # remove day_ and .json from the file name and print in a list format
            if file.endswith(".json"):
                available_days.append(file[4:-5])
    # order teh list in ascending order of date
    available_days.sort()
    print(available_days)

# print_available_file_name()