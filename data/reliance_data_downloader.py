"""
Reliance NSE Historical Data Downloader using Fyers API
Downloads 10 years of daily OHLCV data for Reliance (NSE:RELIANCE-EQ)
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import pytz
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.fyers_session_management import get_fyers_session
from utilities.date_utilities import (
    get_current_epoch_timestamp, 
    get_n_years_ago_epoch_timestamp,
    epoch_to_datetime_ist
)


class RelianceDataDownloader:
    def __init__(self, data_dir="data/raw"):
        """
        Initialize the Reliance data downloader
        
        Args:
            data_dir (str): Directory to save the downloaded data
        """
        self.symbol = "NSE:RELIANCE-EQ"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fyers = None
        
    def _get_fyers_session(self):
        """Get Fyers session with error handling"""
        try:
            self.fyers = get_fyers_session()
            return True
        except Exception as e:
            print(f"Error getting Fyers session: {e}")
            return False
    
    def fetch_historical_data(self, years_back=10, resolution="D"):
        """
        Fetch historical data for Reliance from Fyers API
        Due to API limitations, fetches data in chunks of 300 days for daily data
        
        Args:
            years_back (int): Number of years of historical data to fetch
            resolution (str): Data resolution - "D" for daily, "60" for hourly, etc.
        
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        if not self._get_fyers_session():
            return None
            
        # For daily data, API limit is 366 days per request
        # We'll use 300 days to be safe
        max_days_per_request = 300 if resolution == "D" else 30
        
        # Calculate total date range
        range_to = get_current_epoch_timestamp()
        range_from = get_n_years_ago_epoch_timestamp(years_back)
        
        print(f"Fetching {years_back} years of {resolution} data for {self.symbol}")
        print(f"Total date range: {epoch_to_datetime_ist(range_from)} to {epoch_to_datetime_ist(range_to)}")
        print(f"Will fetch in chunks of {max_days_per_request} days due to API limitations")
        
        all_data = []
        current_end = range_to
        
        # Calculate number of chunks needed
        total_days = (range_to - range_from) // (24 * 3600)  # Convert seconds to days
        chunks_needed = int(np.ceil(total_days / max_days_per_request))
        
        print(f"Total days to fetch: {total_days}, Chunks needed: {chunks_needed}")
        
        for chunk in range(chunks_needed):
            # Calculate chunk start time (going backwards from current_end)
            chunk_start = current_end - (max_days_per_request * 24 * 3600)  # 300 days in seconds
            
            # Make sure we don't go before our target start date
            if chunk_start < range_from:
                chunk_start = range_from
            
            print(f"Fetching chunk {chunk + 1}/{chunks_needed}: "
                  f"{epoch_to_datetime_ist(chunk_start)} to {epoch_to_datetime_ist(current_end)}")
            
            # Prepare API request for this chunk
            data = {
                "symbol": self.symbol,
                "resolution": resolution,
                "date_format": "0",
                "range_from": chunk_start,
                "range_to": current_end,
                "cont_flag": "1"
            }
            
            try:
                response = self.fyers.history(data)
                
                # Check for errors in the response
                if response.get('s') != 'ok':
                    print(f"Error fetching chunk {chunk + 1}: {response.get('message', 'Unknown error')}")
                    continue
                
                # Parse response into DataFrame
                candles = response.get('candles', [])
                if candles:
                    chunk_df = pd.DataFrame(candles, columns=[
                        "timestamp", "open", "high", "low", "close", "volume"
                    ])
                    all_data.append(chunk_df)
                    print(f"  → Fetched {len(candles)} records")
                else:
                    print(f"  → No data for this chunk")
                
                # Move to next chunk (going backwards in time)
                current_end = chunk_start - 1  # Move 1 second back to avoid overlap
                
                # If we've reached the start date, break
                if current_end <= range_from:
                    break
                    
            except Exception as e:
                print(f"Error fetching chunk {chunk + 1}: {e}")
                continue
        
        if not all_data:
            print("No data retrieved from any chunks")
            return None
        
        # Combine all chunks
        print("Combining all chunks...")
        df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates based on timestamp
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Convert timestamp to IST datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        ist = pytz.timezone('Asia/Kolkata')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(ist)
        
        # Set timestamp as index and rename columns to standard format
        df.set_index('timestamp', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Sort by date (oldest first)
        df = df.sort_index()
        
        print(f"Successfully fetched {len(df)} total records")
        print(f"Final date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def save_data(self, df, filename=None):
        """
        Save DataFrame to CSV file
        
        Args:
            df (pd.DataFrame): Data to save
            filename (str): Optional filename, auto-generated if None
        
        Returns:
            str: Path to saved file
        """
        if df is None or df.empty:
            print("No data to save")
            return None
            
        if filename is None:
            start_date = df.index[0].strftime('%Y%m%d')
            end_date = df.index[-1].strftime('%Y%m%d')
            filename = f"RELIANCE_NSE_{start_date}_to_{end_date}.csv"
        
        filepath = self.data_dir / filename
        
        try:
            df.to_csv(filepath)
            print(f"Data saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"Error saving data: {e}")
            return None
    
    def load_existing_data(self, filename=None):
        """
        Load existing data from CSV file
        
        Args:
            filename (str): Optional filename, will search for latest if None
        
        Returns:
            pd.DataFrame: Loaded data
        """
        if filename is None:
            # Find the most recent Reliance data file
            csv_files = list(self.data_dir.glob("RELIANCE_NSE_*.csv"))
            if not csv_files:
                print("No existing data files found")
                return None
            filename = max(csv_files, key=os.path.getctime).name
        
        filepath = self.data_dir / filename
        
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Loaded data from: {filepath}")
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def get_data_summary(self, df):
        """
        Print summary statistics of the data
        
        Args:
            df (pd.DataFrame): Data to summarize
        """
        if df is None or df.empty:
            print("No data to summarize")
            return
            
        print("\n" + "="*50)
        print("RELIANCE NSE DATA SUMMARY")
        print("="*50)
        print(f"Symbol: {self.symbol}")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print("\nBasic Statistics:")
        print(df.describe())
        
        print("\nPrice Information:")
        print(f"Highest Close: ₹{df['Close'].max():.2f} on {df['Close'].idxmax().strftime('%Y-%m-%d')}")
        print(f"Lowest Close: ₹{df['Close'].min():.2f} on {df['Close'].idxmin().strftime('%Y-%m-%d')}")
        print(f"Latest Close: ₹{df['Close'].iloc[-1]:.2f}")
        print(f"Average Daily Volume: {df['Volume'].mean():,.0f}")
        
        # Calculate some basic technical indicators
        df_copy = df.copy()
        df_copy['Daily_Return'] = df_copy['Close'].pct_change()
        df_copy['Volatility_20d'] = df_copy['Daily_Return'].rolling(20).std() * np.sqrt(252)
        
        print(f"\nRisk Metrics:")
        print(f"Average Daily Return: {df_copy['Daily_Return'].mean()*100:.4f}%")
        print(f"Daily Volatility: {df_copy['Daily_Return'].std()*100:.4f}%")
        print(f"Current 20-day Volatility: {df_copy['Volatility_20d'].iloc[-1]*100:.2f}%")
        print("="*50)
    
    def download_and_save_reliance_data(self, years_back=10):
        """
        Complete workflow: Download, save, and summarize Reliance data
        
        Args:
            years_back (int): Number of years of data to download
        
        Returns:
            tuple: (DataFrame, filepath)
        """
        print("Starting Reliance NSE data download...")
        
        # Download data
        df = self.fetch_historical_data(years_back=years_back)
        if df is None:
            return None, None
        
        # Save data
        filepath = self.save_data(df)
        
        # Show summary
        self.get_data_summary(df)
        
        return df, filepath


def main():
    """Main function to demonstrate usage"""
    downloader = RelianceDataDownloader()
    
    # Download 10 years of data
    df, filepath = downloader.download_and_save_reliance_data(years_back=10)
    
    if df is not None:
        print(f"\nData successfully downloaded and saved!")
        print(f"File: {filepath}")
        print(f"Shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nLast few rows:")
        print(df.tail())
        
        return df
    else:
        print("Failed to download data. Please check your Fyers API configuration.")
        return None


if __name__ == "__main__":
    main()
