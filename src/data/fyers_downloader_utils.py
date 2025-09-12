"""
Professional Intraday Data Downloader using Fyers API
Downloads 10-minute candle data for realistic backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append('/Users/bogalaxminarayana/myGit/ai_stock_predictions/src')

# Import Fyers if available, otherwise use fallback
try:
    from api.fyers_data_fetcher import FyersDataFetcher
    FYERS_AVAILABLE = True
except ImportError:
    FYERS_AVAILABLE = False
    print("⚠️ Fyers API not available, will use Yahoo Finance as fallback")

import yfinance as yf


class IntradayDataDownloader:
    """
    Downloads 10-minute intraday data for backtesting
    """
    
    def __init__(self, data_dir='data/intraday'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        if FYERS_AVAILABLE:
            self.fyers = FyersDataFetcher()
            print("✅ Fyers API initialized for intraday data")
        else:
            self.fyers = None
            print("📊 Using Yahoo Finance for intraday data")
    
    def download_intraday_data(self, 
                              symbol='NSE:RELIANCE-EQ',
                              start_date='2024-01-01',
                              end_date='2024-12-31',
                              interval='10m'):
        """
        Download 10-minute intraday data
        
        Args:
            symbol: Trading symbol (NSE:RELIANCE-EQ for Fyers)
            start_date: Start date
            end_date: End date  
            interval: Candle interval (10m for 10 minutes)
        """
        
        print(f"📈 Downloading {interval} data for {symbol}")
        print(f"📅 Period: {start_date} to {end_date}")
        
        filename = f"{symbol.replace(':', '_').replace('-', '_')}_{interval}_{start_date}_{end_date}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            print(f"✅ File already exists: {filepath}")
            return pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
        
        try:
            if self.fyers and FYERS_AVAILABLE:
                data = self._download_from_fyers(symbol, start_date, end_date, interval)
            else:
                data = self._download_from_yahoo(symbol, start_date, end_date, interval)
            
            if data is not None and not data.empty:
                # Save to file for future use
                data.to_csv(filepath)
                print(f"💾 Saved {len(data)} candles to {filepath}")
                return data
            else:
                print("❌ No data downloaded")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return None
    
    def _download_from_fyers(self, symbol, start_date, end_date, interval):
        """Download from Fyers API"""
        
        try:
            # Convert interval format for Fyers
            fyers_interval = self._convert_interval_fyers(interval)
            
            # Fyers date format
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            print(f"🔄 Fetching from Fyers API...")
            
            data = self.fyers.get_historical_data(
                symbol=symbol,
                resolution=fyers_interval,
                from_date=start_dt.strftime('%Y-%m-%d'),
                to_date=end_dt.strftime('%Y-%m-%d')
            )
            
            if data and 'candles' in data:
                # Convert Fyers format to DataFrame
                df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
                df = df.set_index('timestamp')
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                print(f"✅ Downloaded {len(df)} candles from Fyers")
                return df
            else:
                print("❌ No data returned from Fyers")
                return None
                
        except Exception as e:
            print(f"❌ Fyers download failed: {e}")
            return None
    
    def _download_from_yahoo(self, symbol, start_date, end_date, interval):
        """Download from Yahoo Finance as fallback"""
        
        try:
            # Convert symbol for Yahoo Finance
            if 'RELIANCE' in symbol.upper():
                yahoo_symbol = 'RELIANCE.NS'
            else:
                yahoo_symbol = symbol
            
            # Convert interval for Yahoo
            yahoo_interval = self._convert_interval_yahoo(interval)
            
            print(f"🔄 Fetching {yahoo_symbol} from Yahoo Finance...")
            
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yahoo_interval,
                prepost=False
            )
            
            if not data.empty:
                # Rename columns to match our format
                data.index.name = 'timestamp'
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                print(f"✅ Downloaded {len(data)} candles from Yahoo Finance")
                return data
            else:
                print("❌ No data from Yahoo Finance")
                return None
                
        except Exception as e:
            print(f"❌ Yahoo Finance download failed: {e}")
            return None
    
    def _convert_interval_fyers(self, interval):
        """Convert interval to Fyers format"""
        mapping = {
            '1m': '1',
            '5m': '5', 
            '10m': '10',
            '15m': '15',
            '30m': '30',
            '1h': '60',
            '1d': 'D'
        }
        return mapping.get(interval, '10')
    
    def _convert_interval_yahoo(self, interval):
        """Convert interval to Yahoo Finance format"""
        mapping = {
            '1m': '1m',
            '2m': '2m',
            '5m': '5m',
            '10m': '10m',  # Note: Yahoo may not support 10m, will use 5m
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d'
        }
        return mapping.get(interval, '5m')  # Default to 5m if 10m not available
    
    def prepare_data_for_backtrader(self, data_file):
        """
        Prepare intraday data for Backtrader format
        """
        
        if isinstance(data_file, str):
            data = pd.read_csv(data_file, parse_dates=['timestamp'], index_col='timestamp')
        else:
            data = data_file
        
        # Ensure required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                print(f"❌ Missing column: {col}")
                return None
        
        # Sort by timestamp
        data = data.sort_index()
        
        # Remove any duplicate timestamps
        data = data[~data.index.duplicated(keep='first')]
        
        # Forward fill any missing values
        data = data.fillna(method='ffill')
        
        print(f"✅ Prepared {len(data)} candles for Backtrader")
        print(f"📊 Date range: {data.index[0]} to {data.index[-1]}")
        
        return data


def main():
    """
    Download intraday data for backtesting
    """
    
    print("🚀 Professional Intraday Data Downloader")
    print("="*60)
    
    downloader = IntradayDataDownloader()
    
    # Download 5-minute data for recent period (Yahoo Finance limitation)
    data = downloader.download_intraday_data(
        symbol='NSE:RELIANCE-EQ',
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='5m'  # Use 5m since Yahoo doesn't support 10m
    )
    
    if data is not None:
        print(f"\n📊 Data Summary:")
        print(f"Total candles: {len(data)}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Columns: {list(data.columns)}")
        print(f"\nFirst few rows:")
        print(data.head())
        
        # Prepare for Backtrader
        bt_data = downloader.prepare_data_for_backtrader(data)
        
        print(f"\n✅ Data ready for professional backtesting!")
        print(f"💡 Next: Use this data with Backtrader + your AI predictions")
        
    else:
        print("❌ Failed to download data")


if __name__ == "__main__":
    main()
