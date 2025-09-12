"""
Fixed 10-Minute Data Downloader with Proper Date Formats
Attempts to get current and recent intraday data from Fyers API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
import pytz

# Add paths
sys.path.append('/Users/bogalaxminarayana/myGit/ai_stock_predictions')
sys.path.append('/Users/bogalaxminarayana/myGit/ai_stock_predictions/api')

from api.fyers_session_management import get_fyers_session


class Fixed10MinDownloader:
    """
    Downloads 10-minute intraday data with proper API parameters
    """
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        os.makedirs(f"{data_dir}/10min", exist_ok=True)
        
        try:
            self.fyers = get_fyers_session()
            print("✅ Fyers API session initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Fyers API: {e}")
            self.fyers = None

    def test_live_data(self, symbol='NSE:RELIANCE-EQ'):
        """Test if we can get live/current day data"""
        
        print(f"\n🔍 TESTING LIVE DATA FOR {symbol}")
        print("=" * 60)
        
        try:
            # Try to get current day data
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Method 1: Try history with today's date
            data = {
                "symbol": symbol,
                "resolution": "10",
                "date_format": "0",  # Use 0 for epoch timestamps
                "range_from": today,
                "range_to": today,
                "cont_flag": "1"
            }
            
            print(f"📅 Trying current day data: {today}")
            response = self.fyers.history(data=data)
            print(f"📊 Response: {response}")
            
            if response['s'] == 'ok' and 'candles' in response:
                candles = response['candles']
                print(f"✅ Live data available: {len(candles)} candles today")
                
                if len(candles) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                    df['time'] = df['datetime'].dt.time
                    
                    print(f"📊 Time range: {df['time'].min()} - {df['time'].max()}")
                    print(f"📊 Sample data:")
                    print(df[['datetime', 'close', 'volume']].head())
                    
                    return df
            
            return None
            
        except Exception as e:
            print(f"❌ Error testing live data: {e}")
            return None

    def test_quotes_api(self, symbol='NSE:RELIANCE-EQ'):
        """Test quotes API for current data"""
        
        print(f"\n🔍 TESTING QUOTES API FOR {symbol}")
        print("=" * 60)
        
        try:
            # Try quotes API
            response = self.fyers.quotes({"symbols": symbol})
            print(f"📊 Quotes response: {response}")
            
            if response['s'] == 'ok':
                return response
            
            return None
            
        except Exception as e:
            print(f"❌ Error testing quotes: {e}")
            return None

    def try_recent_history_epoch(self, symbol='NSE:RELIANCE-EQ', days_back=7):
        """Try recent history with epoch timestamps"""
        
        print(f"\n🔄 TRYING RECENT {days_back} DAYS WITH EPOCH TIMESTAMPS")
        print("=" * 60)
        
        try:
            # Calculate epoch timestamps
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Convert to IST timezone and then to epoch
            ist = pytz.timezone('Asia/Kolkata')
            
            # Set to market hours
            start_dt = start_time.replace(hour=9, minute=15, second=0, microsecond=0)
            end_dt = end_time.replace(hour=15, minute=30, second=0, microsecond=0)
            
            start_ist = ist.localize(start_dt)
            end_ist = ist.localize(end_dt)
            
            start_epoch = int(start_ist.timestamp())
            end_epoch = int(end_ist.timestamp())
            
            print(f"📅 Period: {start_dt} to {end_dt}")
            print(f"🕐 Epoch: {start_epoch} to {end_epoch}")
            
            data = {
                "symbol": symbol,
                "resolution": "10",
                "date_format": "0",  # Epoch timestamps
                "range_from": str(start_epoch),
                "range_to": str(end_epoch),
                "cont_flag": "1"
            }
            
            response = self.fyers.history(data=data)
            print(f"📊 API Response: {response}")
            
            if response['s'] == 'ok' and 'candles' in response:
                candles = response['candles']
                
                if len(candles) > 0:
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                    df['date'] = df['datetime'].dt.date
                    df['time'] = df['datetime'].dt.time
                    
                    print(f"✅ Downloaded {len(df)} candles across {len(df['date'].unique())} days")
                    print(f"📊 Time range: {df['time'].min()} - {df['time'].max()}")
                    
                    # Check candles per day
                    daily_counts = df.groupby('date').size()
                    print(f"📊 Candles per day:")
                    for date, count in daily_counts.items():
                        print(f"    {date}: {count} candles")
                    
                    return df
            
            return None
            
        except Exception as e:
            print(f"❌ Error trying recent epoch data: {e}")
            return None

    def try_different_resolutions(self, symbol='NSE:RELIANCE-EQ'):
        """Try different time resolutions to see what's available"""
        
        print(f"\n🔍 TESTING DIFFERENT RESOLUTIONS FOR {symbol}")
        print("=" * 60)
        
        resolutions = ["1", "5", "10", "15", "30", "60", "D"]
        today = datetime.now().strftime('%Y-%m-%d')
        
        for resolution in resolutions:
            try:
                data = {
                    "symbol": symbol,
                    "resolution": resolution,
                    "date_format": "0",
                    "range_from": today,
                    "range_to": today,
                    "cont_flag": "1"
                }
                
                response = self.fyers.history(data=data)
                
                if response['s'] == 'ok' and 'candles' in response:
                    candles = response['candles']
                    print(f"✅ Resolution {resolution}: {len(candles)} candles")
                else:
                    print(f"❌ Resolution {resolution}: {response.get('message', 'No data')}")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Resolution {resolution}: Error - {e}")


def main():
    """Main function to test different approaches"""
    
    downloader = Fixed10MinDownloader()
    
    if not downloader.fyers:
        print("❌ Cannot proceed without Fyers API")
        return
    
    print("🚀 TESTING FYERS API DATA AVAILABILITY")
    print("=" * 80)
    
    # Test 1: Live data
    live_data = downloader.test_live_data()
    
    # Test 2: Quotes API
    quotes_data = downloader.test_quotes_api()
    
    # Test 3: Recent history with epoch timestamps
    recent_data = downloader.try_recent_history_epoch(days_back=7)
    
    if recent_data is not None:
        # Save the data we got
        filename = f"data/raw/10min/RELIANCE_NSE_10min_recent_working.csv"
        recent_data['date_str'] = recent_data['datetime'].dt.strftime('%Y-%m-%d')
        recent_data.to_csv(filename, index=False)
        print(f"💾 Saved working data to {filename}")
    
    # Test 4: Different resolutions
    downloader.try_different_resolutions()
    
    print("\n🎯 API TESTING COMPLETE")
    print("Check the output above to understand API limitations!")


if __name__ == "__main__":
    main()
