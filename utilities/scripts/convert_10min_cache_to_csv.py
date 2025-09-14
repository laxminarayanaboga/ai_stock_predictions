"""
Convert cached 10-minute data to CSV for validation
Combines quarterly JSON files into single CSV for AI prediction validation
"""

import pandas as pd
import json
import os
from datetime import datetime


def convert_cached_10min_to_csv():
    """Convert cached JSON data to CSV format"""
    
    cache_dir = "cache_raw_data_all_intraday/NSE:RELIANCE-EQ/interval_10/"
    output_dir = "data/raw/10min/"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔄 CONVERTING 10-MINUTE DATA TO CSV")
    print("=" * 60)
    
    all_data = []
    
    # JSON files in chronological order (2017-2025 - maximum available data)
    json_files = [
        # 2017 (earliest data available)
        "day_2017-07-01_to_2017-09-30.json",
        "day_2017-10-01_to_2017-12-31.json",
        # 2018
        "day_2018-01-01_to_2018-03-31.json",
        "day_2018-04-01_to_2018-06-30.json",
        "day_2018-07-01_to_2018-09-30.json",
        "day_2018-10-01_to_2018-12-31.json",
        # 2019
        "day_2019-01-01_to_2019-03-31.json",
        "day_2019-04-01_to_2019-06-30.json",
        "day_2019-07-01_to_2019-09-30.json",
        "day_2019-10-01_to_2019-12-31.json",
        # 2020
        "day_2020-01-01_to_2020-03-31.json",
        "day_2020-04-01_to_2020-06-30.json",
        "day_2020-07-01_to_2020-09-30.json",
        "day_2020-10-01_to_2020-12-31.json",
        # 2021
        "day_2021-01-01_to_2021-03-31.json",
        "day_2021-04-01_to_2021-06-30.json",
        "day_2021-07-01_to_2021-09-30.json",
        "day_2021-10-01_to_2021-12-31.json",
        # 2022
        "day_2022-01-01_to_2022-03-31.json",
        "day_2022-04-01_to_2022-06-30.json",
        "day_2022-07-01_to_2022-09-30.json",
        "day_2022-10-01_to_2022-12-31.json",
        # 2023
        "day_2023-08-01_to_2023-09-30.json",
        "day_2023-10-01_to_2023-12-31.json", 
        # 2024
        "day_2024-01-01_to_2024-03-31.json",
        "day_2024-04-01_to_2024-06-30.json",
        "day_2024-07-01_to_2024-09-30.json",
        "day_2024-10-01_to_2024-12-31.json",
        # 2025
        "day_2025-01-01_to_2025-03-31.json",
        "day_2025-04-01_to_2025-06-30.json",
        "day_2025-07-01_to_2025-08-31.json",
        "day_2025-09-01_to_2025-09-05.json"  # Latest data
    ]
    
    total_candles = 0
    
    for json_file in json_files:
        file_path = os.path.join(cache_dir, json_file)
        
        if os.path.exists(file_path):
            print(f"📦 Processing {json_file}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'candles' in data and data['candles']:
                # Convert to DataFrame
                df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Convert timestamp to IST (Indian Standard Time)
                df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                df['datetime_ist'] = df['datetime_utc'].dt.tz_convert('Asia/Kolkata')
                df['time_ist'] = df['datetime_ist'].dt.time
                df['date_ist'] = df['datetime_ist'].dt.date
                
                # Filter to market hours in IST (9:15 AM to 3:30 PM IST)
                market_start = pd.to_datetime('09:15', format='%H:%M').time()
                market_end = pd.to_datetime('15:30', format='%H:%M').time()
                df_filtered = df[(df['time_ist'] >= market_start) & (df['time_ist'] <= market_end)]
                
                candles_count = len(df_filtered)
                days_count = len(df_filtered['date_ist'].unique())
                
                print(f"  ✅ {candles_count} candles across {days_count} days")
                print(f"  📊 Time range: {df_filtered['time_ist'].min()} - {df_filtered['time_ist'].max()}")
                
                # Sample analysis for first file
                if json_file == json_files[0]:
                    sample_day = df_filtered['date_ist'].iloc[0] if len(df_filtered) > 0 else None
                    if sample_day:
                        sample_data = df_filtered[df_filtered['date_ist'] == sample_day]
                        print(f"  📝 Sample day {sample_day}: {len(sample_data)} candles")
                        print(f"      Times: {sample_data['time_ist'].min()} to {sample_data['time_ist'].max()}")
                
                all_data.append(df_filtered)
                total_candles += candles_count
            else:
                print(f"  ⚠️ No candle data in {json_file}")
        else:
            print(f"  ❌ File not found: {json_file}")
    
    if all_data:
        # Combine all data
        print(f"\n🔗 COMBINING DATA")
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Final stats
        total_days = len(combined_df['date_ist'].unique())
        first_date = combined_df['date_ist'].iloc[0]
        last_date = combined_df['date_ist'].iloc[-1]
        
        print(f"📊 FINAL STATISTICS:")
        print(f"  Total candles: {len(combined_df)}")
        print(f"  Trading days: {total_days}")
        print(f"  Date range: {first_date} to {last_date}")
        print(f"  Time range: {combined_df['time_ist'].min()} to {combined_df['time_ist'].max()}")
        print(f"  Avg candles/day: {len(combined_df)/total_days:.1f}")
        
        # Check if we have full trading day data
        sample_days = combined_df.groupby('date_ist').size().head(10)
        print(f"\n📝 Sample days candle count:")
        for date, count in sample_days.items():
            print(f"    {date}: {count} candles")
        
        # Save to CSV with only essential columns (no datetime strings)
        final_df = combined_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        output_file = f"{output_dir}RELIANCE_NSE_10min_20170701_to_20250905.csv"
        final_df.to_csv(output_file, index=False)
        
        print(f"\n💾 SAVED TO: {output_file}")
        print(f"📁 File size: {os.path.getsize(output_file)/1024:.1f} KB")
        
        return output_file
    else:
        print("❌ No data to convert")
        return None


if __name__ == "__main__":
    convert_cached_10min_to_csv()
