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
    
    # JSON files in order
    json_files = [
        "day_2023-08-01_to_2023-09-30.json",
        "day_2023-10-01_to_2023-12-31.json", 
        "day_2024-01-01_to_2024-03-31.json",
        "day_2024-04-01_to_2024-06-30.json",
        "day_2024-07-01_to_2024-09-30.json",
        "day_2024-10-01_to_2024-12-31.json",
        "day_2025-01-01_to_2025-03-31.json",
        "day_2025-04-01_to_2025-06-30.json",
        "day_2025-07-01_to_2025-08-31.json"
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
                
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['date'] = df['datetime'].dt.date
                df['time'] = df['datetime'].dt.time
                
                # Format datetime as string
                df['date_str'] = df['datetime'].dt.strftime('%Y-%m-%d')
                df['datetime_str'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Filter to market hours in UTC (Indian market: 9:15 AM to 3:30 PM IST = 3:45 AM to 10:00 AM UTC)
                market_start = pd.to_datetime('03:45', format='%H:%M').time()
                market_end = pd.to_datetime('10:00', format='%H:%M').time()
                df_filtered = df[(df['time'] >= market_start) & (df['time'] <= market_end)]
                
                candles_count = len(df_filtered)
                days_count = len(df_filtered['date'].unique())
                
                print(f"  ✅ {candles_count} candles across {days_count} days")
                print(f"  📊 Time range: {df_filtered['time'].min()} - {df_filtered['time'].max()}")
                
                # Sample analysis for first file
                if json_file == json_files[0]:
                    sample_day = df_filtered['date_str'].iloc[0] if len(df_filtered) > 0 else None
                    if sample_day:
                        sample_data = df_filtered[df_filtered['date_str'] == sample_day]
                        print(f"  📝 Sample day {sample_day}: {len(sample_data)} candles")
                        print(f"      Times: {sample_data['time'].min()} to {sample_data['time'].max()}")
                
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
        total_days = len(combined_df['date'].unique())
        first_date = combined_df['date_str'].iloc[0]
        last_date = combined_df['date_str'].iloc[-1]
        
        print(f"📊 FINAL STATISTICS:")
        print(f"  Total candles: {len(combined_df)}")
        print(f"  Trading days: {total_days}")
        print(f"  Date range: {first_date} to {last_date}")
        print(f"  Time range: {combined_df['time'].min()} to {combined_df['time'].max()}")
        print(f"  Avg candles/day: {len(combined_df)/total_days:.1f}")
        
        # Check if we have full trading day data
        sample_days = combined_df.groupby('date_str').size().head(10)
        print(f"\n📝 Sample days candle count:")
        for date, count in sample_days.items():
            print(f"    {date}: {count} candles")
        
        # Save to CSV
        output_columns = ['date_str', 'datetime_str', 'time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        final_df = combined_df[output_columns].copy()
        final_df.columns = ['date', 'datetime', 'time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        output_file = f"{output_dir}RELIANCE_NSE_10min_20230801_to_20250831.csv"
        final_df.to_csv(output_file, index=False)
        
        print(f"\n💾 SAVED TO: {output_file}")
        print(f"📁 File size: {os.path.getsize(output_file)/1024:.1f} KB")
        
        return output_file
    else:
        print("❌ No data to convert")
        return None


if __name__ == "__main__":
    convert_cached_10min_to_csv()
