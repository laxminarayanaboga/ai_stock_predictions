"""
Convert daily CSV file from datetime strings to Unix timestamps
Fixes data inconsistency between 10-minute and daily files
"""

import pandas as pd
import os
from datetime import datetime


def convert_daily_csv_to_unix_timestamp():
    """Convert RELIANCE daily CSV from datetime strings to Unix timestamps"""
    
    input_file = "data/raw/daily/RELIANCE_NSE_daily_20150911_to_20250831.csv"
    output_file = "data/raw/daily/RELIANCE_NSE_daily_20150911_to_20250831_unix.csv"
    backup_file = "data/raw/daily/RELIANCE_NSE_daily_20150911_to_20250831_original.csv"
    
    print("🔄 CONVERTING DAILY CSV TO UNIX TIMESTAMPS")
    print("=" * 60)
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output file: {output_file}")
    print(f"💾 Backup file: {backup_file}")
    
    # Read the CSV file
    print("📖 Reading CSV file...")
    df = pd.read_csv(input_file)
    
    print(f"📊 Total records: {len(df)}")
    print(f"📅 Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    # Convert timestamp column to Unix timestamp
    print("🔄 Converting timestamps...")
    
    # Parse the datetime string with timezone
    df['datetime_parsed'] = pd.to_datetime(df['timestamp'])
    
    # Convert to Unix timestamp (seconds since epoch)
    df['unix_timestamp'] = df['datetime_parsed'].astype('int64') // 10**9
    
    # Create new dataframe with Unix timestamps
    df_converted = df[['unix_timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_converted.rename(columns={'unix_timestamp': 'timestamp'}, inplace=True)
    
    # Make column names consistent with 10-minute files (lowercase)
    df_converted.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Display sample conversions
    print("📝 Sample conversions:")
    for i in [0, 1, -2, -1]:
        original_time = df['timestamp'].iloc[i]
        unix_time = df_converted['timestamp'].iloc[i]
        converted_back = datetime.fromtimestamp(unix_time)
        print(f"  {original_time} → {unix_time} → {converted_back}")
    
    # Create backup of original file
    print(f"💾 Creating backup: {backup_file}")
    import shutil
    shutil.copy2(input_file, backup_file)
    
    # Save converted file
    print(f"💾 Saving converted file: {output_file}")
    df_converted.to_csv(output_file, index=False)
    
    # Replace original file with converted version
    print(f"🔄 Replacing original file with converted version...")
    shutil.move(output_file, input_file)
    
    print("✅ Conversion completed successfully!")
    print(f"📁 Original file backed up as: {backup_file}")
    print(f"📁 Converted file: {input_file}")
    
    # Verification
    print("\n🔍 VERIFICATION:")
    df_verify = pd.read_csv(input_file)
    print(f"📊 Records in converted file: {len(df_verify)}")
    print(f"📋 Columns: {list(df_verify.columns)}")
    print(f"🕐 First timestamp: {df_verify['timestamp'].iloc[0]} ({datetime.fromtimestamp(df_verify['timestamp'].iloc[0])})")
    print(f"🕐 Last timestamp: {df_verify['timestamp'].iloc[-1]} ({datetime.fromtimestamp(df_verify['timestamp'].iloc[-1])})")
    
    # Show first few rows
    print("\n📝 First 3 rows of converted file:")
    print(df_verify.head(3).to_string(index=False))
    
    print("\n🎉 Daily CSV file now uses Unix timestamps - consistent with 10-minute files!")
    
    return input_file


if __name__ == "__main__":
    convert_daily_csv_to_unix_timestamp()