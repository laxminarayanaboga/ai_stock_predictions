
import os
import json
from datetime import datetime, timezone
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(filename='create_daywise_cache.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def process_file(source_file, source_dir, target_dir):
    """Process a single JSON file and create day-wise cache."""
    # Extract symbol and interval from the path
    parts = os.path.relpath(os.path.dirname(source_file), source_dir).split(os.sep)
    if len(parts) < 2:
        return f"Skipping file {source_file}: Invalid path structure."
    
    symbol = parts[0]
    interval = parts[1]

    try:
        with open(source_file, 'r') as f:
            data = json.load(f)

        # Skip files without 'candles' or with empty data
        if not data or 'candles' not in data or not data['candles']:
            return f"Skipping empty or corrupted file: {source_file}"

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return f"Error reading file {source_file}: {e}"

    # Organize data by day
    daywise_data = {}
    for candle in data['candles']:
        timestamp = candle[0]
        # Convert timestamp to a date string
        date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d')

        if date_str not in daywise_data:
            daywise_data[date_str] = []
        daywise_data[date_str].append(candle)

    # Save day-wise data in the target directory
    for date_str, candles in daywise_data.items():
        target_folder = os.path.join(target_dir, symbol, interval)
        target_file = os.path.join(target_folder, f"day_{date_str}.json")
        os.makedirs(target_folder, exist_ok=True)
        with open(target_file, 'w') as f:
            json.dump({'candles': candles}, f)

    # Delete the processed source file to save disk space
    os.remove(source_file)

    return f"Processed and deleted file: {source_file}"

def create_daywise_cache_parallel(source_dir, target_dir, max_workers=4):
    """Process JSON files in parallel."""
    # Collect all JSON files to process
    files_to_process = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.json'):
                files_to_process.append(os.path.join(root, file))

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file, source_dir, target_dir): file for file in files_to_process}
        
        for future in as_completed(futures):
            result = future.result()
            print(result)
            logging.info(result)

    print("Processing completed.")

# Example usage
source_directory = 'cache_raw_data_all_intraday'
target_directory = 'cache2'
create_daywise_cache_parallel(source_directory, target_directory, max_workers=10)  # Adjust max_workers based on your system's capacity
