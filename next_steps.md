v9 model is promising. 
- Add code to integrate the fyers trading api and place orders according to the given strategy, symbol, quantity
- You may need one click trigger script, that will,
   - get new access token if expired - use the refresh token
   - historical data is available 2025-09-12 -- fetch the delta missing data using src/data/data_fetch.py. Update if anything required.
   - infer the model and generate next day predictions
   - according to given strategy, place trade orders using fyers apis. Orders may need to be placed in intelligent way, according to the strategies. 

- Trade limited 4 symbols. Each needs to be executed in its own strategy. 
   TATAMOTORS,tech_confirm_standard_target_based,Technical Confirmation Standard
   NTPC,tech_confirm_ultra_aggressive,Technical Confirmation Ultra
   TCS,tech_confirm_ultra_aggressive,Technical Confirmation Ultra
   RELIANCE,tech_confirm_ultra_aggressive,Technical Confirmation Ultra

Additonal Notes:
- Raw data available till 2025-09-12 -- data/raw
   - Partial scripts or some code are available at 
      - src/data/historical_data_cache_management.py -- downloads the data to cache folder..
      - src/data/fyers_downloader_utils.py