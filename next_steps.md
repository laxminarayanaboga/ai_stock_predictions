v9 model is promising. 
- Add code to integrate the fyers trading api and place orders according to the given strategy, symbol, quantity
- You may need one click trigger script, that will,
   - get new access token if expired - use the refresh token
   - refresh the historical data with latest day
   - infer the model and generate next day predictions
   - according to given strategy, place trade orders using fyers apis
- Start the actual trading from Wed - 17 Sep 2025