# Scalping module

This package contains a self-contained scalping research toolkit:

- strategies/: indicators and simple strategies (ORB, EMA pullback, VWAP reversion, RSI reversal, MACD cross, Bollinger reversion)
- backtester.py: a simple ORB/EMA/VWAP day-runner using IntradaySimulator with SL/TP and cost-aware PnL
- scripts/:
  - run_scalping_week.py: combine cached 5s JSON to CSV and run the backtester for the cache range
  - run_multi_strategy.py: run a small parameter grid over multiple strategies, simulate on 5s bars, and write a summary CSV

Results are saved under src/scalping/results/<SYMBOL>/...

Assumptions:
- You already have cached 5s data under cache_raw_data_all_intraday/<symbol>/interval_5S/day_YYYY-MM-DD.json
- Use src/data/historical_data_cache_management.ensure_5s_csv() to build a combined CSV if needed.

Notes:
- This module is isolated from the legacy simulator/AI code to avoid interference.
- Expand grids or add strategies in strategies/common_strategies.py as needed.
