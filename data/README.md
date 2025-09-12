# Data Organization Structure

## Directory Layout

```
data/
├── raw/                          # Raw market data from APIs
│   ├── daily/                    # Daily OHLCV data
│   │   ├── RELIANCE_NSE_daily_20150911_to_20250910.csv  # Training data (10 years)
│   │   └── RELIANCE_NSE_daily_20230831_to_20250829.csv  # Validation data (2 years)
│   └── intraday/                 # Intraday/minute-level data
│       └── RELIANCE_NSE_10min_20241101_to_20241130.csv  # 10-minute candles
├── processed/                    # Processed/engineered features
│   └── (model-ready datasets)
└── predictions/                  # AI model predictions
    └── backtest_predictions.pkl

```

## File Naming Convention

### Format: `{SYMBOL}_{EXCHANGE}_{TIMEFRAME}_{STARTDATE}_to_{ENDDATE}.csv`

**Components:**
- `SYMBOL`: Stock symbol (e.g., RELIANCE)
- `EXCHANGE`: Exchange code (NSE, BSE)
- `TIMEFRAME`: Data frequency (daily, 1min, 5min, 10min, 1hour)
- `STARTDATE`: Start date in YYYYMMDD format
- `ENDDATE`: End date in YYYYMMDD format

**Examples:**
- `RELIANCE_NSE_daily_20230831_to_20250829.csv` - Daily data for validation
- `RELIANCE_NSE_10min_20241101_to_20241130.csv` - 10-minute intraday data
- `RELIANCE_NSE_daily_20150911_to_20250910.csv` - Daily data for training

## Data Sources

- **Daily Data**: Fyers API via professional data downloader
- **Intraday Data**: Fyers API 10-minute candles
- **Real-time**: Fyers API live feeds

## Data Quality

- **Validation Data**: 496 trading days (95% completeness)
- **Training Data**: 10+ years of historical data
- **Format**: OHLCV with timestamps and volume
- **Clean**: Deduplicated, sorted, verified

## Usage

- `raw/daily/` - Use for model training and validation
- `raw/intraday/` - Use for backtesting and strategy development
- `processed/` - Store feature-engineered datasets
- Model predictions should reference the corresponding raw data files

Last Updated: September 11, 2025
