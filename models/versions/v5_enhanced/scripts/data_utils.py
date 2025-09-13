"""
Simplified data utilities for V5 model training
"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class SimpleStockDataset(Dataset):
    """Simple PyTorch dataset for stock data"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_reliance_data(use_intraday=True):
    """Load Reliance stock data - both daily and intraday"""
    daily_path = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/data/raw/daily/RELIANCE_NSE_daily_20150911_to_20250910.csv"
    intraday_path = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/data/raw/10min/RELIANCE_NSE_10min_20170701_to_20250831.csv"
    
    try:
        # Load daily data
        daily_df = pd.read_csv(daily_path)
        daily_df['timestamp'] = pd.to_datetime(daily_df['timestamp'])
        daily_df = daily_df.sort_values('timestamp').reset_index(drop=True)
        
        if not use_intraday:
            return daily_df
        
        # Load 10-minute intraday data
        print("Loading 10-minute intraday data...")
        intraday_df = pd.read_csv(intraday_path)
        intraday_df['timestamp'] = pd.to_datetime(intraday_df['timestamp'], unit='s')
        intraday_df = intraday_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Daily data shape: {daily_df.shape}")
        print(f"Intraday data shape: {intraday_df.shape}")
        print(f"Intraday date range: {intraday_df['timestamp'].min()} to {intraday_df['timestamp'].max()}")
        
        return daily_df, intraday_df
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        return None


def aggregate_intraday_features(intraday_df, target_date):
    """Extract features from 10-minute data for a specific date"""
    
    # Get data for the target date
    date_str = target_date.strftime('%Y-%m-%d')
    daily_data = intraday_df[intraday_df['timestamp'].dt.date == target_date.date()]
    
    if len(daily_data) == 0:
        return {}
    
    # Normalize column names
    daily_data.columns = daily_data.columns.str.lower()
    
    features = {}
    
    # Basic OHLCV aggregates
    features['intraday_open'] = daily_data['open'].iloc[0]
    features['intraday_high'] = daily_data['high'].max()
    features['intraday_low'] = daily_data['low'].min()
    features['intraday_close'] = daily_data['close'].iloc[-1]
    features['intraday_volume'] = daily_data['volume'].sum()
    
    # Intraday price movements
    features['intraday_range'] = features['intraday_high'] - features['intraday_low']
    features['intraday_body'] = abs(features['intraday_close'] - features['intraday_open'])
    features['intraday_upper_wick'] = features['intraday_high'] - max(features['intraday_open'], features['intraday_close'])
    features['intraday_lower_wick'] = min(features['intraday_open'], features['intraday_close']) - features['intraday_low']
    
    # Volatility measures
    returns = daily_data['close'].pct_change().dropna()
    features['intraday_volatility'] = returns.std() if len(returns) > 1 else 0
    features['intraday_avg_return'] = returns.mean() if len(returns) > 1 else 0
    
    # Volume patterns
    features['volume_weighted_price'] = (daily_data['close'] * daily_data['volume']).sum() / daily_data['volume'].sum()
    features['volume_std'] = daily_data['volume'].std()
    features['volume_trend'] = daily_data['volume'].corr(pd.Series(range(len(daily_data)))) if len(daily_data) > 2 else 0
    
    # Price momentum within day
    price_changes = daily_data['close'].diff().dropna()
    features['positive_moves'] = (price_changes > 0).sum()
    features['negative_moves'] = (price_changes < 0).sum()
    features['momentum_ratio'] = features['positive_moves'] / len(price_changes) if len(price_changes) > 0 else 0.5
    
    # Time-based patterns
    daily_data = daily_data.copy()  # Make explicit copy to avoid warnings
    daily_data.loc[:, 'hour'] = daily_data['timestamp'].dt.hour
    morning_data = daily_data[daily_data['hour'] < 12]
    afternoon_data = daily_data[daily_data['hour'] >= 12]
    
    if len(morning_data) > 0 and len(afternoon_data) > 0:
        features['morning_avg_price'] = morning_data['close'].mean()
        features['afternoon_avg_price'] = afternoon_data['close'].mean()
        features['morning_afternoon_ratio'] = features['afternoon_avg_price'] / features['morning_avg_price']
    else:
        features['morning_avg_price'] = features['intraday_close']
        features['afternoon_avg_price'] = features['intraday_close']
        features['morning_afternoon_ratio'] = 1.0
    
    # Gap analysis (if we have previous day data)
    features['gap_up'] = 0
    features['gap_down'] = 0
    
    return features


def create_enhanced_features(df, intraday_df=None):
    """Create enhanced features for V5 model with optional intraday data"""
    data = df.copy()
    
    # Normalize column names to lowercase
    data.columns = data.columns.str.lower()
    
    # Basic price features
    data['OC_Ratio'] = (data['close'] - data['open']) / data['open']
    data['HL_Ratio'] = (data['high'] - data['low']) / data['low']
    data['Intraday_Range'] = (data['high'] - data['low']) / data['close']
    data['Open_to_Close'] = (data['close'] - data['open']) / data['open']
    
    # Add intraday features if available
    if intraday_df is not None:
        print("Incorporating 10-minute intraday features...")
        
        # Get unique dates from daily data to avoid processing duplicates
        unique_dates = data['timestamp'].dt.date.unique()
        date_to_features = {}
        
        print(f"Processing intraday data for {len(unique_dates)} unique dates...")
        
        # Process each unique date once
        for date in unique_dates:
            intraday_features = aggregate_intraday_features(intraday_df, pd.to_datetime(date))
            date_to_features[date] = intraday_features
        
        # Map features to all rows
        intraday_features_list = []
        for idx, row in data.iterrows():
            date = pd.to_datetime(row['timestamp']).date()
            intraday_features_list.append(date_to_features.get(date, {}))
        
        # Convert to DataFrame and merge
        intraday_features_df = pd.DataFrame(intraday_features_list)
        
        # Add intraday features to main dataframe
        for col in intraday_features_df.columns:
            if col not in ['intraday_open', 'intraday_high', 'intraday_low', 'intraday_close']:  # Avoid duplicating OHLC
                data[f'intraday_{col}'] = intraday_features_df[col].fillna(0)
        
        print(f"Added {len(intraday_features_df.columns)} intraday features")
    
    # Continue with existing feature engineering...
    
    # Body and shadow analysis
    data['Body'] = abs(data['close'] - data['open'])
    data['Upper_Shadow'] = data['high'] - np.maximum(data['open'], data['close'])
    data['Lower_Shadow'] = np.minimum(data['open'], data['close']) - data['low']
    data['Body_to_Range'] = data['Body'] / (data['high'] - data['low'])
    data['Upper_Shadow'] = data['Upper_Shadow'] / (data['high'] - data['low'])
    data['Lower_Shadow'] = data['Lower_Shadow'] / (data['high'] - data['low'])
    
    # Price gaps
    data['Price_Gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Moving averages and trends
    for period in [5, 10, 20]:
        data[f'SMA_{period}'] = data['close'].rolling(period).mean()
        data[f'EMA_{period}'] = data['close'].ewm(span=period).mean()
        data[f'SMA_{period}_Ratio'] = data['close'] / data[f'SMA_{period}']
        data[f'Trend_{period}'] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period)
    
    # EMA slopes
    for period in [5, 10]:
        ema_col = f'EMA_{period}'
        data[f'EMA_{period}_Slope'] = (data[ema_col] - data[ema_col].shift(1)) / data[ema_col].shift(1)
    
    # Volume indicators
    data['Volume_Normalized'] = data['volume'] / data['volume'].rolling(20).mean()
    data['Volume_Price_Trend'] = (data['close'] - data['close'].shift(1)) * data['volume']
    
    # On-Balance Volume
    data['OBV'] = np.where(data['close'] > data['close'].shift(1), 
                          data['volume'], 
                          np.where(data['close'] < data['close'].shift(1), 
                                  -data['volume'], 0)).cumsum()
    data['OBV_Slope'] = (data['OBV'] - data['OBV'].shift(5)) / data['OBV'].shift(5)
    
    # VWAP
    data['Typical_Price'] = (data['high'] + data['low'] + data['close']) / 3
    data['VWAP'] = (data['Typical_Price'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
    data['VWAP_Deviation'] = (data['close'] - data['VWAP']) / data['VWAP']
    
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    data['RSI_14'] = calculate_rsi(data['close'])
    
    # MACD
    ema_12 = data['close'].ewm(span=12).mean()
    ema_26 = data['close'].ewm(span=26).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    data['MACD_Strength'] = abs(data['MACD']) / data['close']
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    data['BB_Middle'] = data['close'].rolling(bb_period).mean()
    bb_std_val = data['close'].rolling(bb_period).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std_val * bb_std)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std_val * bb_std)
    data['BB_Position'] = (data['close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    data['BB_Squeeze'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    
    # Stochastic Oscillator
    high_14 = data['high'].rolling(14).max()
    low_14 = data['low'].rolling(14).min()
    data['Stochastic_K'] = 100 * (data['close'] - low_14) / (high_14 - low_14)
    
    # Volatility measures
    data['Volatility'] = data['close'].pct_change().rolling(20).std()
    data['ATR'] = data[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], 
                     abs(x['high'] - x['close']), 
                     abs(x['low'] - x['close'])), axis=1
    ).rolling(14).mean()
    data['ATR_Ratio'] = data['ATR'] / data['close']
    
    # Volatility regime
    vol_ma = data['Volatility'].rolling(50).mean()
    data['Volatility_Regime'] = data['Volatility'] / vol_ma
    
    # Momentum indicators
    for period in [3, 5, 10]:
        data[f'Momentum_{period}'] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period)
    
    # Momentum quality (consistency)
    data['Momentum_5_Quality'] = data['Momentum_5'].rolling(5).std()
    
    # Support and Resistance levels
    window = 20
    data['Resistance'] = data['high'].rolling(window).max()
    data['Support'] = data['low'].rolling(window).min()
    data['Resistance_Break'] = (data['high'] > data['Resistance'].shift(1)).astype(int)
    data['Support_Break'] = (data['low'] < data['Support'].shift(1)).astype(int)
    data['S_R_Distance'] = (data['Resistance'] - data['Support']) / data['close']
    
    # Clean up intermediate columns
    drop_cols = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20',
                'Body', 'Typical_Price', 'VWAP', 'MACD', 'MACD_Signal',
                'BB_Middle', 'BB_Upper', 'BB_Lower', 'Volatility', 'ATR',
                'Resistance', 'Support', 'OBV']
    
    existing_drop_cols = [col for col in drop_cols if col in data.columns]
    data = data.drop(columns=existing_drop_cols)
    
    # Clean the data - handle infinity and NaN values
    print(f"Data shape before cleaning: {data.shape}")
    
    # Replace infinite values with NaN first
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values using forward fill then backward fill
    data = data.ffill().bfill()
    
    # Final check for any remaining NaN or inf values
    inf_cols = data.columns[np.isinf(data).any()].tolist()
    nan_cols = data.columns[data.isna().any()].tolist()
    
    if inf_cols:
        print(f"Warning: Infinite values found in columns: {inf_cols}")
        data[inf_cols] = data[inf_cols].replace([np.inf, -np.inf], 0)
    
    if nan_cols:
        print(f"Warning: NaN values found in columns: {nan_cols}")
        data[nan_cols] = data[nan_cols].fillna(0)
    
    print(f"Data shape after cleaning: {data.shape}")
    print(f"Final feature count: {len(data.columns)}")
    
    return data


def prepare_sequences(df, lookback_days=15, target_cols=['open', 'high', 'low', 'close']):
    """Prepare sequences for training"""
    
    # Get feature columns (exclude timestamp and target columns)
    exclude_cols = ['timestamp'] + target_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")  # Show first 10
    
    sequences = []
    targets = []
    
    for i in range(lookback_days, len(df)):
        # Features for the sequence
        seq_features = df[feature_cols].iloc[i-lookback_days:i].values
        
        # Target for next day
        target = df[target_cols].iloc[i].values
        
        sequences.append(seq_features)
        targets.append(target)
    
    return np.array(sequences), np.array(targets), feature_cols