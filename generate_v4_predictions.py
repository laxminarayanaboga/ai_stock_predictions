#!/usr/bin/env python3
"""
Generate Enhanced v4 Model Predictions
Use our newly trained v4_minimal model to generate predictions for strategy testing
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, time
import json
import pickle

# Add project root to path
sys.path.append('/Users/bogalaxminarayana/myGit/ai_stock_predictions')


class MinimalEnhancedLSTM_V4(nn.Module):
    """Same model architecture as training script"""
    
    def __init__(self, input_size=16, hidden_size=128, num_layers=3, dropout=0.2):
        super(MinimalEnhancedLSTM_V4, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 4)  # OHLC
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        final_output = attn_out[:, -1, :]
        price_pred = self.price_predictor(final_output)
        return price_pred


def create_simple_features(data: pd.DataFrame) -> tuple:
    """Same feature engineering as training"""
    data = data.copy()
    
    # Basic features
    base_features = ['open', 'high', 'low', 'close', 'volume']
    
    # Additional simple features
    data['price_range'] = data['high'] - data['low']
    data['close_open_ratio'] = data['close'] / data['open']
    data['high_close_ratio'] = data['high'] / data['close']
    data['low_close_ratio'] = data['low'] / data['close']
    data['volume_ma5'] = data['volume'].rolling(5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma5']
    
    # Simple moving averages
    data['close_ma5'] = data['close'].rolling(5).mean()
    data['close_ma10'] = data['close'].rolling(10).mean()
    data['ma_ratio'] = data['close_ma5'] / data['close_ma10']
    
    # Price changes
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Fill NaN values
    data = data.bfill().ffill()
    
    # Feature columns
    feature_cols = base_features + [
        'price_range', 'close_open_ratio', 'high_close_ratio', 'low_close_ratio',
        'volume_ma5', 'volume_ratio', 'close_ma5', 'close_ma10', 'ma_ratio',
        'price_change', 'volume_change'
    ]
    
    return data, feature_cols


def generate_v4_predictions():
    """Generate predictions using v4_minimal model"""
    
    print("🔮 GENERATING ENHANCED V4 MODEL PREDICTIONS")
    print("=" * 60)
    
    # Load trained model
    model_path = 'models/enhanced_model_v4_minimal.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Train the model first!")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    scaler = checkpoint['scaler']
    feature_cols = checkpoint['feature_cols']
    
    print(f"✅ Loaded model with {config['input_size']} features")
    print(f"📊 Features: {feature_cols}")
    
    # Initialize model
    model = MinimalEnhancedLSTM_V4(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    data_file = 'data/raw/10min/RELIANCE_NSE_10min_20170701_to_20250831.csv'
    print(f"📊 Loading data from: {data_file}")
    
    df = pd.read_csv(data_file)
    df['datetime_ist'] = pd.to_datetime(df['timestamp'], unit='s')
    df['datetime_ist'] = df['datetime_ist'] + pd.Timedelta(hours=5, minutes=30)
    
    print(f"✅ Loaded {len(df):,} candles")
    
    # Filter market hours
    market_hours = df[
        (df['datetime_ist'].dt.time >= time(9, 15)) & 
        (df['datetime_ist'].dt.time <= time(15, 25))
    ].copy()
    
    print(f"🕒 Market hours data: {len(market_hours):,} candles")
    
    # Create enhanced features
    enhanced_data, _ = create_simple_features(market_hours)
    
    # Generate predictions for each trading date
    predictions = {}
    trading_dates = sorted(enhanced_data['datetime_ist'].dt.date.unique())
    lookback_days = config['lookback_days']
    
    print(f"🔮 Generating predictions for {len(trading_dates)} trading dates...")
    
    successful_predictions = 0
    
    for date in trading_dates[lookback_days:]:  # Skip early dates
        date_str = date.strftime('%Y-%m-%d')
        
        try:
            # Get historical data for sequence
            end_idx = enhanced_data[enhanced_data['datetime_ist'].dt.date <= date].index[-1]
            start_idx = max(0, end_idx - lookback_days + 1)
            
            sequence_data = enhanced_data.iloc[start_idx:end_idx+1]
            if len(sequence_data) < lookback_days:
                continue
            
            # Extract features
            sequence_features = sequence_data[feature_cols].values[-lookback_days:]
            
            # Scale features
            sequence_scaled = scaler.transform(sequence_features.reshape(-1, len(feature_cols))).reshape(sequence_features.shape)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0)  # Add batch dimension
            
            # Generate prediction
            with torch.no_grad():
                pred = model(X_tensor)
                predicted_ohlc = pred[0].numpy()
            
            # Get last actual values for context
            last_day_data = enhanced_data[enhanced_data['datetime_ist'].dt.date == date]
            if not last_day_data.empty:
                last_actual = {
                    'Open': float(last_day_data['open'].iloc[-1]),
                    'High': float(last_day_data['high'].max()),
                    'Low': float(last_day_data['low'].min()),
                    'Close': float(last_day_data['close'].iloc[-1])
                }
            else:
                last_actual = {'Open': 0, 'High': 0, 'Low': 0, 'Close': 0}
            
            # Store prediction
            predictions[date_str] = {
                'prediction_date': date_str,
                'last_actual': last_actual,
                'predicted': {
                    'Open': float(predicted_ohlc[0]),
                    'High': float(predicted_ohlc[1]),
                    'Low': float(predicted_ohlc[2]),
                    'Close': float(predicted_ohlc[3])
                },
                'model_version': 'v4_minimal'
            }
            
            successful_predictions += 1
            
            if successful_predictions % 100 == 0:
                print(f"✅ Generated {successful_predictions} predictions...")
        
        except Exception as e:
            continue
    
    print(f"✅ Successfully generated {successful_predictions} predictions")
    
    # Save predictions
    output_file = 'data/predictions/backtest_predictions_v4_minimal.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"📁 Predictions saved: {output_file}")
    
    # Show sample predictions
    sample_dates = list(predictions.keys())[-5:]  # Last 5 predictions
    print(f"\n🔮 Sample predictions (last 5 dates):")
    for date in sample_dates:
        pred = predictions[date]
        print(f"  {date}: O:{pred['predicted']['Open']:.2f} H:{pred['predicted']['High']:.2f} L:{pred['predicted']['Low']:.2f} C:{pred['predicted']['Close']:.2f}")
    
    print(f"\n🎉 Enhanced v4 predictions generation completed!")
    print(f"📊 Total predictions: {len(predictions)}")
    print(f"📅 Date range: {min(predictions.keys())} to {max(predictions.keys())}")
    
    return predictions


if __name__ == "__main__":
    generate_v4_predictions()