"""
Connect Real AI Predictions to Professional Backtrader Strategy
Integrates your trained LSTM model with professional backtesting
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
import torch
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.enhanced_lstm import EnhancedStockLSTM
from src.preprocessing.data_preprocessor import StockDataPreprocessor


def generate_daily_predictions_for_backtest(start_date='2023-08-31', end_date='2025-08-31'):
    """
    Generate daily AI predictions for the backtest period
    This simulates having predictions available before market open
    """
    
    print("🧠 Generating AI predictions for 2-YEAR period...")
    print("📅 Period: August 2023 to August 2025 (2 full years)")
    
    try:
        # Load the trained model and metadata
        model_path = project_root / "models/enhanced_stock_lstm.pth"
        metadata_path = project_root / "models/model_metadata.json"
        
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            print("💡 Please train the model first by running: python src/models/enhanced_lstm.py")
            return None
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model = EnhancedStockLSTM(
            input_size=metadata['input_size'],
            hidden_size=metadata['hidden_size'],
            num_layers=metadata['num_layers'],
            output_size=metadata['output_size']
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Setup preprocessor
        preprocessor = StockDataPreprocessor(
            lookback_days=metadata['lookback_days'],
            prediction_days=1,
            scale_method='minmax'
        )
        
        # Load latest data
        data_dir = project_root / "data/raw"
        data_files = list(data_dir.glob("RELIANCE_NSE_*.csv"))
        if not data_files:
            print(f"❌ No data files found in {data_dir}")
            return None
        
        # Get the most recent data file
        data_file = max(data_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        print(f"📊 Loaded {len(df)} days of data from {data_file.name}")
        print(f"📅 Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        # Generate predictions for each day in backtest period
        predictions = {}
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Use actual historical data for the period we're backtesting
        # This simulates making predictions day by day
        
        for i, date in enumerate(pd.date_range(start_dt, end_dt, freq='D')):
            
            # Skip weekends
            if date.weekday() >= 5:
                continue
            
            date_str = date.strftime('%Y-%m-%d')
            
            try:
                # Use data up to the previous day for prediction
                historical_cutoff = date - timedelta(days=1)
                
                # Fix timezone issues
                if df.index.tz is not None:
                    historical_cutoff = historical_cutoff.tz_localize(df.index.tz)
                
                available_data = df[df.index <= historical_cutoff]
                
                if len(available_data) < metadata['lookback_days']:
                    print(f"⚠️  Not enough data for {date_str}")
                    continue
                
                # Prepare data for prediction
                recent_data = available_data.tail(80)  # Extra for technical indicators
                enhanced_data = preprocessor.add_technical_indicators(recent_data)
                feature_data = preprocessor.prepare_features(enhanced_data)
                feature_data = feature_data.dropna()
                
                if len(feature_data) < metadata['lookback_days']:
                    continue
                
                # Fit scaler and prepare input
                preprocessor.fit_scaler(feature_data)
                last_window = feature_data.tail(preprocessor.lookback_days)
                scaled_window = preprocessor.transform_data(last_window)
                
                # Make prediction
                input_tensor = torch.tensor(scaled_window.values, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model(input_tensor)
                    prediction = prediction.cpu().numpy()
                
                # Convert back to original scale
                prediction_orig = preprocessor.inverse_transform_targets(prediction)[0]
                predicted_close = prediction_orig[3]  # Close price
                
                # Get current close for confidence calculation
                current_close = available_data['Close'].iloc[-1]
                predicted_change = abs((predicted_close - current_close) / current_close)
                
                # Calculate confidence (higher confidence for smaller changes)
                confidence = max(0.1, min(0.9, 1.0 - (predicted_change * 3)))
                
                predictions[date_str] = {
                    'predicted_close': float(predicted_close),
                    'confidence': float(confidence),
                    'actual_close_prev_day': float(current_close),
                    'predicted_ohlc': prediction_orig.tolist()
                }
                
                if i % 20 == 0:  # Print every 20th prediction to avoid spam
                    print(f"📅 {date_str}: Pred: ₹{predicted_close:.2f}, "
                          f"Conf: {confidence:.3f}, Prev: ₹{current_close:.2f}")
                
            except Exception as e:
                print(f"⚠️  Error predicting for {date_str}: {e}")
                continue
        
        print(f"✅ Generated {len(predictions)} predictions")
        
        # Save predictions for backtest
        predictions_file = project_root / 'data/predictions/backtest_predictions.pkl'
        predictions_file.parent.mkdir(exist_ok=True)
        joblib.dump(predictions, predictions_file)
        print(f"💾 Saved predictions to {predictions_file}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Failed to generate predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_backtest_predictions():
    """Load pre-generated backtest predictions"""
    
    try:
        predictions_file = project_root / 'data/backtest_predictions.pkl'
        if predictions_file.exists():
            predictions = joblib.load(predictions_file)
            print(f"📊 Loaded {len(predictions)} pre-generated predictions")
            return predictions
        else:
            print("🔄 No pre-generated predictions found, creating new ones...")
            return generate_daily_predictions_for_backtest()
    
    except Exception as e:
        print(f"❌ Failed to load predictions: {e}")
        return None


if __name__ == "__main__":
    print("🧠 AI Prediction Generator for 2-YEAR Period")
    print("="*60)
    
    # Generate predictions for 2 YEARS (Aug 2023 - Aug 2025)
    predictions = generate_daily_predictions_for_backtest()
    
    if predictions:
        print(f"\n📊 2-YEAR PREDICTION SUMMARY:")
        print(f"Total predictions: {len(predictions)}")
        print(f"Expected trading days: ~500 days")
        
        # Show sample predictions
        sample_dates = list(predictions.keys())
        print(f"\nSample predictions:")
        for i in [0, len(sample_dates)//4, len(sample_dates)//2, len(sample_dates)*3//4, -1]:
            if i < len(sample_dates):
                date = sample_dates[i]
                pred = predictions[date]
                print(f"  {date}: ₹{pred['predicted_close']:.2f} (conf: {pred['confidence']:.3f})")
        
        print("✅ Ready for 2-YEAR backtesting with proper validation!")
    else:
        print("❌ Failed to generate predictions")
