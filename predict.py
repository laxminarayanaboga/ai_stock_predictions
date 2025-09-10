"""
Simple Inference Script for Quick Predictions
Can be used to quickly predict next day's OHLC for Reliance
"""

import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.enhanced_lstm import EnhancedStockLSTM
from src.preprocessing.data_preprocessor import StockDataPreprocessor


def quick_predict():
    """Quick prediction function"""
    print("🔮 Reliance Stock Price Predictor")
    print("=" * 40)
    
    # Check if model exists
    model_path = "models/enhanced_stock_lstm.pth"
    metadata_path = "models/model_metadata.json"
    
    if not Path(model_path).exists():
        print("❌ Model not found! Please train the model first.")
        print("Run: python src/models/enhanced_lstm.py")
        return None
    
    try:
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
        data_file = Path("data/raw").glob("RELIANCE_NSE_*.csv")
        data_file = max(data_file, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        print(f"📊 Using data up to: {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"📈 Latest Close Price: ₹{df['Close'].iloc[-1]:.2f}")
        
        # Prepare data for prediction
        recent_data = df.tail(80)  # Extra for technical indicators
        enhanced_data = preprocessor.add_technical_indicators(recent_data)
        feature_data = preprocessor.prepare_features(enhanced_data)
        feature_data = feature_data.dropna()
        
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
        
        # Get current values
        current = df.iloc[-1][['Open', 'High', 'Low', 'Close']].values
        
        # Calculate changes
        changes = prediction_orig - current
        change_pcts = (changes / current) * 100
        
        print("\n🎯 NEXT DAY PREDICTION")
        print("-" * 40)
        print(f"{'Price':<8} {'Current':<10} {'Predicted':<10} {'Change':<8} {'%':<8}")
        print("-" * 40)
        
        labels = ['Open', 'High', 'Low', 'Close']
        for i, label in enumerate(labels):
            color = "🟢" if changes[i] > 0 else "🔴" if changes[i] < 0 else "⚪"
            print(f"{label:<8} ₹{current[i]:<9.2f} ₹{prediction_orig[i]:<9.2f} {changes[i]:>7.2f} {change_pcts[i]:>6.1f}% {color}")
        
        print("-" * 40)
        
        # Overall sentiment
        close_change = change_pcts[3]  # Close price change
        if close_change > 1:
            sentiment = "🚀 STRONG BUY"
        elif close_change > 0.5:
            sentiment = "🟢 BUY"
        elif close_change > -0.5:
            sentiment = "⚪ HOLD"
        elif close_change > -1:
            sentiment = "🔴 SELL"
        else:
            sentiment = "📉 STRONG SELL"
        
        print(f"💡 Sentiment: {sentiment}")
        print(f"📊 Expected Close: ₹{prediction_orig[3]:.2f} ({change_pcts[3]:+.1f}%)")
        
        # Model info
        print(f"\n📋 Model Info:")
        print(f"   • Trained on: {metadata['training_date'][:10]}")
        print(f"   • Test RMSE: {metadata['test_rmse']:.2f}")
        print(f"   • Features: {len(metadata['features'])}")
        
        return {
            'predicted_ohlc': prediction_orig.tolist(),
            'current_ohlc': current.tolist(),
            'changes': changes.tolist(),
            'change_percentages': change_pcts.tolist(),
            'sentiment': sentiment
        }
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None


if __name__ == "__main__":
    result = quick_predict()
    
    if result:
        print("\n✅ Prediction completed successfully!")
        print("💡 Remember: This is for educational purposes only.")
        print("📊 Always do your own research before investing.")
    else:
        print("\n❌ Prediction failed.")
