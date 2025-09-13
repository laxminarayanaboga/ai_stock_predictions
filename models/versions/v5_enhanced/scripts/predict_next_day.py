"""
Quick Next Day Prediction Script for V5 Model
Provides simple interface for getting next day OHLC predictions
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from models.versions.v5_enhanced.model_v5 import create_enhanced_model_v5
from models.versions.v5_enhanced.scripts.data_utils import (
    load_reliance_data, create_enhanced_features, prepare_sequences
)


class V5Predictor:
    """Quick predictor for V5 model"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or "/Users/bogalaxminarayana/myGit/ai_stock_predictions/models/versions/v5_enhanced/best_model_v5.pth"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.config = None
        
        print(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load trained model and metadata"""
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}")
            print("Please train the model first using train_v5.py")
            return False
        
        print(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.feature_cols = checkpoint['feature_cols']
        self.scaler = checkpoint['scaler']
        
        # Create and load model
        self.model = create_enhanced_model_v5(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Model trained on {len(self.feature_cols)} features")
        print(f"Lookback days: {self.config['lookback_days']}")
        
        return True
    
    def get_latest_data(self, days=30):
        """Get latest data for prediction"""
        df = load_reliance_data()
        if df is None:
            return None
        
        # Get enhanced features
        df_features = create_enhanced_features(df)
        
        # Get the most recent data
        recent_data = df_features.tail(days)
        
        return recent_data
    
    def predict_next_day(self, confidence_samples=10):
        """Predict next day OHLC with confidence estimation"""
        
        if self.model is None:
            print("Model not loaded. Cannot make predictions.")
            return None
        
        # Get recent data
        recent_data = self.get_latest_data(days=self.config['lookback_days'] + 10)
        if recent_data is None:
            print("Could not load recent data")
            return None
        
        # Prepare sequence
        target_cols = ['open', 'high', 'low', 'close']
        sequences, _, feature_cols = prepare_sequences(
            recent_data, 
            self.config['lookback_days'], 
            target_cols
        )
        
        if len(sequences) == 0:
            print("Not enough data for prediction")
            return None
        
        # Use the most recent sequence
        latest_sequence = sequences[-1:] # Shape: (1, lookback_days, features)
        
        # Scale the sequence
        original_shape = latest_sequence.shape
        sequence_reshaped = latest_sequence.reshape(-1, latest_sequence.shape[-1])
        sequence_scaled = self.scaler.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(original_shape)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_scaled).to(self.device)
        
        # Get predictions with uncertainty
        predictions = []
        confidences = []
        
        self.model.eval()
        with torch.no_grad():
            # Standard prediction
            pred = self.model(sequence_tensor)
            predictions.append(pred.cpu().numpy())
            
            # Get confidence if model supports it
            try:
                pred_conf, conf = self.model(sequence_tensor, return_confidence=True)
                confidences.append(conf.cpu().numpy())
            except:
                confidences.append(np.array([[0.5]]))  # Default confidence
            
            # Monte Carlo dropout for uncertainty estimation
            mean_pred, std_pred = self.model.predict_with_uncertainty(
                sequence_tensor, num_samples=confidence_samples
            )
        
        # Process results
        prediction = predictions[0][0]  # Shape: (4,) for OHLC
        confidence = confidences[0][0][0] if confidences[0].size > 0 else 0.5
        uncertainty = std_pred.cpu().numpy()[0]  # Uncertainty for each OHLC
        
        # Get last actual prices for reference
        last_actual = recent_data[target_cols].iloc[-1].values
        
        # Calculate prediction changes
        pred_changes = (prediction - last_actual) / last_actual * 100
        
        # Create result dictionary
        result = {
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': 'v5_enhanced',
            'last_actual': {
                'Open': float(last_actual[0]),
                'High': float(last_actual[1]),
                'Low': float(last_actual[2]),
                'Close': float(last_actual[3])
            },
            'predicted': {
                'Open': float(prediction[0]),
                'High': float(prediction[1]),
                'Low': float(prediction[2]),
                'Close': float(prediction[3])
            },
            'prediction_changes': {
                'Open': float(pred_changes[0]),
                'High': float(pred_changes[1]),
                'Low': float(pred_changes[2]),
                'Close': float(pred_changes[3])
            },
            'uncertainty': {
                'Open': float(uncertainty[0]),
                'High': float(uncertainty[1]),
                'Low': float(uncertainty[2]),
                'Close': float(uncertainty[3])
            },
            'confidence': float(confidence),
            'uncertainty_score': float(np.mean(uncertainty)),
            'prediction_quality': 'High' if confidence > 0.7 and np.mean(uncertainty) < 10 else 
                                 'Medium' if confidence > 0.5 and np.mean(uncertainty) < 20 else 'Low'
        }
        
        return result
    
    def save_prediction(self, prediction, filename=None):
        """Save prediction to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"prediction_v5_{timestamp}.json"
        
        predictions_dir = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/models/versions/v5_enhanced/predictions"
        os.makedirs(predictions_dir, exist_ok=True)
        
        filepath = os.path.join(predictions_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(prediction, f, indent=2)
        
        print(f"Prediction saved to: {filepath}")
        return filepath
    
    def print_prediction(self, prediction):
        """Print prediction in a formatted way"""
        print("\n" + "="*60)
        print("RELIANCE NEXT DAY PREDICTION - V5 ENHANCED MODEL")
        print("="*60)
        
        print(f"Prediction Date: {prediction['prediction_date']}")
        print(f"Prediction Time: {prediction['prediction_time']}")
        print(f"Model Version: {prediction['model_version']}")
        print(f"Confidence: {prediction['confidence']:.1%}")
        print(f"Prediction Quality: {prediction['prediction_quality']}")
        
        print("\nLAST ACTUAL PRICES:")
        for key, value in prediction['last_actual'].items():
            print(f"  {key}: ₹{value:.2f}")
        
        print("\nPREDICTED PRICES:")
        for key, value in prediction['predicted'].items():
            change = prediction['prediction_changes'][key]
            uncertainty = prediction['uncertainty'][key]
            direction = "↑" if change > 0 else "↓" if change < 0 else "→"
            print(f"  {key}: ₹{value:.2f} ({direction} {change:+.2f}%) ±{uncertainty:.2f}")
        
        print(f"\nOverall Uncertainty Score: {prediction['uncertainty_score']:.2f}")
        print("="*60)


def main():
    """Main prediction function"""
    print("V5 Enhanced Model - Next Day Prediction")
    print("Loading model and making prediction...")
    
    # Create predictor
    predictor = V5Predictor()
    
    # Make prediction
    prediction = predictor.predict_next_day(confidence_samples=15)
    
    if prediction is None:
        print("Failed to make prediction")
        return
    
    # Print prediction
    predictor.print_prediction(prediction)
    
    # Save prediction
    filepath = predictor.save_prediction(prediction)
    
    # Save as latest prediction
    latest_path = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/models/versions/v5_enhanced/latest_prediction_v5.json"
    with open(latest_path, 'w') as f:
        json.dump(prediction, f, indent=2)
    
    print(f"\nPrediction also saved as: {latest_path}")


if __name__ == "__main__":
    main()