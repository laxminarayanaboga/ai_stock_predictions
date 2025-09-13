"""
V5 Enhanced Model Prediction Script
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from models.versions.v5_enhanced.model_v5 import EnhancedLSTMV5, create_enhanced_model_v5
from models.versions.v5_enhanced.scripts.data_utils import load_reliance_data, create_enhanced_features, prepare_sequences


class V5Predictor:
    """V5 Enhanced model predictor"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.config = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            # Load with weights_only=False for compatibility with sklearn objects
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract model configuration
            self.config = checkpoint['config']
            self.feature_cols = checkpoint['feature_cols'] 
            self.scaler = checkpoint['scaler']
            
            # Create model using the same function as training
            self.model = create_enhanced_model_v5(self.config)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully!")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Features: {len(self.feature_cols)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def prepare_data(self):
        """Prepare data for prediction"""
        # Load data with intraday features
        data_result = load_reliance_data(use_intraday=True)
        if data_result is None:
            raise ValueError("Could not load data")
        
        if isinstance(data_result, tuple):
            df, intraday_df = data_result
            print("Using both daily and 10-minute intraday data")
        else:
            df = data_result
            intraday_df = None
            print("Using daily data only")
        
        # Create enhanced features
        df_features = create_enhanced_features(df, intraday_df)
        
        # Prepare sequences
        lookback = self.config['lookback_days']
        target_cols = ['open', 'high', 'low', 'close']
        
        sequences, targets, feature_cols = prepare_sequences(df_features, lookback, target_cols)
        
        # Scale features using the saved scaler
        X_reshaped = sequences.reshape(-1, sequences.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(sequences.shape)
        
        print(f"Prepared {len(sequences)} sequences")
        print(f"Data date range: {df.index[0]} to {df.index[-1]}")
        
        return X_scaled, targets, df.index[lookback:]
    
    def predict(self, X, return_confidence=True):
        """Make predictions"""
        self.model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for i in range(0, len(X), 32):  # Batch size 32
                batch = X[i:i+32]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                if return_confidence:
                    pred, confidence = self.model(batch_tensor, return_confidence=True)
                    confidences.extend(confidence.cpu().numpy())
                else:
                    pred = self.model(batch_tensor, return_confidence=False)
                
                predictions.extend(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        confidences = np.array(confidences) if confidences else None
        
        return predictions, confidences
    
    def evaluate_predictions(self, predictions, actual, dates):
        """Evaluate prediction performance"""
        # Calculate metrics
        mse = np.mean((predictions - actual) ** 2)
        mae = np.mean(np.abs(predictions - actual))
        
        # Directional accuracy
        pred_direction = np.sign(np.diff(predictions[:, 3]))  # Close price direction
        actual_direction = np.sign(np.diff(actual[:, 3]))
        direction_acc = np.mean(pred_direction == actual_direction)
        
        # Correlation
        correlation = np.corrcoef(predictions.flatten(), actual.flatten())[0, 1]
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'direction_accuracy': direction_acc,
            'correlation': correlation,
            'total_predictions': len(predictions)
        }
        
        print("\n" + "="*50)
        print("V5 ENHANCED MODEL PERFORMANCE")
        print("="*50)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Directional Accuracy: {direction_acc:.4f} ({direction_acc*100:.1f}%)")
        print(f"Correlation: {correlation:.4f}")
        print(f"Total Predictions: {len(predictions)}")
        print("="*50)
        
        return metrics
    
    def plot_predictions(self, predictions, actual, dates, save_path=None):
        """Plot predictions vs actual"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        target_names = ['Open', 'High', 'Low', 'Close']
        
        # Recent data (last 100 days)
        recent_idx = -100
        recent_dates = dates[recent_idx:]
        recent_pred = predictions[recent_idx:]
        recent_actual = actual[recent_idx:]
        
        for i, name in enumerate(target_names):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            ax.plot(recent_dates, recent_actual[:, i], label=f'Actual {name}', alpha=0.7)
            ax.plot(recent_dates, recent_pred[:, i], label=f'Predicted {name}', alpha=0.7)
            ax.set_title(f'{name} Price Predictions (Last 100 Days)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def save_predictions(self, predictions, actual, dates, confidences=None):
        """Save predictions to file"""
        # Create results directory
        results_dir = os.path.join(os.path.dirname(self.model_path), 'predictions')
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare data for saving
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'V5_Enhanced',
            'predictions': predictions.tolist(),
            'actual': actual.tolist(),
            'dates': [str(d) for d in dates],
            'target_columns': ['open', 'high', 'low', 'close']
        }
        
        if confidences is not None:
            results['confidences'] = confidences.tolist()
        
        # Save to JSON
        json_path = os.path.join(results_dir, f'v5_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Predictions saved to: {json_path}")
        return json_path


def main():
    """Main prediction function"""
    # Model path
    model_dir = os.path.dirname(__file__).replace('/scripts', '')
    model_path = os.path.join(model_dir, 'best_model_v5.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return
    
    # Create predictor
    predictor = V5Predictor(model_path)
    
    # Load model
    if not predictor.load_model():
        return
    
    # Prepare data
    print("Preparing data...")
    X, targets, dates = predictor.prepare_data()
    
    # Make predictions
    print("Making predictions...")
    predictions, confidences = predictor.predict(X, return_confidence=True)
    
    # Evaluate
    metrics = predictor.evaluate_predictions(predictions, targets, dates)
    
    # Plot results
    plot_path = os.path.join(os.path.dirname(model_path), 'v5_predictions_plot.png')
    predictor.plot_predictions(predictions, targets, dates, save_path=plot_path)
    
    # Save predictions
    predictor.save_predictions(predictions, targets, dates, confidences)
    
    # Compare with previous models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print("V5 Enhanced Results:")
    print(f"  Correlation: {metrics['correlation']:.4f}")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.1%}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print("\nPrevious Model Results (for comparison):")
    print("  V2 Attention: Correlation ~0.576")
    print("  Target: >0.65 correlation")
    
    if metrics['correlation'] > 0.65:
        print("🎉 TARGET ACHIEVED! V5 Enhanced exceeds 0.65 correlation!")
    elif metrics['correlation'] > 0.576:
        print("✅ IMPROVEMENT! V5 Enhanced beats V2 model!")
    else:
        print("📈 Room for improvement compared to V2 baseline")
    print("="*50)


if __name__ == "__main__":
    main()