"""
Model Evaluation and Prediction Module
Provides comprehensive evaluation metrics and prediction capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.enhanced_lstm import EnhancedStockLSTM
from src.preprocessing.data_preprocessor import StockDataPreprocessor


class ModelEvaluator:
    def __init__(self, model_path="models/enhanced_stock_lstm.pth", metadata_path="models/model_metadata.json"):
        """
        Initialize model evaluator
        
        Args:
            model_path (str): Path to saved model
            metadata_path (str): Path to model metadata
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.preprocessor = None
        self.metadata = None
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def load_model(self):
        """Load the trained model"""
        # Initialize model with saved parameters
        self.model = EnhancedStockLSTM(
            input_size=self.metadata['input_size'],
            hidden_size=self.metadata['hidden_size'],
            num_layers=self.metadata['num_layers'],
            output_size=self.metadata['output_size']
        )
        
        # Load trained weights
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.eval()
        
        print("Model loaded successfully!")
        return self.model
    
    def setup_preprocessor(self):
        """Setup preprocessor with same configuration"""
        self.preprocessor = StockDataPreprocessor(
            lookback_days=self.metadata['lookback_days'],
            prediction_days=1,
            scale_method='minmax'
        )
        return self.preprocessor
    
    def calculate_metrics(self, predictions, actuals):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            predictions (np.array): Model predictions
            actuals (np.array): Actual values
            
        Returns:
            dict: Dictionary of metrics
        """
        # Basic regression metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        # Percentage errors
        percentage_errors = np.abs((predictions - actuals) / actuals) * 100
        mape = np.mean(percentage_errors)
        
        # R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Directional accuracy (for Close price)
        actual_direction = np.sign(np.diff(actuals[:, 3]))  # Close price changes
        pred_direction = np.sign(np.diff(predictions[:, 3]))
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }
    
    def plot_predictions(self, predictions, actuals, save_path="evaluation_plots.png"):
        """
        Create comprehensive prediction plots
        
        Args:
            predictions (np.array): Model predictions
            actuals (np.array): Actual values
            save_path (str): Path to save plots
        """
        ohlc_names = ['Open', 'High', 'Low', 'Close']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Predictions vs Actual Values', fontsize=16)
        
        for i, (ax, name) in enumerate(zip(axes.flat, ohlc_names)):
            # Plot predictions vs actuals
            ax.plot(actuals[:, i], label=f'Actual {name}', alpha=0.7)
            ax.plot(predictions[:, i], label=f'Predicted {name}', alpha=0.7)
            ax.set_title(f'{name} Price Prediction')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price (₹)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Scatter plots for correlation
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Prediction Correlation Analysis', fontsize=16)
        
        for i, (ax, name) in enumerate(zip(axes.flat, ohlc_names)):
            ax.scatter(actuals[:, i], predictions[:, i], alpha=0.6)
            ax.plot([actuals[:, i].min(), actuals[:, i].max()], 
                   [actuals[:, i].min(), actuals[:, i].max()], 'r--', lw=2)
            ax.set_xlabel(f'Actual {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name} Correlation')
            ax.grid(True, alpha=0.3)
            
            # Calculate and display R²
            r2 = np.corrcoef(actuals[:, i], predictions[:, i])[0, 1] ** 2
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_correlation.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_errors(self, predictions, actuals):
        """
        Analyze prediction errors in detail
        
        Args:
            predictions (np.array): Model predictions
            actuals (np.array): Actual values
        """
        errors = predictions - actuals
        percentage_errors = (errors / actuals) * 100
        ohlc_names = ['Open', 'High', 'Low', 'Close']
        
        print("=== ERROR ANALYSIS ===")
        print(f"{'Metric':<15} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10}")
        print("-" * 60)
        
        # Mean errors
        mean_errors = np.mean(errors, axis=0)
        print(f"{'Mean Error':<15}", end="")
        for error in mean_errors:
            print(f"{error:>9.2f} ", end="")
        print()
        
        # Standard deviation of errors
        std_errors = np.std(errors, axis=0)
        print(f"{'Std Error':<15}", end="")
        for error in std_errors:
            print(f"{error:>9.2f} ", end="")
        print()
        
        # Mean absolute percentage errors
        mape_values = np.mean(np.abs(percentage_errors), axis=0)
        print(f"{'MAPE (%)':<15}", end="")
        for mape in mape_values:
            print(f"{mape:>9.2f} ", end="")
        print()
        
        # 95th percentile errors
        p95_errors = np.percentile(np.abs(errors), 95, axis=0)
        print(f"{'95th Pct Err':<15}", end="")
        for error in p95_errors:
            print(f"{error:>9.2f} ", end="")
        print()
        
        print("-" * 60)
        
        # Error distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Error Distribution Analysis', fontsize=16)
        
        for i, (ax, name) in enumerate(zip(axes.flat, ohlc_names)):
            ax.hist(percentage_errors[:, i], bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{name} Percentage Error Distribution')
            ax.set_xlabel('Percentage Error (%)')
            ax.set_ylabel('Frequency')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_pe = np.mean(percentage_errors[:, i])
            std_pe = np.std(percentage_errors[:, i])
            ax.text(0.02, 0.98, f'Mean: {mean_pe:.2f}%\nStd: {std_pe:.2f}%', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('models/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


class NextDayPredictor:
    def __init__(self, model_evaluator):
        """
        Initialize next day predictor
        
        Args:
            model_evaluator: ModelEvaluator instance
        """
        self.evaluator = model_evaluator
        self.model = model_evaluator.model
        self.preprocessor = model_evaluator.preprocessor
    
    def predict_next_day(self, recent_data_days=30):
        """
        Predict next day's OHLC values
        
        Args:
            recent_data_days (int): Number of recent days to use for prediction
            
        Returns:
            dict: Next day predictions
        """
        # Load latest data
        data_file = Path("data/raw").glob("RELIANCE_NSE_*.csv")
        data_file = max(data_file, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        # Get recent data
        recent_data = df.tail(recent_data_days + 50)  # Extra for technical indicators
        
        # Add technical indicators
        enhanced_data = self.preprocessor.add_technical_indicators(recent_data)
        
        # Prepare features
        feature_data = self.preprocessor.prepare_features(enhanced_data)
        feature_data = feature_data.dropna()
        
        # Take last 30 days (lookback window)
        last_window = feature_data.tail(self.preprocessor.lookback_days)
        
        # Fit scaler on recent data (simple approach for demo)
        self.preprocessor.fit_scaler(feature_data)
        
        # Scale the data
        scaled_window = self.preprocessor.transform_data(last_window)
        
        # Convert to tensor
        input_tensor = torch.tensor(scaled_window.values, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = prediction.cpu().numpy()
        
        # Convert back to original scale
        prediction_orig = self.preprocessor.inverse_transform_targets(prediction)
        
        # Get last actual values for comparison
        last_actual = df.iloc[-1][['Open', 'High', 'Low', 'Close']].values
        
        # Calculate predicted changes
        predicted_values = prediction_orig[0]
        changes = predicted_values - last_actual
        change_percentages = (changes / last_actual) * 100
        
        result = {
            'prediction_date': (df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d'),
            'last_actual': {
                'Open': float(last_actual[0]),
                'High': float(last_actual[1]),
                'Low': float(last_actual[2]),
                'Close': float(last_actual[3])
            },
            'predicted': {
                'Open': float(predicted_values[0]),
                'High': float(predicted_values[1]),
                'Low': float(predicted_values[2]),
                'Close': float(predicted_values[3])
            },
            'changes': {
                'Open': float(changes[0]),
                'High': float(changes[1]),
                'Low': float(changes[2]),
                'Close': float(changes[3])
            },
            'change_percentages': {
                'Open': float(change_percentages[0]),
                'High': float(change_percentages[1]),
                'Low': float(change_percentages[2]),
                'Close': float(change_percentages[3])
            }
        }
        
        return result
    
    def print_prediction(self, prediction):
        """Print formatted prediction"""
        print("=== NEXT DAY PREDICTION ===")
        print(f"Prediction Date: {prediction['prediction_date']}")
        print()
        
        print(f"{'Metric':<8} {'Last Actual':<12} {'Predicted':<12} {'Change':<10} {'Change %':<10}")
        print("-" * 62)
        
        for metric in ['Open', 'High', 'Low', 'Close']:
            last = prediction['last_actual'][metric]
            pred = prediction['predicted'][metric]
            change = prediction['changes'][metric]
            change_pct = prediction['change_percentages'][metric]
            
            print(f"{metric:<8} ₹{last:<11.2f} ₹{pred:<11.2f} {change:>9.2f} {change_pct:>9.2f}%")
        
        print("-" * 62)
        
        # Overall sentiment
        close_change = prediction['change_percentages']['Close']
        if close_change > 1:
            sentiment = "🟢 BULLISH"
        elif close_change < -1:
            sentiment = "🔴 BEARISH"
        else:
            sentiment = "🟡 NEUTRAL"
        
        print(f"Overall Sentiment: {sentiment}")
        print(f"Expected Close Change: {close_change:.2f}%")


def main():
    """Main evaluation function"""
    print("=== Model Evaluation and Prediction ===")
    
    # Check if model exists
    if not Path("models/enhanced_stock_lstm.pth").exists():
        print("Model not found. Please train the model first using enhanced_lstm.py")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    evaluator.load_model()
    evaluator.setup_preprocessor()
    
    # Load test data and make predictions (simplified for demo)
    print("Loading test data...")
    data_file = Path("data/raw").glob("RELIANCE_NSE_*.csv")
    data_file = max(data_file, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Use last 500 days as test data
    test_data = df.tail(500)
    processed_data = evaluator.preprocessor.prepare_data_for_training(test_data, test_size=0.2, val_size=0.1)
    
    # Make predictions on test set
    X_test = torch.tensor(processed_data['X_test'], dtype=torch.float32)
    y_test = processed_data['y_test']
    
    evaluator.model.eval()
    with torch.no_grad():
        predictions = evaluator.model(X_test).numpy()
    
    # Convert back to original scale
    predictions_orig = evaluator.preprocessor.inverse_transform_targets(predictions)
    actuals_orig = evaluator.preprocessor.inverse_transform_targets(y_test)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(predictions_orig, actuals_orig)
    
    print("\n=== MODEL PERFORMANCE METRICS ===")
    for metric, value in metrics.items():
        if metric == 'Directional_Accuracy':
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.4f}")
    
    # Plot predictions
    evaluator.plot_predictions(predictions_orig, actuals_orig, "models/evaluation_plots.png")
    
    # Analyze errors
    evaluator.analyze_errors(predictions_orig, actuals_orig)
    
    # Make next day prediction
    print("\n" + "="*50)
    predictor = NextDayPredictor(evaluator)
    next_day_pred = predictor.predict_next_day()
    predictor.print_prediction(next_day_pred)
    
    # Save prediction
    with open('models/latest_prediction.json', 'w') as f:
        json.dump(next_day_pred, f, indent=2)
    
    print(f"\nLatest prediction saved to: models/latest_prediction.json")
    
    return evaluator, predictor


if __name__ == "__main__":
    main()
