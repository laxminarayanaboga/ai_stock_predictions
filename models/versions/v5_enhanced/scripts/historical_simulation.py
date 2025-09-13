"""
Historical Simulation Script for V5 Model
Comprehensive backtesting and performance analysis
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from models.versions.v5_enhanced.model_v5 import create_enhanced_model_v5
from models.versions.v5_enhanced.scripts.data_utils import (
    load_reliance_data, create_enhanced_features, prepare_sequences
)


class V5HistoricalSimulator:
    """Comprehensive historical simulation for V5 model"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or "/Users/bogalaxminarayana/myGit/ai_stock_predictions/models/versions/v5_enhanced/best_model_v5.pth"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.config = None
        self.results_dir = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/models/versions/v5_enhanced/analysis"
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load trained model and metadata"""
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}")
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
        return True
    
    def prepare_simulation_data(self, start_date=None, end_date=None):
        """Prepare data for historical simulation"""
        df = load_reliance_data()
        if df is None:
            return None, None, None
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range if specified
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        print(f"Simulation period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total days: {len(df)}")
        
        # Create enhanced features
        df_features = create_enhanced_features(df)
        
        # Prepare sequences
        target_cols = ['open', 'high', 'low', 'close']
        sequences, targets, feature_cols = prepare_sequences(
            df_features, 
            self.config['lookback_days'], 
            target_cols
        )
        
        # Get corresponding dates for each sequence
        dates = df_features['timestamp'].iloc[self.config['lookback_days']:].values
        
        return sequences, targets, dates
    
    def run_historical_simulation(self, start_date=None, end_date=None, save_individual=False):
        """Run comprehensive historical simulation"""
        print("Starting historical simulation...")
        
        # Prepare data
        sequences, targets, dates = self.prepare_simulation_data(start_date, end_date)
        if sequences is None:
            print("Failed to prepare simulation data")
            return None
        
        print(f"Running simulation on {len(sequences)} data points")
        
        # Scale sequences
        original_shape = sequences.shape
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled = self.scaler.transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(original_shape)
        
        # Run predictions
        all_predictions = []
        all_confidences = []
        all_uncertainties = []
        
        batch_size = 64
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(sequences_scaled), batch_size):
                batch_end = min(i + batch_size, len(sequences_scaled))
                batch_sequences = sequences_scaled[i:batch_end]
                
                # Convert to tensor
                batch_tensor = torch.FloatTensor(batch_sequences).to(self.device)
                
                # Get predictions
                predictions = self.model(batch_tensor)
                all_predictions.append(predictions.cpu().numpy())
                
                # Get uncertainties using Monte Carlo dropout
                mean_pred, std_pred = self.model.predict_with_uncertainty(
                    batch_tensor, num_samples=5
                )
                all_uncertainties.append(std_pred.cpu().numpy())
                
                # Try to get confidence scores
                try:
                    _, conf = self.model(batch_tensor, return_confidence=True)
                    all_confidences.append(conf.cpu().numpy())
                except:
                    all_confidences.append(np.ones((len(batch_sequences), 1)) * 0.5)
        
        # Combine results
        predictions = np.concatenate(all_predictions)
        uncertainties = np.concatenate(all_uncertainties)
        confidences = np.concatenate(all_confidences)
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(targets, predictions, uncertainties, confidences)
        
        # Create detailed results
        results = {
            'simulation_info': {
                'model_version': 'v5_enhanced',
                'simulation_date': datetime.now().isoformat(),
                'start_date': dates[0].isoformat() if len(dates) > 0 else None,
                'end_date': dates[-1].isoformat() if len(dates) > 0 else None,
                'total_predictions': len(predictions),
                'model_config': self.config
            },
            'metrics': metrics,
            'daily_predictions': []
        }
        
        # Store individual predictions if requested
        if save_individual:
            for i, date in enumerate(dates):
                daily_result = {
                    'date': date.isoformat()[:10],
                    'actual': {
                        'Open': float(targets[i][0]),
                        'High': float(targets[i][1]),
                        'Low': float(targets[i][2]),
                        'Close': float(targets[i][3])
                    },
                    'predicted': {
                        'Open': float(predictions[i][0]),
                        'High': float(predictions[i][1]),
                        'Low': float(predictions[i][2]),
                        'Close': float(predictions[i][3])
                    },
                    'uncertainty': {
                        'Open': float(uncertainties[i][0]),
                        'High': float(uncertainties[i][1]),
                        'Low': float(uncertainties[i][2]),
                        'Close': float(uncertainties[i][3])
                    },
                    'confidence': float(confidences[i][0]),
                    'errors': {
                        'Open': float(abs(predictions[i][0] - targets[i][0])),
                        'High': float(abs(predictions[i][1] - targets[i][1])),
                        'Low': float(abs(predictions[i][2] - targets[i][2])),
                        'Close': float(abs(predictions[i][3] - targets[i][3]))
                    }
                }
                results['daily_predictions'].append(daily_result)
        
        print("Historical simulation completed!")
        return results
    
    def _calculate_comprehensive_metrics(self, targets, predictions, uncertainties, confidences):
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic regression metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        metrics['overall'] = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse)
        }
        
        # Per-target metrics (OHLC)
        target_names = ['Open', 'High', 'Low', 'Close']
        metrics['per_target'] = {}
        
        for i, name in enumerate(target_names):
            target_mse = mean_squared_error(targets[:, i], predictions[:, i])
            target_mae = mean_absolute_error(targets[:, i], predictions[:, i])
            target_rmse = np.sqrt(target_mse)
            
            # Correlation
            correlation = np.corrcoef(targets[:, i], predictions[:, i])[0, 1]
            
            # Percentage errors
            mape = np.mean(np.abs((targets[:, i] - predictions[:, i]) / targets[:, i])) * 100
            
            metrics['per_target'][name] = {
                'mse': float(target_mse),
                'mae': float(target_mae),
                'rmse': float(target_rmse),
                'correlation': float(correlation),
                'mape': float(mape)
            }
        
        # Direction accuracy (for Close price)
        actual_direction = np.sign(np.diff(targets[:, 3]))  # Close price changes
        pred_direction = np.sign(np.diff(predictions[:, 3]))
        direction_accuracy = np.mean(actual_direction == pred_direction)
        
        metrics['direction_accuracy'] = float(direction_accuracy)
        
        # Confidence and uncertainty analysis
        metrics['confidence_analysis'] = {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'mean_uncertainty': float(np.mean(uncertainties)),
            'std_uncertainty': float(np.std(uncertainties))
        }
        
        # Trading simulation metrics
        trading_metrics = self._calculate_trading_metrics(targets[:, 3], predictions[:, 3])
        metrics['trading'] = trading_metrics
        
        return metrics
    
    def _calculate_trading_metrics(self, actual_close, predicted_close):
        """Calculate trading-specific metrics"""
        # Simple trading strategy: buy if predicted > current, sell otherwise
        positions = []
        returns = []
        
        for i in range(1, len(predicted_close)):
            # Predict direction
            predicted_change = predicted_close[i] - actual_close[i-1]
            actual_change = actual_close[i] - actual_close[i-1]
            
            # Position: 1 for long, -1 for short, 0 for hold
            if predicted_change > 0:
                position = 1
            elif predicted_change < 0:
                position = -1
            else:
                position = 0
            
            positions.append(position)
            
            # Calculate return
            if position != 0:
                return_pct = (actual_change / actual_close[i-1]) * position
                returns.append(return_pct)
            else:
                returns.append(0)
        
        returns = np.array(returns)
        
        # Calculate trading metrics
        total_return = np.sum(returns)
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
        avg_return = np.mean(returns) if len(returns) > 0 else 0
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns)
        
        return {
            'total_return': float(total_return),
            'win_rate': float(win_rate),
            'avg_return': float(avg_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'num_trades': len(returns)
        }
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return float(np.min(drawdown))
    
    def create_analysis_plots(self, results):
        """Create comprehensive analysis plots"""
        if not results['daily_predictions']:
            print("No daily predictions available for plotting")
            return
        
        # Extract data for plotting
        dates = [pred['date'] for pred in results['daily_predictions']]
        actual_close = [pred['actual']['Close'] for pred in results['daily_predictions']]
        predicted_close = [pred['predicted']['Close'] for pred in results['daily_predictions']]
        errors_close = [pred['errors']['Close'] for pred in results['daily_predictions']]
        confidences = [pred['confidence'] for pred in results['daily_predictions']]
        
        # Convert dates to datetime
        dates = pd.to_datetime(dates)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Actual vs Predicted Close Prices
        axes[0, 0].plot(dates, actual_close, label='Actual', alpha=0.8, linewidth=1)
        axes[0, 0].plot(dates, predicted_close, label='Predicted', alpha=0.8, linewidth=1)
        axes[0, 0].set_title('Actual vs Predicted Close Prices (V5)', fontsize=14)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price (₹)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction Errors Over Time
        axes[0, 1].plot(dates, errors_close, alpha=0.7, color='red')
        axes[0, 1].set_title('Prediction Errors Over Time', fontsize=14)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Absolute Error (₹)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Confidence Scores Over Time
        axes[1, 0].plot(dates, confidences, alpha=0.7, color='green')
        axes[1, 0].set_title('Model Confidence Over Time', fontsize=14)
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Confidence Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Scatter Plot - Actual vs Predicted
        axes[1, 1].scatter(actual_close, predicted_close, alpha=0.6, s=1)
        
        # Add perfect prediction line
        min_val = min(min(actual_close), min(predicted_close))
        max_val = max(max(actual_close), max(predicted_close))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        axes[1, 1].set_title('Actual vs Predicted Scatter Plot', fontsize=14)
        axes[1, 1].set_xlabel('Actual Close (₹)')
        axes[1, 1].set_ylabel('Predicted Close (₹)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(actual_close, predicted_close)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, 'historical_simulation_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Analysis plots saved to: {plot_path}")
        
        return plot_path
    
    def save_results(self, results, filename=None):
        """Save simulation results"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'historical_simulation_v5_{timestamp}.json'
        
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def print_summary(self, results):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("V5 ENHANCED MODEL - HISTORICAL SIMULATION SUMMARY")
        print("="*80)
        
        metrics = results['metrics']
        sim_info = results['simulation_info']
        
        print(f"Simulation Period: {sim_info['start_date']} to {sim_info['end_date']}")
        print(f"Total Predictions: {sim_info['total_predictions']}")
        print(f"Model Version: {sim_info['model_version']}")
        
        print("\nOVERALL PERFORMANCE:")
        print(f"  RMSE: {metrics['overall']['rmse']:.2f}")
        print(f"  MAE: {metrics['overall']['mae']:.2f}")
        print(f"  Direction Accuracy: {metrics['direction_accuracy']:.1%}")
        
        print("\nPER-TARGET PERFORMANCE:")
        for target, target_metrics in metrics['per_target'].items():
            print(f"  {target}:")
            print(f"    MAE: {target_metrics['mae']:.2f}")
            print(f"    MAPE: {target_metrics['mape']:.2f}%")
            print(f"    Correlation: {target_metrics['correlation']:.3f}")
        
        print("\nCONFIDENCE ANALYSIS:")
        conf_metrics = metrics['confidence_analysis']
        print(f"  Mean Confidence: {conf_metrics['mean_confidence']:.1%}")
        print(f"  Mean Uncertainty: {conf_metrics['mean_uncertainty']:.2f}")
        
        print("\nTRADING SIMULATION:")
        trading_metrics = metrics['trading']
        print(f"  Total Return: {trading_metrics['total_return']:.2%}")
        print(f"  Win Rate: {trading_metrics['win_rate']:.1%}")
        print(f"  Sharpe Ratio: {trading_metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {trading_metrics['max_drawdown']:.2%}")
        print(f"  Number of Trades: {trading_metrics['num_trades']}")
        
        print("="*80)


def main():
    """Main simulation function"""
    print("V5 Enhanced Model - Historical Simulation")
    
    # Create simulator
    simulator = V5HistoricalSimulator()
    
    # Run simulation for last 1 year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Running simulation from {start_date.date()} to {end_date.date()}")
    
    # Run simulation
    results = simulator.run_historical_simulation(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        save_individual=True
    )
    
    if results is None:
        print("Simulation failed")
        return
    
    # Print summary
    simulator.print_summary(results)
    
    # Create plots
    simulator.create_analysis_plots(results)
    
    # Save results
    filepath = simulator.save_results(results)
    
    print(f"\nSimulation completed successfully!")
    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    main()