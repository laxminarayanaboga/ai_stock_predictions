"""
Multi-Model Training and Comparison Framework
Trains multiple model versions and compares their performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.model_version_manager import ModelVersionManager
from src.models.improved_architectures import get_model_and_config, MODEL_CONFIGS, create_loss_function
from src.preprocessing.improved_preprocessor import ImprovedStockDataPreprocessor
from src.evaluation.model_evaluator import ModelEvaluator


class ImprovedStockLSTMTrainer:
    """Enhanced trainer with more advanced features"""
    
    def __init__(self, model, device='cpu', loss_type='mse'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.loss_type = loss_type
        
    def train_epoch(self, train_loader, criterion, optimizer, scheduler=None):
        """Train for one epoch with optional learning rate scheduling"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Step-wise learning rate scheduling
            if scheduler and hasattr(scheduler, 'step') and 'step' in str(type(scheduler)):
                scheduler.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, config):
        """Enhanced training loop"""
        epochs = config.get('epochs', 50)
        lr = config.get('lr', 0.001)
        patience = config.get('patience', 10)
        
        # Loss function
        criterion = create_loss_function(self.loss_type)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 2,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        patience_counter = 0
        best_epoch = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Loss type: {self.loss_type}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, criterion, optimizer, scheduler)
            
            # Validation
            val_loss = self.validate_epoch(val_loader, criterion)
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch < 5:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Learning Rate: {current_lr:.8f}")
                print(f"  Best Val Loss: {self.best_val_loss:.6f} (Epoch {best_epoch+1})")
                print()
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"Training completed. Best model from epoch {best_epoch+1} loaded.")
        
        return {
            'best_epoch': best_epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1],
            'total_epochs': len(self.train_losses)
        }
    
    def predict(self, test_loader):
        """Make predictions on test data"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predictions.append(output.cpu().numpy())
                actuals.append(target.cpu().numpy())
        
        return np.vstack(predictions), np.vstack(actuals)


class MultiModelExperiment:
    """Manages training and comparison of multiple model versions"""
    
    def __init__(self, data_file: str = None):
        """Initialize the experiment"""
        self.version_manager = ModelVersionManager()
        self.results = {}
        
        # Load and prepare data
        if data_file is None:
            data_files = list(Path("data/raw/daily").glob("RELIANCE_NSE_*.csv"))
            if not data_files:
                raise FileNotFoundError("No Reliance data files found in data/raw/daily/")
            data_file = max(data_files, key=lambda x: x.stat().st_mtime)
        
        print(f"Loading data from: {data_file}")
        self.df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        # Initialize preprocessor with improved settings
        self.preprocessor = ImprovedStockDataPreprocessor(
            lookback_days=15,  # Reduced for more recent patterns
            prediction_days=1,
            scale_method='robust'  # More robust to outliers
        )
        
        # Prepare data
        self.processed_data = self.preprocessor.prepare_data_for_training(self.df)
        print(f"Data prepared: {self.processed_data['X_train'].shape[0]} training samples")
    
    def create_data_loaders(self, batch_size=32):
        """Create PyTorch data loaders"""
        X_train = torch.tensor(self.processed_data['X_train'], dtype=torch.float32)
        y_train = torch.tensor(self.processed_data['y_train'], dtype=torch.float32)
        X_val = torch.tensor(self.processed_data['X_val'], dtype=torch.float32)
        y_val = torch.tensor(self.processed_data['y_val'], dtype=torch.float32)
        X_test = torch.tensor(self.processed_data['X_test'], dtype=torch.float32)
        y_test = torch.tensor(self.processed_data['y_test'], dtype=torch.float32)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model_version(self, version_name: str, loss_type: str = 'mse'):
        """Train a specific model version"""
        print(f"\n🚀 Training model version: {version_name}")
        print("=" * 50)
        
        # Create version in version manager
        self.version_manager.create_version(
            version_name,
            MODEL_CONFIGS[version_name]['description'],
            copy_current=False
        )
        
        # Get model and config
        input_size = self.processed_data['X_train'].shape[2]
        model, config = get_model_and_config(version_name, input_size)
        
        # Create data loaders
        batch_size = config['training']['batch_size']
        train_loader, val_loader, test_loader = self.create_data_loaders(batch_size)
        
        # Initialize trainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = ImprovedStockLSTMTrainer(model, device, loss_type)
        
        # Train model
        training_info = trainer.train(train_loader, val_loader, config['training'])
        
        # Evaluate on test set
        predictions, actuals = trainer.predict(test_loader)
        
        # Convert back to original scale
        predictions_orig = self.preprocessor.inverse_transform_targets(predictions)
        actuals_orig = self.preprocessor.inverse_transform_targets(actuals)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(predictions_orig, actuals_orig)
        
        # Prepare metadata
        metadata = {
            'model_type': config['model_class'].__name__,
            'input_size': input_size,
            'output_size': 4,
            'lookback_days': self.preprocessor.lookback_days,
            'features': self.processed_data['feature_columns'],
            'loss_type': loss_type,
            'training_date': datetime.now().isoformat(),
            **config['params'],
            **metrics
        }
        
        # Save trained model
        self.version_manager.save_trained_model(
            version_name,
            model,
            metadata,
            config['training'],
            metrics
        )
        
        # Store results
        self.results[version_name] = {
            'metrics': metrics,
            'training_info': training_info,
            'predictions': predictions_orig,
            'actuals': actuals_orig,
            'config': config
        }
        
        print(f"✅ Completed training {version_name}")
        self.print_metrics(version_name, metrics)
        
        return trainer, predictions_orig, actuals_orig
    
    def calculate_metrics(self, predictions, actuals):
        """Calculate comprehensive evaluation metrics"""
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        # Percentage errors for each OHLC
        percentage_errors = np.abs((predictions - actuals) / actuals) * 100
        ohlc_names = ['Open', 'High', 'Low', 'Close']
        
        # Direction accuracy (for price movements)
        actual_changes = actuals[:, 3] - actuals[:, 0]  # Close - Open
        pred_changes = predictions[:, 3] - predictions[:, 0]
        
        direction_accuracy = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
        
        # Correlation for each OHLC
        correlations = []
        for i in range(4):
            corr = np.corrcoef(predictions[:, i], actuals[:, i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        return {
            'test_mse': float(mse),
            'test_mae': float(mae),
            'test_rmse': float(rmse),
            'direction_accuracy': float(direction_accuracy),
            'ohlc_mae': [float(np.mean(percentage_errors[:, i])) for i in range(4)],
            'ohlc_correlations': correlations,
            'avg_correlation': float(np.mean(correlations))
        }
    
    def print_metrics(self, version_name, metrics):
        """Print formatted metrics"""
        print(f"\n📊 {version_name} Results:")
        print(f"  MSE: {metrics['test_mse']:.4f}")
        print(f"  MAE: {metrics['test_mae']:.4f}")
        print(f"  RMSE: {metrics['test_rmse']:.4f}")
        print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
        print(f"  Average Correlation: {metrics['avg_correlation']:.4f}")
        
        ohlc_names = ['Open', 'High', 'Low', 'Close']
        print(f"  OHLC MAE%: " + ", ".join([f"{name}: {mae:.2f}%" 
                                           for name, mae in zip(ohlc_names, metrics['ohlc_mae'])]))
    
    def run_all_experiments(self, loss_types=['mse', 'huber']):
        """Run experiments for all model versions"""
        print("🧪 Starting Multi-Model Experiments")
        print("=" * 60)
        
        for version_name in MODEL_CONFIGS.keys():
            for loss_type in loss_types:
                full_version_name = f"{version_name}_{loss_type}"
                try:
                    self.train_model_version(version_name, loss_type)
                except Exception as e:
                    print(f"❌ Failed to train {full_version_name}: {str(e)}")
                    continue
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report"""
        if not self.results:
            print("No results to compare")
            return
        
        print("\n📊 MODEL COMPARISON REPORT")
        print("=" * 60)
        
        # Create comparison DataFrame
        comparison_data = []
        for version_name, result in self.results.items():
            metrics = result['metrics']
            config = result['config']
            
            comparison_data.append({
                'Version': version_name,
                'Architecture': config['model_class'].__name__,
                'Parameters': config['params'].get('hidden_size', 'N/A'),
                'MSE': metrics['test_mse'],
                'MAE': metrics['test_mae'],
                'Direction Acc': metrics['direction_accuracy'],
                'Avg Correlation': metrics['avg_correlation'],
                'Close MAE%': metrics['ohlc_mae'][3] if len(metrics['ohlc_mae']) > 3 else 0
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('MAE')  # Sort by MAE (lower is better)
        
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison
        df.to_csv('models/model_comparison.csv', index=False)
        print(f"\n💾 Comparison saved to: models/model_comparison.csv")
        
        # Best model recommendation
        best_model = df.iloc[0]
        print(f"\n🏆 Best Performing Model: {best_model['Version']}")
        print(f"   Architecture: {best_model['Architecture']}")
        print(f"   MAE: {best_model['MAE']:.4f}")
        print(f"   Direction Accuracy: {best_model['Direction Acc']:.2%}")


def main():
    """Main execution function"""
    print("🔬 Stock Prediction Model Experiments")
    print("=" * 50)
    
    # Initialize experiment
    experiment = MultiModelExperiment()
    
    # Train specific models (you can modify this list)
    models_to_train = ['v2_attention', 'v3_simpler', 'v4_conv_lstm']
    loss_types = ['mse', 'huber']
    
    for model_name in models_to_train:
        for loss_type in loss_types:
            try:
                experiment.train_model_version(model_name, loss_type)
            except Exception as e:
                print(f"❌ Failed to train {model_name} with {loss_type}: {str(e)}")
                continue
    
    # Generate comparison
    experiment.generate_comparison_report()


if __name__ == "__main__":
    main()