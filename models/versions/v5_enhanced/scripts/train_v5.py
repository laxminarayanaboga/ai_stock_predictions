"""
Training script for Enhanced LSTM V5 Model
Builds upon the successful V2 attention model with significant improvements
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from models.versions.v5_enhanced.model_v5 import EnhancedLSTMV5, create_enhanced_model_v5
from models.versions.v5_enhanced.scripts.data_utils import (
    SimpleStockDataset, load_reliance_data, create_enhanced_features, prepare_sequences
)


class V5Trainer:
    """Enhanced trainer for V5 model with advanced features"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'direction_accuracy': [],
            'correlation': []
        }
        
        print(f"Using device: {self.device}")
        
    def load_and_prepare_data(self):
        """Load and prepare data with enhanced features including intraday data"""
        print("Loading and preparing data...")
        
        # Load raw data (both daily and intraday)
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
        
        # Enhanced feature engineering with intraday data
        df_features = create_enhanced_features(df, intraday_df)
        
        # Prepare sequences
        lookback = self.config['lookback_days']
        target_cols = ['open', 'high', 'low', 'close']
        
        sequences, targets, feature_cols = prepare_sequences(df_features, lookback, target_cols)
        
        print(f"Created {len(sequences)} sequences with shape {sequences.shape}")
        print(f"Using {len(feature_cols)} features (including intraday features)")
        
        # Update input size in config to match actual features
        self.config['input_size'] = len(feature_cols)
        
        # Split data
        split_idx = int(len(sequences) * 0.8)
        val_split_idx = int(len(sequences) * 0.9)
        
        X_train = sequences[:split_idx]
        y_train = targets[:split_idx]
        X_val = sequences[split_idx:val_split_idx]
        y_val = targets[split_idx:val_split_idx]
        X_test = sequences[val_split_idx:]
        y_test = targets[val_split_idx:]
        
        # Scale features
        X_train_scaled = self._scale_features(X_train, fit=True)
        X_val_scaled = self._scale_features(X_val)
        X_test_scaled = self._scale_features(X_test)
        
        # Create datasets
        train_dataset = SimpleStockDataset(X_train_scaled, y_train)
        val_dataset = SimpleStockDataset(X_val_scaled, y_val)
        test_dataset = SimpleStockDataset(X_test_scaled, y_test)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, 
                                     batch_size=self.config['batch_size'], 
                                     shuffle=True)
        self.val_loader = DataLoader(val_dataset, 
                                   batch_size=self.config['batch_size'], 
                                   shuffle=False)
        self.test_loader = DataLoader(test_dataset, 
                                    batch_size=self.config['batch_size'], 
                                    shuffle=False)
        
        # Store feature columns for later use
        self.feature_cols = feature_cols
        
        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    
    def _scale_features(self, X, fit=False):
        """Scale features maintaining sequence structure"""
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)
        
        return X_scaled.reshape(original_shape)
    
    def create_model(self):
        """Create and initialize the V5 model"""
        print("Creating Enhanced LSTM V5 model...")
        
        self.model = create_enhanced_model_v5(self.config)
        self.model.to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model created with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler"""
        # AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['epochs'] // 4,
            T_mult=2,
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        # Loss function - Huber loss for robustness
        self.criterion = nn.HuberLoss(delta=1.0)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for batch_X, batch_y in self.train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            mae = torch.mean(torch.abs(predictions - batch_y)).item()
            total_mae += mae
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                mae = torch.mean(torch.abs(predictions - batch_y)).item()
                total_mae += mae
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        # Calculate additional metrics
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        # Direction accuracy (for Close price)
        pred_direction = np.sign(predictions[:, 3])  # Close price is index 3
        true_direction = np.sign(targets[:, 3])
        direction_acc = np.mean(pred_direction == true_direction)
        
        # Correlation
        correlations = []
        for i in range(4):  # OHLC
            corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        avg_correlation = np.mean(correlations) if correlations else 0
        
        return avg_loss, avg_mae, direction_acc, avg_correlation
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_mae = self.train_epoch()
            
            # Validate
            val_loss, val_mae, direction_acc, correlation = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Save metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_mae'].append(train_mae)
            self.training_history['val_mae'].append(val_mae)
            self.training_history['direction_accuracy'].append(direction_acc)
            self.training_history['correlation'].append(correlation)
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.config['epochs'] - 1:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.config['epochs']}:")
                print(f"  Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
                print(f"  Direction Acc: {direction_acc:.4f}, Correlation: {correlation:.4f}")
                print(f"  Learning Rate: {lr:.6f}")
                print()
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model(epoch, val_loss, is_best=True)
        
        print("Training completed!")
        return self.training_history
    
    def save_model(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        model_dir = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/models/versions/v5_enhanced"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'feature_cols': self.feature_cols,
            'scaler': self.scaler,
            'training_history': self.training_history
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(model_dir, 'best_model_v5.pth'))
            print(f"Best model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(model_dir, 'latest_checkpoint_v5.pth'))
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE plot
        axes[0, 1].plot(self.training_history['train_mae'], label='Train MAE')
        axes[0, 1].plot(self.training_history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Direction accuracy
        axes[1, 0].plot(self.training_history['direction_accuracy'])
        axes[1, 0].set_title('Direction Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True)
        
        # Correlation
        axes[1, 1].plot(self.training_history['correlation'])
        axes[1, 1].set_title('Average Correlation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        model_dir = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/models/versions/v5_enhanced"
        plt.savefig(os.path.join(model_dir, 'training_history_v5.png'), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main training function"""
    # Model configuration
    config = {
        'input_size': 30,  # Will be updated based on actual features
        'hidden_size': 128,
        'num_layers': 3,
        'output_size': 4,  # OHLC
        'dropout': 0.15,
        'attention_heads': 8,
        'prediction_horizon': 1,
        'lookback_days': 15,
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 200
    }
    
    print("=" * 60)
    print("Enhanced LSTM V5 Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = V5Trainer(config)
    
    # Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_and_prepare_data()
    
    # Update input size based on actual features
    config['input_size'] = X_train.shape[-1]
    trainer.config = config
    
    # Create model
    model = trainer.create_model()
    
    # Create optimizer and scheduler
    trainer.create_optimizer_and_scheduler()
    
    # Train the model
    training_history = trainer.train()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save final metadata
    metadata = {
        'model_type': 'EnhancedLSTMV5',
        'config': config,
        'training_date': datetime.now().isoformat(),
        'final_val_loss': trainer.best_loss,
        'feature_count': len(trainer.feature_cols),
        'features': trainer.feature_cols,
        'training_completed': True
    }
    
    model_dir = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/models/versions/v5_enhanced"
    with open(os.path.join(model_dir, 'metadata_v5.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Training completed successfully!")
    print(f"Best validation loss: {trainer.best_loss:.4f}")
    print(f"Model saved to: {model_dir}")


if __name__ == "__main__":
    main()