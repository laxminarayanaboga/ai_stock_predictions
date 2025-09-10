"""
Enhanced LSTM Model for Stock Price Prediction
Improved version of the original start-sudo-code.py with advanced features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.preprocessing.data_preprocessor import StockDataPreprocessor


class EnhancedStockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Enhanced LSTM model with improvements over the original
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Hidden layer size
            num_layers (int): Number of LSTM layers
            output_size (int): Number of outputs (4 for OHLC)
            dropout (float): Dropout rate for regularization
        """
        super(EnhancedStockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Multiple fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Batch normalization (only if batch size > 1)
        if out.size(0) > 1:
            out = self.batch_norm(out)
        
        # Fully connected layers with dropout and activation
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out


class StockLSTMTrainer:
    def __init__(self, model, device='cpu'):
        """
        Trainer class for the LSTM model
        
        Args:
            model: The LSTM model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
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
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, patience=10):
        """
        Complete training loop with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs (int): Maximum number of epochs
            lr (float): Learning rate
            patience (int): Early stopping patience
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss = self.validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Learning Rate: {current_lr:.8f}")
                print(f"  Best Val Loss: {self.best_val_loss:.6f}")
                print()
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/best_model.pth'))
        print("Training completed. Best model loaded.")
    
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
    
    def plot_training_history(self, save_path="training_history.png"):
        """Plot training and validation losses"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Model Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training history saved to {save_path}")


def create_data_loaders(processed_data, batch_size=32):
    """Create PyTorch data loaders"""
    # Convert to tensors
    X_train = torch.tensor(processed_data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(processed_data['y_train'], dtype=torch.float32)
    X_val = torch.tensor(processed_data['X_val'], dtype=torch.float32)
    y_val = torch.tensor(processed_data['y_val'], dtype=torch.float32)
    X_test = torch.tensor(processed_data['X_test'], dtype=torch.float32)
    y_test = torch.tensor(processed_data['y_test'], dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function"""
    print("=== Enhanced LSTM Stock Prediction Model ===")
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_file = Path("data/raw").glob("RELIANCE_NSE_*.csv")
    data_file = max(data_file, key=lambda x: x.stat().st_mtime)
    
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Initialize preprocessor
    preprocessor = StockDataPreprocessor(
        lookback_days=30,
        prediction_days=1,
        scale_method='minmax'
    )
    
    # Prepare data
    processed_data = preprocessor.prepare_data_for_training(df)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(processed_data, batch_size=32)
    
    # Model parameters
    input_size = processed_data['X_train'].shape[2]  # Number of features
    hidden_size = 128
    num_layers = 3
    output_size = 4  # OHLC
    
    print(f"\nModel Configuration:")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of layers: {num_layers}")
    print(f"Output size: {output_size}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedStockLSTM(input_size, hidden_size, num_layers, output_size)
    
    # Initialize trainer
    trainer = StockLSTMTrainer(model, device)
    
    # Train model
    trainer.train(
        train_loader, 
        val_loader, 
        epochs=50, 
        lr=0.001, 
        patience=10
    )
    
    # Plot training history
    trainer.plot_training_history("models/training_history.png")
    
    # Make predictions
    print("Making predictions on test set...")
    predictions, actuals = trainer.predict(test_loader)
    
    # Convert predictions back to original scale
    predictions_orig = preprocessor.inverse_transform_targets(predictions)
    actuals_orig = preprocessor.inverse_transform_targets(actuals)
    
    # Calculate metrics
    mse = np.mean((predictions_orig - actuals_orig) ** 2)
    mae = np.mean(np.abs(predictions_orig - actuals_orig))
    rmse = np.sqrt(mse)
    
    print(f"\nTest Results:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Calculate percentage errors for each OHLC
    percentage_errors = np.abs((predictions_orig - actuals_orig) / actuals_orig) * 100
    ohlc_names = ['Open', 'High', 'Low', 'Close']
    
    print(f"\nAverage Percentage Errors:")
    for i, name in enumerate(ohlc_names):
        print(f"{name}: {np.mean(percentage_errors[:, i]):.2f}%")
    
    # Save model and results
    torch.save(model.state_dict(), "models/enhanced_stock_lstm.pth")
    
    # Save training metadata
    metadata = {
        'model_type': 'EnhancedStockLSTM',
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'output_size': output_size,
        'lookback_days': preprocessor.lookback_days,
        'features': processed_data['feature_columns'],
        'test_mse': float(mse),
        'test_mae': float(mae),
        'test_rmse': float(rmse),
        'training_date': datetime.now().isoformat()
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n=== Training Complete! ===")
    print("Model saved to: models/enhanced_stock_lstm.pth")
    print("Metadata saved to: models/model_metadata.json")
    
    return model, preprocessor, predictions_orig, actuals_orig


if __name__ == "__main__":
    main()
