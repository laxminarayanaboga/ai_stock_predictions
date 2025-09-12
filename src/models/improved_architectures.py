"""
Improved Model Architectures for Stock Prediction
Different approaches to address the current model's limitations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
import json
from datetime import datetime


class AttentionLSTM(nn.Module):
    """LSTM with Attention mechanism for better temporal feature learning"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Layer normalization and residual connection
        lstm_out = self.layer_norm(lstm_out + attn_out)
        
        # Take the last time step
        out = lstm_out[:, -1, :]
        
        # Final layers
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class SimplerLSTM(nn.Module):
    """Simpler LSTM to reduce overfitting"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(SimplerLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Smaller LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Simple output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out


class ConvLSTM(nn.Module):
    """CNN + LSTM hybrid for capturing both local and temporal patterns"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(ConvLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1D Convolutional layers for local pattern detection
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, hidden_size, kernel_size=3, padding=1)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Transpose for conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Transpose back: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Last time step
        
        # Output layers
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class ResidualLSTM(nn.Module):
    """LSTM with residual connections"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(ResidualLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection to match hidden size
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_size, hidden_size, 1, batch_first=True, dropout=0)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Project input to hidden size
        x = self.input_proj(x)
        
        # Apply LSTM layers with residual connections
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            residual = x
            lstm_out, _ = lstm(x)
            
            # Residual connection and layer normalization
            x = norm(lstm_out + residual)
            x = self.dropout(x)
        
        # Take last time step
        out = x[:, -1, :]
        
        # Output layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class EnsembleLSTM(nn.Module):
    """Ensemble of different LSTM architectures"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(EnsembleLSTM, self).__init__()
        
        # Three different sub-models
        self.model1 = SimplerLSTM(input_size, hidden_size//2, 2, output_size, dropout)
        self.model2 = AttentionLSTM(input_size, hidden_size//2, 2, output_size, dropout)
        self.model3 = ConvLSTM(input_size, hidden_size//2, 2, output_size, dropout)
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        # Get predictions from each model
        pred1 = self.model1(x)
        pred2 = self.model2(x)
        pred3 = self.model3(x)
        
        # Weighted ensemble
        weights = torch.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = (weights[0] * pred1 + 
                        weights[1] * pred2 + 
                        weights[2] * pred3)
        
        return ensemble_pred


# Model configurations for different experiments
MODEL_CONFIGS = {
    "v2_attention": {
        "model_class": AttentionLSTM,
        "description": "LSTM with multi-head attention mechanism",
        "params": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1
        },
        "training": {
            "epochs": 100,
            "lr": 0.0005,
            "batch_size": 64,
            "patience": 15
        }
    },
    
    "v3_simpler": {
        "model_class": SimplerLSTM,
        "description": "Simplified LSTM to reduce overfitting",
        "params": {
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.05
        },
        "training": {
            "epochs": 150,
            "lr": 0.001,
            "batch_size": 32,
            "patience": 20
        }
    },
    
    "v4_conv_lstm": {
        "model_class": ConvLSTM,
        "description": "CNN + LSTM hybrid for pattern detection",
        "params": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.15
        },
        "training": {
            "epochs": 100,
            "lr": 0.0007,
            "batch_size": 48,
            "patience": 15
        }
    },
    
    "v5_residual": {
        "model_class": ResidualLSTM,
        "description": "LSTM with residual connections",
        "params": {
            "hidden_size": 64,
            "num_layers": 3,
            "dropout": 0.1
        },
        "training": {
            "epochs": 100,
            "lr": 0.0008,
            "batch_size": 64,
            "patience": 15
        }
    },
    
    "v6_ensemble": {
        "model_class": EnsembleLSTM,
        "description": "Ensemble of multiple architectures",
        "params": {
            "hidden_size": 96,  # Will be split among sub-models
            "num_layers": 2,
            "dropout": 0.1
        },
        "training": {
            "epochs": 80,
            "lr": 0.0005,
            "batch_size": 32,
            "patience": 12
        }
    }
}


def get_model_and_config(version_name: str, input_size: int, output_size: int = 4):
    """
    Get a model instance and its configuration
    
    Args:
        version_name: Name of the model version
        input_size: Number of input features
        output_size: Number of outputs (4 for OHLC)
        
    Returns:
        Tuple of (model_instance, config_dict)
    """
    if version_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model version: {version_name}")
    
    config = MODEL_CONFIGS[version_name]
    model_class = config["model_class"]
    params = config["params"].copy()
    
    # Create model instance
    model = model_class(
        input_size=input_size,
        output_size=output_size,
        **params
    )
    
    return model, config


def create_loss_function(loss_type: str = "mse"):
    """Create different loss functions for experimentation"""
    
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.SmoothL1Loss()
    elif loss_type == "weighted_mse":
        # Custom weighted MSE that penalizes direction errors more
        class WeightedMSE(nn.Module):
            def forward(self, pred, target):
                # Standard MSE
                mse = nn.MSELoss()(pred, target)
                
                # Additional penalty for wrong direction
                pred_direction = torch.sign(pred - target[:, [0]])  # Compare to Open
                target_direction = torch.sign(target - target[:, [0]])
                
                direction_penalty = torch.mean((pred_direction != target_direction).float())
                
                return mse + 0.1 * direction_penalty
        
        return WeightedMSE()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Demo of different model architectures
    print("🏗️  Stock Prediction Model Architectures")
    print("=" * 50)
    
    input_size = 26  # Current number of features
    output_size = 4  # OHLC
    
    for version_name, config in MODEL_CONFIGS.items():
        print(f"\n📦 {version_name}:")
        print(f"   Description: {config['description']}")
        print(f"   Parameters: {config['params']}")
        
        model, _ = get_model_and_config(version_name, input_size)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   Total Parameters: {param_count:,}")