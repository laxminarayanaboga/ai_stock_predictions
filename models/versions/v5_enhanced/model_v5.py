"""
Enhanced LSTM with Multi-Scale Attention for Stock Price Prediction - Version 5
Based on the promising V2 attention model with significant improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MultiScaleAttention(nn.Module):
    """Multi-scale temporal attention mechanism"""
    
    def __init__(self, hidden_size, num_heads=8, scales=[1, 3, 5]):
        super(MultiScaleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.scales = scales
        self.head_dim = hidden_size // num_heads
        
        # Multi-scale attention layers
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads, dropout=0.1, batch_first=True)
            for _ in scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(len(scales) * hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = x.size()
        
        scale_outputs = []
        for i, (scale, attention) in enumerate(zip(self.scales, self.scale_attentions)):
            # Apply attention at different scales
            if scale == 1:
                scale_x = x
            else:
                # Downsample for larger scales
                step = max(1, seq_len // (seq_len // scale))
                indices = torch.arange(0, seq_len, step, device=x.device)[:seq_len//scale]
                if len(indices) < 2:
                    indices = torch.arange(0, min(2, seq_len), device=x.device)
                scale_x = x[:, indices, :]
            
            # Apply attention
            attn_output, _ = attention(scale_x, scale_x, scale_x)
            
            # Upsample back to original sequence length if needed
            if scale_x.size(1) != seq_len:
                attn_output = F.interpolate(
                    attn_output.transpose(1, 2), 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(attn_output)
        
        # Fuse multi-scale outputs
        fused = torch.cat(scale_outputs, dim=-1)
        fused = self.scale_fusion(fused)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + fused)
        
        return output


class AdaptiveDropout(nn.Module):
    """Adaptive dropout based on market volatility"""
    
    def __init__(self, base_dropout=0.1):
        super(AdaptiveDropout, self).__init__()
        self.base_dropout = base_dropout
        
    def forward(self, x, volatility_indicator=None):
        if not self.training:
            return x
            
        if volatility_indicator is not None:
            # Adjust dropout based on volatility
            # Higher volatility -> higher dropout to prevent overfitting to noise
            adaptive_rate = self.base_dropout * (1 + volatility_indicator.mean().item())
            adaptive_rate = min(adaptive_rate, 0.5)  # Cap at 50%
        else:
            adaptive_rate = self.base_dropout
            
        return F.dropout(x, p=adaptive_rate, training=self.training)


class EnhancedLSTMV5(nn.Module):
    """
    Enhanced LSTM model with multi-scale attention and advanced features
    Building upon the successful V2 attention architecture
    """
    
    def __init__(self, input_size=30, hidden_size=128, num_layers=3, output_size=4, 
                 dropout=0.1, attention_heads=8, prediction_horizon=1):
        super(EnhancedLSTMV5, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.prediction_horizon = prediction_horizon
        
        # Feature engineering layer
        self.feature_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-layer LSTM with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_size if i == 0 else hidden_size, 
                   hidden_size, 
                   batch_first=True, 
                   dropout=dropout if i < num_layers-1 else 0)
            for i in range(num_layers)
        ])
        
        # Multi-scale attention
        self.attention = MultiScaleAttention(hidden_size, attention_heads)
        
        # Adaptive dropout
        self.adaptive_dropout = AdaptiveDropout(dropout)
        
        # Residual connection layers
        self.residual_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Output layers with skip connections
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_size // 4, output_size)
        )
        
        # Confidence estimation layer
        self.confidence_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x, return_confidence=False):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # Feature transformation
        x_reshaped = x.view(-1, self.input_size)
        x_transformed = self.feature_transform(x_reshaped)
        x = x_transformed.view(batch_size, seq_len, self.hidden_size)
        
        # Extract volatility indicator for adaptive dropout
        # Assuming volatility is one of the features (e.g., index -1)
        volatility = x[:, :, -1:].mean(dim=1)  # Use last feature as volatility proxy
        
        # Multi-layer LSTM with residual connections
        lstm_output = x
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.residual_norms)):
            residual = lstm_output
            lstm_out, _ = lstm(lstm_output)
            
            # Apply adaptive dropout
            lstm_out = self.adaptive_dropout(lstm_out, volatility)
            
            # Residual connection (skip first layer for dimension matching)
            if i > 0:
                lstm_output = norm(lstm_out + residual)
            else:
                lstm_output = norm(lstm_out)
        
        # Multi-scale attention
        attended_output = self.attention(lstm_output)
        
        # Take the last time step
        final_hidden = attended_output[:, -1, :]
        
        # Generate predictions
        predictions = self.output_layers(final_hidden)
        
        if return_confidence:
            confidence = self.confidence_layer(final_hidden)
            return predictions, confidence
        
        return predictions
    
    def predict_with_uncertainty(self, x, num_samples=10):
        """Monte Carlo dropout for uncertainty estimation"""
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        self.eval()  # Disable dropout
        
        return mean_pred, std_pred


def create_enhanced_model_v5(config):
    """Factory function to create V5 model with configuration"""
    return EnhancedLSTMV5(
        input_size=config.get('input_size', 30),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 3),
        output_size=config.get('output_size', 4),
        dropout=config.get('dropout', 0.1),
        attention_heads=config.get('attention_heads', 8),
        prediction_horizon=config.get('prediction_horizon', 1)
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the model
    config = {
        'input_size': 30,
        'hidden_size': 128,
        'num_layers': 3,
        'output_size': 4,
        'dropout': 0.15,
        'attention_heads': 8,
        'prediction_horizon': 1
    }
    
    model = create_enhanced_model_v5(config)
    
    # Test forward pass
    batch_size, seq_len, input_size = 32, 15, 30
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Standard prediction
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Prediction with confidence
    output, confidence = model(x, return_confidence=True)
    print(f"Output shape: {output.shape}, Confidence shape: {confidence.shape}")
    
    # Uncertainty estimation
    mean_pred, std_pred = model.predict_with_uncertainty(x, num_samples=5)
    print(f"Mean prediction shape: {mean_pred.shape}, Std shape: {std_pred.shape}")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")