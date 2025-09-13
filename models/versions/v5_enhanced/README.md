# Model V5 Enhanced

**Creation Date**: September 13, 2025  
**Based on**: V2 Attention model (best performing so far)  
**Status**: Development

## Overview

This version builds upon the promising V2 Attention model which achieved:
- Average Correlation: 0.5762
- MAE: 60.45
- Direction Accuracy: 50%
- Close MAE%: 3.62

## Key Improvements in V5

1. **Enhanced Architecture**
   - Improved attention mechanism with multi-scale temporal attention
   - Residual connections for better gradient flow
   - Adaptive dropout based on market volatility
   - Advanced feature engineering with market regime detection

2. **Better Data Utilization**
   - Integration of 10-minute intraday data for fine-tuning
   - Advanced technical indicators
   - Market microstructure features
   - Calendar and seasonal effects

3. **Risk-Aware Predictions**
   - Confidence intervals for predictions
   - Market regime-dependent predictions
   - Volatility-adjusted forecasts

## Model Architecture

```
Input Layer (30+ features) 
→ Feature Engineering Layer
→ Multi-Scale Attention LSTM (3 layers, 128 hidden units)
→ Residual Connections
→ Adaptive Dropout
→ Dense Layers with Batch Normalization
→ Output Layer (OHLC + Confidence)
```

## Features (30+)

### Price Action Features (8)
- OC_Ratio, HL_Ratio, Intraday_Range
- Open_to_Close, Body_to_Range
- Upper_Shadow, Lower_Shadow, Price_Gap

### Trend Features (6)
- Trend_5, Trend_10, Trend_20
- SMA_5_Ratio, SMA_10_Ratio, SMA_20_Ratio

### Momentum Features (8)
- RSI_14, MACD_Histogram, MACD_Strength
- Momentum_3, Momentum_5, Momentum_10
- Momentum_5_Quality, Stochastic_K

### Volume Features (4)
- Volume_Normalized, Volume_Price_Trend
- OBV_Slope, VWAP_Deviation

### Volatility Features (4)
- BB_Position, BB_Squeeze, Volatility_Regime
- ATR_Ratio

### Support/Resistance Features (3)
- Resistance_Break, Support_Break, S_R_Distance

## Scripts Included

- `train_v5.py` - Main training script
- `predict_next_day.py` - Quick next-day prediction
- `historical_simulation.py` - Comprehensive backtesting
- `model_analysis.py` - Performance analysis and visualization
- `feature_importance.py` - Feature analysis

## Performance Targets

- Average Correlation: > 0.65
- Direction Accuracy: > 55%
- Close MAE%: < 3.0
- Sharpe Ratio (trading): > 1.5