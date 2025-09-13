# V5 Enhanced Model - Usage Guide

## Quick Start

### 1. Training the Model

```bash
cd models/versions/v5_enhanced/scripts
python train_v5.py
```

This will:
- Load Reliance historical data
- Create 30+ enhanced features
- Train the V5 model with multi-scale attention
- Save the best model and training history
- Generate training plots

### 2. Making Next Day Predictions

```bash
cd models/versions/v5_enhanced/scripts
python predict_next_day.py
```

Output includes:
- OHLC predictions for next day
- Confidence scores and uncertainty estimates
- Percentage changes from last actual prices
- Prediction quality assessment

### 3. Running Historical Simulation

```bash
cd models/versions/v5_enhanced/scripts
python historical_simulation.py
```

This provides:
- Comprehensive backtesting over the last year
- Performance metrics (MAE, RMSE, correlations)
- Trading simulation results
- Detailed analysis plots
- Direction accuracy assessment

## Model Features

### Architecture Improvements over V2
- **Multi-scale attention**: Captures patterns at different time scales
- **Residual connections**: Better gradient flow for deeper networks
- **Adaptive dropout**: Adjusts based on market volatility
- **Advanced feature engineering**: 30+ sophisticated features

### Enhanced Features (30+)
1. **Price Action** (8): OC_Ratio, HL_Ratio, Intraday_Range, Body analysis
2. **Trend Analysis** (6): Multiple timeframe trends and slopes
3. **Momentum** (8): RSI, MACD, momentum quality indicators
4. **Volume Analysis** (4): Normalized volume, OBV, VWAP deviations
5. **Volatility** (4): Bollinger Bands, ATR, volatility regimes
6. **Support/Resistance** (3): Break detection, distance measures

### Performance Targets
- **Correlation**: > 0.65 (vs V2's 0.58)
- **Direction Accuracy**: > 55% (vs V2's 50%)
- **Close MAE%**: < 3.0% (vs V2's 3.62%)
- **Sharpe Ratio**: > 1.5 (trading strategy)

## File Structure

```
models/versions/v5_enhanced/
├── README.md                 # Main documentation
├── USAGE.md                 # This usage guide
├── model_v5.py              # Enhanced model architecture
├── scripts/
│   ├── train_v5.py          # Training script
│   ├── predict_next_day.py  # Quick prediction
│   ├── historical_simulation.py # Backtesting
│   └── data_utils.py        # Data processing utilities
├── analysis/                # Results and plots
├── predictions/             # Individual predictions
├── best_model_v5.pth       # Best trained model
├── latest_checkpoint_v5.pth # Latest checkpoint
├── metadata_v5.json        # Model metadata
└── latest_prediction_v5.json # Latest prediction
```

## Configuration

### Model Configuration
```python
config = {
    'input_size': 30,           # Number of features
    'hidden_size': 128,         # LSTM hidden units
    'num_layers': 3,            # LSTM layers
    'output_size': 4,           # OHLC outputs
    'dropout': 0.15,            # Base dropout rate
    'attention_heads': 8,       # Multi-head attention
    'lookback_days': 15,        # Sequence length
    'batch_size': 64,           # Training batch size
    'learning_rate': 0.001,     # Initial learning rate
    'epochs': 200               # Training epochs
}
```

### Data Configuration
- **Lookback Period**: 15 days
- **Features**: 30+ technical indicators
- **Target**: Next day OHLC
- **Training Split**: 80% train, 10% validation, 10% test

## Understanding the Output

### Prediction Output
```json
{
  "prediction_date": "2025-09-13",
  "model_version": "v5_enhanced",
  "last_actual": {
    "Open": 1250.00,
    "High": 1265.50,
    "Low": 1245.30,
    "Close": 1260.75
  },
  "predicted": {
    "Open": 1262.30,
    "High": 1278.40,
    "Low": 1255.60,
    "Close": 1270.20
  },
  "prediction_changes": {
    "Open": 0.98,
    "High": 1.02,
    "Low": 0.82,
    "Close": 0.75
  },
  "uncertainty": {
    "Open": 5.2,
    "High": 7.1,
    "Low": 4.8,
    "Close": 6.3
  },
  "confidence": 0.78,
  "prediction_quality": "High"
}
```

### Metrics Interpretation
- **MAE**: Mean Absolute Error in rupees
- **MAPE**: Mean Absolute Percentage Error
- **Correlation**: Linear correlation (-1 to 1)
- **Direction Accuracy**: % of correct price direction predictions
- **Confidence**: Model's confidence in prediction (0-1)
- **Uncertainty**: Standard deviation of Monte Carlo predictions

## Best Practices

### 1. Model Usage
- Always check prediction quality before trading decisions
- Use uncertainty estimates for risk management
- Consider market conditions and external factors
- Combine with other analysis methods

### 2. Performance Monitoring
- Track prediction accuracy over time
- Monitor confidence scores and uncertainty
- Regular retraining on new data
- Compare against benchmark models

### 3. Trading Application
- Use predictions as directional guidance
- Apply position sizing based on confidence
- Set stop-losses using uncertainty estimates
- Consider transaction costs and market impact

## Troubleshooting

### Common Issues
1. **Model not found**: Run `train_v5.py` first
2. **Data loading errors**: Check data file paths
3. **CUDA errors**: Ensure GPU compatibility or use CPU
4. **Memory issues**: Reduce batch size in configuration

### Performance Issues
- **Low accuracy**: Retrain with more recent data
- **High uncertainty**: Check data quality and market volatility
- **Poor direction accuracy**: Consider additional features or model adjustments

## Next Steps

### Potential Improvements
1. **Ensemble Methods**: Combine multiple models
2. **Alternative Architectures**: Transformer-based models
3. **Additional Data**: Market sentiment, news, economic indicators
4. **Risk Management**: Volatility forecasting, portfolio optimization

### Monitoring and Maintenance
1. Regular performance evaluation
2. Periodic retraining (monthly/quarterly)
3. Feature importance analysis
4. Model drift detection