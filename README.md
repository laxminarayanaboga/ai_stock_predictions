# 🚀 AI Stock Predictions for Reliance NSE

**Advanced LSTM-based Stock Price Prediction & Trading Simulation System**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![License](https://img.shields.io/badge/License-Educational-yellow)

## 🎯 **Project Overview**

This project implements a comprehensive AI-powered stock prediction and trading simulation system for **Reliance Industries (NSE:RELIANCE-EQ)** using advanced LSTM neural networks. The system features:

- 🧠 **AI Price Prediction**: Advanced LSTM model analyzing 10 years of data and 26 technical indicators
- 🏗️ **Trading Simulation Framework**: Multi-strategy backtesting with real trading costs
- 📊 **Strategy Comparison**: Automated testing of multiple trading strategies with detailed analytics
- 💰 **Professional Integration**: Fyers brokerage API with accurate charge calculations

## ✨ **Key Features**

### 🧠 **AI Prediction Engine**
- 📊 **10 Years of Data**: 2,475 trading days from 2015-2025
- 🧠 **Advanced LSTM Model**: 3-layer architecture with 354K parameters
- 📈 **26 Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, etc.
- 🎯 **High Accuracy**: MAPE of 1.69% and R² of 0.71
- ⚡ **Real-time Predictions**: Next-day OHLC forecasting

### 🏗️ **Trading Simulation Framework**
- 🎪 **Multi-Strategy Testing**: Run 9+ different trading strategies simultaneously
- 📊 **Comprehensive Analytics**: Win rates, PnL tracking, risk metrics
- 🔄 **Automated Backtesting**: Historical performance analysis with real market conditions
- 📈 **Strategy Comparison**: Visual charts and detailed reports
- 💰 **Real Trading Costs**: Accurate brokerage charges and taxes

### 🚀 **Professional Features**
- 📱 **User-friendly Interface**: Command-line tools for easy execution
- 💰 **Fyers Integration**: Real brokerage API with accurate charge calculations
- 📊 **Performance Analytics**: Comprehensive trading metrics and risk analysis
- 🗂️ **Organized Results**: Timestamped folders with automatic cleanup
- 📋 **Export Capabilities**: CSV, JSON, and visualization outputs

## 🧹 **Recent Updates (September 2025)**

✅ **Multi-Strategy Trading Framework**
- Added comprehensive trading simulation system with 9+ strategies
- Implemented automated strategy comparison with detailed analytics
- Created timestamped result folders with automatic cleanup
- Added visual comparison charts and detailed CSV exports

✅ **Project Cleaned & Optimized**
- Fixed all import errors and broken dependencies
- Added professional trading charges calculator (`trading_charges.py`)
- Added comprehensive performance metrics (`performance_metrics.py`)
- Removed unnecessary files and Python cache
- Simplified strategy management system
- All core functionality verified and working

📋 See [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) for detailed cleanup information.

## 🏆 **Model Performance**

| Metric | Value | Description |
|--------|-------|-------------|
| **RMSE** | 29.13 | Root Mean Square Error |
| **MAE** | 24.19 | Mean Absolute Error |
| **MAPE** | 1.69% | Mean Absolute Percentage Error |
| **R²** | 0.71 | Coefficient of Determination |
| **Directional Accuracy** | 48.33% | Price direction prediction |

## 🚀 **Quick Start**

### 1. **Installation**
```bash
git clone https://github.com/yourusername/ai_stock_predictions.git
cd ai_stock_predictions
pip install -r requirements.txt
```

### 2. **Configure Fyers API**
Edit `config.ini` with your Fyers API credentials:
```ini
[Fyers_APP]
client_id = YOUR_CLIENT_ID
secret_key = YOUR_SECRET_KEY
# ... other credentials
```

### 3. **Download Data**
```bash
python src/data/reliance_data_downloader.py
```

### 4. **Train Model**
```bash
python src/models/enhanced_lstm.py
```

### 5. **Make Predictions**
```bash
python predict.py
```

### 6. **Run Trading Simulations**
```bash
# Single strategy simulation
python src/simulator/strategy_simulator.py

# Multi-strategy comparison (recommended)
python src/simulator/multi_strategy_runner.py
```

## 🎪 **Usage Examples**

### **Quick Prediction**
```bash
$ python predict.py

🔮 Reliance Stock Price Predictor
========================================
📊 Using data up to: 2025-09-10
📈 Latest Close Price: ₹1377.00

🎯 NEXT DAY PREDICTION
----------------------------------------
Price    Current    Predicted  Change   %       
----------------------------------------
Open     ₹1383.90   ₹1373.73    -10.17   -0.7% 🔴
High     ₹1388.50   ₹1385.84     -2.66   -0.2% 🔴
Low      ₹1374.10   ₹1367.01     -7.09   -0.5% 🔴
Close    ₹1377.00   ₹1375.88     -1.12   -0.1% 🔴
----------------------------------------
💡 Sentiment: ⚪ HOLD
📊 Expected Close: ₹1375.88 (-0.1%)
```

### **Multi-Strategy Trading Simulation**
```bash
$ python src/simulator/multi_strategy_runner.py

🚀 Multi-Strategy Trading Simulator
================================================================================
📊 Loaded 439 days of market data
🤖 Loaded 490 AI predictions
🎯 Testing 9 different trading strategies...

🚀 Running Strategy: Strategy 06 - High Confidence
📊 Conservative: SL=1.5%, TP=4%, Conf=80%
...
📊 STRATEGY RESULTS:
   Signals Generated: 490
   Total Trades: 339
   Win Rate: 47.2%
   Total PnL: ₹-21,185.03
   Avg PnL/Trade: ₹-62.49

================================================================================
📊 STRATEGY COMPARISON SUMMARY
================================================================================
                     Strategy  Trades Win Rate   Total PnL  Avg PnL
       Strategy 02 - Tight SL     446    46.4% ₹-25,434.40  ₹-57.03
Strategy 06 - High Confidence     339    47.2% ₹-21,185.03  ₹-62.49
       Strategy 09 - Balanced     442    47.7% ₹-23,171.31  ₹-52.42
================================================================================

🔄 Generating comprehensive strategy comparison report...
✅ Comparison report generated!
📄 Text report: src/simulator/results/run_20250912_210555/strategy_comparison_report.txt
📊 Charts: src/simulator/results/run_20250912_210555/strategy_comparison_charts.png
📋 Detailed CSV: src/simulator/results/run_20250912_210555/strategy_comparison_detailed.csv

🏆 Best Strategy: Strategy 06 - High Confidence
💰 Best PnL: ₹-21,185
📈 Best Win Rate: 48.4%
```

### **Model Evaluation**
```bash
python src/evaluation/model_evaluator.py
```

### **Data Management**
```bash
# Verify downloaded data
python src/data/verify_data.py

# Convert cache to CSV format
python convert_10min_cache_to_csv.py

# Clean old cache files
python src/data/clear_files.py
```

## 🏗️ **Project Architecture**

```
ai_stock_predictions/
├── 📊 data/                    # Raw market data
│   ├── raw/                   # Downloaded OHLCV data
│   │   ├── 10min/            # Intraday data
│   │   └── daily/            # Daily data
│   └── processed/            # Preprocessed data
├── 🧠 src/                     # Source code
│   ├── data/                 # Data handling & downloaders
│   │   ├── reliance_data_downloader.py
│   │   ├── verify_data.py
│   │   └── data_fetch.py
│   ├── preprocessing/        # Feature engineering
│   │   └── data_preprocessor.py
│   ├── models/              # LSTM implementation
│   │   └── enhanced_lstm.py
│   ├── evaluation/          # Model evaluation
│   │   └── model_evaluator.py
│   └── simulator/           # Trading simulation framework
│       ├── multi_strategy_runner.py    # 🎯 Main multi-strategy runner
│       ├── strategy_simulator.py       # Single strategy simulator
│       ├── strategy_base.py           # Strategy framework
│       ├── intraday_core.py          # Core trading logic
│       └── pnl_calculator.py         # P&L calculations
├── 🎯 models/                  # Trained models & results
│   ├── enhanced_stock_lstm.pth      # Trained LSTM model
│   ├── model_metadata.json         # Model configuration
│   ├── latest_prediction.json      # Latest predictions
│   └── *.png                       # Training visualizations
├── 🔧 api/                     # Fyers API integration
│   ├── fyers_data_api.py           # Market data API
│   ├── fyers_session_management.py # Authentication
│   └── generate_accesstoken.py     # Token generation
├── 🛠️ utilities/              # Helper functions
│   └── date_utilities.py
├── 📈 simulation_results/      # Trading simulation outputs
│   ├── trades.csv              # Trade history
│   ├── equity_curve.csv        # Portfolio performance
│   └── performance_report.json # Detailed metrics
├── ⚡ predict.py              # 🎯 Quick prediction script
├── 📋 requirements.txt        # Dependencies
└── 🗂️ src/simulator/results/  # Multi-strategy results
    └── run_YYYYMMDD_HHMMSS/   # Timestamped result folders
        ├── strategy_comparison_report.txt
        ├── strategy_comparison_charts.png
        ├── strategy_XX_trades.csv
        └── strategy_XX_summary.json
```

## 🎯 **Available Commands & Scripts**

### **🧠 AI Prediction Commands**
```bash
# Quick next-day prediction
python predict.py

# Train the LSTM model
python src/models/enhanced_lstm.py

# Evaluate model performance
python src/evaluation/model_evaluator.py
```

### **📊 Data Management Commands**
```bash
# Download fresh market data
python src/data/reliance_data_downloader.py

# Verify data integrity
python src/data/verify_data.py

# Convert 10-min cache to CSV
python convert_10min_cache_to_csv.py

# Clean old cache files
python src/data/clear_files.py

# Test API limits
python src/data/test_fyers_api_limits.py
```

### **🏗️ Trading Simulation Commands**
```bash
# 🎯 Run multi-strategy comparison (RECOMMENDED)
python src/simulator/multi_strategy_runner.py

# Run single strategy simulation
python src/simulator/strategy_simulator.py
```

### **🔧 API & Authentication Commands**
```bash
# Generate Fyers access token
python api/generate_accesstoken.py

# Test Fyers API connection
python api/fyers_data_api.py
```

### **📈 Simulation Output Structure**
After running `multi_strategy_runner.py`, results are saved in timestamped folders:
```
src/simulator/results/run_YYYYMMDD_HHMMSS/
├── strategy_comparison_report.txt       # 📄 Detailed text analysis
├── strategy_comparison_charts.png       # 📊 Visual comparisons
├── strategy_comparison_detailed.csv     # 📋 Detailed CSV data
├── strategy_XX_trades.csv              # Individual strategy trades
├── strategy_XX_summary.json            # Strategy performance metrics
└── strategy_XX_equity_curve.csv        # Portfolio value over time
```

## 🧠 **Model Architecture**

```python
EnhancedStockLSTM(
  (lstm): LSTM(26, 128, num_layers=3, batch_first=True, dropout=0.2)
  (dropout): Dropout(p=0.2)
  (batch_norm): BatchNorm1d(128)
  (fc1): Linear(128, 64)
  (fc2): Linear(64, 32)
  (fc3): Linear(32, 4)  # OHLC outputs
)
```

**Parameters**: 354,788 trainable parameters

## 🎪 **Trading Strategies Available**

The multi-strategy runner includes 9 different trading strategies:

| Strategy | Description | Stop Loss | Take Profit | Confidence |
|----------|-------------|-----------|-------------|------------|
| **Strategy 02** | Tight SL | 1.0% | 3.0% | 70% |
| **Strategy 03** | Reduced TP | 1.5% | 2.5% | 70% |
| **Strategy 04** | Aggressive | 1.5% | 3.5% | 65% |
| **Strategy 05** | Conservative | 2.0% | 4.0% | 75% |
| **Strategy 06** | High Confidence | 1.5% | 4.0% | 80% |
| **Strategy 07** | Wide Stops | 2.5% | 5.0% | 65% |
| **Strategy 08** | Scalping | 0.5% | 1.0% | 60% |
| **Strategy 09** | Balanced | 1.5% | 3.0% | 65% |
| **Strategy 10** | Trend Following | 2.5% | 7.5% | 60% |

Each strategy is automatically tested and compared with detailed performance metrics.

## 📈 **Features Used**

### **Basic OHLCV**
- Open, High, Low, Close, Volume

### **Technical Indicators**
- **Moving Averages**: SMA(5,10,20,50), EMA(5,10,20)
- **Oscillators**: RSI, MACD, MACD Histogram
- **Bands**: Bollinger Bands position
- **Volatility**: 10-day, 20-day volatility
- **Momentum**: 5-day, 10-day momentum
- **Volume**: Volume ratio analysis
- **Price Position**: Support/Resistance levels

## 📊 **Data Pipeline**

1. **Data Collection**: Fyers API integration with chunked requests
2. **Feature Engineering**: 26 technical indicators calculation
3. **Preprocessing**: MinMax scaling and sequence generation
4. **Model Training**: LSTM with early stopping and learning rate scheduling
5. **Evaluation**: Comprehensive metrics and visualizations
6. **Prediction**: Real-time next-day forecasting

## 🔧 **API Configuration**

The project uses **Fyers API** for real-time NSE data. Configure your credentials in `config.ini`:

```ini
[Fyers_APP]
redirect_uri = http://127.0.0.1
client_id = YOUR_CLIENT_ID
secret_key = YOUR_SECRET_KEY
access_token = YOUR_ACCESS_TOKEN
refresh_token = YOUR_REFRESH_TOKEN
```

## 📈 **Training Details**

- **Lookback Window**: 30 days
- **Training Data**: 1,668 sequences
- **Validation Data**: 212 sequences  
- **Test Data**: 456 sequences
- **Batch Size**: 32
- **Optimizer**: Adam with weight decay
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Early Stopping**: Patience of 10 epochs
- **Training Time**: ~5 minutes on CPU

## 🎯 **Results & Visualizations**

The model generates comprehensive visualizations:

- **Training History**: Loss curves and convergence analysis
- **Prediction Plots**: Actual vs predicted OHLC values
- **Correlation Analysis**: R² values for each price component
- **Error Distribution**: Statistical analysis of prediction errors

## ⚠️ **Important Disclaimers**

- 📚 **Educational Purpose**: This project is for learning and research
- 💰 **Not Financial Advice**: Do not use for actual trading decisions
- 📊 **Market Risk**: Stock markets are inherently unpredictable
- 🎯 **Past Performance**: Does not guarantee future results

## 🚀 **Future Enhancements**

### **🔄 AI Model Improvements**
- Multi-timeframe predictions (weekly, monthly)
- Ensemble models (LSTM + Transformer + XGBoost)
- Sentiment analysis from news and social media
- Real-time streaming predictions

### **🏗️ Trading Framework Enhancements**
- Portfolio optimization across multiple stocks
- Risk management with position sizing
- Options trading strategies
- Real-time paper trading mode

### **� Analytics & Visualization**
- Web dashboard for interactive visualization
- Real-time strategy performance monitoring
- Advanced backtesting with market conditions
- Strategy optimization using genetic algorithms

### **🔧 Technical Improvements**
- Docker containerization
- Cloud deployment (AWS/GCP)
- REST API for predictions
- Real-time data streaming

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 **License**

This project is for educational purposes. Please ensure compliance with your local financial regulations.

## 🙏 **Acknowledgments**

- **Fyers API** for providing high-quality market data
- **PyTorch** team for the excellent deep learning framework
- **scikit-learn** for preprocessing utilities
- **Reliance Industries** for being an excellent stock to analyze

---

⭐ **Star this repo if you found it helpful!** ⭐

**Built with ❤️ for learning and education. Happy coding! 🚀**

## 📞 **Support & Contact**

- 🐛 **Issues**: [GitHub Issues](https://github.com/laxminarayanaboga/ai_stock_predictions/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/laxminarayanaboga/ai_stock_predictions/discussions)
- 📧 **Email**: [Contact Developer](mailto:your.email@example.com)

---