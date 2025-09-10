# 🚀 AI Stock Predictions for Reliance NSE

**Advanced LSTM-based Stock Price Prediction System**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![License](https://img.shields.io/badge/License-Educational-yellow)

## 🎯 **Project Overview**

This project implements a sophisticated AI-powered stock prediction system for **Reliance Industries (NSE:RELIANCE-EQ)** using advanced LSTM neural networks. The system analyzes 10 years of historical data and 26 technical indicators to predict next-day OHLC (Open, High, Low, Close) prices.

## ✨ **Key Features**

- 📊 **10 Years of Data**: 2,475 trading days from 2015-2025
- 🧠 **Advanced LSTM Model**: 3-layer architecture with 354K parameters
- 📈 **26 Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, etc.
- 🎯 **High Accuracy**: MAPE of 1.69% and R² of 0.71
- ⚡ **Real-time Predictions**: Next-day OHLC forecasting
- 📱 **User-friendly Interface**: Simple command-line prediction tool

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
python data/reliance_data_downloader.py
```

### 4. **Train Model**
```bash
python src/models/enhanced_lstm.py
```

### 5. **Make Predictions**
```bash
python predict.py
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

### **Model Evaluation**
```bash
python src/evaluation/model_evaluator.py
```

## 🏗️ **Project Architecture**

```
ai_stock_predictions/
├── 📊 data/                    # Data handling
│   ├── raw/                   # Downloaded stock data
│   ├── reliance_data_downloader.py
│   └── verify_data.py
├── 🧠 src/                     # Source code
│   ├── preprocessing/         # Feature engineering
│   ├── models/               # LSTM implementation
│   └── evaluation/           # Model evaluation
├── 🎯 models/                  # Trained models
│   ├── enhanced_stock_lstm.pth
│   ├── model_metadata.json
│   └── *.png                 # Visualizations
├── 🔧 api/                     # Fyers API integration
├── 🛠️ utilities/              # Helper functions
├── ⚡ predict.py              # Quick prediction script
└── 📋 requirements.txt        # Dependencies
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

- 🔄 **Multi-timeframe predictions** (weekly, monthly)
- 🔄 **Ensemble models** (LSTM + Transformer + XGBoost)
- 🔄 **Sentiment analysis** from news and social media
- 🔄 **Portfolio optimization** across multiple stocks
- 🔄 **Web dashboard** for interactive visualization
- 🔄 **Real-time streaming** predictions

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

**Built with ❤️ for learning and education. Happy coding! 🚀**