# 🚀 Setup Instructions

## 📋 **Prerequisites**
- Python 3.10 or higher
- Fyers API account (for data access)

## 🔧 **Quick Setup**

### 1. **Clone and Install**
```bash
git clone https://github.com/your-username/ai_stock_predictions.git
cd ai_stock_predictions
pip install -r requirements.txt
```

### 2. **Configure Fyers API**
```bash
# Copy the example config
cp config.ini.example config.ini

# Edit config.ini with your Fyers API credentials
# Get credentials from: https://myapi.fyers.in/
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

## 📁 **Project Structure**
```
ai_stock_predictions/
├── 📊 data/                    # Data handling
├── 🧠 src/                     # Source code
├── 🎯 models/                  # Model outputs
├── 🔧 api/                     # Fyers API integration
├── 🛠️ utilities/              # Helper functions
├── ⚡ predict.py              # Quick prediction
└── 📋 requirements.txt        # Dependencies
```

## ⚠️ **Important Notes**

- **Educational Purpose**: This is for learning, not financial advice
- **API Limits**: Fyers API has rate limits and daily quotas
- **Model Files**: Large .pth files are excluded from git - retrain locally
- **Data Privacy**: Raw data files are not included in repository

## 🆘 **Troubleshooting**

### Common Issues:
1. **API Authentication**: Check your config.ini credentials
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Import Errors**: Make sure you're in the project root directory
4. **Model Not Found**: Train the model first using `python src/models/enhanced_lstm.py`

## 📞 **Support**
- Check existing issues on GitHub
- Review the comprehensive README.md
- Ensure all prerequisites are met
