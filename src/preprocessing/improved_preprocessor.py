"""
Improved Data Preprocessor with Enhanced Feature Engineering
Addresses issues that might be causing poor model performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class ImprovedStockDataPreprocessor:
    """Enhanced preprocessor with better feature engineering"""
    
    def __init__(self, lookback_days=20, prediction_days=1, scale_method='robust'):
        """
        Initialize the preprocessor
        
        Args:
            lookback_days (int): Number of days to look back for predictions
            prediction_days (int): Number of days ahead to predict
            scale_method (str): 'minmax', 'standard', or 'robust' scaling
        """
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.scale_method = scale_method
        self.scaler = None
        self.feature_columns = None
        self.target_columns = ['Open', 'High', 'Low', 'Close']
        
    def add_improved_technical_indicators(self, df):
        """
        Add improved technical indicators with better signal-to-noise ratio
        """
        data = df.copy()
        
        # Standardize column names to uppercase
        data.columns = data.columns.str.title()
        
        # Basic price features (normalized)
        data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close']
        data['OC_Ratio'] = (data['Close'] - data['Open']) / data['Open']
        
        # Trend strength indicators
        for period in [5, 10, 20]:
            data[f'Trend_{period}'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
            data[f'HL_Trend_{period}'] = (data['High'] - data['Low']) / data['Close'].rolling(period).mean()
        
        # Multi-timeframe moving averages (more robust)
        for period in [5, 10, 20]:
            sma = data['Close'].rolling(period).mean()
            data[f'SMA_{period}_Ratio'] = data['Close'] / sma
            
            # EMA with trend strength
            ema = data['Close'].ewm(span=period).mean()
            data[f'EMA_{period}_Ratio'] = data['Close'] / ema
            data[f'EMA_{period}_Slope'] = (ema - ema.shift(3)) / ema.shift(3)
        
        # Volume-price relationship (key for direction prediction)
        data['Volume_Price_Trend'] = data['Volume'] * data['OC_Ratio']
        data['Volume_Normalized'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        # Volatility regime indicators
        returns = data['Close'].pct_change()
        data['Volatility_Regime'] = returns.rolling(10).std() / returns.rolling(50).std()
        
        # RSI with multiple timeframes
        for period in [7, 14, 21]:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD with signal strength
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        data['MACD_Strength'] = abs(data['MACD_Histogram']) / data['Close']
        
        # Bollinger Bands with position and squeeze
        bb_middle = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_Position'] = (data['Close'] - bb_middle) / (2 * bb_std)
        data['BB_Squeeze'] = bb_std / bb_middle  # Volatility measure
        
        # Support/Resistance breakout indicators
        high_20 = data['High'].rolling(20).max()
        low_20 = data['Low'].rolling(20).min()
        data['Resistance_Break'] = (data['Close'] > high_20.shift(1)).astype(float)
        data['Support_Break'] = (data['Close'] < low_20.shift(1)).astype(float)
        
        # Price momentum quality
        for period in [3, 5, 10]:
            momentum = data['Close'] / data['Close'].shift(period) - 1
            data[f'Momentum_{period}'] = momentum
            data[f'Momentum_{period}_Quality'] = momentum * data['Volume_Normalized']
        
        # Intraday patterns
        data['Intraday_Range'] = (data['High'] - data['Low']) / data['Open']
        data['Open_to_Close'] = (data['Close'] - data['Open']) / data['Open']
        data['Body_to_Range'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low'])
        
        # Market microstructure
        data['Upper_Shadow'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / data['Close']
        data['Lower_Shadow'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data['Close']
        
        return data
    
    def select_best_features(self, data):
        """
        Select the most predictive features, avoiding overfitting
        """
        # Core price features
        core_features = [
            'OC_Ratio', 'HL_Ratio', 'Intraday_Range', 'Open_to_Close',
            'Body_to_Range', 'Upper_Shadow', 'Lower_Shadow'
        ]
        
        # Trend features
        trend_features = [
            'Trend_5', 'Trend_10', 'Trend_20',
            'SMA_5_Ratio', 'SMA_10_Ratio', 'SMA_20_Ratio',
            'EMA_5_Slope', 'EMA_10_Slope'
        ]
        
        # Volume features
        volume_features = [
            'Volume_Normalized', 'Volume_Price_Trend'
        ]
        
        # Technical indicators
        technical_features = [
            'RSI_14', 'MACD_Histogram', 'MACD_Strength',
            'BB_Position', 'BB_Squeeze', 'Volatility_Regime'
        ]
        
        # Momentum features
        momentum_features = [
            'Momentum_3', 'Momentum_5', 'Momentum_10',
            'Momentum_5_Quality'
        ]
        
        # Breakout features
        breakout_features = [
            'Resistance_Break', 'Support_Break'
        ]
        
        selected_features = (core_features + trend_features + volume_features + 
                           technical_features + momentum_features + breakout_features)
        
        return [col for col in selected_features if col in data.columns]
    
    def prepare_data_for_training(self, df):
        """
        Prepare data for training with improved preprocessing
        """
        print("🔧 Preparing data with improved preprocessing...")
        
        # Add technical indicators
        data = self.add_improved_technical_indicators(df)
        
        # Select best features
        self.feature_columns = self.select_best_features(data)
        
        # Remove rows with NaN values
        data = data.dropna()
        
        print(f"📊 Using {len(self.feature_columns)} features:")
        for i, feature in enumerate(self.feature_columns):
            if i % 5 == 0:
                print()
            print(f"  {feature}", end="")
        print()
        
        # Prepare feature matrix
        feature_data = data[self.feature_columns].values
        target_data = data[self.target_columns].values
        
        # Create sequences for LSTM
        X, y = self.create_sequences(feature_data, target_data)
        
        # Split data (80% train, 10% val, 10% test)
        train_size = int(0.8 * len(X))
        val_size = int(0.1 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # Fit scaler on training data only
        if self.scale_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scale_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scale_method == 'robust':
            self.scaler = RobustScaler()
        
        # Reshape for scaling
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        # Scale all sets
        X_train = self.scale_sequences(X_train)
        X_val = self.scale_sequences(X_val)
        X_test = self.scale_sequences(X_test)
        
        # Scale targets
        self.target_scaler = RobustScaler()
        y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
        self.target_scaler.fit(y_train_reshaped)
        
        y_train = self.target_scaler.transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
        y_val = self.target_scaler.transform(y_val.reshape(-1, y_val.shape[-1])).reshape(y_val.shape)
        y_test = self.target_scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
        
        print(f"📈 Data prepared:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples") 
        print(f"  Test: {X_test.shape[0]} samples")
        print(f"  Features: {X_train.shape[2]}")
        print(f"  Sequence length: {X_train.shape[1]}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_columns': self.feature_columns
        }
    
    def create_sequences(self, feature_data, target_data):
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(feature_data) - self.lookback_days - self.prediction_days + 1):
            # Features: lookback_days of historical data
            X.append(feature_data[i:(i + self.lookback_days)])
            
            # Target: OHLC for next day
            y.append(target_data[i + self.lookback_days])
        
        return np.array(X), np.array(y)
    
    def scale_sequences(self, sequences):
        """Scale sequence data"""
        original_shape = sequences.shape
        sequences_reshaped = sequences.reshape(-1, original_shape[-1])
        sequences_scaled = self.scaler.transform(sequences_reshaped)
        return sequences_scaled.reshape(original_shape)
    
    def inverse_transform_targets(self, targets):
        """Convert scaled targets back to original scale"""
        original_shape = targets.shape
        targets_reshaped = targets.reshape(-1, original_shape[-1])
        targets_orig = self.target_scaler.inverse_transform(targets_reshaped)
        return targets_orig.reshape(original_shape)


# Update the multi_model_trainer to use the improved preprocessor
def create_improved_experiment():
    """Create experiment with improved preprocessor"""
    from pathlib import Path
    import pandas as pd
    
    # Load data
    data_files = list(Path("data/raw/daily").glob("RELIANCE_NSE_*.csv"))
    if not data_files:
        raise FileNotFoundError("No Reliance data files found in data/raw/daily/")
    data_file = max(data_files, key=lambda x: x.stat().st_mtime)
    
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Use improved preprocessor
    preprocessor = ImprovedStockDataPreprocessor(
        lookback_days=15,  # Shorter lookback for more recent patterns
        prediction_days=1,
        scale_method='robust'
    )
    
    processed_data = preprocessor.prepare_data_for_training(df)
    return preprocessor, processed_data


if __name__ == "__main__":
    # Test the improved preprocessor
    preprocessor, data = create_improved_experiment()
    print("✅ Improved preprocessor working correctly!")