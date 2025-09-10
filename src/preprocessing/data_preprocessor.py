"""
Advanced Data Preprocessor for Stock Market Data
Handles feature engineering, technical indicators, and data preparation for LSTM models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class StockDataPreprocessor:
    def __init__(self, lookback_days=30, prediction_days=1, scale_method='minmax'):
        """
        Initialize the preprocessor
        
        Args:
            lookback_days (int): Number of days to look back for predictions
            prediction_days (int): Number of days ahead to predict
            scale_method (str): 'minmax' or 'standard' scaling
        """
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.scale_method = scale_method
        self.scaler = None
        self.feature_columns = None
        self.target_columns = ['Open', 'High', 'Low', 'Close']
        
    def add_technical_indicators(self, df):
        """
        Add comprehensive technical indicators to the dataframe
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Enhanced data with technical indicators
        """
        data = df.copy()
        
        # Price-based indicators
        data['Price_Range'] = data['High'] - data['Low']
        data['Price_Change'] = data['Close'] - data['Open']
        data['Price_Change_Pct'] = data['Price_Change'] / data['Open']
        
        # Moving Averages
        for period in [5, 10, 20, 50]:
            data[f'SMA_{period}'] = data['Close'].rolling(period).mean()
            data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
            
        # Price relative to moving averages
        data['Close_to_SMA20'] = data['Close'] / data['SMA_20']
        data['Close_to_SMA50'] = data['Close'] / data['SMA_50']
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Volatility indicators
        data['Returns'] = data['Close'].pct_change()
        data['Volatility_10'] = data['Returns'].rolling(10).std()
        data['Volatility_20'] = data['Returns'].rolling(20).std()
        
        # Support and Resistance levels
        data['Resistance_20'] = data['High'].rolling(20).max()
        data['Support_20'] = data['Low'].rolling(20).min()
        data['Price_Position'] = (data['Close'] - data['Support_20']) / (data['Resistance_20'] - data['Support_20'])
        
        # Momentum indicators
        data['Momentum_5'] = data['Close'] / data['Close'].shift(5)
        data['Momentum_10'] = data['Close'] / data['Close'].shift(10)
        
        # Gap indicators
        data['Gap_Up'] = (data['Open'] > data['High'].shift(1)).astype(int)
        data['Gap_Down'] = (data['Open'] < data['Low'].shift(1)).astype(int)
        
        return data
    
    def prepare_features(self, df):
        """
        Select and prepare features for model training
        
        Args:
            df (pd.DataFrame): Data with technical indicators
            
        Returns:
            pd.DataFrame: Prepared features
        """
        # Select features for the model
        basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        technical_features = [
            'Price_Range', 'Price_Change_Pct',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_20',
            'Close_to_SMA20', 'Close_to_SMA50',
            'BB_Position', 'RSI', 'MACD', 'MACD_Histogram',
            'Volume_Ratio', 'Volatility_10', 'Volatility_20',
            'Price_Position', 'Momentum_5', 'Momentum_10'
        ]
        
        # Combine all features
        self.feature_columns = basic_features + technical_features
        
        # Select available features (some might not exist due to insufficient data)
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        return df[available_features].copy()
    
    def create_sequences(self, data, target_data=None):
        """
        Create sequences for LSTM training
        
        Args:
            data (pd.DataFrame): Feature data
            target_data (pd.DataFrame): Target data (if None, uses OHLC from data)
            
        Returns:
            tuple: (X, y) sequences
        """
        if target_data is None:
            target_data = data[self.target_columns]
        
        X, y = [], []
        
        for i in range(len(data) - self.lookback_days - self.prediction_days + 1):
            # Features: lookback_days of all features
            X.append(data.iloc[i:i + self.lookback_days].values)
            
            # Target: next day's OHLC
            target_idx = i + self.lookback_days + self.prediction_days - 1
            y.append(target_data.iloc[target_idx].values)
        
        return np.array(X), np.array(y)
    
    def fit_scaler(self, data):
        """
        Fit the scaler on training data
        
        Args:
            data (pd.DataFrame): Training data
        """
        if self.scale_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        self.scaler.fit(data)
        print(f"Scaler fitted with {self.scale_method} method")
        
    def transform_data(self, data):
        """
        Transform data using fitted scaler
        
        Args:
            data (pd.DataFrame): Data to transform
            
        Returns:
            pd.DataFrame: Scaled data
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        scaled_data = self.scaler.transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    
    def inverse_transform_targets(self, predictions):
        """
        Inverse transform predictions back to original scale
        
        Args:
            predictions (np.array): Scaled predictions
            
        Returns:
            np.array: Original scale predictions
        """
        # Create a dummy array with all features
        dummy = np.zeros((len(predictions), len(self.feature_columns)))
        
        # Find indices of OHLC in feature columns
        ohlc_indices = [self.feature_columns.index(col) for col in self.target_columns]
        
        # Place predictions in correct positions
        dummy[:, ohlc_indices] = predictions
        
        # Inverse transform
        inverse_transformed = self.scaler.inverse_transform(dummy)
        
        # Return only OHLC columns
        return inverse_transformed[:, ohlc_indices]
    
    def prepare_data_for_training(self, df, test_size=0.2, val_size=0.1):
        """
        Complete preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            test_size (float): Test set proportion
            val_size (float): Validation set proportion
            
        Returns:
            dict: Processed data ready for training
        """
        print("Starting data preprocessing pipeline...")
        
        # Add technical indicators
        print("Adding technical indicators...")
        enhanced_data = self.add_technical_indicators(df)
        
        # Prepare features
        print("Preparing features...")
        feature_data = self.prepare_features(enhanced_data)
        
        # Remove rows with NaN values (due to technical indicators)
        feature_data = feature_data.dropna()
        print(f"Data shape after removing NaN: {feature_data.shape}")
        
        # Time-based split (important for time series)
        total_samples = len(feature_data)
        train_end = int(total_samples * (1 - test_size - val_size))
        val_end = int(total_samples * (1 - test_size))
        
        train_data = feature_data.iloc[:train_end]
        val_data = feature_data.iloc[train_end:val_end]
        test_data = feature_data.iloc[val_end:]
        
        print(f"Train: {len(train_data)} samples")
        print(f"Validation: {len(val_data)} samples") 
        print(f"Test: {len(test_data)} samples")
        
        # Fit scaler on training data only
        self.fit_scaler(train_data)
        
        # Transform all datasets
        train_scaled = self.transform_data(train_data)
        val_scaled = self.transform_data(val_data)
        test_scaled = self.transform_data(test_data)
        
        # Create sequences
        print("Creating sequences...")
        X_train, y_train = self.create_sequences(train_scaled)
        X_val, y_val = self.create_sequences(val_scaled)
        X_test, y_test = self.create_sequences(test_scaled)
        
        print(f"Training sequences: X={X_train.shape}, y={y_train.shape}")
        print(f"Validation sequences: X={X_val.shape}, y={y_val.shape}")
        print(f"Test sequences: X={X_test.shape}, y={y_test.shape}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'raw_data': {
                'train': train_data,
                'val': val_data,
                'test': test_data
            },
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }


def main():
    """Test the preprocessor with Reliance data"""
    from pathlib import Path
    
    # Load the data
    data_file = Path("data/raw").glob("RELIANCE_NSE_*.csv")
    data_file = max(data_file, key=lambda x: x.stat().st_mtime)
    
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Initialize preprocessor
    preprocessor = StockDataPreprocessor(
        lookback_days=30,
        prediction_days=1,
        scale_method='minmax'
    )
    
    # Prepare data
    processed_data = preprocessor.prepare_data_for_training(df)
    
    print("\n=== Preprocessing Complete ===")
    print("Data is ready for LSTM model training!")
    
    return processed_data, preprocessor


if __name__ == "__main__":
    main()
