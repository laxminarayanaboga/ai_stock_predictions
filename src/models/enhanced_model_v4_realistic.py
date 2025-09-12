"""
Enhanced Stock Prediction Model v4 - Realistic Implementation
Works with available 3-year dataset and focuses on practical improvements
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

class RealisticStockLSTM_V4(nn.Module):
    """
    Realistic LSTM model v4 that enhances v2_attention with:
    1. Better intraday pattern recognition (using available data)
    2. Enhanced technical indicators
    3. Multi-timeframe analysis (10min + hourly + daily aggregations)
    4. Improved feature engineering
    
    Note: Uses ONLY available data - no pre-market or extended historical data
    """
    
    def __init__(self, config):
        super(RealisticStockLSTM_V4, self).__init__()
        
        # Feature dimensions (realistic based on available data)
        self.base_features = config.get('base_features', 29)  # From v2_attention
        self.technical_features = config.get('technical_features', 15)  # Enhanced technicals
        self.pattern_features = config.get('pattern_features', 10)  # Pattern recognition
        self.timeframe_features = config.get('timeframe_features', 12)  # Multi-timeframe
        
        # Total realistic features - use config override if provided
        if 'total_features' in config:
            self.total_features = config['total_features']
            print(f"🔧 Using configured total_features: {self.total_features}")
        else:
            self.total_features = (self.base_features + self.technical_features + 
                                  self.pattern_features + self.timeframe_features)
            print(f"🔧 Calculated total_features: {self.total_features}")
        
        # Architecture (keep similar to v2 for compatibility)
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.2)
        self.lookback_days = config.get('lookback_days', 15)
        
        # Enhanced LSTM with multi-head attention (like v2)
        self.lstm = nn.LSTM(
            input_size=self.total_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Multi-head attention (enhanced from v2)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=self.dropout
        )
        
        # Feature importance weighting
        self.feature_weights = nn.Parameter(torch.ones(4))  # 4 feature groups
        
        # Output layers (same as v2 for compatibility)
        self.price_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 4)  # OHLC
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.direction_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 4, 3),  # up, down, sideways
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        """
        Forward pass - compatible with v2_attention interface
        
        Args:
            x: (batch, sequence, total_features)
        """
        batch_size = x.size(0)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply multi-head attention
        lstm_out_permuted = lstm_out.permute(1, 0, 2)  # (seq, batch, features)
        attended_out, attention_weights = self.attention(
            lstm_out_permuted, lstm_out_permuted, lstm_out_permuted
        )
        
        # Take the last timestep
        final_hidden = attended_out[-1]  # (batch, hidden_size)
        
        # Generate predictions (same format as v2)
        price_pred = self.price_predictor(final_hidden)
        confidence_pred = self.confidence_predictor(final_hidden)
        direction_pred = self.direction_predictor(final_hidden)
        
        return {
            'price_prediction': price_pred,
            'confidence': confidence_pred,
            'direction': direction_pred,
            'attention_weights': attention_weights,
            'features': final_hidden
        }


class RealisticDataPreprocessor_V4:
    """
    Enhanced data preprocessor that works with available 3-year dataset
    Focuses on better feature engineering with existing data
    """
    
    def __init__(self, lookback_days=15, max_historical_days=30):
        self.lookback_days = lookback_days
        self.max_historical_days = max_historical_days  # Limited by our 3-year dataset
        self.logger = logging.getLogger(__name__)
    
    def extract_enhanced_technical_indicators(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract enhanced technical indicators using available intraday data
        """
        if len(data) < 20:  # Need minimum data
            return np.zeros(15)
        
        features = []
        
        # 1. Enhanced RSI (multiple periods)
        close_prices = data['close'].values
        
        # RSI 14-period
        gains = np.diff(close_prices)
        gains[gains < 0] = 0
        losses = -np.diff(close_prices)
        losses[losses < 0] = 0
        
        if len(gains) >= 14:
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi_14 = 100 - (100 / (1 + rs))
            features.append(rsi_14 / 100)  # Normalize
        else:
            features.append(0.5)
        
        # 2. MACD
        if len(close_prices) >= 26:
            ema_12 = self._calculate_ema(close_prices, 12)
            ema_26 = self._calculate_ema(close_prices, 26)
            macd = ema_12 - ema_26
            signal = self._calculate_ema([macd], 9)[0] if len([macd]) >= 9 else macd
            macd_histogram = macd - signal
            
            # Normalize MACD features
            price_std = np.std(close_prices[-26:])
            features.extend([
                macd / price_std if price_std > 0 else 0,
                signal / price_std if price_std > 0 else 0,
                macd_histogram / price_std if price_std > 0 else 0
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 3. Bollinger Bands
        if len(close_prices) >= 20:
            sma_20 = np.mean(close_prices[-20:])
            std_20 = np.std(close_prices[-20:])
            upper_band = sma_20 + (2 * std_20)
            lower_band = sma_20 - (2 * std_20)
            current_price = close_prices[-1]
            
            # BB position and width
            bb_position = (current_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
            bb_width = (upper_band - lower_band) / sma_20 if sma_20 > 0 else 0
            
            features.extend([bb_position, bb_width])
        else:
            features.extend([0.5, 0.0])
        
        # 4. Stochastic Oscillator
        if len(data) >= 14:
            high_14 = data['high'].rolling(14).max().iloc[-1]
            low_14 = data['low'].rolling(14).min().iloc[-1]
            current_close = data['close'].iloc[-1]
            
            stoch_k = (current_close - low_14) / (high_14 - low_14) * 100 if (high_14 - low_14) > 0 else 50
            features.append(stoch_k / 100)  # Normalize
        else:
            features.append(0.5)
        
        # 5. ATR (Average True Range)
        if len(data) >= 14:
            tr_values = []
            for i in range(1, min(len(data), 15)):
                high_low = data['high'].iloc[-i] - data['low'].iloc[-i]
                high_close_prev = abs(data['high'].iloc[-i] - data['close'].iloc[-i-1])
                low_close_prev = abs(data['low'].iloc[-i] - data['close'].iloc[-i-1])
                tr = max(high_low, high_close_prev, low_close_prev)
                tr_values.append(tr)
            
            atr = np.mean(tr_values) if tr_values else 0
            atr_ratio = atr / data['close'].iloc[-1] if data['close'].iloc[-1] > 0 else 0
            features.append(atr_ratio)
        else:
            features.append(0.0)
        
        # 6. Volume indicators
        volumes = data['volume'].values
        if len(volumes) >= 10:
            # Volume SMA ratio
            vol_sma_10 = np.mean(volumes[-10:])
            current_vol = volumes[-1]
            vol_ratio = current_vol / vol_sma_10 if vol_sma_10 > 0 else 1
            
            # Volume trend
            vol_trend = (np.mean(volumes[-5:]) - np.mean(volumes[-10:-5])) / np.mean(volumes[-10:-5]) if np.mean(volumes[-10:-5]) > 0 else 0
            
            features.extend([vol_ratio, vol_trend])
        else:
            features.extend([1.0, 0.0])
        
        # 7. Price momentum indicators
        if len(close_prices) >= 10:
            # Rate of change
            roc_5 = (close_prices[-1] - close_prices[-6]) / close_prices[-6] * 100 if len(close_prices) > 5 else 0
            roc_10 = (close_prices[-1] - close_prices[-11]) / close_prices[-11] * 100 if len(close_prices) > 10 else 0
            
            features.extend([roc_5 / 100, roc_10 / 100])  # Normalize
        else:
            features.extend([0.0, 0.0])
        
        # 8. Support/Resistance levels
        if len(data) >= 20:
            recent_highs = data['high'].rolling(5).max().dropna()
            recent_lows = data['low'].rolling(5).min().dropna()
            current_price = data['close'].iloc[-1]
            
            if len(recent_highs) > 0 and len(recent_lows) > 0:
                resistance = recent_highs.max()
                support = recent_lows.min()
                
                resistance_distance = (resistance - current_price) / current_price if current_price > 0 else 0
                support_distance = (current_price - support) / current_price if current_price > 0 else 0
                
                features.extend([resistance_distance, support_distance])
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features[:15])  # Ensure exactly 15 features
    
    def extract_pattern_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract candlestick pattern and price action features
        """
        if len(data) < 5:
            return np.zeros(10)
        
        features = []
        recent_data = data.tail(5)  # Last 5 candles
        
        # 1. Doji patterns
        doji_count = 0
        for _, candle in recent_data.iterrows():
            body = abs(candle['close'] - candle['open'])
            range_hl = candle['high'] - candle['low']
            if range_hl > 0 and body / range_hl < 0.1:  # Doji threshold
                doji_count += 1
        features.append(doji_count / 5)  # Normalize
        
        # 2. Hammer/Shooting star patterns
        hammer_count = 0
        for _, candle in recent_data.iterrows():
            body = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['close'], candle['open'])
            lower_shadow = min(candle['close'], candle['open']) - candle['low']
            range_hl = candle['high'] - candle['low']
            
            if range_hl > 0:
                if lower_shadow > 2 * body and upper_shadow < body:  # Hammer
                    hammer_count += 1
                elif upper_shadow > 2 * body and lower_shadow < body:  # Shooting star
                    hammer_count += 1
        features.append(hammer_count / 5)  # Normalize
        
        # 3. Engulfing patterns
        engulfing_signals = 0
        for i in range(1, len(recent_data)):
            prev_candle = recent_data.iloc[i-1]
            curr_candle = recent_data.iloc[i]
            
            prev_body = abs(prev_candle['close'] - prev_candle['open'])
            curr_body = abs(curr_candle['close'] - curr_candle['open'])
            
            if curr_body > prev_body * 1.5:  # Current candle engulfs previous
                engulfing_signals += 1
        features.append(engulfing_signals / 4)  # Normalize
        
        # 4. Gap analysis
        gaps = 0
        for i in range(1, len(recent_data)):
            prev_close = recent_data.iloc[i-1]['close']
            curr_open = recent_data.iloc[i]['open']
            gap_pct = abs(curr_open - prev_close) / prev_close * 100
            if gap_pct > 0.5:  # Significant gap
                gaps += 1
        features.append(gaps / 4)  # Normalize
        
        # 5. Price action patterns
        consecutive_up = 0
        consecutive_down = 0
        for i in range(len(recent_data)):
            candle = recent_data.iloc[i]
            if candle['close'] > candle['open']:
                consecutive_up += 1
            elif candle['close'] < candle['open']:
                consecutive_down += 1
        
        features.extend([consecutive_up / 5, consecutive_down / 5])  # Normalize
        
        # 6. Volume-price confirmation
        vol_price_confirm = 0
        for i in range(1, len(recent_data)):
            prev_vol = recent_data.iloc[i-1]['volume']
            curr_vol = recent_data.iloc[i]['volume']
            prev_close = recent_data.iloc[i-1]['close']
            curr_close = recent_data.iloc[i]['close']
            
            if curr_vol > prev_vol and abs(curr_close - prev_close) / prev_close > 0.01:
                vol_price_confirm += 1
        features.append(vol_price_confirm / 4)  # Normalize
        
        # 7. High-low analysis
        if len(recent_data) >= 3:
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
            lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
            
            features.extend([higher_highs / 4, lower_lows / 4])  # Normalize
        else:
            features.extend([0.0, 0.0])
        
        # 8. Intraday strength
        if not recent_data.empty:
            last_candle = recent_data.iloc[-1]
            range_hl = last_candle['high'] - last_candle['low']
            close_position = (last_candle['close'] - last_candle['low']) / range_hl if range_hl > 0 else 0.5
            features.append(close_position)
        else:
            features.append(0.5)
        
        return np.array(features[:10])  # Ensure exactly 10 features
    
    def extract_timeframe_features(self, data: pd.DataFrame, target_date: str) -> np.ndarray:
        """
        Extract multi-timeframe features using data aggregation
        """
        if len(data) < 30:  # Need sufficient data
            return np.zeros(12)
        
        target_date_obj = pd.to_datetime(target_date).date()
        
        # Get historical data before target date
        historical_data = data[data['datetime_ist'].dt.date < target_date_obj].copy()
        
        if historical_data.empty:
            return np.zeros(12)
        
        features = []
        
        # 1. Hourly aggregation (if we have enough intraday data)
        try:
            hourly_data = historical_data.set_index('datetime_ist').resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min', 
                'close': 'last',
                'volume': 'sum'
            }).dropna().tail(24)  # Last 24 hours
            
            if len(hourly_data) >= 6:  # Need at least 6 hours
                hourly_returns = hourly_data['close'].pct_change().dropna()
                hourly_vol_trend = (hourly_data['volume'].tail(6).mean() - 
                                  hourly_data['volume'].head(6).mean()) / hourly_data['volume'].head(6).mean() if hourly_data['volume'].head(6).mean() > 0 else 0
                
                features.extend([
                    hourly_returns.mean(),
                    hourly_returns.std(),
                    hourly_vol_trend
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        except:
            features.extend([0.0, 0.0, 0.0])
        
        # 2. Daily aggregation
        try:
            daily_data = historical_data.groupby(historical_data['datetime_ist'].dt.date).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).tail(min(30, self.max_historical_days))  # Limited by our dataset
            
            if len(daily_data) >= 5:
                daily_returns = daily_data['close'].pct_change().dropna()
                daily_momentum = (daily_data['close'].iloc[-1] - daily_data['close'].iloc[0]) / daily_data['close'].iloc[0] if len(daily_data) > 1 else 0
                daily_volatility = daily_returns.std() if len(daily_returns) > 1 else 0
                
                # Volume analysis
                avg_volume = daily_data['volume'].mean()
                recent_volume = daily_data['volume'].tail(5).mean()
                volume_trend = (recent_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
                
                features.extend([
                    daily_momentum,
                    daily_volatility,
                    volume_trend
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        except:
            features.extend([0.0, 0.0, 0.0])
        
        # 3. Weekly patterns (if we have enough data)
        try:
            # Group by week
            historical_data['week'] = historical_data['datetime_ist'].dt.isocalendar().week
            weekly_data = historical_data.groupby('week').agg({
                'close': 'last',
                'volume': 'sum'
            }).tail(4)  # Last 4 weeks
            
            if len(weekly_data) >= 2:
                weekly_returns = weekly_data['close'].pct_change().dropna()
                weekly_momentum = weekly_returns.mean() if len(weekly_returns) > 0 else 0
                weekly_vol_change = (weekly_data['volume'].iloc[-1] - weekly_data['volume'].iloc[0]) / weekly_data['volume'].iloc[0] if len(weekly_data) > 1 and weekly_data['volume'].iloc[0] > 0 else 0
                
                features.extend([weekly_momentum, weekly_vol_change])
            else:
                features.extend([0.0, 0.0])
        except:
            features.extend([0.0, 0.0])
        
        # 4. Trend consistency across timeframes
        try:
            if len(daily_data) >= 5:
                short_trend = (daily_data['close'].tail(3).mean() - daily_data['close'].head(3).mean()) / daily_data['close'].head(3).mean()
                long_trend = (daily_data['close'].tail(5).mean() - daily_data['close'].head(5).mean()) / daily_data['close'].head(5).mean()
                
                trend_alignment = 1 if (short_trend > 0 and long_trend > 0) or (short_trend < 0 and long_trend < 0) else 0
                trend_strength = abs(short_trend) + abs(long_trend)
                
                features.extend([trend_alignment, trend_strength])
            else:
                features.extend([0.0, 0.0])
        except:
            features.extend([0.0, 0.0])
        
        # 5. Market regime detection
        try:
            if len(daily_data) >= 10:
                recent_volatility = daily_data['close'].pct_change().tail(5).std()
                historical_volatility = daily_data['close'].pct_change().std()
                
                volatility_regime = recent_volatility / historical_volatility if historical_volatility > 0 else 1
                features.append(min(volatility_regime, 3.0))  # Cap at 3x
            else:
                features.append(1.0)
        except:
            features.append(1.0)
        
        return np.array(features[:12])  # Ensure exactly 12 features
    
    def _calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def prepare_v4_training_data(self, data: pd.DataFrame, predictions_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare enhanced training data while maintaining v2 compatibility
        """
        X = []
        y = []
        
        # Get unique trading dates
        trading_dates = sorted(data['datetime_ist'].dt.date.unique())
        
        # Skip early dates that don't have enough historical context
        start_index = max(5, self.max_historical_days // 7)  # Start after we have some history
        
        for date in trading_dates[start_index:]:
            date_str = date.strftime('%Y-%m-%d')
            
            # Skip if no prediction available
            if date_str not in predictions_dict:
                continue
            
            try:
                # Get day's data
                day_data = data[data['datetime_ist'].dt.date == date].copy()
                if len(day_data) < self.lookback_days:
                    continue
                
                # Get market hours data (9:15 AM onwards)
                market_data = day_data[day_data['datetime_ist'].dt.time >= time(9, 15)]
                if len(market_data) < self.lookback_days:
                    continue
                
                # Extract base features (from existing v2 logic - simplified here)
                base_features = self._extract_base_features(market_data)
                
                # Extract enhanced features
                technical_features = self.extract_enhanced_technical_indicators(market_data)
                pattern_features = self.extract_pattern_features(market_data)
                timeframe_features = self.extract_timeframe_features(data, date_str)
                
                # Combine all features
                all_features = np.concatenate([
                    base_features.flatten(),
                    technical_features,
                    pattern_features,
                    timeframe_features
                ])
                
                # Reshape for LSTM (sequence_length, features)
                sequence_features = all_features.reshape(self.lookback_days, -1)
                
                # Get target values (same format as v2)
                prediction = predictions_dict[date_str]
                target = [
                    prediction['predicted_open'],
                    prediction['predicted_high'],
                    prediction['predicted_low'],
                    prediction['predicted_close']
                ]
                
                X.append(sequence_features)
                y.append(target)
                
            except Exception as e:
                self.logger.warning(f"Error processing date {date_str}: {e}")
                continue
        
        return np.array(X), np.array(y)
    
    def _extract_base_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract base features (placeholder - would use your existing v2 logic)"""
        # This would use your existing v2_attention feature extraction
        # For now, returning basic OHLCV features to maintain structure
        features = []
        
        for _, row in market_data.head(self.lookback_days).iterrows():
            # Basic OHLCV + some derived features to match v2's 29 features
            row_features = [
                row['open'], row['high'], row['low'], row['close'], row['volume'],
                # Add derived features to reach 29 total
                row['high'] - row['low'],  # Range
                abs(row['close'] - row['open']),  # Body
                (row['close'] - row['open']) / row['open'] if row['open'] > 0 else 0,  # Return
                row['volume'] / 1000,  # Volume scaled
            ]
            # Pad to reach expected base features count
            while len(row_features) < 29:
                row_features.append(0.0)
            
            features.extend(row_features[:29])  # Ensure exactly 29 per row
        
        # Pad or truncate to expected size
        expected_size = self.lookback_days * 29
        if len(features) < expected_size:
            features.extend([0] * (expected_size - len(features)))
        
        return np.array(features[:expected_size]).reshape(self.lookback_days, 29)


def create_v4_model_config():
    """Create configuration for v4 model (realistic enhancement)"""
    return {
        'base_features': 29,  # Keep compatible with v2
        'technical_features': 15,  # Enhanced technical indicators
        'pattern_features': 10,  # Pattern recognition
        'timeframe_features': 12,  # Multi-timeframe analysis
        'hidden_size': 128,  # Same as v2
        'num_layers': 3,  # Same as v2
        'dropout': 0.2,  # Same as v2
        'lookback_days': 15,  # Same as v2
        'max_historical_days': 30,  # Limited by our 3-year dataset
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'model_version': 'v4_realistic'
    }


if __name__ == "__main__":
    # Example usage
    config = create_v4_model_config()
    model = RealisticStockLSTM_V4(config)
    preprocessor = RealisticDataPreprocessor_V4()
    
    print("Realistic Stock Prediction Model v4 Created!")
    print(f"Total input features: {model.total_features}")
    print("Features breakdown:")
    print(f"  - Base (v2 compatible): {model.base_features}")
    print(f"  - Enhanced Technical: {model.technical_features}")
    print(f"  - Pattern Recognition: {model.pattern_features}")
    print(f"  - Multi-timeframe: {model.timeframe_features}")
    print(f"\nModel maintains v2_attention compatibility")
    print(f"Uses only available 3-year dataset (2022-2025)")
    print(f"No pre-market or extended historical data required")