"""
Signal Generators - Pure signal generation from AI predictions
Each signal generator takes predictions and market data, returns trading signals
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import time
import pandas as pd


@dataclass
class TradingSignal:
    """Standard trading signal format"""
    date: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    entry_price: float
    target_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    signal_strength: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseSignalGenerator(ABC):
    """Base class for all signal generators"""
    
    def __init__(self, name: str, min_confidence: float = 0.6, 
                 validate_open_price: bool = False, max_open_error_pct: float = 0.03):
        self.name = name
        self.min_confidence = min_confidence
        self.validate_open_price = validate_open_price
        self.max_open_error_pct = max_open_error_pct
    
    @abstractmethod
    def generate_signal(self, date_str: str, predictions: Dict, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate trading signal for a given date"""
        pass
    
    def should_trade(self, signal: TradingSignal) -> bool:
        """Check if signal meets minimum criteria for trading"""
        if signal is None or signal.confidence < self.min_confidence:
            return False
        
        # Add open price validation if enabled
        if self.validate_open_price:
            return self._validate_open_price(signal)
        
        return True
    
    def _validate_open_price(self, signal: TradingSignal) -> bool:
        """Validate predicted open price against actual opening price"""
        if not signal.metadata or 'actual_open' not in signal.metadata:
            # If no actual open price available, skip validation
            return True
        
        predicted_open = signal.metadata.get('predicted_open', signal.entry_price)
        actual_open = signal.metadata['actual_open']
        
        # Calculate error percentage
        error_pct = abs(predicted_open - actual_open) / actual_open
        
        # Check if error is within tolerance
        is_valid = error_pct <= self.max_open_error_pct
        
        # Store validation result in metadata
        signal.metadata['open_validation'] = {
            'predicted_open': predicted_open,
            'actual_open': actual_open,
            'error_pct': error_pct,
            'is_valid': is_valid,
            'max_allowed_error_pct': self.max_open_error_pct
        }
        
        return is_valid


class OpenCloseSignalGenerator(BaseSignalGenerator):
    """
    Traditional strategy: Buy if predicted close > predicted open, else sell
    Entry at market open
    """
    
    def __init__(self, min_confidence: float = 0.6, min_change_pct: float = 0.005,
                 validate_open_price: bool = False, max_open_error_pct: float = 0.03):
        super().__init__("OpenClose", min_confidence, validate_open_price, max_open_error_pct)
        self.min_change_pct = min_change_pct
    
    def generate_signal(self, date_str: str, predictions: Dict, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        # Get prediction for this date
        prediction = predictions.get(date_str)
        if not prediction:
            return None
        
        # Extract prediction data
        predicted = prediction['predicted']
        confidence = prediction.get('confidence', 0.5) / 100.0  # Convert to 0-1 scale
        
        predicted_open = predicted['Open']
        predicted_close = predicted['Close']
        
        # Calculate expected change
        expected_change_pct = (predicted_close - predicted_open) / predicted_open
        
        # Skip if change is too small
        if abs(expected_change_pct) < self.min_change_pct:
            return None
        
        # Determine signal direction
        signal_type = 'BUY' if predicted_close > predicted_open else 'SELL'
        
        return TradingSignal(
            date=date_str,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=predicted_open,
            target_price=predicted_close,
            signal_strength=abs(expected_change_pct),
            metadata={
                'predicted_open': predicted_open,
                'predicted_close': predicted_close,
                'expected_change_pct': expected_change_pct
            }
        )


class HighLowSignalGenerator(BaseSignalGenerator):
    """
    High-Low strategy: Enter at predicted low/high, target opposite extreme
    For bullish days: enter at low, target high
    For bearish days: enter at high, target low
    """
    
    def __init__(self, min_confidence: float = 0.6, min_hl_spread_pct: float = 0.01,
                 validate_open_price: bool = False, max_open_error_pct: float = 0.03):
        super().__init__("HighLow", min_confidence, validate_open_price, max_open_error_pct)
        self.min_hl_spread_pct = min_hl_spread_pct
    
    def generate_signal(self, date_str: str, predictions: Dict, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        # Get prediction for this date
        prediction = predictions.get(date_str)
        if not prediction:
            return None
        
        # Extract prediction data
        predicted = prediction['predicted']
        confidence = prediction.get('confidence', 0.5) / 100.0  # Convert to 0-1 scale
        
        predicted_open = predicted['Open']
        predicted_high = predicted['High']
        predicted_low = predicted['Low']
        predicted_close = predicted['Close']
        
        # Calculate high-low spread
        hl_spread_pct = (predicted_high - predicted_low) / predicted_open
        
        # Skip if spread is too small
        if hl_spread_pct < self.min_hl_spread_pct:
            return None
        
        # Determine if bullish or bearish
        is_bullish = predicted_close > predicted_open
        
        if is_bullish:
            # Bullish: Enter at low, target high
            signal_type = 'BUY'
            entry_price = predicted_low
            target_price = predicted_high
            signal_strength = (predicted_close - predicted_open) / predicted_open
        else:
            # Bearish: Enter at high, target low
            signal_type = 'SELL'
            entry_price = predicted_high
            target_price = predicted_low
            signal_strength = (predicted_open - predicted_close) / predicted_open
        
        return TradingSignal(
            date=date_str,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            signal_strength=abs(signal_strength),
            metadata={
                'predicted_open': predicted_open,
                'predicted_high': predicted_high,
                'predicted_low': predicted_low,
                'predicted_close': predicted_close,
                'is_bullish': is_bullish,
                'hl_spread_pct': hl_spread_pct
            }
        )


class MeanReversionSignalGenerator(BaseSignalGenerator):
    """
    Mean reversion strategy: Trade against extreme predictions
    If predicted high is very high, sell. If predicted low is very low, buy.
    """
    
    def __init__(self, min_confidence: float = 0.7, extreme_threshold_pct: float = 0.02,
                 validate_open_price: bool = False, max_open_error_pct: float = 0.03):
        super().__init__("MeanReversion", min_confidence, validate_open_price, max_open_error_pct)
        self.extreme_threshold_pct = extreme_threshold_pct
    
    def generate_signal(self, date_str: str, predictions: Dict, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        # Get prediction for this date
        prediction = predictions.get(date_str)
        if not prediction:
            return None
        
        # Extract prediction data
        predicted = prediction['predicted']
        confidence = prediction.get('confidence', 0.5) / 100.0
        
        predicted_open = predicted['Open']
        predicted_high = predicted['High']
        predicted_low = predicted['Low']
        
        # Calculate extremes
        high_move_pct = (predicted_high - predicted_open) / predicted_open
        low_move_pct = (predicted_open - predicted_low) / predicted_open
        
        # Check for extreme moves
        if high_move_pct > self.extreme_threshold_pct:
            # High is extreme, sell at high expecting reversion
            return TradingSignal(
                date=date_str,
                signal_type='SELL',
                confidence=confidence,
                entry_price=predicted_high,
                target_price=predicted_open,  # Target back to open
                signal_strength=high_move_pct,
                metadata={
                    'predicted_open': predicted_open,
                    'predicted_high': predicted_high,
                    'predicted_low': predicted_low,
                    'extreme_type': 'high',
                    'extreme_move_pct': high_move_pct
                }
            )
        
        elif low_move_pct > self.extreme_threshold_pct:
            # Low is extreme, buy at low expecting reversion
            return TradingSignal(
                date=date_str,
                signal_type='BUY',
                confidence=confidence,
                entry_price=predicted_low,
                target_price=predicted_open,  # Target back to open
                signal_strength=low_move_pct,
                metadata={
                    'predicted_open': predicted_open,
                    'predicted_high': predicted_high,
                    'predicted_low': predicted_low,
                    'extreme_type': 'low',
                    'extreme_move_pct': low_move_pct
                }
            )
        
        return None


class BreakoutSignalGenerator(BaseSignalGenerator):
    """
    Breakout strategy: Enter when predicted move is very large
    High confidence + large predicted range = strong breakout signal
    """
    
    def __init__(self, min_confidence: float = 0.8, min_range_pct: float = 0.025,
                 validate_open_price: bool = False, max_open_error_pct: float = 0.03):
        super().__init__("Breakout", min_confidence, validate_open_price, max_open_error_pct)
        self.min_range_pct = min_range_pct
    
    def generate_signal(self, date_str: str, predictions: Dict, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        # Get prediction for this date
        prediction = predictions.get(date_str)
        if not prediction:
            return None
        
        # Extract prediction data
        predicted = prediction['predicted']
        confidence = prediction.get('confidence', 0.5) / 100.0
        
        predicted_open = predicted['Open']
        predicted_high = predicted['High']
        predicted_low = predicted['Low']
        predicted_close = predicted['Close']
        
        # Calculate range
        range_pct = (predicted_high - predicted_low) / predicted_open
        
        # Skip if range is too small
        if range_pct < self.min_range_pct:
            return None
        
        # Determine breakout direction based on close relative to range
        close_position = (predicted_close - predicted_low) / (predicted_high - predicted_low)
        
        if close_position > 0.7:  # Close in upper 30% of range
            # Bullish breakout
            signal_type = 'BUY'
            entry_price = predicted_open
            target_price = predicted_high
        elif close_position < 0.3:  # Close in lower 30% of range
            # Bearish breakout
            signal_type = 'SELL'
            entry_price = predicted_open
            target_price = predicted_low
        else:
            # No clear breakout direction
            return None
        
        return TradingSignal(
            date=date_str,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            signal_strength=range_pct,
            metadata={
                'predicted_open': predicted_open,
                'predicted_high': predicted_high,
                'predicted_low': predicted_low,
                'predicted_close': predicted_close,
                'range_pct': range_pct,
                'close_position': close_position
            }
        )


class GapStrategySignalGenerator(BaseSignalGenerator):
    """
    Gap Strategy: Compare predicted open vs actual gap from previous close
    If market gaps too much against prediction, skip trade (unfavorable risk-reward)
    """
    
    def __init__(self, min_confidence: float = 0.6, max_gap_deviation_pct: float = 0.02,
                 validate_open_price: bool = True, max_open_error_pct: float = 0.03):
        super().__init__("GapStrategy", min_confidence, validate_open_price, max_open_error_pct)
        self.max_gap_deviation_pct = max_gap_deviation_pct
    
    def generate_signal(self, date_str: str, predictions: Dict, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        # Get prediction for this date
        prediction = predictions.get(date_str)
        if not prediction:
            return None
        
        # Extract prediction data
        predicted = prediction['predicted']
        confidence = prediction.get('confidence', 0.5) / 100.0
        
        predicted_open = predicted['Open']
        predicted_close = predicted['Close']
        
        # Get previous day's close from market data
        day_data = market_data[market_data['date_str'] == date_str]
        if day_data.empty:
            return None
        
        # Get actual opening price
        actual_open = day_data.iloc[0]['close']  # First candle's close as opening price
        
        # Find previous trading day to get previous close
        all_dates = sorted(market_data['date_str'].unique())
        try:
            date_idx = all_dates.index(date_str)
            if date_idx == 0:  # No previous day
                return None
            prev_date = all_dates[date_idx - 1]
            prev_day_data = market_data[market_data['date_str'] == prev_date]
            if prev_day_data.empty:
                return None
            prev_close = prev_day_data.iloc[-1]['close']  # Last candle's close
        except (ValueError, IndexError):
            return None
        
        # Calculate gaps
        predicted_gap_pct = (predicted_open - prev_close) / prev_close
        actual_gap_pct = (actual_open - prev_close) / prev_close
        gap_deviation = abs(actual_gap_pct - predicted_gap_pct)
        
        # Skip if gap deviation is too large
        if gap_deviation > self.max_gap_deviation_pct:
            return None
        
        # Determine signal based on predicted direction, adjusted for gap
        expected_change_pct = (predicted_close - predicted_open) / predicted_open
        
        # Adjust for gap impact - if we gap in our favor, reduce target
        gap_adjusted_target = predicted_close
        if (expected_change_pct > 0 and actual_gap_pct > 0) or (expected_change_pct < 0 and actual_gap_pct < 0):
            # Gap in our favor, be more conservative
            gap_adjusted_target = predicted_open + (predicted_close - predicted_open) * 0.7
        
        signal_type = 'BUY' if predicted_close > predicted_open else 'SELL'
        
        return TradingSignal(
            date=date_str,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=actual_open,  # Enter at actual open
            target_price=gap_adjusted_target,
            signal_strength=abs(expected_change_pct),
            metadata={
                'predicted_open': predicted_open,
                'predicted_close': predicted_close,
                'actual_open': actual_open,
                'prev_close': prev_close,
                'predicted_gap_pct': predicted_gap_pct,
                'actual_gap_pct': actual_gap_pct,
                'gap_deviation': gap_deviation,
                'gap_adjusted_target': gap_adjusted_target
            }
        )


class BracketOrderSignalGenerator(BaseSignalGenerator):
    """
    Pure Bracket Order Strategy: Enter at open with SL=predicted Low, TP=predicted High
    Uses entire OHLC prediction as a day's trading bracket
    """
    
    def __init__(self, min_confidence: float = 0.6, min_bracket_size_pct: float = 0.015,
                 validate_open_price: bool = False, max_open_error_pct: float = 0.03):
        super().__init__("BracketOrder", min_confidence, validate_open_price, max_open_error_pct)
        self.min_bracket_size_pct = min_bracket_size_pct
    
    def generate_signal(self, date_str: str, predictions: Dict, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        # Get prediction for this date
        prediction = predictions.get(date_str)
        if not prediction:
            return None
        
        # Extract prediction data
        predicted = prediction['predicted']
        confidence = prediction.get('confidence', 0.5) / 100.0
        
        predicted_open = predicted['Open']
        predicted_high = predicted['High']
        predicted_low = predicted['Low']
        predicted_close = predicted['Close']
        
        # Calculate bracket size
        bracket_size_pct = (predicted_high - predicted_low) / predicted_open
        
        # Skip if bracket is too small
        if bracket_size_pct < self.min_bracket_size_pct:
            return None
        
        # Determine direction based on where close is expected relative to open
        expected_direction = 'BUY' if predicted_close > predicted_open else 'SELL'
        
        # For bracket orders, we always enter at open but set asymmetric targets
        if expected_direction == 'BUY':
            # Bullish: Larger upside target, closer downside stop
            target_price = predicted_high
            stop_loss_price = predicted_low
        else:
            # Bearish: We'll short, so "target" is the low, "stop" is the high
            signal_type = 'SELL'
            target_price = predicted_low
            stop_loss_price = predicted_high
        
        return TradingSignal(
            date=date_str,
            signal_type=expected_direction,
            confidence=confidence,
            entry_price=predicted_open,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
            signal_strength=bracket_size_pct,
            metadata={
                'predicted_open': predicted_open,
                'predicted_high': predicted_high,
                'predicted_low': predicted_low,
                'predicted_close': predicted_close,
                'bracket_size_pct': bracket_size_pct,
                'upside_potential_pct': (predicted_high - predicted_open) / predicted_open,
                'downside_risk_pct': (predicted_open - predicted_low) / predicted_open
            }
        )


class TechnicalConfirmationSignalGenerator(BaseSignalGenerator):
    """
    Technical Confirmation Strategy: Use AI prediction for bias, confirm with simple technicals
    Only trade when both AI and technical indicator agree on direction
    """
    
    def __init__(self, min_confidence: float = 0.6, lookback_periods: int = 5,
                 validate_open_price: bool = False, max_open_error_pct: float = 0.03):
        super().__init__("TechnicalConfirmation", min_confidence, validate_open_price, max_open_error_pct)
        self.lookback_periods = lookback_periods
    
    def _calculate_simple_technicals(self, market_data: pd.DataFrame, current_date: str) -> Dict[str, float]:
        """Calculate simple technical indicators"""
        # Get data up to current date
        date_filter = market_data['date_str'] <= current_date
        historical_data = market_data[date_filter].copy()
        
        if len(historical_data) < self.lookback_periods + 1:
            return {}
        
        # Get daily closes (last close of each day)
        daily_data = historical_data.groupby('date_str')['close'].last().reset_index()
        daily_data.columns = ['date', 'close']
        
        if len(daily_data) < self.lookback_periods + 1:
            return {}
        
        # Calculate simple moving average
        recent_closes = daily_data['close'].tail(self.lookback_periods).values
        sma = recent_closes.mean()
        current_close = recent_closes[-1]
        
        # Calculate momentum (rate of change)
        if len(daily_data) >= 3:
            momentum = (current_close - daily_data['close'].iloc[-3]) / daily_data['close'].iloc[-3]
        else:
            momentum = 0
        
        # Simple RSI-like calculation
        price_changes = daily_data['close'].tail(self.lookback_periods).diff().dropna()
        if len(price_changes) > 0:
            gains = price_changes[price_changes > 0].mean() if len(price_changes[price_changes > 0]) > 0 else 0
            losses = abs(price_changes[price_changes < 0].mean()) if len(price_changes[price_changes < 0]) > 0 else 0
            rs = gains / losses if losses != 0 else 100
            rsi_like = 100 - (100 / (1 + rs))
        else:
            rsi_like = 50
        
        return {
            'sma': sma,
            'current_close': current_close,
            'momentum': momentum,
            'rsi_like': rsi_like,
            'sma_signal': 1 if current_close > sma else -1,
            'momentum_signal': 1 if momentum > 0.005 else (-1 if momentum < -0.005 else 0),
            'rsi_signal': 1 if rsi_like < 30 else (-1 if rsi_like > 70 else 0)  # Oversold/overbought
        }
    
    def generate_signal(self, date_str: str, predictions: Dict, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        # Get prediction for this date
        prediction = predictions.get(date_str)
        if not prediction:
            return None
        
        # Extract prediction data
        predicted = prediction['predicted']
        confidence = prediction.get('confidence', 0.5) / 100.0
        
        predicted_open = predicted['Open']
        predicted_close = predicted['Close']
        
        # Get AI signal direction
        ai_signal = 1 if predicted_close > predicted_open else -1
        expected_change_pct = (predicted_close - predicted_open) / predicted_open
        
        # Skip if change is too small
        if abs(expected_change_pct) < 0.005:
            return None
        
        # Calculate technical indicators
        technicals = self._calculate_simple_technicals(market_data, date_str)
        if not technicals:
            return None
        
        # Combine technical signals
        tech_score = technicals['sma_signal'] + technicals['momentum_signal'] + technicals['rsi_signal']
        tech_signal = 1 if tech_score > 0 else (-1 if tech_score < 0 else 0)
        
        # Only trade if AI and technicals agree
        if ai_signal * tech_signal <= 0:  # No agreement or neutral
            return None
        
        signal_type = 'BUY' if ai_signal > 0 else 'SELL'
        
        # Boost confidence when technicals strongly agree
        tech_boost = min(abs(tech_score) * 0.1, 0.2)  # Up to 20% boost
        adjusted_confidence = min(confidence + tech_boost, 1.0)
        
        return TradingSignal(
            date=date_str,
            signal_type=signal_type,
            confidence=adjusted_confidence,
            entry_price=predicted_open,
            target_price=predicted_close,
            signal_strength=abs(expected_change_pct),
            metadata={
                'predicted_open': predicted_open,
                'predicted_close': predicted_close,
                'ai_signal': ai_signal,
                'tech_signal': tech_signal,
                'tech_score': tech_score,
                'technicals': technicals,
                'tech_boost': tech_boost,
                'original_confidence': confidence
            }
        )


class AdaptivePositionSignalGenerator(BaseSignalGenerator):
    """
    Adaptive Position Sizing Strategy: Vary position size based on confidence and prediction quality
    Higher confidence and better track record = larger positions
    """
    
    def __init__(self, min_confidence: float = 0.5, base_position_size: float = 1000,
                 confidence_multiplier: float = 2.0, validate_open_price: bool = False, max_open_error_pct: float = 0.03):
        super().__init__("AdaptivePosition", min_confidence, validate_open_price, max_open_error_pct)
        self.base_position_size = base_position_size
        self.confidence_multiplier = confidence_multiplier
    
    def generate_signal(self, date_str: str, predictions: Dict, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        # Get prediction for this date
        prediction = predictions.get(date_str)
        if not prediction:
            return None
        
        # Extract prediction data
        predicted = prediction['predicted']
        confidence = prediction.get('confidence', 0.5) / 100.0
        
        predicted_open = predicted['Open']
        predicted_close = predicted['Close']
        
        # Calculate expected change
        expected_change_pct = (predicted_close - predicted_open) / predicted_open
        
        # Skip if change is too small
        if abs(expected_change_pct) < 0.003:
            return None
        
        # Calculate adaptive position size
        confidence_factor = (confidence - self.min_confidence) / (1.0 - self.min_confidence)
        strength_factor = min(abs(expected_change_pct) / 0.02, 2.0)  # Cap at 2x for 2% moves
        
        position_multiplier = 1.0 + (confidence_factor * strength_factor * (self.confidence_multiplier - 1.0))
        adaptive_position_size = self.base_position_size * position_multiplier
        
        signal_type = 'BUY' if predicted_close > predicted_open else 'SELL'
        
        return TradingSignal(
            date=date_str,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=predicted_open,
            target_price=predicted_close,
            signal_strength=abs(expected_change_pct),
            metadata={
                'predicted_open': predicted_open,
                'predicted_close': predicted_close,
                'expected_change_pct': expected_change_pct,
                'base_position_size': self.base_position_size,
                'position_multiplier': position_multiplier,
                'adaptive_position_size': adaptive_position_size,
                'confidence_factor': confidence_factor,
                'strength_factor': strength_factor
            }
        )


def create_signal_generators() -> Dict[str, BaseSignalGenerator]:
    """Factory function to create all signal generators"""
    
    generators = {
        # Open-Close strategies with different confidence levels
        'openclose_conservative': OpenCloseSignalGenerator(min_confidence=0.7, min_change_pct=0.008),
        'openclose_standard': OpenCloseSignalGenerator(min_confidence=0.6, min_change_pct=0.005),
        'openclose_aggressive': OpenCloseSignalGenerator(min_confidence=0.5, min_change_pct=0.003),
        
        # High-Low strategies with different parameters
        'highlow_conservative': HighLowSignalGenerator(min_confidence=0.7, min_hl_spread_pct=0.015),
        'highlow_standard': HighLowSignalGenerator(min_confidence=0.6, min_hl_spread_pct=0.01),
        'highlow_aggressive': HighLowSignalGenerator(min_confidence=0.5, min_hl_spread_pct=0.008),
        
        # Mean reversion strategies
        'meanrev_conservative': MeanReversionSignalGenerator(min_confidence=0.8, extreme_threshold_pct=0.025),
        'meanrev_standard': MeanReversionSignalGenerator(min_confidence=0.7, extreme_threshold_pct=0.02),
        
        # Breakout strategies
        'breakout_conservative': BreakoutSignalGenerator(min_confidence=0.85, min_range_pct=0.03),
        'breakout_standard': BreakoutSignalGenerator(min_confidence=0.8, min_range_pct=0.025),
        
        # NEW: Gap-based strategies
        'gap_conservative': GapStrategySignalGenerator(min_confidence=0.7, max_gap_deviation_pct=0.015),
        'gap_standard': GapStrategySignalGenerator(min_confidence=0.6, max_gap_deviation_pct=0.02),
        'gap_relaxed': GapStrategySignalGenerator(min_confidence=0.5, max_gap_deviation_pct=0.03),
        
        # NEW: Bracket order strategies
        'bracket_conservative': BracketOrderSignalGenerator(min_confidence=0.7, min_bracket_size_pct=0.02),
        'bracket_standard': BracketOrderSignalGenerator(min_confidence=0.6, min_bracket_size_pct=0.015),
        'bracket_aggressive': BracketOrderSignalGenerator(min_confidence=0.5, min_bracket_size_pct=0.01),
        
        # NEW: Technical confirmation strategies
        'tech_confirm_conservative': TechnicalConfirmationSignalGenerator(min_confidence=0.7, lookback_periods=7),
        'tech_confirm_standard': TechnicalConfirmationSignalGenerator(min_confidence=0.6, lookback_periods=5),
        'tech_confirm_responsive': TechnicalConfirmationSignalGenerator(min_confidence=0.5, lookback_periods=3),
        
        # NEW: Adaptive position sizing strategies
        'adaptive_conservative': AdaptivePositionSignalGenerator(min_confidence=0.6, confidence_multiplier=1.5),
        'adaptive_standard': AdaptivePositionSignalGenerator(min_confidence=0.5, confidence_multiplier=2.0),
        'adaptive_aggressive': AdaptivePositionSignalGenerator(min_confidence=0.4, confidence_multiplier=3.0),
        
        # Realistic trading strategies with open price validation
        'openclose_realistic': OpenCloseSignalGenerator(min_confidence=0.6, min_change_pct=0.005, 
                                                       validate_open_price=True, max_open_error_pct=0.03),
        'openclose_strict_validation': OpenCloseSignalGenerator(min_confidence=0.6, min_change_pct=0.005,
                                                              validate_open_price=True, max_open_error_pct=0.01),
        'openclose_relaxed_validation': OpenCloseSignalGenerator(min_confidence=0.6, min_change_pct=0.005,
                                                               validate_open_price=True, max_open_error_pct=0.05),
    }
    
    return generators