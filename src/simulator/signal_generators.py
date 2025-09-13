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
        
        # Realistic trading strategies with open price validation
        'openclose_realistic': OpenCloseSignalGenerator(min_confidence=0.6, min_change_pct=0.005, 
                                                       validate_open_price=True, max_open_error_pct=0.03),
        'openclose_strict_validation': OpenCloseSignalGenerator(min_confidence=0.6, min_change_pct=0.005,
                                                              validate_open_price=True, max_open_error_pct=0.01),
        'openclose_relaxed_validation': OpenCloseSignalGenerator(min_confidence=0.6, min_change_pct=0.005,
                                                               validate_open_price=True, max_open_error_pct=0.05),
    }
    
    return generators