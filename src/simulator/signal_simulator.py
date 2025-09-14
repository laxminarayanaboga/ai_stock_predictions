"""
Enhanced Intraday Simulator - Takes signals and parameters, returns PnL
Clean separation of concerns: signals come from generators, simulator just executes
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import time
import pandas as pd
from .intraday_core import IntradaySimulator, TradeEntry, TradeResult
from .signal_generators import TradingSignal


@dataclass
class TradingParameters:
    """Trading execution parameters
    Position sizing supports two modes:
    - Fixed shares via position_size (legacy)
    - Capital-based sizing via capital_per_trade (preferred)
    If capital_per_trade > 0, it takes precedence and position_size is computed at entry.
    """
    stop_loss_pct: float = 0.01  # 1% stop loss
    take_profit_pct: float = 0.02  # 2% take profit
    position_size: Optional[int] = None  # Fixed number of shares (legacy)
    capital_per_trade: Optional[float] = 100000.0  # INR capital to deploy per trade
    entry_time: time = time(9, 15)  # Market open time
    
    # Advanced parameters
    use_signal_target: bool = False  # Use signal's target price instead of fixed TP
    max_hold_time_minutes: int = 360  # Max 6 hours (360 min)
    entry_tolerance_pct: float = 0.005  # 0.5% tolerance for entry price
    wait_for_entry_price: bool = False  # Wait for better entry price
    
    # Realistic trading parameters
    delayed_entry: bool = False  # Enter at delayed time (e.g., 9:25 instead of 9:15)
    delayed_entry_time: time = time(9, 25)  # Delayed entry time (10 minutes after open)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, time):
                result[field_name] = field_value.strftime('%H:%M:%S')
            else:
                result[field_name] = field_value
        return result


@dataclass
class SignalExecutionResult:
    """Result of executing a trading signal"""
    signal: TradingSignal
    parameters: TradingParameters
    trade_result: Optional[TradeResult]
    execution_status: str  # 'EXECUTED', 'NO_ENTRY', 'SKIPPED'
    execution_notes: str = ""
    
    @property
    def pnl(self) -> float:
        """Get PnL from trade result"""
        return self.trade_result.pnl if self.trade_result else 0.0
    
    @property
    def was_profitable(self) -> bool:
        """Check if trade was profitable"""
        return self.pnl > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis"""
        result = {
            'date': self.signal.date,
            'signal_type': self.signal.signal_type,
            'confidence': self.signal.confidence,
            'signal_strength': self.signal.signal_strength,
            'execution_status': self.execution_status,
            'execution_notes': self.execution_notes,
            'pnl': self.pnl,
            'was_profitable': self.was_profitable
        }
        
        # Add trade details if executed
        if self.trade_result:
            result.update({
                'entry_time': self.trade_result.entry_time.strftime('%H:%M:%S'),
                'exit_time': self.trade_result.exit_time.strftime('%H:%M:%S'),
                'entry_price': self.trade_result.entry_price,
                'exit_price': self.trade_result.exit_price,
                'direction': self.trade_result.direction,
                'exit_reason': self.trade_result.exit_reason,
                'duration_minutes': self.trade_result.duration_minutes,
                'position_size': self.trade_result.position_size,
                'capital_used': round(self.trade_result.entry_price * self.trade_result.position_size, 2)
            })
        
        # Add signal metadata
        if self.signal.metadata:
            for key, value in self.signal.metadata.items():
                result[f'signal_{key}'] = value
        
        # Add parameters
        for key, value in self.parameters.to_dict().items():
            result[f'param_{key}'] = value
        
        return result


class SignalBasedSimulator:
    """
    Executes trading signals with given parameters
    Clean interface: signal + parameters -> execution result
    """
    
    def __init__(self, market_close_time: time = time(15, 15)):
        self.core_simulator = IntradaySimulator(market_close_time)
        self.market_close_time = market_close_time
    
    def execute_signal(self, signal: TradingSignal, parameters: TradingParameters, 
                      day_candles: pd.DataFrame) -> SignalExecutionResult:
        """
        Execute a trading signal with given parameters
        
        Args:
            signal: Trading signal to execute
            parameters: Trading parameters (SL, TP, position size, etc.)
            day_candles: Day's market data
        
        Returns:
            SignalExecutionResult with execution details and PnL
        """
        
        if day_candles.empty:
            return SignalExecutionResult(
                signal=signal,
                parameters=parameters,
                trade_result=None,
                execution_status='NO_ENTRY',
                execution_notes='No market data available'
            )
        
        # Find optimal entry point
        entry_result = self._find_entry_point(signal, parameters, day_candles)
        if not entry_result['success']:
            return SignalExecutionResult(
                signal=signal,
                parameters=parameters,
                trade_result=None,
                execution_status='NO_ENTRY',
                execution_notes=entry_result['reason']
            )
        
        # Sizing: compute position size from capital if provided, else use fixed size
        entry_price = entry_result['entry_price']
        dynamic_qty: Optional[int] = None
        if getattr(parameters, 'capital_per_trade', None):
            if parameters.capital_per_trade and parameters.capital_per_trade > 0:
                dynamic_qty = int(parameters.capital_per_trade // entry_price)
                if dynamic_qty < 1:
                    return SignalExecutionResult(
                        signal=signal,
                        parameters=parameters,
                        trade_result=None,
                        execution_status='NO_ENTRY',
                        execution_notes=f"Insufficient capital {parameters.capital_per_trade:.2f} for entry price {entry_price:.2f}"
                    )
                entry_result['position_size'] = dynamic_qty
        
        # Create trade entry
        trade_entry = self._create_trade_entry(signal, parameters, entry_result)
        
        # Execute trade using core simulator
        trade_result = self.core_simulator.simulate_trade(trade_entry, day_candles)
        
        if trade_result:
            return SignalExecutionResult(
                signal=signal,
                parameters=parameters,
                trade_result=trade_result,
                execution_status='EXECUTED',
                execution_notes=f"Entry at {entry_result['entry_price']:.2f}"
            )
        else:
            return SignalExecutionResult(
                signal=signal,
                parameters=parameters,
                trade_result=None,
                execution_status='NO_ENTRY',
                execution_notes='Trade simulation failed'
            )
    
    def _find_entry_point(self, signal: TradingSignal, parameters: TradingParameters, 
                         day_candles: pd.DataFrame) -> Dict[str, Any]:
        """Find optimal entry point based on signal and parameters"""
        
        # Determine actual entry time (delayed or immediate)
        actual_entry_time = parameters.delayed_entry_time if parameters.delayed_entry else parameters.entry_time
        
        # Get candles from entry time onwards
        entry_candles = day_candles[day_candles['time_ist'] >= actual_entry_time]
        if entry_candles.empty:
            return {'success': False, 'reason': f'No candles after entry time {actual_entry_time}'}
        
        target_price = signal.entry_price
        
        if not parameters.wait_for_entry_price:
            # Enter immediately at first available candle
            first_candle = entry_candles.iloc[0]
            return {
                'success': True,
                'entry_time': first_candle['time_ist'],
                'entry_price': first_candle['close'],  # Use market price, not signal price
                'candle_index': 0,
                'delayed_entry_used': parameters.delayed_entry
            }
        
        # Wait for price to be within tolerance of target
        tolerance = parameters.entry_tolerance_pct
        max_wait_candles = min(6, len(entry_candles))  # Wait max 1 hour for entry
        
        for i, (idx, candle) in enumerate(entry_candles.head(max_wait_candles).iterrows()):
            candle_low = candle['low']
            candle_high = candle['high']
            candle_close = candle['close']
            
            # Check if target price is within this candle's range
            if candle_low <= target_price <= candle_high:
                # Price hit our target, enter here
                entry_price = min(target_price * (1 + tolerance), 
                                max(target_price * (1 - tolerance), candle_close))
                return {
                    'success': True,
                    'entry_time': candle['time_ist'],
                    'entry_price': entry_price,
                    'candle_index': i
                }
            
            # Check if close price is within tolerance
            price_diff_pct = abs(candle_close - target_price) / target_price
            if price_diff_pct <= tolerance:
                return {
                    'success': True,
                    'entry_time': candle['time_ist'],
                    'entry_price': candle_close,
                    'candle_index': i
                }
        
        # If we couldn't find good entry within wait time, enter at market
        if len(entry_candles) > 0:
            last_waited_candle = entry_candles.iloc[min(max_wait_candles-1, len(entry_candles)-1)]
            return {
                'success': True,
                'entry_time': last_waited_candle['time_ist'],
                'entry_price': last_waited_candle['close'],
                'candle_index': max_wait_candles-1,
                'notes': 'Entered at market after wait timeout'
            }
        
        return {'success': False, 'reason': 'No suitable entry point found'}
    
    def _create_trade_entry(self, signal: TradingSignal, parameters: TradingParameters, 
                           entry_result: Dict[str, Any]) -> TradeEntry:
        """Create TradeEntry from signal and parameters"""
        
        entry_price = entry_result['entry_price']
        
        # Determine direction
        direction = 'LONG' if signal.signal_type == 'BUY' else 'SHORT'
        
        # Determine position size (capital-based sizing takes precedence)
        position_size = entry_result.get('position_size')
        if position_size is None:
            # Fallback to fixed position size from parameters; if None, default to 1 share
            position_size = parameters.position_size if parameters.position_size is not None else 1
        
        # Calculate take profit
        if parameters.use_signal_target and signal.target_price:
            # Use signal's target price to calculate TP percentage
            if direction == 'LONG':
                tp_pct = (signal.target_price - entry_price) / entry_price
            else:
                tp_pct = (entry_price - signal.target_price) / entry_price
            
            # Ensure reasonable TP (between 0.5% and 5%)
            tp_pct = max(0.005, min(tp_pct, 0.05))
        else:
            # Use fixed TP from parameters
            tp_pct = parameters.take_profit_pct
        
        return TradeEntry(
            entry_time=entry_result['entry_time'],
            entry_price=entry_price,
            direction=direction,
            stop_loss_pct=parameters.stop_loss_pct,
            take_profit_pct=tp_pct,
            position_size=position_size
        )
    
    def execute_multiple_signals(self, signals_with_params: list, day_candles: pd.DataFrame) -> list:
        """
        Execute multiple signals for the same day
        
        Args:
            signals_with_params: List of (signal, parameters) tuples
            day_candles: Day's market data
        
        Returns:
            List of SignalExecutionResult
        """
        
        results = []
        for signal, parameters in signals_with_params:
            result = self.execute_signal(signal, parameters, day_candles)
            results.append(result)
        
        return results


def create_parameter_sets() -> Dict[str, TradingParameters]:
    """Factory function to create different parameter sets"""
    
    parameter_sets = {
        # Conservative parameters
        'conservative': TradingParameters(
            stop_loss_pct=0.01,      # 1% SL
            take_profit_pct=0.02,    # 2% TP
            capital_per_trade=100000.0,
            use_signal_target=False,
            wait_for_entry_price=False
        ),
        
        # Aggressive parameters
        'aggressive': TradingParameters(
            stop_loss_pct=0.005,     # 0.5% SL
            take_profit_pct=0.03,    # 3% TP
            capital_per_trade=100000.0,
            use_signal_target=False,
            wait_for_entry_price=False
        ),
        
        # Target-based parameters (use signal targets)
        'target_based': TradingParameters(
            stop_loss_pct=0.005,     # 0.5% SL
            take_profit_pct=0.02,    # Fallback TP
            capital_per_trade=100000.0,
            use_signal_target=True,  # Use signal's target price
            wait_for_entry_price=True,
            entry_tolerance_pct=0.003
        ),
        
        # Tight parameters
        'tight': TradingParameters(
            stop_loss_pct=0.003,     # 0.3% SL
            take_profit_pct=0.015,   # 1.5% TP
            capital_per_trade=100000.0,
            use_signal_target=False,
            wait_for_entry_price=True,
            entry_tolerance_pct=0.002
        ),
        
        # Loose parameters
        'loose': TradingParameters(
            stop_loss_pct=0.015,     # 1.5% SL
            take_profit_pct=0.04,    # 4% TP
            capital_per_trade=100000.0,
            use_signal_target=False,
            wait_for_entry_price=False
        ),
        
        # Realistic trading parameters (like Strategy 17)
        'realistic': TradingParameters(
            stop_loss_pct=0.015,     # 1.5% SL
            take_profit_pct=0.03,    # 3% TP
            capital_per_trade=100000.0,
            use_signal_target=False,
            wait_for_entry_price=False,
            delayed_entry=True,      # Enter at 9:25 instead of 9:15
            delayed_entry_time=time(9, 25)
        ),
        
        # Realistic with strict validation
        'realistic_strict': TradingParameters(
            stop_loss_pct=0.015,     # 1.5% SL
            take_profit_pct=0.03,    # 3% TP
            capital_per_trade=100000.0,
            use_signal_target=False,
            wait_for_entry_price=False,
            delayed_entry=True,      # Enter at 9:25 instead of 9:15
            delayed_entry_time=time(9, 25)
        ),
        
        # Realistic with relaxed validation
        'realistic_relaxed': TradingParameters(
            stop_loss_pct=0.015,     # 1.5% SL
            take_profit_pct=0.03,    # 3% TP
            capital_per_trade=100000.0,
            use_signal_target=False,
            wait_for_entry_price=False,
            delayed_entry=True,      # Enter at 9:25 instead of 9:15
            delayed_entry_time=time(9, 25)
        )
    }
    
    return parameter_sets