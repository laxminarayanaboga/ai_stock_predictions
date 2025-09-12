"""
Core simulator for intraday trading
Handles individual trade simulation with entry/exit logic
"""

import pandas as pd
from datetime import time, datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List
from .pnl_calculator import get_npl

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class TradeEntry:
    """Trade entry parameters"""
    entry_time: time
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    stop_loss_pct: float  # e.g., 0.01 for 1%
    take_profit_pct: float  # e.g., 0.03 for 3%
    position_size: int = 100  # number of shares


@dataclass
class TradeResult:
    """Trade result after simulation"""
    entry_time: time
    entry_price: float
    exit_time: time
    exit_price: float
    direction: str
    position_size: int
    pnl: float
    exit_reason: str  # 'STOP_LOSS', 'TAKE_PROFIT', 'END_OF_DAY'
    duration_minutes: int


class IntradaySimulator:
    """
    Simple intraday trade simulator for a single day
    Takes entry parameters and day's candle data, returns exit result
    """
    
    def __init__(self, market_close_time: time = time(15, 15)):
        self.market_close_time = market_close_time
    
    def simulate_trade(self, trade_entry: TradeEntry, day_candles: pd.DataFrame) -> Optional[TradeResult]:
        """
        Simulate a single intraday trade
        
        Args:
            trade_entry: Trade entry parameters
            day_candles: DataFrame with columns ['timestamp', 'datetime_ist', 'time_ist', 'open', 'high', 'low', 'close']
        
        Returns:
            TradeResult or None if trade couldn't be executed
        """
        
        # Validate input data
        if day_candles.empty:
            return None
        
        # Find entry candle
        entry_candles = day_candles[day_candles['time_ist'] >= trade_entry.entry_time]
        if entry_candles.empty:
            return None
        
        # Get the first available candle at or after entry time
        entry_candle = entry_candles.iloc[0]
        actual_entry_time = entry_candle['time_ist']
        actual_entry_price = trade_entry.entry_price  # Use specified entry price
        
        # Calculate stop loss and take profit levels
        if trade_entry.direction == 'LONG':
            stop_loss_price = actual_entry_price * (1 - trade_entry.stop_loss_pct)
            take_profit_price = actual_entry_price * (1 + trade_entry.take_profit_pct)
        else:  # SHORT
            stop_loss_price = actual_entry_price * (1 + trade_entry.stop_loss_pct)
            take_profit_price = actual_entry_price * (1 - trade_entry.take_profit_pct)
        
        # Simulate through remaining candles
        remaining_candles = day_candles[day_candles['time_ist'] > actual_entry_time]
        
        for idx, candle in remaining_candles.iterrows():
            candle_time = candle['time_ist']
            candle_high = candle['high']
            candle_low = candle['low']
            candle_close = candle['close']
            
            # Check for stop loss or take profit hits within this candle
            if trade_entry.direction == 'LONG':
                # For LONG: check if low hit stop loss or high hit take profit
                if candle_low <= stop_loss_price:
                    # Stop loss hit
                    exit_price = stop_loss_price
                    if trade_entry.direction == 'LONG':
                        pnl = get_npl(actual_entry_price, exit_price, trade_entry.position_size)
                    else:  # SHORT
                        pnl = get_npl(exit_price, actual_entry_price, trade_entry.position_size)
                    duration = self._calculate_duration(actual_entry_time, candle_time)
                    
                    return TradeResult(
                        entry_time=actual_entry_time,
                        entry_price=actual_entry_price,
                        exit_time=candle_time,
                        exit_price=exit_price,
                        direction=trade_entry.direction,
                        position_size=trade_entry.position_size,
                        pnl=pnl,
                        exit_reason='STOP_LOSS',
                        duration_minutes=duration
                    )
                
                elif candle_high >= take_profit_price:
                    # Take profit hit
                    exit_price = take_profit_price
                    if trade_entry.direction == 'LONG':
                        pnl = get_npl(actual_entry_price, exit_price, trade_entry.position_size)
                    else:  # SHORT
                        pnl = get_npl(exit_price, actual_entry_price, trade_entry.position_size)
                    duration = self._calculate_duration(actual_entry_time, candle_time)
                    
                    return TradeResult(
                        entry_time=actual_entry_time,
                        entry_price=actual_entry_price,
                        exit_time=candle_time,
                        exit_price=exit_price,
                        direction=trade_entry.direction,
                        position_size=trade_entry.position_size,
                        pnl=pnl,
                        exit_reason='TAKE_PROFIT',
                        duration_minutes=duration
                    )
            
            else:  # SHORT
                # For SHORT: check if high hit stop loss or low hit take profit
                if candle_high >= stop_loss_price:
                    # Stop loss hit
                    exit_price = stop_loss_price
                    if trade_entry.direction == 'LONG':
                        pnl = get_npl(actual_entry_price, exit_price, trade_entry.position_size)
                    else:  # SHORT
                        pnl = get_npl(exit_price, actual_entry_price, trade_entry.position_size)
                    duration = self._calculate_duration(actual_entry_time, candle_time)
                    
                    return TradeResult(
                        entry_time=actual_entry_time,
                        entry_price=actual_entry_price,
                        exit_time=candle_time,
                        exit_price=exit_price,
                        direction=trade_entry.direction,
                        position_size=trade_entry.position_size,
                        pnl=pnl,
                        exit_reason='STOP_LOSS',
                        duration_minutes=duration
                    )
                
                elif candle_low <= take_profit_price:
                    # Take profit hit
                    exit_price = take_profit_price
                    if trade_entry.direction == 'LONG':
                        pnl = get_npl(actual_entry_price, exit_price, trade_entry.position_size)
                    else:  # SHORT
                        pnl = get_npl(exit_price, actual_entry_price, trade_entry.position_size)
                    duration = self._calculate_duration(actual_entry_time, candle_time)
                    
                    return TradeResult(
                        entry_time=actual_entry_time,
                        entry_price=actual_entry_price,
                        exit_time=candle_time,
                        exit_price=exit_price,
                        direction=trade_entry.direction,
                        position_size=trade_entry.position_size,
                        pnl=pnl,
                        exit_reason='TAKE_PROFIT',
                        duration_minutes=duration
                    )
            
            # Check if it's market close time
            if candle_time >= self.market_close_time:
                # Force exit at market close
                exit_price = candle_close
                
                # Use realistic PnL calculation with charges
                if trade_entry.direction == 'LONG':
                    pnl = get_npl(actual_entry_price, exit_price, trade_entry.position_size)
                else:  # SHORT
                    pnl = get_npl(exit_price, actual_entry_price, trade_entry.position_size)
                
                duration = self._calculate_duration(actual_entry_time, candle_time)
                
                return TradeResult(
                    entry_time=actual_entry_time,
                    entry_price=actual_entry_price,
                    exit_time=candle_time,
                    exit_price=exit_price,
                    direction=trade_entry.direction,
                    position_size=trade_entry.position_size,
                    pnl=pnl,
                    exit_reason='END_OF_DAY',
                    duration_minutes=duration
                )
        
        # If we reach here, no exit condition was met (shouldn't happen in normal cases)
        # Force exit with last available price
        last_candle = remaining_candles.iloc[-1] if not remaining_candles.empty else entry_candle
        exit_price = last_candle['close']
        
        # Use realistic PnL calculation with charges
        if trade_entry.direction == 'LONG':
            pnl = get_npl(actual_entry_price, exit_price, trade_entry.position_size)
        else:  # SHORT
            pnl = get_npl(exit_price, actual_entry_price, trade_entry.position_size)
        
        duration = self._calculate_duration(actual_entry_time, last_candle['time_ist'])
        
        return TradeResult(
            entry_time=actual_entry_time,
            entry_price=actual_entry_price,
            exit_time=last_candle['time_ist'],
            exit_price=exit_price,
            direction=trade_entry.direction,
            position_size=trade_entry.position_size,
            pnl=pnl,
            exit_reason='END_OF_DAY',
            duration_minutes=duration
        )
    
    def _calculate_duration(self, entry_time: time, exit_time: time) -> int:
        """Calculate duration in minutes between entry and exit times"""
        
        # Convert time objects to minutes since midnight
        entry_minutes = entry_time.hour * 60 + entry_time.minute
        exit_minutes = exit_time.hour * 60 + exit_time.minute
        
        return max(0, exit_minutes - entry_minutes)
    
    def simulate_multiple_trades(self, trades: List[TradeEntry], day_candles: pd.DataFrame) -> List[TradeResult]:
        """
        Simulate multiple trades for the same day
        
        Args:
            trades: List of trade entries
            day_candles: Day's candle data
        
        Returns:
            List of trade results
        """
        
        results = []
        for trade in trades:
            result = self.simulate_trade(trade, day_candles)
            if result:
                results.append(result)
        
        return results


def test_simulator():
    """Test the simulator with sample data"""
    
    # Create sample intraday data
    sample_data = {
        'timestamp': [1693542900, 1693543500, 1693544100, 1693544700, 1693545300],  # 10-min intervals
        'open': [1200.0, 1205.0, 1210.0, 1208.0, 1195.0],
        'high': [1208.0, 1212.0, 1215.0, 1210.0, 1200.0],
        'low': [1198.0, 1203.0, 1207.0, 1195.0, 1190.0],
        'close': [1205.0, 1210.0, 1208.0, 1195.0, 1198.0]
    }
    
    df = pd.DataFrame(sample_data)
    df['datetime_ist'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
    df['time_ist'] = df['datetime_ist'].dt.time
    
    # Create simulator
    simulator = IntradaySimulator()
    
    # Test LONG trade
    long_trade = TradeEntry(
        entry_time=time(9, 15),
        entry_price=1200.0,
        direction='LONG',
        stop_loss_pct=0.01,  # 1%
        take_profit_pct=0.02,  # 2%
        position_size=100
    )
    
    result = simulator.simulate_trade(long_trade, df)
    
    if result:
        print("✅ Test Trade Result:")
        print(f"   Entry: {result.entry_time} at ₹{result.entry_price}")
        print(f"   Exit:  {result.exit_time} at ₹{result.exit_price}")
        print(f"   PnL:   ₹{result.pnl:.2f}")
        print(f"   Reason: {result.exit_reason}")
        print(f"   Duration: {result.duration_minutes} minutes")
    else:
        print("❌ Test trade failed")


if __name__ == "__main__":
    test_simulator()