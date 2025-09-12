"""
Strategy 1: Open-to-Close Trading
Current implementation - buy/sell at market open, exit at market close
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .base_strategy import BaseTradingStrategy


class OpenToCloseStrategy(BaseTradingStrategy):
    """
    Traditional intraday strategy:
    - Entry: Market open
    - Exit: Market close
    - Signal: Based on predicted close vs current open price
    """
    
    def __init__(self, min_confidence: float = 0.005):
        super().__init__(
            name="Open-to-Close",
            description="Enter at market open, exit at market close based on daily predictions"
        )
        self.min_confidence = min_confidence
    
    def generate_signal(self, 
                       current_price: float, 
                       predicted_prices: np.ndarray, 
                       daily_data: pd.Series,
                       confidence: float) -> Tuple[str, Dict]:
        """Generate signal based on predicted vs current price"""
        
        if not self.should_trade(confidence, self.min_confidence):
            return 'HOLD', {}
        
        predicted_close = predicted_prices[3]  # Close price prediction
        price_change = predicted_close - current_price
        price_change_pct = price_change / current_price
        
        if price_change_pct > self.min_confidence:
            signal = 'BUY'
        elif price_change_pct < -self.min_confidence:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        execution_details = self.calculate_execution_prices(signal, daily_data, predicted_prices)
        
        return signal, execution_details
    
    def calculate_execution_prices(self, 
                                  signal: str, 
                                  daily_data: pd.Series,
                                  predicted_prices: np.ndarray) -> Dict:
        """Calculate entry and exit prices for open-to-close strategy"""
        
        if signal == 'HOLD':
            return {}
        
        actual_open = daily_data['Open']
        actual_close = daily_data['Close']
        
        if signal == 'BUY':
            return {
                'entry_price': actual_open,
                'exit_price': actual_close,
                'entry_time': 'market_open',
                'exit_time': 'market_close',
                'direction': 'long'
            }
        else:  # SELL
            return {
                'entry_price': actual_open,  # Short sell at open
                'exit_price': actual_close,  # Buy back at close
                'entry_time': 'market_open',
                'exit_time': 'market_close',
                'direction': 'short'
            }
