"""
Base Trading Strategy Interface
All trading strategies should inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


class BaseTradingStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.trades = []
        
    @abstractmethod
    def generate_signal(self, 
                       current_price: float, 
                       predicted_prices: np.ndarray, 
                       daily_data: pd.Series,
                       confidence: float) -> Tuple[str, Dict]:
        """
        Generate trading signal based on strategy logic
        
        Args:
            current_price: Previous day's close price
            predicted_prices: [open, high, low, close] predictions for today
            daily_data: Today's actual OHLCV data
            confidence: Model's prediction confidence
            
        Returns:
            Tuple of (signal, execution_details)
            signal: 'BUY', 'SELL', or 'HOLD'
            execution_details: Dict with entry_price, exit_price, etc.
        """
        pass
    
    @abstractmethod
    def calculate_execution_prices(self, 
                                  signal: str, 
                                  daily_data: pd.Series,
                                  predicted_prices: np.ndarray) -> Dict:
        """
        Calculate actual entry and exit prices based on strategy
        
        Args:
            signal: Trading signal ('BUY' or 'SELL')
            daily_data: Today's actual OHLCV data
            predicted_prices: Model predictions
            
        Returns:
            Dict with entry_price, exit_price, entry_time, exit_time
        """
        pass
    
    def should_trade(self, confidence: float, min_confidence: float = 0.0001) -> bool:
        """Check if confidence meets minimum threshold"""
        return confidence >= min_confidence
    
    def get_strategy_info(self) -> Dict:
        """Return strategy information"""
        return {
            'name': self.name,
            'description': self.description,
            'total_trades': len(self.trades)
        }
