"""
Base strategy framework for building reusable trading strategies.
Supports both configuration-driven and custom logic strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import time
from typing import Dict, Optional, Any, List
import pandas as pd
import json
from pathlib import Path


@dataclass
class StrategyConfig:
    """Base configuration for trading strategies"""
    strategy_id: str
    name: str
    description: str
    stop_loss_pct: float
    take_profit_pct: float
    min_confidence: float = 0.6
    min_price_change: float = 0.005
    position_size: int = 100
    entry_time: time = time(9, 15)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = asdict(self)
        # Convert time object to string for JSON serialization
        result['entry_time'] = self.entry_time.strftime('%H:%M:%S')
        return result


@dataclass
class StrategyResults:
    """Container for strategy performance results"""
    strategy_id: str
    strategy_name: str
    strategy_description: str
    config: StrategyConfig
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_pnl: float
    avg_pnl_per_trade: float
    trades: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization"""
        result = asdict(self)
        result['config'] = self.config.to_dict()
        return result


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Provides common functionality and enforces interface consistency.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.results = None
    
    @abstractmethod
    def generate_signal(self, date_str: str, opening_price: float, 
                       predictions: Dict, market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal for a given day.
        
        Args:
            date_str: Date in 'YYYY-MM-DD' format
            opening_price: Opening price for the day
            predictions: AI predictions dictionary
            market_data: Historical market data
        
        Returns:
            Signal dictionary with direction, confidence, etc. or None
        """
        pass
    
    def should_enter_trade(self, signal: Dict[str, Any]) -> bool:
        """
        Determine if we should enter a trade based on signal.
        Can be overridden for custom logic.
        """
        if not signal:
            return False
        
        confidence = signal.get('confidence', 0.0)
        expected_change = signal.get('expected_change_pct', 0.0)
        
        return (confidence >= self.config.min_confidence and 
                abs(expected_change) >= self.config.min_price_change)
    
    def create_trade_entry(self, signal: Dict[str, Any], opening_price: float):
        """
        Create trade entry from signal.
        Can be overridden for custom entry logic.
        """
        from .intraday_core import TradeEntry
        
        return TradeEntry(
            entry_time=self.config.entry_time,
            entry_price=opening_price,
            direction=signal['direction'],
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
            position_size=self.config.position_size
        )
    
    def calculate_results(self, trades: List[Dict[str, Any]]) -> StrategyResults:
        """Calculate strategy performance results"""
        total_trades = len(trades)
        if total_trades == 0:
            return StrategyResults(
                strategy_id=self.config.strategy_id,
                strategy_name=self.config.name,
                strategy_description=self.config.description,
                config=self.config,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0.0,
                total_pnl=0.0,
                avg_pnl_per_trade=0.0,
                trades=[]
            )
        
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        total_pnl = sum(trade['pnl'] for trade in trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        return StrategyResults(
            strategy_id=self.config.strategy_id,
            strategy_name=self.config.name,
            strategy_description=self.config.description,
            config=self.config,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            total_pnl=total_pnl,
            avg_pnl_per_trade=avg_pnl,
            trades=trades
        )


class PredictionBasedStrategy(BaseStrategy):
    """
    Strategy that uses AI predictions to generate signals.
    Configurable via StrategyConfig for different parameters.
    """
    
    def generate_signal(self, date_str: str, opening_price: float, 
                       predictions: Dict, market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate signal based on AI predictions"""
        
        # Check if we have prediction for this date
        prediction = predictions.get(date_str)
        if not prediction:
            return None
        
        # Get confidence
        confidence = prediction.get('confidence', 0.5)
        
        # Get predicted close price - handle different data formats
        predicted_close = None
        if 'predicted_close' in prediction:
            predicted_close = prediction['predicted_close']
        elif 'Close' in prediction:
            predicted_close = prediction['Close']
        elif 'predicted' in prediction and 'Close' in prediction['predicted']:
            predicted_close = prediction['predicted']['Close']
        else:
            return None  # No predicted close price available
        
        # Calculate expected price change
        expected_change_pct = (predicted_close - opening_price) / opening_price
        
        # Determine direction
        direction = 'LONG' if expected_change_pct > 0 else 'SHORT'
        
        return {
            'direction': direction,
            'confidence': confidence,
            'expected_change_pct': expected_change_pct,
            'predicted_close': predicted_close,
            'opening_price': opening_price,
            'signal_strength': abs(expected_change_pct) * confidence
        }


class CustomLogicStrategy(BaseStrategy):
    """
    Base class for strategies with completely custom logic.
    Override generate_signal() for custom signal generation.
    """
    pass


class StrategyRegistry:
    """
    Registry to manage multiple strategies and their results.
    Prevents result overwrites and enables batch processing.
    """
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.results: Dict[str, StrategyResults] = {}
    
    def register_strategy(self, strategy: BaseStrategy):
        """Register a strategy in the registry"""
        strategy_id = strategy.config.strategy_id
        if strategy_id in self.strategies:
            raise ValueError(f"Strategy '{strategy_id}' already registered")
        
        self.strategies[strategy_id] = strategy
        print(f"✅ Registered strategy: {strategy_id} - {strategy.config.name}")
    
    def get_strategy(self, strategy_id: str) -> Optional[BaseStrategy]:
        """Get strategy by ID"""
        return self.strategies.get(strategy_id)
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy IDs"""
        return list(self.strategies.keys())
    
    def add_results(self, strategy_id: str, results: StrategyResults):
        """Add results for a strategy"""
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy '{strategy_id}' not registered")
        
        self.results[strategy_id] = results
        print(f"💾 Stored results for strategy: {strategy_id}")
    
    def get_results(self, strategy_id: str) -> Optional[StrategyResults]:
        """Get results for a strategy"""
        return self.results.get(strategy_id)
    
    def get_all_results(self) -> Dict[str, StrategyResults]:
        """Get all strategy results"""
        return self.results.copy()
    
    def save_results(self, output_dir: str = None):
        """Save all results to separate files"""
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        for strategy_id, results in self.results.items():
            # Save trades CSV
            if results.trades:
                trades_df = pd.DataFrame(results.trades)
                trades_file = output_dir / f"{strategy_id}_trades.csv"
                trades_df.to_csv(trades_file, index=False)
                print(f"💾 Trades saved: {trades_file}")
            
            # Save summary JSON
            summary_file = output_dir / f"{strategy_id}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
            print(f"💾 Summary saved: {summary_file}")
            
            # Create equity curve
            if results.trades:
                equity_data = []
                running_pnl = 0
                
                for trade in results.trades:
                    running_pnl += trade['pnl']
                    equity_data.append({
                        'date': trade['date'],
                        'daily_pnl': trade['pnl'],
                        'cumulative_pnl': running_pnl
                    })
                
                equity_df = pd.DataFrame(equity_data)
                equity_file = output_dir / f"{strategy_id}_equity_curve.csv"
                equity_df.to_csv(equity_file, index=False)
                print(f"💾 Equity curve saved: {equity_file}")
    
    def print_summary(self):
        """Print summary of all strategies"""
        if not self.results:
            print("No strategy results available")
            return
        
        print("\n" + "="*80)
        print("📊 STRATEGY COMPARISON SUMMARY")
        print("="*80)
        
        summary_data = []
        for strategy_id, results in self.results.items():
            summary_data.append({
                'Strategy': results.strategy_name,
                'Trades': results.total_trades,
                'Win Rate': f"{results.win_rate_pct:.1f}%",
                'Total PnL': f"₹{results.total_pnl:,.2f}",
                'Avg PnL': f"₹{results.avg_pnl_per_trade:.2f}"
            })
        
        # Create comparison table
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print("="*80)