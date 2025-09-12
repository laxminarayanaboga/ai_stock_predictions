"""
Strategy Manager - Compare and run multiple trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add src to path for imports
sys.path.append('/Users/bogalaxminarayana/myGit/ai_stock_predictions/src')

from trading.strategies.open_to_close import OpenToCloseStrategy
from trading.strategies.high_low_reversion import HighLowReversionStrategy
from trading.strategies.breakout_strategy import BreakoutStrategy
from trading.strategies.volatility_strategy import VolatilityStrategy
from trading.strategies.time_based_strategy import TimeBasedStrategy
from trading.trading_charges import FyersChargesCalculator
from trading.performance_metrics import TradingMetricsCalculator


class StrategyManager:
    """
    Manages multiple trading strategies and compares their performance
    """
    
    def __init__(self, initial_capital: float = 500000):
        self.initial_capital = initial_capital
        self.charges_calculator = FyersChargesCalculator()
        self.metrics_calculator = TradingMetricsCalculator()
        
        # Initialize all strategies
        self.strategies = {
            'open_to_close': OpenToCloseStrategy(),
            'high_low_reversion': HighLowReversionStrategy(),
            'breakout': BreakoutStrategy(),
            'volatility': VolatilityStrategy(),
            'time_based': TimeBasedStrategy()
        }
    
    def run_strategy_comparison(self, 
                              predictions_df: pd.DataFrame,
                              data_df: pd.DataFrame,
                              quantity: int = 100) -> Dict:
        """
        Run all strategies and compare their performance
        """
        
        results = {}
        
        for strategy_name, strategy in self.strategies.items():
            print(f"\n--- Running {strategy.name} Strategy ---")
            
            strategy_results = self._run_single_strategy(
                strategy, predictions_df, data_df, quantity
            )
            
            results[strategy_name] = strategy_results
            
            # Print summary for this strategy
            total_pnl = strategy_results['total_pnl']
            total_trades = len(strategy_results['trades'])
            win_rate = strategy_results['performance_metrics'].get('win_rate', 0.0)
            
            print(f"Total P&L: ₹{total_pnl:,.2f}")
            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Return: {(total_pnl/self.initial_capital)*100:.3f}%")
        
        return results
    
    def _run_single_strategy(self, 
                           strategy, 
                           predictions_df: pd.DataFrame,
                           data_df: pd.DataFrame,
                           quantity: int) -> Dict:
        """
        Run a single strategy and return detailed results
        """
        
        trades = []
        current_capital = self.initial_capital
        
        for idx, prediction_row in predictions_df.iterrows():
            if idx not in data_df.index:
                continue
            
            daily_data = data_df.loc[idx]
            current_price = daily_data['Open']  # Use open as "current" price
            
            predicted_prices = np.array([
                prediction_row['Predicted_Open'],
                prediction_row['Predicted_High'],
                prediction_row['Predicted_Low'],
                prediction_row['Predicted_Close']
            ])
            
            # Calculate prediction confidence (based on MAPE from our model)
            confidence = 1.0 - abs(predicted_prices[3] - daily_data['Close']) / daily_data['Close']
            confidence = max(0.0, min(1.0, confidence))
            
            # Generate signal
            signal, execution_details = strategy.generate_signal(
                current_price, predicted_prices, daily_data, confidence
            )
            
            if signal != 'HOLD' and execution_details:
                trade = self._execute_trade(
                    signal, execution_details, daily_data, quantity, idx
                )
                if trade:
                    trades.append(trade)
        
        # Calculate performance metrics
        if trades:
            report = self.metrics_calculator.generate_comprehensive_report(trades, self.initial_capital)
            performance_metrics = {
                'win_rate': report['profitability'].get('win_rate', 0.0),
                'profit_factor': report['profitability'].get('profit_factor', 0.0),
                'max_drawdown_pct': report['risk'].get('max_drawdown_pct', 0.0),
                'sharpe_ratio': report['risk'].get('sharpe_ratio', 0.0)
            }
            total_pnl = report['profitability'].get('net_profit', 0.0)
        else:
            performance_metrics = {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0
            }
            total_pnl = 0.0
        
        return {
            'strategy_name': strategy.name,
            'trades': trades,
            'total_pnl': total_pnl,
            'performance_metrics': performance_metrics,
            'final_capital': self.initial_capital + total_pnl
        }
    
    def _execute_trade(self, 
                      signal: str, 
                      execution_details: Dict,
                      daily_data: pd.Series,
                      quantity: int,
                      date_idx) -> Dict:
        """
        Execute a trade with realistic charges
        """
        
        entry_price = execution_details['entry_price']
        exit_price = execution_details['exit_price']
        direction = execution_details['direction']
        
        # Calculate trade value
        trade_value = entry_price * quantity
        
        # Calculate charges
        charges_result = self.charges_calculator.calculate_total_charges(
            entry_price, exit_price, quantity
        )
        
        total_charges = charges_result['total_charges']
        
        # Calculate P&L
        if direction == 'long':
            gross_pnl = (exit_price - entry_price) * quantity
        else:  # short
            gross_pnl = (entry_price - exit_price) * quantity
        
        net_pnl = gross_pnl - total_charges
        
        return {
            'date': date_idx,
            'signal': signal,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'trade_value': trade_value,
            'gross_pnl': gross_pnl,
            'total_charges': total_charges,
            'pnl': net_pnl,  # Performance metrics expects 'pnl' column
            'net_pnl': net_pnl,  # Keep for backwards compatibility
            'entry_time': execution_details.get('entry_time', 'unknown'),
            'exit_time': execution_details.get('exit_time', 'unknown')
        }
    
    def generate_comparison_report(self, results: Dict) -> str:
        """
        Generate a comprehensive comparison report
        """
        
        report = "\n" + "="*80 + "\n"
        report += "TRADING STRATEGY COMPARISON REPORT\n"
        report += "="*80 + "\n"
        
        # Strategy performance summary
        summary_data = []
        for strategy_name, result in results.items():
            summary_data.append({
                'Strategy': result['strategy_name'],
                'Total P&L': f"₹{result['total_pnl']:,.2f}",
                'Trades': len(result['trades']),
                'Return %': f"{(result['total_pnl']/self.initial_capital)*100:.3f}%",
                'Win Rate': f"{result['performance_metrics'].get('win_rate', 0):.1f}%" if result['performance_metrics'] else "0.0%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        report += "\nSTRATEGY PERFORMANCE SUMMARY:\n"
        report += summary_df.to_string(index=False)
        
        # Detailed analysis for each strategy
        for strategy_name, result in results.items():
            report += f"\n\n{'-'*60}\n"
            report += f"{result['strategy_name'].upper()} STRATEGY DETAILS\n"
            report += f"{'-'*60}\n"
            
            if result['trades']:
                metrics = result['performance_metrics']
                report += f"Total Trades: {len(result['trades'])}\n"
                report += f"Total P&L: ₹{result['total_pnl']:,.2f}\n"
                report += f"Average P&L per Trade: ₹{result['total_pnl']/len(result['trades']):,.2f}\n"
                report += f"Win Rate: {metrics.get('win_rate', 0):.1f}%\n"
                report += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
                report += f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%\n"
                report += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            else:
                report += "No trades executed for this strategy.\n"
        
        # Best strategy recommendation
        if results:
            best_strategy = max(results.items(), key=lambda x: x[1]['total_pnl'])
            report += f"\n\n{'='*60}\n"
            report += "RECOMMENDATION\n"
            report += f"{'='*60}\n"
            report += f"Best Performing Strategy: {best_strategy[1]['strategy_name']}\n"
            report += f"Total Return: ₹{best_strategy[1]['total_pnl']:,.2f} ({(best_strategy[1]['total_pnl']/self.initial_capital)*100:.3f}%)\n"
        
        return report


if __name__ == "__main__":
    # This will be called from a separate script
    print("Strategy Manager initialized. Use run_strategy_comparison.py to execute.")
