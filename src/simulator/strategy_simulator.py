"""
Trading Strategy Implementation
Uses the core simulator to test different trading strategies
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import time, datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from simulator.intraday_core import IntradaySimulator, TradeEntry, TradeResult


class StrategyConfig:
    """Strategy configuration"""
    def __init__(self, 
                 name: str,
                 description: str,
                 stop_loss_pct: float,
                 take_profit_pct: float,
                 min_confidence: float = 0.6,
                 min_price_change: float = 0.005,
                 position_size: int = 100,
                 entry_time: time = time(9, 15)):
        
        self.name = name
        self.description = description
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_confidence = min_confidence
        self.min_price_change = min_price_change
        self.position_size = position_size
        self.entry_time = entry_time


class StrategySimulator:
    """
    Strategy-level simulator that processes multiple days
    """
    
    def __init__(self, data_file: str, predictions_file: str):
        self.data_file = Path(data_file)
        self.predictions_file = Path(predictions_file)
        self.simulator = IntradaySimulator()
        
        # Load data
        self.market_data = self._load_market_data()
        self.predictions = self._load_predictions()
        
        print(f"📊 Loaded {len(self.market_data)} market candles")
        print(f"🧠 Loaded {len(self.predictions)} AI predictions")
    
    def _load_market_data(self) -> pd.DataFrame:
        """Load and process market data"""
        
        df = pd.read_csv(self.data_file)
        df['datetime_ist'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
        df['time_ist'] = df['datetime_ist'].dt.time
        df['date_str'] = df['datetime_ist'].dt.strftime('%Y-%m-%d')
        
        # Filter for market hours (9:15 AM to 3:15 PM)
        market_hours_filter = (
            (df['time_ist'] >= time(9, 15)) & 
            (df['time_ist'] <= time(15, 15))
        )
        
        return df[market_hours_filter].copy()
    
    def _load_predictions(self) -> Dict:
        """Load AI predictions"""
        
        with open(self.predictions_file, 'rb') as f:
            return pickle.load(f)
    
    def _get_trade_signal(self, date_str: str, opening_price: float, config: StrategyConfig) -> Optional[TradeEntry]:
        """
        Generate trade signal for a given day
        
        Args:
            date_str: Date in 'YYYY-MM-DD' format
            opening_price: Opening price for the day
            config: Strategy configuration
        
        Returns:
            TradeEntry if signal generated, None otherwise
        """
        
        # Check if we have prediction for this date
        prediction = self.predictions.get(date_str)
        if not prediction:
            return None
        
        # Check confidence threshold
        confidence = prediction.get('confidence', 0.5)
        if confidence < config.min_confidence:
            return None
        
        # Get predicted close price
        predicted_close = prediction.get('predicted_close', prediction.get('Close', opening_price))
        
        # Calculate expected price change
        expected_change_pct = (predicted_close - opening_price) / opening_price
        
        # Check if expected change meets minimum threshold
        if abs(expected_change_pct) < config.min_price_change:
            return None
        
        # Determine direction
        direction = 'LONG' if expected_change_pct > 0 else 'SHORT'
        
        # Create trade entry
        return TradeEntry(
            entry_time=config.entry_time,
            entry_price=opening_price,
            direction=direction,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            position_size=config.position_size
        )
    
    def run_strategy(self, config: StrategyConfig) -> Dict:
        """
        Run complete strategy simulation
        
        Args:
            config: Strategy configuration
        
        Returns:
            Dictionary with results
        """
        
        print(f"\n🚀 Running Strategy: {config.name}")
        print(f"📊 {config.description}")
        print(f"⚙️  SL: {config.stop_loss_pct*100:.1f}%, TP: {config.take_profit_pct*100:.1f}%, Conf: {config.min_confidence*100:.0f}%")
        print("-" * 60)
        
        # Get unique trading dates
        trading_dates = sorted(self.market_data['date_str'].unique())
        
        all_trades = []
        daily_results = []
        total_pnl = 0.0
        winning_trades = 0
        
        for date_str in trading_dates:
            # Get day's data
            day_data = self.market_data[self.market_data['date_str'] == date_str].copy()
            
            if day_data.empty:
                continue
            
            # Get opening price (first candle of the day)
            opening_price = day_data.iloc[0]['close']  # Use close of first candle as opening
            
            # Generate trade signal
            trade_entry = self._get_trade_signal(date_str, opening_price, config)
            
            if not trade_entry:
                # No signal for this day
                continue
            
            # Simulate the trade
            result = self.simulator.simulate_trade(trade_entry, day_data)
            
            if result:
                # Record the trade
                trade_record = {
                    'date': date_str,
                    'entry_time': result.entry_time.strftime('%H:%M:%S'),
                    'exit_time': result.exit_time.strftime('%H:%M:%S'),
                    'entry_price': result.entry_price,
                    'exit_price': result.exit_price,
                    'direction': result.direction,
                    'position_size': result.position_size,
                    'pnl': result.pnl,
                    'exit_reason': result.exit_reason,
                    'duration_minutes': result.duration_minutes
                }
                
                all_trades.append(trade_record)
                total_pnl += result.pnl
                
                if result.pnl > 0:
                    winning_trades += 1
                    status = "🟢"
                else:
                    status = "🔴"
                
                print(f"{status} {date_str}: {result.direction} | "
                      f"₹{result.entry_price:.2f} → ₹{result.exit_price:.2f} | "
                      f"PnL: ₹{result.pnl:+.2f} | {result.exit_reason}")
        
        # Calculate summary statistics
        total_trades = len(all_trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # Create results summary
        results = {
            'strategy_name': config.name,
            'strategy_description': config.description,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate_pct': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'config': {
                'stop_loss_pct': config.stop_loss_pct,
                'take_profit_pct': config.take_profit_pct,
                'min_confidence': config.min_confidence,
                'min_price_change': config.min_price_change,
                'position_size': config.position_size
            },
            'trades': all_trades
        }
        
        # Print summary
        print(f"\n📊 STRATEGY RESULTS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total PnL: ₹{total_pnl:,.2f}")
        print(f"   Avg PnL/Trade: ₹{avg_pnl:.2f}")
        
        return results
    
    def save_results(self, results: Dict, output_dir: str = None):
        """Save results to files"""
        
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        strategy_name = results['strategy_name']
        
        # Save trades CSV
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_file = output_dir / f"{strategy_name}_trades.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"💾 Trades saved: {trades_file}")
        
        # Save summary JSON
        summary_file = output_dir / f"{strategy_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Summary saved: {summary_file}")
        
        # Create equity curve
        if results['trades']:
            equity_data = []
            running_pnl = 0
            
            for trade in results['trades']:
                running_pnl += trade['pnl']
                equity_data.append({
                    'date': trade['date'],
                    'daily_pnl': trade['pnl'],
                    'cumulative_pnl': running_pnl
                })
            
            equity_df = pd.DataFrame(equity_data)
            equity_file = output_dir / f"{strategy_name}_equity_curve.csv"
            equity_df.to_csv(equity_file, index=False)
            print(f"💾 Equity curve saved: {equity_file}")


def main():
    """Main function to test the strategy simulator"""
    
    # Define data paths
    data_file = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/data/raw/10min/RELIANCE_NSE_10min_20230801_to_20250831.csv"
    predictions_file = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/data/predictions/backtest_predictions.pkl"
    
    # Create strategy simulator
    strategy_sim = StrategySimulator(data_file, predictions_file)
    
    # Define Strategy 2 (tight stop loss)
    strategy_2 = StrategyConfig(
        name="strategy_02_tight_sl",
        description="Tight stop loss: SL=1%, TP=3%, Conf=60%",
        stop_loss_pct=0.01,
        take_profit_pct=0.03,
        min_confidence=0.6,
        min_price_change=0.005,
        position_size=100
    )
    
    # Run the strategy
    results = strategy_sim.run_strategy(strategy_2)
    
    # Save results
    strategy_sim.save_results(results)
    
    print(f"\n✅ Strategy simulation completed!")


if __name__ == "__main__":
    main()