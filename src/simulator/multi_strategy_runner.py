"""
Multi-Strategy Trading Simulator
Uses the strategy framework to run multiple strategies and compare results
"""

import pandas as pd
import pickle
from datetime import time, datetime
from pathlib import Path
from typing import Dict, List
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from simulator.strategy_base import (
    StrategyConfig, PredictionBasedStrategy, StrategyRegistry
)
from simulator.intraday_core import IntradaySimulator


class MultiStrategySimulator:
    """
    Runs multiple strategies against the same dataset and compares results
    """
    
    def __init__(self, data_file: str, predictions_file: str, output_dir: str = None):
        self.data_file = Path(data_file)
        self.predictions_file = Path(predictions_file)
        self.simulator = IntradaySimulator()
        self.registry = StrategyRegistry()
        
        # Create timestamped output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("src/simulator/results") / f"run_{timestamp}"
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Results will be saved to: {self.output_dir}")
        
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
    
    def add_strategy(self, config: StrategyConfig, strategy_class=PredictionBasedStrategy):
        """Add a strategy to the simulator"""
        strategy = strategy_class(config)
        self.registry.register_strategy(strategy)
    
    def run_strategy(self, strategy_id: str) -> Dict:
        """Run a single strategy and return results"""
        strategy = self.registry.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy '{strategy_id}' not found")
        
        config = strategy.config
        print(f"\n🚀 Running Strategy: {config.name}")
        print(f"📊 {config.description}")
        print(f"⚙️  SL: {config.stop_loss_pct*100:.1f}%, TP: {config.take_profit_pct*100:.1f}%, Conf: {config.min_confidence*100:.0f}%")
        print("-" * 60)
        
        # Get unique trading dates
        trading_dates = sorted(self.market_data['date_str'].unique())
        
        all_trades = []
        total_pnl = 0.0
        winning_trades = 0
        signals_generated = 0
        
        for date_str in trading_dates:
            # Get day's data
            day_data = self.market_data[self.market_data['date_str'] == date_str].copy()
            
            if day_data.empty:
                continue
            
            # Get opening price (first candle of the day)
            opening_price = day_data.iloc[0]['close']  # Use close of first candle as opening
            
            # Generate signal using strategy
            signal = strategy.generate_signal(date_str, opening_price, 
                                            self.predictions, self.market_data)
            
            if signal:
                signals_generated += 1
            
            # Check if we should enter trade
            if not strategy.should_enter_trade(signal):
                continue
            
            # Create trade entry
            trade_entry = strategy.create_trade_entry(signal, opening_price)
            
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
                    'duration_minutes': result.duration_minutes,
                    'signal_strength': signal.get('signal_strength', 0),
                    'confidence': signal.get('confidence', 0)
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
        
        # Calculate results using strategy framework
        results = strategy.calculate_results(all_trades)
        
        # Add metadata
        results.metadata = {
            'signals_generated': signals_generated,
            'signal_to_trade_ratio': len(all_trades) / signals_generated if signals_generated > 0 else 0,
            'data_period': f"{trading_dates[0]} to {trading_dates[-1]}",
            'trading_days': len(trading_dates)
        }
        
        # Store results in registry
        self.registry.add_results(strategy_id, results)
        
        # Print summary
        print(f"\n📊 STRATEGY RESULTS:")
        print(f"   Signals Generated: {signals_generated}")
        print(f"   Total Trades: {results.total_trades}")
        print(f"   Win Rate: {results.win_rate_pct:.1f}%")
        print(f"   Total PnL: ₹{results.total_pnl:,.2f}")
        print(f"   Avg PnL/Trade: ₹{results.avg_pnl_per_trade:.2f}")
        
        return results.to_dict()
    
    def run_all_strategies(self) -> Dict[str, Dict]:
        """Run all registered strategies"""
        results = {}
        strategy_ids = self.registry.list_strategies()
        
        if not strategy_ids:
            print("⚠️ No strategies registered")
            return results
        
        print(f"\n🔄 Running {len(strategy_ids)} strategies...")
        
        for strategy_id in strategy_ids:
            try:
                result = self.run_strategy(strategy_id)
                results[strategy_id] = result
            except Exception as e:
                print(f"❌ Error running strategy {strategy_id}: {e}")
        
        return results
    
    def save_all_results(self, output_dir: str = None):
        """Save all strategy results"""
        if output_dir is None:
            output_dir = str(self.output_dir)
        self.registry.save_results(output_dir)
    
    def print_comparison(self):
        """Print comparison of all strategies"""
        self.registry.print_summary()
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report with charts"""
        print("\n🔄 Generating comprehensive strategy comparison report...")
        
        # Load strategy data
        strategies = []
        for strategy_id in self.registry.list_strategies():
            results = self.registry.get_results(strategy_id)
            if results:
                config = self.registry.get_strategy(strategy_id).config
                strategies.append({
                    'Strategy ID': strategy_id,
                    'Strategy Name': config.name,
                    'Description': config.description,
                    'Stop Loss %': config.stop_loss_pct * 100,
                    'Take Profit %': config.take_profit_pct * 100,
                    'Min Confidence %': config.min_confidence * 100,
                    'Total Trades': results.total_trades,
                    'Winning Trades': results.winning_trades,
                    'Losing Trades': results.losing_trades,
                    'Win Rate %': round(results.win_rate_pct, 2),
                    'Total PnL': round(results.total_pnl, 2),
                    'Avg PnL/Trade': round(results.avg_pnl_per_trade, 2),
                })
        
        if not strategies:
            print("⚠️ No strategy data found for report generation")
            return
        
        df = pd.DataFrame(strategies)
        df = df.sort_values('Total PnL', ascending=False)
        
        # Generate text report
        report_text = self._create_text_report(df)
        
        # Save text report
        report_path = self.output_dir / "strategy_comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Save detailed CSV
        csv_path = self.output_dir / "strategy_comparison_detailed.csv"
        df.to_csv(csv_path, index=False)
        
        # Create visualizations
        chart_path = self._create_visualizations(df)
        
        print(f"✅ Comparison report generated!")
        print(f"📄 Text report: {report_path}")
        print(f"📊 Charts: {chart_path}")
        print(f"📋 Detailed CSV: {csv_path}")
        
        # Print quick summary
        if len(df) > 0:
            best = df.iloc[0]
            print(f"\n🏆 Best Strategy: {best['Strategy Name']}")
            print(f"💰 Best PnL: ₹{best['Total PnL']:,.0f}")
            print(f"📈 Best Win Rate: {df['Win Rate %'].max():.1f}%")
        
        return report_path, chart_path
    
    def _create_text_report(self, df: pd.DataFrame) -> str:
        """Create detailed text report"""
        report = []
        report.append("=" * 100)
        report.append("📊 COMPREHENSIVE STRATEGY COMPARISON REPORT")
        report.append("=" * 100)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Strategies Analyzed: {len(df)}")
        report.append("")
        
        # Performance Summary Table
        report.append("📈 PERFORMANCE RANKING (Best to Worst by Total PnL)")
        report.append("-" * 100)
        report.append(f"{'Rank':<4} {'Strategy':<25} {'Trades':<7} {'Win%':<6} {'Total PnL':<12} {'Avg PnL':<10} {'SL%':<5} {'TP%':<5} {'Conf%':<6}")
        report.append("-" * 100)
        
        for i, (_, row) in enumerate(df.iterrows()):
            rank = i + 1
            report.append(f"{rank:<4} {row['Strategy Name']:<25} {row['Total Trades']:<7} {row['Win Rate %']:<6} ₹{row['Total PnL']:<11,.0f} ₹{row['Avg PnL/Trade']:<9,.0f} {row['Stop Loss %']:<5.1f} {row['Take Profit %']:<5.1f} {row['Min Confidence %']:<6.0f}")
        
        report.append("")
        
        # Best Strategy Analysis
        if len(df) > 0:
            best = df.iloc[0]
            report.append("🏆 BEST PERFORMING STRATEGY")
            report.append("-" * 50)
            report.append(f"Strategy: {best['Strategy Name']}")
            report.append(f"Description: {best['Description']}")
            report.append(f"Parameters: SL={best['Stop Loss %']:.1f}%, TP={best['Take Profit %']:.1f}%, Confidence={best['Min Confidence %']:.0f}%")
            report.append(f"Performance: {best['Total Trades']} trades, {best['Win Rate %']:.1f}% win rate")
            report.append(f"Total PnL: ₹{best['Total PnL']:,.0f} (₹{best['Avg PnL/Trade']:,.0f} per trade)")
            report.append("")
            
            # Worst Strategy Analysis
            if len(df) > 1:
                worst = df.iloc[-1]
                report.append("📉 WORST PERFORMING STRATEGY")
                report.append("-" * 50)
                report.append(f"Strategy: {worst['Strategy Name']}")
                report.append(f"Description: {worst['Description']}")
                report.append(f"Parameters: SL={worst['Stop Loss %']:.1f}%, TP={worst['Take Profit %']:.1f}%, Confidence={worst['Min Confidence %']:.0f}%")
                report.append(f"Performance: {worst['Total Trades']} trades, {worst['Win Rate %']:.1f}% win rate")
                report.append(f"Total PnL: ₹{worst['Total PnL']:,.0f} (₹{worst['Avg PnL/Trade']:,.0f} per trade)")
                report.append("")
        
        # Key Insights
        report.append("💡 KEY INSIGHTS")
        report.append("-" * 50)
        
        # Win rate analysis
        avg_win_rate = df['Win Rate %'].mean()
        best_win_rate = df.loc[df['Win Rate %'].idxmax()]
        report.append(f"• Average win rate across all strategies: {avg_win_rate:.1f}%")
        report.append(f"• Highest win rate: {best_win_rate['Win Rate %']:.1f}% ({best_win_rate['Strategy Name']})")
        
        # Trade volume analysis
        avg_trades = df['Total Trades'].mean()
        report.append(f"• Average trades per strategy: {avg_trades:.0f}")
        
        # Parameter analysis
        report.append("")
        report.append("📊 PARAMETER ANALYSIS")
        report.append("-" * 50)
        
        # Stop Loss analysis
        sl_performance = df.groupby('Stop Loss %')['Total PnL'].mean().sort_values(ascending=False)
        report.append("Stop Loss % Performance:")
        for sl, pnl in sl_performance.items():
            report.append(f"  • {sl:.1f}%: ₹{pnl:,.0f} average PnL")
        
        # Take Profit analysis
        tp_performance = df.groupby('Take Profit %')['Total PnL'].mean().sort_values(ascending=False)
        report.append("\nTake Profit % Performance:")
        for tp, pnl in tp_performance.items():
            report.append(f"  • {tp:.1f}%: ₹{pnl:,.0f} average PnL")
        
        # Confidence analysis
        conf_performance = df.groupby('Min Confidence %')['Total PnL'].mean().sort_values(ascending=False)
        report.append("\nMin Confidence % Performance:")
        for conf, pnl in conf_performance.items():
            report.append(f"  • {conf:.0f}%: ₹{pnl:,.0f} average PnL")
        
        report.append("")
        report.append("🎯 RECOMMENDATIONS")
        report.append("-" * 50)
        if len(df) > 0:
            best = df.iloc[0]
            report.append("Based on the analysis:")
            report.append(f"1. Use {best['Strategy Name']} as your primary strategy")
            report.append(f"2. Optimal parameters appear to be: SL={best['Stop Loss %']:.1f}%, TP={best['Take Profit %']:.1f}%, Conf={best['Min Confidence %']:.0f}%")
            
            # Find best confidence threshold
            best_conf = conf_performance.index[0] if len(conf_performance) > 0 else 60
            report.append(f"3. Higher confidence thresholds ({best_conf:.0f}%+) tend to perform better")
            
            # Find best stop loss
            best_sl = sl_performance.index[0] if len(sl_performance) > 0 else 1.5
            report.append(f"4. Optimal stop loss appears to be around {best_sl:.1f}%")
            
            report.append("5. Consider further model improvements to increase prediction accuracy")
        
        report.append("")
        report.append(f"📁 Results saved to: {self.output_dir}")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def _create_visualizations(self, df: pd.DataFrame):
        """Create comparison charts"""
        if len(df) == 0:
            return None
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Strategy Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Total PnL Comparison
        df_sorted = df.sort_values('Total PnL', ascending=True)
        colors = ['red' if x < 0 else 'green' for x in df_sorted['Total PnL']]
        ax1.barh(range(len(df_sorted)), df_sorted['Total PnL'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(df_sorted)))
        ax1.set_yticklabels([name.replace('Strategy ', 'S') for name in df_sorted['Strategy Name']], fontsize=9)
        ax1.set_xlabel('Total PnL (₹)')
        ax1.set_title('Total PnL by Strategy')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(df_sorted['Total PnL']):
            ax1.text(v + (max(df_sorted['Total PnL']) - min(df_sorted['Total PnL'])) * 0.01, i, f'₹{v:,.0f}', 
                    ha='left' if v < 0 else 'left', va='center', fontsize=8)
        
        # 2. Win Rate vs Total Trades
        scatter = ax2.scatter(df['Total Trades'], df['Win Rate %'], 
                             c=df['Total PnL'], cmap='RdYlGn', s=100, alpha=0.7)
        ax2.set_xlabel('Total Trades')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Win Rate vs Trade Volume')
        ax2.grid(alpha=0.3)
        
        # Add strategy labels
        for i, (_, row) in enumerate(df.iterrows()):
            ax2.annotate(row['Strategy Name'].split(' - ')[1] if ' - ' in row['Strategy Name'] else row['Strategy Name'], 
                        (row['Total Trades'], row['Win Rate %']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Total PnL (₹)')
        
        # 3. Parameter Impact - Stop Loss
        if len(df['Stop Loss %'].unique()) > 1:
            sl_grouped = df.groupby('Stop Loss %')['Total PnL'].mean().reset_index()
            ax3.bar(sl_grouped['Stop Loss %'], sl_grouped['Total PnL'], alpha=0.7, color='skyblue')
            ax3.set_xlabel('Stop Loss (%)')
            ax3.set_ylabel('Average Total PnL (₹)')
            ax3.set_title('Stop Loss Impact on Performance')
            ax3.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, row in sl_grouped.iterrows():
                ax3.text(row['Stop Loss %'], row['Total PnL'] + abs(row['Total PnL']) * 0.05, 
                        f'₹{row["Total PnL"]:,.0f}', ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Not enough data\nfor Stop Loss analysis', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Stop Loss Impact')
        
        # 4. Confidence Threshold Impact
        if len(df['Min Confidence %'].unique()) > 1:
            conf_grouped = df.groupby('Min Confidence %')['Total PnL'].mean().reset_index()
            ax4.bar(conf_grouped['Min Confidence %'], conf_grouped['Total PnL'], alpha=0.7, color='lightcoral')
            ax4.set_xlabel('Min Confidence (%)')
            ax4.set_ylabel('Average Total PnL (₹)')
            ax4.set_title('Confidence Threshold Impact')
            ax4.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, row in conf_grouped.iterrows():
                ax4.text(row['Min Confidence %'], row['Total PnL'] + abs(row['Total PnL']) * 0.05, 
                        f'₹{row["Total PnL"]:,.0f}', ha='center', va='bottom', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'Not enough data\nfor Confidence analysis', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Confidence Impact')
        
        plt.tight_layout()
        
        # Save the plot
        chart_path = self.output_dir / "strategy_comparison_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path


# Define predefined strategies
def create_predefined_strategies() -> List[StrategyConfig]:
    """Create a set of predefined strategies for testing"""
    
    strategies = [
        # Original strategies
        StrategyConfig(
            strategy_id="strategy_02_tight_sl",
            name="Strategy 02 - Tight SL",
            description="Tight stop loss: SL=1%, TP=3%, Conf=60%",
            stop_loss_pct=0.01,
            take_profit_pct=0.03,
            min_confidence=0.6,
            min_price_change=0.005,
            position_size=100
        ),
        
        StrategyConfig(
            strategy_id="strategy_03_reduced_tp",
            name="Strategy 03 - Reduced TP", 
            description="Reduced target: SL=1%, TP=2%, Conf=60%",
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            min_confidence=0.6,
            min_price_change=0.005,
            position_size=100
        ),
        
        # New strategies with different parameters
        StrategyConfig(
            strategy_id="strategy_04_aggressive",
            name="Strategy 04 - Aggressive",
            description="Higher risk: SL=2%, TP=5%, Conf=50%",
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            min_confidence=0.5,
            min_price_change=0.005,
            position_size=100
        ),
        
        StrategyConfig(
            strategy_id="strategy_05_conservative",
            name="Strategy 05 - Conservative",
            description="Very safe: SL=0.5%, TP=1.5%, Conf=70%",
            stop_loss_pct=0.005,
            take_profit_pct=0.015,
            min_confidence=0.7,
            min_price_change=0.005,
            position_size=100
        ),
        
        StrategyConfig(
            strategy_id="strategy_06_high_conf",
            name="Strategy 06 - High Confidence",
            description="AI selective: SL=1.5%, TP=4%, Conf=80%",
            stop_loss_pct=0.015,
            take_profit_pct=0.04,
            min_confidence=0.8,
            min_price_change=0.01,
            position_size=100
        ),
        
        StrategyConfig(
            strategy_id="strategy_07_wide_stops",
            name="Strategy 07 - Wide Stops",
            description="Patient trader: SL=3%, TP=6%, Conf=55%",
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            min_confidence=0.55,
            min_price_change=0.005,
            position_size=100
        ),
        
        StrategyConfig(
            strategy_id="strategy_08_scalping",
            name="Strategy 08 - Scalping",
            description="Quick profits: SL=0.5%, TP=1%, Conf=65%",
            stop_loss_pct=0.005,
            take_profit_pct=0.01,
            min_confidence=0.65,
            min_price_change=0.003,
            position_size=100
        ),
        
        StrategyConfig(
            strategy_id="strategy_09_balanced",
            name="Strategy 09 - Balanced",
            description="Middle ground: SL=1.5%, TP=3%, Conf=65%",
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
            min_confidence=0.65,
            min_price_change=0.005,
            position_size=100
        ),
        
        StrategyConfig(
            strategy_id="strategy_10_trend_follow",
            name="Strategy 10 - Trend Following",
            description="Big moves: SL=2.5%, TP=7.5%, Conf=60%",
            stop_loss_pct=0.025,
            take_profit_pct=0.075,
            min_confidence=0.6,
            min_price_change=0.01,
            position_size=100
        )
    ]
    
    return strategies


def main():
    """Main function to run multiple strategies"""
    
    # Clean up old results (optional - keep last 5 runs)
    results_base_dir = Path("src/simulator/results")
    if results_base_dir.exists():
        # Get all run directories
        run_dirs = [d for d in results_base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        run_dirs.sort(key=lambda x: x.name, reverse=True)  # Sort by timestamp, newest first
        
        # Keep only the 5 most recent runs, delete older ones
        for old_dir in run_dirs[5:]:
            print(f"🗑️ Cleaning up old run: {old_dir.name}")
            shutil.rmtree(old_dir)
    
    # Define data paths
    data_file = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/data/raw/10min/RELIANCE_NSE_10min_20230801_to_20250831.csv"
    predictions_file = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/data/predictions/backtest_predictions.pkl"
    
    # Create multi-strategy simulator
    simulator = MultiStrategySimulator(data_file, predictions_file)
    
    # Add predefined strategies
    predefined_strategies = create_predefined_strategies()
    for strategy_config in predefined_strategies:
        simulator.add_strategy(strategy_config)
    
    # Run all strategies
    results = simulator.run_all_strategies()
    
    # Print comparison
    simulator.print_comparison()
    
    # Save all results
    simulator.save_all_results()
    
    # Generate comprehensive comparison report
    simulator.generate_comparison_report()
    
    print(f"\n✅ Multi-strategy simulation completed!")
    print(f"📊 {len(results)} strategies executed")
    print(f"📁 All results saved to: {simulator.output_dir}")


if __name__ == "__main__":
    main()