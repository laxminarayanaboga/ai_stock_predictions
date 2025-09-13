"""
Strategy Runner - Clean orchestration of signal generators + parameters
Runs multiple strategies with different parameter combinations
"""

import pandas as pd
import json
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import itertools
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulator.signal_generators import create_signal_generators, BaseSignalGenerator
from src.simulator.signal_simulator import SignalBasedSimulator, create_parameter_sets, TradingParameters
from src.simulator.advanced_analytics import AdvancedAnalytics


@dataclass
class StrategyConfiguration:
    """Complete strategy configuration"""
    strategy_id: str
    name: str
    description: str
    signal_generator: BaseSignalGenerator
    parameters: TradingParameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'description': self.description,
            'signal_generator_name': self.signal_generator.name,
            'signal_generator_config': {
                'min_confidence': self.signal_generator.min_confidence,
                'type': type(self.signal_generator).__name__
            },
            'parameters': self.parameters.to_dict()
        }


@dataclass
class StrategyResults:
    """Results for a single strategy"""
    config: StrategyConfiguration
    total_signals: int
    executed_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate_pct: float
    avg_pnl_per_trade: float
    avg_pnl_per_signal: float
    execution_rate_pct: float
    trade_details: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'config': self.config.to_dict(),
            'performance': {
                'total_signals': self.total_signals,
                'executed_trades': self.executed_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'total_pnl': self.total_pnl,
                'win_rate_pct': self.win_rate_pct,
                'avg_pnl_per_trade': self.avg_pnl_per_trade,
                'avg_pnl_per_signal': self.avg_pnl_per_signal,
                'execution_rate_pct': self.execution_rate_pct
            },
            'trade_details': self.trade_details
        }
        return result


class StrategyRunner:
    """
    Main strategy runner - clean and maintainable
    Orchestrates signal generation, execution, and result analysis
    """
    
    def __init__(self, data_file: str, predictions_file: str, output_dir: str = None):
        self.data_file = Path(data_file)
        self.predictions_file = Path(predictions_file)
        
        # Create timestamped output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("src/simulator/results") / f"run_{timestamp}"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize simulator
        self.simulator = SignalBasedSimulator()
        
        # Initialize advanced analytics
        self.analytics = AdvancedAnalytics()
        
        # Load data
        self.market_data = self._load_market_data()
        self.predictions = self._load_predictions()
        
        print(f"📁 Results will be saved to: {self.output_dir}")
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
        if str(self.predictions_file).endswith('.json'):
            with open(self.predictions_file, 'r') as f:
                return json.load(f)
        else:
            import pickle
            with open(self.predictions_file, 'rb') as f:
                return pickle.load(f)
    
    def run_strategy(self, config: StrategyConfiguration) -> StrategyResults:
        """Run a single strategy configuration"""
        
        print(f"\n🚀 Running Strategy: {config.name}")
        print(f"📊 {config.description}")
        print(f"⚙️  SL: {config.parameters.stop_loss_pct*100:.1f}%, TP: {config.parameters.take_profit_pct*100:.1f}%")
        print(f"🎯 Signal: {config.signal_generator.name}, Min Conf: {config.signal_generator.min_confidence*100:.0f}%")
        print("-" * 80)
        
        # Get unique trading dates
        trading_dates = sorted(self.market_data['date_str'].unique())
        
        all_execution_results = []
        total_signals = 0
        executed_trades = 0
        total_pnl = 0.0
        
        for date_str in trading_dates:
            # Get day's market data first
            day_data = self.market_data[self.market_data['date_str'] == date_str].copy()
            if day_data.empty:
                continue
            
            # Get actual opening price for validation
            actual_open = day_data.iloc[0]['close']  # First candle's close as opening price
            
            # Generate signal for this date
            signal = config.signal_generator.generate_signal(date_str, self.predictions, self.market_data)
            
            if not signal:
                continue
            
            # Add actual open price to signal metadata for validation
            if signal.metadata is None:
                signal.metadata = {}
            signal.metadata['actual_open'] = actual_open
            
            total_signals += 1
            
            # Check if signal meets trading criteria (including open price validation)
            if not config.signal_generator.should_trade(signal):
                continue
            
            # Execute signal
            result = self.simulator.execute_signal(signal, config.parameters, day_data)
            all_execution_results.append(result)
            
            if result.execution_status == 'EXECUTED':
                executed_trades += 1
                total_pnl += result.pnl
                
                # Print trade result
                status = "🟢" if result.pnl > 0 else "🔴"
                trade = result.trade_result
                print(f"{status} {date_str}: {signal.signal_type} | "
                      f"₹{trade.entry_price:.2f} → ₹{trade.exit_price:.2f} | "
                      f"PnL: ₹{result.pnl:+.2f} | {trade.exit_reason}")
                
                # Print validation info if enabled
                if config.signal_generator.validate_open_price and 'open_validation' in signal.metadata:
                    validation = signal.metadata['open_validation']
                    print(f"     Open Validation: Predicted ₹{validation['predicted_open']:.2f}, "
                          f"Actual ₹{validation['actual_open']:.2f}, "
                          f"Error {validation['error_pct']*100:.1f}% ({'✅' if validation['is_valid'] else '❌'})")
        
        # Calculate results
        winning_trades = sum(1 for r in all_execution_results if r.was_profitable)
        losing_trades = executed_trades - winning_trades
        win_rate = (winning_trades / executed_trades * 100) if executed_trades > 0 else 0
        avg_pnl_per_trade = total_pnl / executed_trades if executed_trades > 0 else 0
        avg_pnl_per_signal = total_pnl / total_signals if total_signals > 0 else 0
        execution_rate = (executed_trades / total_signals * 100) if total_signals > 0 else 0
        
        # Convert execution results to trade details
        trade_details = [result.to_dict() for result in all_execution_results]
        
        results = StrategyResults(
            config=config,
            total_signals=total_signals,
            executed_trades=executed_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            win_rate_pct=win_rate,
            avg_pnl_per_trade=avg_pnl_per_trade,
            avg_pnl_per_signal=avg_pnl_per_signal,
            execution_rate_pct=execution_rate,
            trade_details=trade_details
        )
        
        print(f"\n📈 Strategy Results:")
        print(f"   Signals Generated: {total_signals}")
        print(f"   Trades Executed: {executed_trades} ({execution_rate:.1f}%)")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total PnL: ₹{total_pnl:+.2f}")
        print(f"   Avg PnL/Trade: ₹{avg_pnl_per_trade:+.2f}")
        print(f"   Avg PnL/Signal: ₹{avg_pnl_per_signal:+.2f}")
        
        return results
    
    def run_multiple_strategies(self, configs: List[StrategyConfiguration]) -> Dict[str, StrategyResults]:
        """Run multiple strategy configurations"""
        
        results = {}
        
        for config in configs:
            try:
                result = self.run_strategy(config)
                results[config.strategy_id] = result
            except Exception as e:
                print(f"❌ Error running strategy {config.strategy_id}: {e}")
                continue
        
        return results
    
    def create_realistic_strategies(self) -> List[StrategyConfiguration]:
        """Create realistic strategy combinations - no stupid multiplication!"""
        
        signal_generators = create_signal_generators()
        parameter_sets = create_parameter_sets()
        
        # Define realistic strategy combinations that make sense
        realistic_strategies = [
            # OpenClose strategies with matching parameters
            {
                'gen': 'openclose_conservative',
                'params': 'conservative',
                'name': 'OpenClose Conservative',
                'desc': 'Conservative open-close strategy with tight stops'
            },
            {
                'gen': 'openclose_standard', 
                'params': 'aggressive',
                'name': 'OpenClose Standard',
                'desc': 'Standard open-close strategy with balanced risk'
            },
            {
                'gen': 'openclose_aggressive',
                'params': 'tight',
                'name': 'OpenClose Aggressive',
                'desc': 'Aggressive open-close strategy with tight management'
            },
            
            # HighLow strategies with appropriate parameters
            {
                'gen': 'highlow_conservative',
                'params': 'conservative', 
                'name': 'HighLow Conservative',
                'desc': 'Conservative high-low strategy with wide stops'
            },
            {
                'gen': 'highlow_standard',
                'params': 'target_based',
                'name': 'HighLow Standard', 
                'desc': 'Standard high-low strategy targeting predicted levels'
            },
            {
                'gen': 'highlow_aggressive',
                'params': 'aggressive',
                'name': 'HighLow Aggressive',
                'desc': 'Aggressive high-low strategy with higher targets'
            },
            
            # MeanReversion strategies
            {
                'gen': 'meanrev_conservative',
                'params': 'tight',
                'name': 'MeanReversion Conservative',
                'desc': 'Conservative mean reversion with tight risk management'
            },
            {
                'gen': 'meanrev_standard',
                'params': 'loose',
                'name': 'MeanReversion Standard',
                'desc': 'Standard mean reversion with loose parameters'
            },
            
            # Breakout strategies  
            {
                'gen': 'breakout_conservative',
                'params': 'target_based',
                'name': 'Breakout Conservative',
                'desc': 'Conservative breakout strategy targeting key levels'
            },
            {
                'gen': 'breakout_standard',
                'params': 'aggressive',
                'name': 'Breakout Standard',
                'desc': 'Standard breakout strategy with aggressive targets'
            },
            
            # Realistic trading strategies (like Strategy 17)
            {
                'gen': 'openclose_realistic',
                'params': 'realistic',
                'name': 'OpenClose Realistic Entry',
                'desc': 'Realistic trading: Validate @ 9:15, Enter @ 9:25, SL=1.5%, TP=3%, Max Open Error=3%'
            },
            {
                'gen': 'openclose_strict_validation',
                'params': 'realistic_strict',
                'name': 'OpenClose Strict Validation',
                'desc': 'Strict open validation: Max Open Error=1%, Delayed Entry, SL=1.5%, TP=3%'
            },
            {
                'gen': 'openclose_relaxed_validation',
                'params': 'realistic_relaxed',
                'name': 'OpenClose Relaxed Validation',
                'desc': 'Relaxed open validation: Max Open Error=5%, Delayed Entry, SL=1.5%, TP=3%'
            }
        ]
        
        configurations = []
        
        for strategy_def in realistic_strategies:
            gen_name = strategy_def['gen']
            param_name = strategy_def['params']
            
            if gen_name not in signal_generators:
                print(f"⚠️  Warning: Signal generator '{gen_name}' not found")
                continue
                
            if param_name not in parameter_sets:
                print(f"⚠️  Warning: Parameter set '{param_name}' not found")
                continue
            
            generator = signal_generators[gen_name]
            parameters = parameter_sets[param_name]
            
            strategy_id = f"{gen_name}_{param_name}"
            
            config = StrategyConfiguration(
                strategy_id=strategy_id,
                name=strategy_def['name'],
                description=strategy_def['desc'],
                signal_generator=generator,
                parameters=parameters
            )
            
            configurations.append(config)
        
        return configurations
    
    def save_results(self, results: Dict[str, StrategyResults]) -> pd.DataFrame:
        """Save detailed results like the old system - individual files per strategy"""
        
        # Create detailed results for each strategy (like old system)
        all_trades_data = []
        strategy_summaries = []
        
        for strategy_id, result in results.items():
            strategy_name = result.config.strategy_id
            
            # 1. Save individual strategy summary
            summary = {
                'strategy_id': strategy_id,
                'strategy_name': result.config.name,
                'description': result.config.description,
                'parameters': {
                    'stop_loss_pct': result.config.parameters.stop_loss_pct,
                    'take_profit_pct': result.config.parameters.take_profit_pct,
                    'position_size': result.config.parameters.position_size
                },
                'performance': {
                    'total_signals': result.total_signals,
                    'executed_trades': result.executed_trades,
                    'execution_rate_pct': result.execution_rate_pct,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'win_rate_pct': result.win_rate_pct,
                    'total_pnl': result.total_pnl,
                    'avg_pnl_per_trade': result.avg_pnl_per_trade,
                    'avg_pnl_per_signal': result.avg_pnl_per_signal
                }
            }
            
            # Save strategy summary JSON
            with open(self.output_dir / f"{strategy_name}_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # 2. Save individual trades CSV
            if result.trade_details:
                trades_df = pd.DataFrame([
                    trade
                    for trade in result.trade_details
                    if trade.get('execution_status') == 'EXECUTED'
                ])
                
                if not trades_df.empty:
                    trades_df.to_csv(self.output_dir / f"{strategy_name}_trades.csv", index=False)
                    
                    # 3. Create equity curve
                    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                    trades_df['trade_number'] = range(1, len(trades_df) + 1)
                    equity_curve = trades_df[['trade_number', 'date', 'pnl', 'cumulative_pnl']].copy()
                    equity_curve.to_csv(self.output_dir / f"{strategy_name}_equity_curve.csv", index=False)
            
            # 4. Calculate and save advanced metrics using AdvancedAnalytics
            if result.trade_details:
                # Prepare trade data for analytics
                trade_records = []
                for trade in result.trade_details:
                    if trade.get('execution_status') == 'EXECUTED':
                        trade_result = trade.get('trade_result', {})
                        
                        trade_record = {
                            'date': trade.get('date', ''),  # Date is directly in trade, not in signal
                            'pnl': trade.get('pnl', 0),
                            'entry_price': trade.get('entry_price', 0),
                            'exit_price': trade.get('exit_price', 0),
                            'exit_reason': trade.get('exit_reason', ''),
                            'duration_minutes': trade.get('duration_minutes', 0),
                            'signal_type': trade.get('signal_type', ''),
                            'confidence': trade.get('confidence', 0)
                        }
                        trade_records.append(trade_record)
                
                # Calculate advanced metrics using AdvancedAnalytics
                if trade_records:
                    advanced_analysis = self.analytics.analyze_strategy(trade_records)
                else:
                    # Empty analysis if no executed trades
                    advanced_analysis = self.analytics._empty_analysis()
            else:
                # No trades to analyze
                advanced_analysis = self.analytics._empty_analysis()
            
            # Save advanced metrics
            with open(self.output_dir / f"{strategy_name}_advanced_metrics.json", 'w') as f:
                json.dump(advanced_analysis, f, indent=2, default=str)
            
            # Collect for summary
            strategy_summaries.append(summary)
            
            # Collect all trades for combined analysis
            for trade in result.trade_details:
                if trade.get('execution_status') == 'EXECUTED':
                    trade_data = trade.copy()
                    trade_data['strategy_id'] = strategy_id
                    trade_data['strategy_name'] = result.config.name
                    all_trades_data.append(trade_data)
        
        # 5. Create overall comparison CSV
        comparison_data = []
        for summary in strategy_summaries:
            comparison_data.append({
                'Strategy ID': summary['strategy_id'],
                'Strategy Name': summary['strategy_name'], 
                'Description': summary['description'],
                'Total Signals': summary['performance']['total_signals'],
                'Executed Trades': summary['performance']['executed_trades'],
                'Execution Rate %': f"{summary['performance']['execution_rate_pct']:.1f}%",
                'Win Rate %': f"{summary['performance']['win_rate_pct']:.1f}%",
                'Total PnL ₹': f"₹{summary['performance']['total_pnl']:+,.2f}",
                'Avg PnL/Trade ₹': f"₹{summary['performance']['avg_pnl_per_trade']:+,.2f}",
                'Avg PnL/Signal ₹': f"₹{summary['performance']['avg_pnl_per_signal']:+,.2f}",
                'SL %': f"{summary['parameters']['stop_loss_pct']*100:.1f}%",
                'TP %': f"{summary['parameters']['take_profit_pct']*100:.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df_numeric = pd.DataFrame([
            {
                'Strategy ID': summary['strategy_id'],
                'Strategy Name': summary['strategy_name'],
                'Total Signals': summary['performance']['total_signals'],
                'Executed Trades': summary['performance']['executed_trades'], 
                'Execution Rate %': summary['performance']['execution_rate_pct'],
                'Win Rate %': summary['performance']['win_rate_pct'],
                'Total PnL ₹': summary['performance']['total_pnl'],
                'Avg PnL/Trade ₹': summary['performance']['avg_pnl_per_trade'],
                'Avg PnL/Signal ₹': summary['performance']['avg_pnl_per_signal']
            }
            for summary in strategy_summaries
        ])
        
        # Sort by total PnL
        comparison_df_numeric = comparison_df_numeric.sort_values('Total PnL ₹', ascending=False)
        comparison_df = comparison_df.sort_values('Total PnL ₹', ascending=False, key=lambda x: x.str.replace('₹', '').str.replace(',', '').str.replace('+', '').astype(float))
        
        # Save comparison files
        comparison_df.to_csv(self.output_dir / "strategy_comparison_detailed.csv", index=False)
        comparison_df_numeric.to_csv(self.output_dir / "strategy_comparison_numeric.csv", index=False)
        
        # 6. Create summary report
        with open(self.output_dir / "strategy_comparison_report.txt", 'w') as f:
            f.write("STRATEGY COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Strategies Tested: {len(results)}\n\n")
            
            f.write("TOP 5 STRATEGIES BY PROFIT:\n")
            f.write("-" * 30 + "\n")
            for i, (_, row) in enumerate(comparison_df_numeric.head().iterrows(), 1):
                f.write(f"{i}. {row['Strategy Name']}: {row['Total PnL ₹']:+,.2f} "
                       f"({row['Win Rate %']:.1f}% win rate, {row['Executed Trades']} trades)\n")
        
        print(f"\n💾 Detailed results saved to {self.output_dir}")
        print(f"   📊 Individual strategy files: *_summary.json, *_trades.csv, *_equity_curve.csv")
        print(f"   📈 Advanced metrics: *_advanced_metrics.json")
        print(f"   📋 Overall comparison: strategy_comparison_detailed.csv")
        print(f"   📝 Summary report: strategy_comparison_report.txt")
        
        return comparison_df_numeric


def main():
    """Main execution function with proper strategy organization"""
    
    # Configuration
    DATA_FILE = "data/raw/10min/RELIANCE_NSE_10min_20220801_to_20250831_3year_simulation.csv"
    PREDICTIONS_FILE = "data/predictions/backtest_predictions_v2_attention_extended.json"
    
    print("🚀 Strategy Runner - Proper Implementation")
    print("=" * 80)
    
    # Create runner
    runner = StrategyRunner(
        data_file=DATA_FILE,
        predictions_file=PREDICTIONS_FILE
    )
    
    # Create realistic strategy combinations (not stupid multiplication!)
    print("\n🔧 Creating realistic strategy combinations...")
    configurations = runner.create_realistic_strategies()
    
    print(f"🎯 Created {len(configurations)} realistic strategy combinations")
    print("\n📋 Strategy List:")
    for i, config in enumerate(configurations, 1):
        print(f"   {i:2d}. {config.name} - {config.description}")
    
    # Run all strategies
    print("\n🏃 Running strategies...")
    print("=" * 80)
    results = runner.run_multiple_strategies(configurations)
    
    # Create detailed results like old system
    print("\n📊 Generating detailed results...")
    print("=" * 80)
    comparison_df = runner.save_results(results)
    
    # Show summary
    print("\n🏆 Top Strategies by Profit:")
    print(comparison_df.head()[['Strategy Name', 'Total PnL ₹', 'Win Rate %', 'Executed Trades']].to_string(index=False))
    
    print("\n✅ Strategy execution completed! Check results in src/simulator/results/")


if __name__ == "__main__":
    main()