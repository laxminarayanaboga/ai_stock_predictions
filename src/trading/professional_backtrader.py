"""
Professional Intraday Trading Strategy with AI Predictions
Pure intraday approach: Enter at market open (9:15 AM), exit at close (3:15 PM)
Uses real Fyers 10-minute data for realistic backtesting
"""

import backtrader as bt
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime, time
from pathlib import Path

# Add paths
sys.path.append('/Users/bogalaxminarayana/myGit/ai_stock_predictions/src')


class IntradayAITradingStrategy(bt.Strategy):
    """
    Pure Intraday AI Trading Strategy
    
    Flow:
    1. Day N-1: AI predicts OHLC for Day N
    2. Day N at 9:15 AM: Enter trade based on AI prediction if confidence > threshold
    3. During day (10-min candles): Monitor for stop loss or take profit hits
    4. Day N at 3:15 PM: Force close any remaining positions
    5. Record daily trade results with complete details
    """
    
    params = (
        ('min_confidence', 0.6),          # 60% minimum prediction confidence
        ('stop_loss_pct', 0.02),          # 2% stop loss
        ('take_profit_pct', 0.03),        # 3% take profit
        ('position_size', 100),           # Number of shares per trade
        ('market_open_time', time(9, 15)), # 9:15 AM IST
        ('market_close_time', time(15, 15)), # 3:15 PM IST
        ('min_price_change', 0.005),      # Minimum 0.5% expected price change
    )
    
    def __init__(self):
        print("🚀 Initializing Pure Intraday AI Trading Strategy")
        
        # Trade tracking
        self.current_order = None
        self.is_exit_order = False  # Flag to track if current order is exit order
        self.daily_trade_started = False
        self.entry_price = None
        self.entry_time = None
        self.current_date = None
        self.pending_exit = None  # Store exit info for recording when order fills
        
        # Daily results tracking
        self.daily_results = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Load AI predictions
        self.daily_predictions = {}
        self._load_ai_predictions()
        
        print(f"✅ Strategy initialized with {len(self.daily_predictions)} daily predictions")
    
    def _load_ai_predictions(self):
        """Load AI predictions from backtest_predictions.pkl"""
        
        try:
            import pickle
            
            project_root = Path(__file__).parent.parent.parent
            predictions_file = project_root / 'data/predictions/backtest_predictions.pkl'
            
            if not predictions_file.exists():
                raise FileNotFoundError(f"AI predictions file not found: {predictions_file}")
            
            with open(predictions_file, 'rb') as f:
                self.daily_predictions = pickle.load(f)
            print(f"🧠 Loaded {len(self.daily_predictions)} REAL AI predictions")
            
            # Show sample predictions
            sample_count = min(3, len(self.daily_predictions))
            for i, (date, pred) in enumerate(list(self.daily_predictions.items())[:sample_count]):
                confidence = pred.get('confidence', 'N/A')
                predicted_close = pred.get('predicted_close', pred.get('Close', 'N/A'))
                print(f"   {date}: Close ₹{predicted_close:.2f} (conf: {confidence})")
                
        except Exception as e:
            print(f"❌ Failed to load AI predictions: {e}")
            print("💡 Please ensure backtest_predictions.pkl exists in data/predictions/")
            raise
    
    def next(self):
        """Called for every 10-minute candle"""
        
        # Get current datetime from unix timestamp (already in IST)
        current_dt = self.data.datetime.datetime()
        current_date_str = current_dt.strftime('%Y-%m-%d')
        current_time = current_dt.time()
        
        # Check if this is a new trading day
        if self.current_date != current_date_str:
            self._start_new_trading_day(current_date_str)
        
        # Skip if outside market hours (9:15 AM to 3:15 PM IST)
        if not self._is_market_hours(current_time):
            return
        
        # Handle existing position - check for SL/TP hits
        if self.position and self.daily_trade_started:
            self._monitor_position(current_dt)
        
        # Force close at market close (3:15 PM)
        if current_time >= self.params.market_close_time and self.position:
            self._force_close_position(current_dt, "Market Close")
            return
        
        # Enter new position at market open if no position exists
        if (not self.position and not self.daily_trade_started and 
            current_time >= self.params.market_open_time):
            self._try_enter_position(current_dt, current_date_str)
    
    def _start_new_trading_day(self, new_date):
        """Initialize new trading day"""
        
        # If there's an existing position from previous day, it means the market close
        # logic didn't fire properly - this should not happen in intraday trading
        if self.position:
            print(f"⚠️ WARNING: Found unclosed position from previous day!")
            print(f"   Position: {self.position.size} shares at entry ₹{self.entry_price}")
            print(f"   This suggests market close logic failed on previous day")
            
            # Force close the position using current price
            current_price = self.data.close[0]
            current_dt = self.data.datetime.datetime()
            self._close_position(current_dt, "Overnight Close (Emergency)", current_price)
        
        # Set the new current date
        self.current_date = new_date
        
        # Reset trade tracking for new day - no overnight positions allowed in intraday trading
        self.daily_trade_started = False
        self.entry_price = None
        self.entry_time = None
        
        print(f"\n📅 New Trading Day: {new_date}")
    
    def _is_market_hours(self, current_time):
        """Check if current time is within market hours (9:15 AM - 3:15 PM)"""
        return self.params.market_open_time <= current_time <= self.params.market_close_time
    
    def _try_enter_position(self, current_dt, date_str):
        """Try to enter position based on AI prediction"""
        
        # Get today's prediction
        prediction = self.daily_predictions.get(date_str)
        if not prediction:
            print(f"📊 No prediction available for {date_str}")
            return
        
        current_price = self.data.close[0]
        
        # Determine trade direction based on AI prediction
        trade_signal = self._get_trade_signal(prediction, current_price)
        
        if trade_signal:
            direction, confidence, predicted_price = trade_signal
            
            self.is_exit_order = False  # This is an entry order
            
            if direction == "BUY":
                self.current_order = self.buy(size=self.params.position_size)
            else:  # SELL
                self.current_order = self.sell(size=self.params.position_size)
            
            self.entry_price = current_price
            self.entry_time = current_dt
            self.daily_trade_started = True
            
            print(f"🎯 {direction} Signal at {current_price:.2f} | "
                  f"Predicted: {predicted_price:.2f} | Confidence: {confidence:.3f}")
    
    def _get_trade_signal(self, prediction, current_price):
        """Determine if we should buy/sell based on AI prediction"""
        
        confidence = prediction.get('confidence', 0.5)
        
        # Skip if confidence is too low
        if confidence < self.params.min_confidence:
            return None
        
        # Get predicted close price
        predicted_close = prediction.get('predicted_close', 
                                       prediction.get('Close', current_price))
        
        # Calculate expected price change
        expected_change_pct = (predicted_close - current_price) / current_price
        
        # Check if expected change meets minimum threshold
        if abs(expected_change_pct) < self.params.min_price_change:
            return None
        
        # Return trade direction
        if expected_change_pct > 0:
            return ("BUY", confidence, predicted_close)
        else:
            return ("SELL", confidence, predicted_close)
    
    def _monitor_position(self, current_dt):
        """Monitor existing position for SL/TP hits"""
        
        if not self.entry_price:
            return
        
        current_price = self.data.close[0]
        position_size = self.position.size
        
        # Calculate SL and TP levels
        if position_size > 0:  # Long position
            stop_loss_price = self.entry_price * (1 - self.params.stop_loss_pct)
            take_profit_price = self.entry_price * (1 + self.params.take_profit_pct)
            
            if current_price <= stop_loss_price:
                self._close_position(current_dt, "Stop Loss Hit", current_price)
            elif current_price >= take_profit_price:
                self._close_position(current_dt, "Take Profit Hit", current_price)
        
        else:  # Short position
            stop_loss_price = self.entry_price * (1 + self.params.stop_loss_pct)
            take_profit_price = self.entry_price * (1 - self.params.take_profit_pct)
            
            if current_price >= stop_loss_price:
                self._close_position(current_dt, "Stop Loss Hit", current_price)
            elif current_price <= take_profit_price:
                self._close_position(current_dt, "Take Profit Hit", current_price)
    
    def _close_position(self, exit_dt, reason, exit_price):
        """Close current position and record trade details"""
        
        if not self.position:
            return
        
        position_size = self.position.size
        
        # Store exit information for recording when order fills
        self.pending_exit = {
            'exit_dt': exit_dt,
            'reason': reason,
            'exit_price': exit_price,
            'position_size': position_size
        }
        
        # Mark this as an exit order
        self.is_exit_order = True
        
        # Place close order
        if position_size > 0:
            self.current_order = self.sell(size=abs(position_size))
        else:
            self.current_order = self.buy(size=abs(position_size))
        
        print(f"🔄 {reason}: Placing close order for {abs(position_size)} shares at ₹{exit_price:.2f}")
    
    def _force_close_position(self, exit_dt, reason):
        """Force close position at market close"""
        
        if self.position:
            current_price = self.data.close[0]
            self._close_position(exit_dt, reason, current_price)
    
    def notify_order(self, order):
        """Handle order status updates"""
        
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status == order.Completed:
            if order.isbuy():
                print(f"✅ BUY Order Filled: {order.executed.size} shares at ₹{order.executed.price:.2f}")
            else:  # Sell order
                print(f"✅ SELL Order Filled: {order.executed.size} shares at ₹{order.executed.price:.2f}")
            
            # If this is an exit order, record the trade
            if self.is_exit_order and self.pending_exit:
                self._record_completed_trade(order)
                    
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"❌ Order {order.status}: {order.ref}")
            # Clear pending exit if order failed
            if self.is_exit_order:
                self.pending_exit = None
        
        # Reset order tracking ONLY AFTER processing the order
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.current_order = None
            self.is_exit_order = False
    
    def _record_completed_trade(self, order):
        """Record a completed trade when exit order is filled"""
        
        if not self.pending_exit or not self.entry_price or not self.entry_time:
            return
        
        exit_info = self.pending_exit
        actual_exit_price = order.executed.price
        position_size = exit_info['position_size']
        
        # Calculate PnL based on actual fills
        if position_size > 0:  # Was long position
            pnl = (actual_exit_price - self.entry_price) * abs(position_size)
        else:  # Was short position  
            pnl = (self.entry_price - actual_exit_price) * abs(position_size)
        
        # Record trade details
        trade_record = {
            'date': self.current_date,
            'entry_time': self.entry_time.strftime('%H:%M:%S'),
            'exit_time': exit_info['exit_dt'].strftime('%H:%M:%S'),
            'entry_price': self.entry_price,
            'exit_price': actual_exit_price,
            'position_size': position_size,
            'direction': 'LONG' if position_size > 0 else 'SHORT',
            'pnl': pnl,
            'exit_reason': exit_info['reason']
        }
        
        self.daily_results.append(trade_record)
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            status_icon = "🟢"
        else:
            status_icon = "🔴"
        
        print(f"{status_icon} {exit_info['reason']}: Entry ₹{self.entry_price:.2f} → Exit ₹{actual_exit_price:.2f} | "
              f"PnL: ₹{pnl:.2f} | Time: {self.entry_time.strftime('%H:%M')} - {exit_info['exit_dt'].strftime('%H:%M')}")
        
        # Reset trade tracking
        self.daily_trade_started = False
        self.entry_price = None
        self.entry_time = None
        self.pending_exit = None
    
    def stop(self):
        """Called at end of backtest - generate comprehensive report"""
        
        print(f"\n{'='*80}")
        print(f"🏁 INTRADAY TRADING BACKTEST COMPLETED")
        print(f"{'='*80}")
        
        # Basic statistics
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
        
        print(f"📊 TRADING STATISTICS:")
        print(f"   Total Trading Days: {len(set(r['date'] for r in self.daily_results))}")
        print(f"   Total Trades: {self.total_trades}")
        print(f"   Winning Trades: {self.winning_trades}")
        print(f"   Losing Trades: {self.total_trades - self.winning_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total PnL: ₹{self.total_pnl:.2f}")
        print(f"   Average PnL per Trade: ₹{avg_pnl:.2f}")
        
        # Portfolio performance
        final_value = self.broker.getvalue()
        initial_value = self.broker.startingcash
        total_return = final_value - initial_value
        return_pct = (total_return / initial_value) * 100
        
        print(f"\n💰 PORTFOLIO PERFORMANCE:")
        print(f"   Starting Capital: ₹{initial_value:,.2f}")
        print(f"   Final Portfolio Value: ₹{final_value:,.2f}")
        print(f"   Total Return: ₹{total_return:,.2f}")
        print(f"   Return Percentage: {return_pct:.2f}%")
        
        # Save detailed results
        self._save_trading_results()
        
        print(f"\n✅ Detailed results saved to simulation_results/")
    
    def _save_trading_results(self):
        """Save detailed trading results to files"""
        
        try:
            # Create results directory
            results_dir = Path(__file__).parent.parent.parent / 'simulation_results'
            results_dir.mkdir(exist_ok=True)
            
            # Save daily trades
            trades_df = pd.DataFrame(self.daily_results)
            if not trades_df.empty:
                trades_df.to_csv(results_dir / 'intraday_trades.csv', index=False)
                
                # Save as JSON for easy reading
                with open(results_dir / 'intraday_trades.json', 'w') as f:
                    json.dump(self.daily_results, f, indent=2, default=str)
            
            # Create equity curve
            equity_data = []
            running_pnl = 0
            
            for trade in self.daily_results:
                running_pnl += trade['pnl']
                equity_data.append({
                    'date': trade['date'],
                    'daily_pnl': trade['pnl'],
                    'cumulative_pnl': running_pnl,
                    'portfolio_value': self.broker.startingcash + running_pnl
                })
            
            if equity_data:
                equity_df = pd.DataFrame(equity_data)
                equity_df.to_csv(results_dir / 'intraday_equity_curve.csv', index=False)
                
                with open(results_dir / 'intraday_equity_curve.json', 'w') as f:
                    json.dump(equity_data, f, indent=2, default=str)
            
            # Performance summary
            performance_summary = {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.total_trades - self.winning_trades,
                'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
                'total_pnl': self.total_pnl,
                'average_pnl_per_trade': self.total_pnl / self.total_trades if self.total_trades > 0 else 0,
                'starting_capital': self.broker.startingcash,
                'final_portfolio_value': self.broker.getvalue(),
                'total_return': self.broker.getvalue() - self.broker.startingcash,
                'return_percentage': ((self.broker.getvalue() / self.broker.startingcash) - 1) * 100
            }
            
            with open(results_dir / 'intraday_performance_report.json', 'w') as f:
                json.dump(performance_summary, f, indent=2)
            
            print(f"💾 Results saved:")
            print(f"   📊 intraday_trades.csv - Detailed trade records")
            print(f"   📈 intraday_equity_curve.csv - Daily portfolio values")
            print(f"   📋 intraday_performance_report.json - Summary statistics")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


class PandasData(bt.feeds.PandasData):
    """
    Custom data feed for intraday data using unix timestamps
    """
    
    params = (
        ('datetime', None),  # Use index as datetime
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )


def run_intraday_backtest():
    """
    Run pure intraday backtest with Fyers data and AI strategy
    Enter at 9:15 AM, exit at 3:15 PM (or when SL/TP hit)
    """
    
    print("🚀 Pure Intraday AI Trading Backtest")
    print("="*70)
    print("📋 Strategy Rules:")
    print("   • Enter: Market open (9:15 AM) based on AI prediction")
    print("   • Exit: SL/TP hit OR market close (3:15 PM)")
    print("   • Max 1 trade per day")
    print("   • Complete intraday approach")
    print("="*70)
    
    # Initialize Backtrader
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(IntradayAITradingStrategy)
    
    # Load intraday data
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / 'data/raw/10min/RELIANCE_NSE_10min_20230801_to_20250831.csv'
    
    try:
        print(f"📊 Loading data from: {data_file}")
        df = pd.read_csv(data_file)
        
        # Convert unix timestamp to datetime in IST
        df['datetime_ist'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
        
        print(f"📊 Loaded {len(df)} intraday candles from {df['datetime_ist'].iloc[0]} to {df['datetime_ist'].iloc[-1]}")
        print(f"🕐 Sample IST timestamps: {df['datetime_ist'].head(3).tolist()}")
        
        # Filter for market hours only (9:15 AM to 3:15 PM IST)
        df['time_ist'] = df['datetime_ist'].dt.time
        market_hours_filter = (
            (df['time_ist'] >= time(9, 15)) & 
            (df['time_ist'] <= time(15, 15))
        )
        
        df_market = df[market_hours_filter].copy()
        print(f"📊 Market hours data: {len(df_market)} candles")
        
        if df_market.empty:
            print("⚠️  No market hours data found. Check time filtering logic!")
            return None
        
        # Set the IST datetime as index for Backtrader
        df_market = df_market.set_index('datetime_ist')
        
        # Create data feed
        data_feed = PandasData(dataname=df_market)
        cerebro.adddata(data_feed)
        
        # Set broker parameters for intraday trading
        cerebro.broker.setcash(1000000.0)  # ₹10 Lakh starting capital
        cerebro.broker.setcommission(commission=0.0006)  # 0.06% total charges (brokerage + taxes)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        print(f"💰 Starting Capital: ₹{cerebro.broker.getvalue():,.2f}")
        print(f"💸 Commission: 0.06% (brokerage + taxes)")
        
        # Run backtest
        print(f"\n🔄 Running pure intraday backtest...")
        print("-" * 50)
        
        results = cerebro.run()
        
        # Print analyzer results
        strat = results[0]
        
        print(f"\n📊 ADVANCED PERFORMANCE METRICS:")
        print("-" * 50)
        
        # Sharpe Ratio
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe_analysis.get('sharperatio')
        if sharpe_ratio:
            print(f"📈 Sharpe Ratio: {sharpe_ratio:.3f}")
        else:
            print(f"📈 Sharpe Ratio: N/A (insufficient data)")
        
        # Drawdown Analysis
        dd_analysis = strat.analyzers.drawdown.get_analysis()
        max_drawdown = dd_analysis.get('max', {}).get('drawdown', 0)
        print(f"📉 Maximum Drawdown: {max_drawdown:.2f}%")
        
        # Trade Analysis
        trade_analysis = strat.analyzers.trades.get_analysis()
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        lost_trades = trade_analysis.get('lost', {}).get('total', 0)
        
        print(f"📊 Total Analyzed Trades: {total_trades}")
        print(f"🟢 Won Trades: {won_trades}")
        print(f"🔴 Lost Trades: {lost_trades}")
        
        if total_trades > 0:
            win_rate = (won_trades / total_trades) * 100
            print(f"🎯 Win Rate: {win_rate:.1f}%")
        
        # Returns Analysis  
        returns_analysis = strat.analyzers.returns.get_analysis()
        total_return = returns_analysis.get('rtot', 0) * 100
        print(f"💹 Total Return: {total_return:.2f}%")
        
        print(f"\n✅ Pure intraday backtest completed!")
        print(f"📁 Check simulation_results/ for detailed trade logs")
        
        return results
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_file}")
        print("💡 Please ensure the Fyers data file exists!")
        return None
    
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_intraday_backtest()
