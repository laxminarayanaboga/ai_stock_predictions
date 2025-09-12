"""
AI Stock Trading Simulator
Simulates intraday trading using LSTM model predictions with realistic Fyers charges
"""

import sys
import os
sys.path.append('/Users/bogalaxminarayana/myGit/ai_stock_predictions')

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

from src.models.enhanced_lstm import EnhancedStockLSTM
from src.preprocessing.data_preprocessor import StockDataPreprocessor
from src.trading.trading_charges import FyersChargesCalculator
from src.trading.performance_metrics import TradingMetricsCalculator


class AITradingSimulator:
    """
    Simulates intraday trading using AI model predictions
    """
    
    def __init__(self, 
                 model_path: str = 'models/enhanced_stock_lstm.pth',
                 initial_capital: float = 500000.0,
                 fixed_quantity: int = 10):
        """
        Initialize the trading simulator
        
        Args:
            model_path: Path to trained LSTM model
            initial_capital: Starting capital amount
            fixed_quantity: Fixed number of shares per trade
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.fixed_quantity = fixed_quantity
        
        # Load model and preprocessor
        self.device = torch.device('cpu')
        self.model = self._load_model(model_path)
        self.preprocessor = StockDataPreprocessor()
        
        # Initialize calculators
        self.charges_calculator = FyersChargesCalculator()
        self.metrics_calculator = TradingMetricsCalculator()
        
        # Trading parameters
        self.sequence_length = 60
        self.min_prediction_confidence = 0.0001  # Minimum price change to trigger trade (0.01%)
        self.max_trades_per_day = 5  # Increase trades per day
        
        # Trading history
        self.trades = []
        self.daily_pnl = []
        self.equity_curve = []
        
        print(f"🤖 AI Trading Simulator Initialized")
        print(f"💰 Initial Capital: ₹{self.initial_capital:,.2f}")
        print(f"📈 Fixed Quantity: {self.fixed_quantity} shares per trade")
        print(f"🎯 Model loaded from: {model_path}")
    
    def _load_model(self, model_path: str) -> EnhancedStockLSTM:
        """Load the trained LSTM model"""
        try:
            # Model parameters (should match training)
            input_size = 26  # Number of features
            hidden_size = 128
            num_layers = 3
            output_size = 4  # OHLC
            dropout = 0.2
            
            model = EnhancedStockLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout
            )
            
            # Load the saved state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            print(f"✅ Model loaded successfully from {model_path}")
            
            # Store the expected input size for validation
            self.expected_input_size = input_size
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _generate_trading_signal(self, current_price: float, predicted_prices: np.ndarray) -> Tuple[str, float]:
        """
        Generate trading signal based on prediction
        
        Args:
            current_price: Current stock price
            predicted_prices: [open, high, low, close] predictions
            
        Returns:
            Tuple of (signal, confidence) where signal is 'BUY', 'SELL', or 'HOLD'
        """
        predicted_close = predicted_prices[3]  # Close price prediction
        price_change = predicted_close - current_price
        price_change_pct = price_change / current_price
        
        # Trading thresholds
        buy_threshold = self.min_prediction_confidence
        sell_threshold = -self.min_prediction_confidence
        
        if price_change_pct > buy_threshold:
            return 'BUY', abs(price_change_pct)
        elif price_change_pct < sell_threshold:
            return 'SELL', abs(price_change_pct)
        else:
            return 'HOLD', 0.0
    
    def _execute_trade(self, 
                      signal: str, 
                      entry_price: float, 
                      exit_price: float, 
                      trade_date: str,
                      confidence: float) -> Optional[Dict]:
        """
        Execute a trade and calculate P&L including all charges
        
        Args:
            signal: 'BUY' or 'SELL'
            entry_price: Entry price
            exit_price: Exit price
            trade_date: Date of trade
            confidence: Confidence level of prediction
            
        Returns:
            Trade record dictionary
        """
        try:
            if signal == 'BUY':
                # Buy at entry, sell at exit
                buy_price = entry_price
                sell_price = exit_price
            elif signal == 'SELL':
                # Short selling - sell at entry, buy at exit
                buy_price = exit_price
                sell_price = entry_price
            else:
                return None
            
            # Calculate charges
            charges = self.charges_calculator.calculate_total_charges(
                buy_price=buy_price,
                sell_price=sell_price,
                quantity=self.fixed_quantity
            )
            
            # Create trade record
            trade_record = {
                'date': trade_date,
                'signal': signal,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'quantity': self.fixed_quantity,
                'confidence': confidence,
                'gross_pnl': charges['gross_pnl'],
                'total_charges': charges['total_charges'],
                'pnl': charges['net_pnl'],
                'charge_percentage': charges['charge_percentage'],
                'turnover': charges['turnover']
            }
            
            # Update capital
            self.current_capital += charges['net_pnl']
            
            return trade_record
            
        except Exception as e:
            print(f"❌ Error executing trade: {e}")
            return None
    
    def _prepare_sequence_data(self, data: pd.DataFrame, end_idx: int) -> Optional[np.ndarray]:
        """
        Prepare sequence data for model prediction
        
        Args:
            data: Preprocessed dataframe
            end_idx: End index for sequence
            
        Returns:
            Sequence array for model input
        """
        try:
            start_idx = end_idx - self.sequence_length + 1
            
            if start_idx < 0:
                return None
                
            # Get all feature columns (excluding only date and timestamp)
            exclude_columns = ['date', 'timestamp']
            feature_columns = [col for col in data.columns if col not in exclude_columns]
            
            if len(feature_columns) == 0:
                print(f"⚠️ No feature columns found. Available columns: {list(data.columns)}")
                return None
            
            if len(feature_columns) != 26:
                print(f"⚠️ Feature mismatch: Expected 26, got {len(feature_columns)}")
                print(f"Available features: {feature_columns}")
                # Try to proceed anyway for debugging
                print(f"🔄 Proceeding with {len(feature_columns)} features for testing...")
                
            sequence_data = data[feature_columns].iloc[start_idx:end_idx + 1].values
            
            if sequence_data.shape[0] != self.sequence_length:
                return None
                
            # Handle any remaining NaN values
            if np.isnan(sequence_data).any():
                sequence_data = np.nan_to_num(sequence_data, nan=0.0)
            
            # Ensure we have the right number of features
            if sequence_data.shape[1] != self.expected_input_size:
                if sequence_data.shape[1] < self.expected_input_size:
                    # Pad with zeros if we have fewer features
                    padding = np.zeros((sequence_data.shape[0], self.expected_input_size - sequence_data.shape[1]))
                    sequence_data = np.concatenate([sequence_data, padding], axis=1)
                else:
                    # Truncate if we have more features
                    sequence_data = sequence_data[:, :self.expected_input_size]
                
            return sequence_data
            
        except Exception as e:
            print(f"❌ Error preparing sequence data: {e}")
            return None
    
    def simulate_trading(self, data_path: str = 'data/raw/RELIANCE_NSE_20150911_to_20250910.csv', 
                        start_date: str = '2024-01-01',
                        end_date: str = '2024-12-31') -> Dict:
        """
        Simulate trading for the specified period
        
        Args:
            data_path: Path to historical data CSV
            start_date: Start date for simulation
            end_date: End date for simulation
            
        Returns:
            Simulation results dictionary
        """
        print(f"\n🚀 Starting AI Trading Simulation")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"📊 Data source: {data_path}")
        
        try:
            # Load and preprocess data
            print("📈 Loading and preprocessing data...")
            raw_data = pd.read_csv(data_path)
            
            # Handle different column names for date/timestamp
            if 'timestamp' in raw_data.columns:
                raw_data['date'] = pd.to_datetime(raw_data['timestamp']).dt.date
            elif 'date' in raw_data.columns:
                raw_data['date'] = pd.to_datetime(raw_data['date']).dt.date
            else:
                raise ValueError("Data must contain either 'date' or 'timestamp' column")
            
            # Ensure OHLCV columns are in expected format (capitalized)
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            # Apply column mapping if needed
            raw_data = raw_data.rename(columns=column_mapping)
            
            # Convert date column to datetime for filtering
            raw_data['date'] = pd.to_datetime(raw_data['date'])
            
            # Filter data for simulation period with sufficient history for indicators
            # Get 6 months of history before start date for technical indicators (accounting for weekends/holidays)
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            history_start = start_dt - timedelta(days=180)  # 6 months history
            
            extended_data = raw_data[
                (raw_data['date'] >= history_start.strftime('%Y-%m-%d')) & 
                (raw_data['date'] <= end_date)
            ].copy().reset_index(drop=True)
            
            print(f"📊 Extended data (with history): {len(extended_data)} trading days")
            print(f"📊 History start: {history_start.strftime('%Y-%m-%d')}")
            print(f"📊 Data availability: {raw_data['date'].min()} to {raw_data['date'].max()}")
            
            if len(extended_data) < self.sequence_length + 90:
                print(f"❌ Need {self.sequence_length + 90} days, have {len(extended_data)} days")
                print(f"📊 Available data from: {raw_data['date'].min()}")
                raise ValueError(f"Insufficient data. Need at least {self.sequence_length + 90} days including history.")
            
            # Preprocess the data
            print("🔧 Adding technical indicators...")
            extended_data_with_indicators = self.preprocessor.add_technical_indicators(extended_data)
            
            # Prepare features but keep the original data with date for reference
            feature_data = self.preprocessor.prepare_features(extended_data_with_indicators)
            
            # Combine feature data with date column from original
            processed_data = feature_data.copy()
            processed_data['date'] = extended_data['date'].iloc[:len(feature_data)]
            
            # Forward fill any NaN values created by indicators
            processed_data = processed_data.fillna(method='ffill').dropna()
            
            # Now filter to actual simulation period after preprocessing
            simulation_data = processed_data[
                (processed_data['date'] >= start_date) & 
                (processed_data['date'] <= end_date)
            ].copy().reset_index(drop=True)
            
            print(f"📊 Simulation data (after preprocessing): {len(simulation_data)} trading days")
            
            # Start simulation
            print("🎯 Starting day-by-day simulation...")
            daily_trades_count = 0
            current_date = None
            
            for i in range(self.sequence_length, len(processed_data)):
                # Check if this row is in our simulation period
                current_date_val = processed_data.iloc[i]['date']
                if current_date_val < pd.to_datetime(start_date) or current_date_val > pd.to_datetime(end_date):
                    continue
                    
                trade_date = current_date_val.strftime('%Y-%m-%d')
                
                # Reset daily trade count for new day
                if current_date != trade_date:
                    daily_trades_count = 0
                    current_date = trade_date
                
                # Skip if max trades per day reached
                if daily_trades_count >= self.max_trades_per_day:
                    continue
                
                # Prepare sequence for prediction
                sequence_data = self._prepare_sequence_data(processed_data, i - 1)
                if sequence_data is None:
                    continue
                
                # Make prediction
                with torch.no_grad():
                    sequence_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
                    prediction = self.model(sequence_tensor)
                    predicted_prices = prediction.cpu().numpy().flatten()
                
                # Get current price (actual close price from previous day)
                current_price = processed_data.iloc[i - 1]['Close']
                
                # Generate trading signal
                signal, confidence = self._generate_trading_signal(current_price, predicted_prices)
                
                if signal != 'HOLD':
                    # Use actual OHLC prices for the trading day
                    actual_open = processed_data.iloc[i]['Open']
                    actual_close = processed_data.iloc[i]['Close']
                    
                    # Entry at market open, exit at market close (intraday)
                    entry_price = actual_open
                    exit_price = actual_close
                    
                    # Execute trade
                    trade_record = self._execute_trade(
                        signal=signal,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        trade_date=trade_date,
                        confidence=confidence
                    )
                    
                    if trade_record:
                        self.trades.append(trade_record)
                        daily_trades_count += 1
                        
                        # Print occasional trade updates
                        if len(self.trades) % 50 == 0:
                            print(f"📊 Executed {len(self.trades)} trades, Current capital: ₹{self.current_capital:,.2f}")
                
                # Record daily equity
                self.equity_curve.append({
                    'date': trade_date,
                    'capital': self.current_capital,
                    'trades_today': daily_trades_count
                })
            
            # Calculate final results
            total_return = self.current_capital - self.initial_capital
            total_return_pct = (total_return / self.initial_capital) * 100
            
            print(f"\n✅ Simulation completed!")
            print(f"📊 Total trades executed: {len(self.trades)}")
            print(f"💰 Final capital: ₹{self.current_capital:,.2f}")
            print(f"📈 Total return: ₹{total_return:,.2f} ({total_return_pct:.2f}%)")
            
            # Generate comprehensive performance report
            performance_report = self.metrics_calculator.generate_comprehensive_report(
                trades=self.trades,
                initial_capital=self.initial_capital
            )
            
            return {
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'performance_report': performance_report,
                'simulation_summary': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'trading_days': len(simulation_data),
                    'total_trades': len(self.trades),
                    'initial_capital': self.initial_capital,
                    'final_capital': self.current_capital,
                    'total_return': total_return,
                    'total_return_pct': total_return_pct
                }
            }
            
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            raise
    
    def save_results(self, results: Dict, output_dir: str = 'simulation_results') -> None:
        """Save simulation results to files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save trades
            trades_file = os.path.join(output_dir, 'trades.json')
            with open(trades_file, 'w') as f:
                json.dump(results['trades'], f, indent=2, default=str)
            
            # Save equity curve
            equity_file = os.path.join(output_dir, 'equity_curve.json')
            with open(equity_file, 'w') as f:
                json.dump(results['equity_curve'], f, indent=2, default=str)
            
            # Save performance report
            report_file = os.path.join(output_dir, 'performance_report.json')
            with open(report_file, 'w') as f:
                json.dump(results['performance_report'], f, indent=2, default=str)
            
            # Save summary as CSV
            trades_df = pd.DataFrame(results['trades'])
            trades_csv = os.path.join(output_dir, 'trades.csv')
            trades_df.to_csv(trades_csv, index=False)
            
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_csv = os.path.join(output_dir, 'equity_curve.csv')
            equity_df.to_csv(equity_csv, index=False)
            
            print(f"💾 Results saved to {output_dir}/")
            print(f"   - trades.json & trades.csv")
            print(f"   - equity_curve.json & equity_curve.csv") 
            print(f"   - performance_report.json")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Main function to run the trading simulation"""
    print("🤖 AI Stock Trading Simulator")
    print("=" * 50)
    
    # Initialize simulator
    simulator = AITradingSimulator(
        initial_capital=500000.0,  # ₹5 lakh
        fixed_quantity=10  # 10 shares per trade
    )
    
    # Run simulation for 2024 (or available data period)
    results = simulator.simulate_trading(
        data_path='data/reliance_10_years.csv',
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    # Print comprehensive performance report
    simulator.metrics_calculator.print_performance_report(results['performance_report'])
    
    # Save results
    simulator.save_results(results)
    
    # Print some key insights
    trades_df = pd.DataFrame(results['trades'])
    if len(trades_df) > 0:
        print(f"\n🔍 QUICK INSIGHTS:")
        print(f"Average charge per trade: ₹{trades_df['total_charges'].mean():.2f}")
        print(f"Average charge percentage: {trades_df['charge_percentage'].mean():.3f}%")
        print(f"Best trade: ₹{trades_df['pnl'].max():.2f}")
        print(f"Worst trade: ₹{trades_df['pnl'].min():.2f}")
        print(f"Trading days with activity: {trades_df['date'].nunique()}")


if __name__ == "__main__":
    main()
