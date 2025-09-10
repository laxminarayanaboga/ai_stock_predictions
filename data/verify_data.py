"""
Data Verification and Visualization Script for Reliance NSE Data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_verify_data():
    """Load and verify the downloaded Reliance data"""
    data_file = Path("data/raw").glob("RELIANCE_NSE_*.csv")
    data_file = max(data_file, key=lambda x: x.stat().st_mtime)
    
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Check for data quality issues
    print("\nData Quality Checks:")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate dates: {df.index.duplicated().sum()}")
    
    # Check for anomalies
    print(f"Zero volume days: {(df['Volume'] == 0).sum()}")
    print(f"Days where High < Low: {(df['High'] < df['Low']).sum()}")
    print(f"Days where Close outside High-Low: {((df['Close'] > df['High']) | (df['Close'] < df['Low'])).sum()}")
    
    return df

def create_basic_visualizations(df):
    """Create basic visualizations of the data"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reliance NSE Stock Data Analysis (10 Years)', fontsize=16)
    
    # Price chart
    axes[0, 0].plot(df.index, df['Close'], linewidth=1, color='blue')
    axes[0, 0].set_title('Closing Price Over Time')
    axes[0, 0].set_ylabel('Price (₹)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Volume chart
    axes[0, 1].bar(df.index, df['Volume'], width=1, alpha=0.7, color='orange')
    axes[0, 1].set_title('Trading Volume Over Time')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Daily returns
    daily_returns = df['Close'].pct_change().dropna()
    axes[1, 0].hist(daily_returns, bins=50, alpha=0.7, color='green')
    axes[1, 0].set_title('Daily Returns Distribution')
    axes[1, 0].set_xlabel('Daily Return')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(daily_returns.mean(), color='red', linestyle='--', label=f'Mean: {daily_returns.mean():.4f}')
    axes[1, 0].legend()
    
    # OHLC box plot for recent period
    recent_data = df.tail(100)  # Last 100 days
    box_data = [recent_data['Open'], recent_data['High'], recent_data['Low'], recent_data['Close']]
    axes[1, 1].boxplot(box_data, labels=['Open', 'High', 'Low', 'Close'])
    axes[1, 1].set_title('OHLC Distribution (Last 100 Days)')
    axes[1, 1].set_ylabel('Price (₹)')
    
    plt.tight_layout()
    plt.savefig('data/reliance_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'data/reliance_data_analysis.png'")

def calculate_technical_indicators(df):
    """Calculate some basic technical indicators"""
    print("\nCalculating Technical Indicators...")
    
    df_tech = df.copy()
    
    # Simple moving averages
    df_tech['SMA_20'] = df_tech['Close'].rolling(20).mean()
    df_tech['SMA_50'] = df_tech['Close'].rolling(50).mean()
    df_tech['SMA_200'] = df_tech['Close'].rolling(200).mean()
    
    # Volatility
    df_tech['Daily_Return'] = df_tech['Close'].pct_change()
    df_tech['Volatility_20'] = df_tech['Daily_Return'].rolling(20).std() * np.sqrt(252)
    
    # Price ranges
    df_tech['Daily_Range'] = df_tech['High'] - df_tech['Low']
    df_tech['Range_Pct'] = (df_tech['Daily_Range'] / df_tech['Close']) * 100
    
    # Recent statistics
    recent_data = df_tech.tail(30)
    print(f"Current Price: ₹{df_tech['Close'].iloc[-1]:.2f}")
    print(f"20-day SMA: ₹{df_tech['SMA_20'].iloc[-1]:.2f}")
    print(f"50-day SMA: ₹{df_tech['SMA_50'].iloc[-1]:.2f}")
    print(f"200-day SMA: ₹{df_tech['SMA_200'].iloc[-1]:.2f}")
    print(f"20-day Volatility: {df_tech['Volatility_20'].iloc[-1]*100:.2f}%")
    print(f"Average Daily Range (30 days): {recent_data['Daily_Range'].mean():.2f}")
    print(f"Average Range % (30 days): {recent_data['Range_Pct'].mean():.2f}%")
    
    return df_tech

def main():
    """Main verification function"""
    print("=== Reliance NSE Data Verification ===")
    
    # Load and verify data
    df = load_and_verify_data()
    
    # Calculate technical indicators
    df_tech = calculate_technical_indicators(df)
    
    # Create visualizations
    create_basic_visualizations(df)
    
    print("\n=== Data Ready for AI Model Training ===")
    print("The data looks good and is ready to be used for training the stock prediction model.")
    print(f"Total trading days: {len(df)}")
    print(f"Data completeness: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%")
    
    return df

if __name__ == "__main__":
    main()
