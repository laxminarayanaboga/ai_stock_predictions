"""
Advanced Analytics Module for Trading Strategy Analysis
Provides comprehensive risk metrics, statistical analysis, and performance evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from scipy import stats


@dataclass
class RiskMetrics:
    """Container for risk-adjusted performance metrics"""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float  # Value at Risk 95%
    expected_shortfall_95: float  # Conditional VaR
    volatility_annualized: float
    downside_deviation: float
    profit_factor: float
    recovery_factor: float
    ulcer_index: float


@dataclass
class TradeDistributionMetrics:
    """Detailed trade distribution analysis"""
    avg_win_size: float
    avg_loss_size: float
    largest_win: float
    largest_loss: float
    win_loss_ratio: float
    consecutive_wins_max: int
    consecutive_losses_max: int
    avg_trade_duration: float
    median_trade_duration: float
    trade_count_by_hour: Dict[int, int]
    trade_count_by_day: Dict[str, int]
    profit_distribution: Dict[str, float]  # percentiles


@dataclass
class TemporalAnalysis:
    """Time-based performance analysis"""
    monthly_returns: Dict[str, float]
    daily_returns: Dict[str, float]
    rolling_sharpe_30d: List[float]
    rolling_max_drawdown_30d: List[float]
    performance_by_month: Dict[str, Dict[str, float]]
    performance_by_weekday: Dict[str, Dict[str, float]]
    best_month: Tuple[str, float]
    worst_month: Tuple[str, float]


@dataclass
class StatisticalSignificance:
    """Statistical tests for strategy validation"""
    t_test_pvalue: float
    t_test_statistic: float
    normality_test_pvalue: float
    runs_test_pvalue: float  # Test for randomness
    autocorrelation_lag1: float
    information_ratio: float
    batting_average: float  # % of periods with positive returns


class AdvancedAnalytics:
    """Advanced analytics engine for trading strategy evaluation"""
    
    def __init__(self, risk_free_rate: float = 0.06):
        """
        Initialize analytics engine
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252
    
    def analyze_strategy(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive analysis of strategy performance
        
        Args:
            trades: List of trade records
            
        Returns:
            Dictionary containing all analysis results
        """
        if not trades:
            return self._empty_analysis()
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)
        df['date'] = pd.to_datetime(df['date'])
        df['pnl_cumulative'] = df['pnl'].cumsum()
        
        # Calculate all metrics
        risk_metrics = self._calculate_risk_metrics(df)
        distribution_metrics = self._calculate_distribution_metrics(df)
        temporal_analysis = self._calculate_temporal_analysis(df)
        statistical_significance = self._calculate_statistical_significance(df)
        
        return {
            'risk_metrics': risk_metrics.__dict__,
            'distribution_metrics': distribution_metrics.__dict__,
            'temporal_analysis': temporal_analysis.__dict__,
            'statistical_significance': statistical_significance.__dict__,
            'summary': self._generate_summary(risk_metrics, distribution_metrics, 
                                            temporal_analysis, statistical_significance)
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> RiskMetrics:
        """Calculate risk-adjusted performance metrics"""
        if df.empty:
            return self._empty_risk_metrics()
        
        returns = df['pnl'].values
        cumulative_returns = df['pnl_cumulative'].values
        
        # Basic statistics
        total_return = cumulative_returns[-1]
        n_trades = len(returns)
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if n_trades > 1 else 0
        
        # Annualized metrics (assume daily trading)
        annual_return = avg_return * self.trading_days_per_year
        annual_volatility = std_return * np.sqrt(self.trading_days_per_year) if std_return > 0 else 0
        
        # Sharpe Ratio
        excess_return = annual_return - self.risk_free_rate
        sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino Ratio (uses downside deviation)
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns, ddof=1) if len(negative_returns) > 1 else 0
        downside_deviation_annual = downside_deviation * np.sqrt(self.trading_days_per_year)
        sortino_ratio = excess_return / downside_deviation_annual if downside_deviation_annual > 0 else 0
        
        # Maximum Drawdown
        peak = cumulative_returns[0]
        max_drawdown = 0
        max_dd_duration = 0
        current_dd_duration = 0
        
        for i, value in enumerate(cumulative_returns):
            if value > peak:
                peak = value
                current_dd_duration = 0
            else:
                drawdown = (peak - value) / abs(peak) if peak != 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
        
        # Calmar Ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        # Expected Shortfall (Conditional VaR)
        returns_below_var = returns[returns <= var_95]
        expected_shortfall_95 = np.mean(returns_below_var) if len(returns_below_var) > 0 else 0
        
        # Profit Factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Recovery Factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Ulcer Index (alternative drawdown measure)
        drawdowns = []
        peak = cumulative_returns[0]
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown_pct = ((peak - value) / abs(peak)) * 100 if peak != 0 else 0
            drawdowns.append(drawdown_pct ** 2)
        ulcer_index = np.sqrt(np.mean(drawdowns)) if drawdowns else 0
        
        return RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown * 100,  # Convert to percentage
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            expected_shortfall_95=expected_shortfall_95,
            volatility_annualized=annual_volatility,
            downside_deviation=downside_deviation_annual,
            profit_factor=profit_factor,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index
        )
    
    def _calculate_distribution_metrics(self, df: pd.DataFrame) -> TradeDistributionMetrics:
        """Calculate trade distribution and pattern metrics"""
        if df.empty:
            return self._empty_distribution_metrics()
        
        returns = df['pnl'].values
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        # Win/Loss statistics
        avg_win_size = np.mean(wins) if len(wins) > 0 else 0
        avg_loss_size = np.mean(losses) if len(losses) > 0 else 0
        largest_win = np.max(wins) if len(wins) > 0 else 0
        largest_loss = np.min(losses) if len(losses) > 0 else 0
        win_loss_ratio = abs(avg_win_size / avg_loss_size) if avg_loss_size != 0 else float('inf')
        
        # Consecutive wins/losses
        consecutive_wins_max = self._max_consecutive(returns > 0)
        consecutive_losses_max = self._max_consecutive(returns < 0)
        
        # Trade duration analysis
        if 'duration_minutes' in df.columns:
            avg_duration = df['duration_minutes'].mean()
            median_duration = df['duration_minutes'].median()
        else:
            avg_duration = median_duration = 0
        
        # Trade timing analysis
        if 'entry_time' in df.columns:
            df['hour'] = pd.to_datetime(df['entry_time'], format='%H:%M:%S').dt.hour
            trade_count_by_hour = df['hour'].value_counts().to_dict()
        else:
            trade_count_by_hour = {}
        
        # Day of week analysis
        df['weekday'] = df['date'].dt.day_name()
        trade_count_by_day = df['weekday'].value_counts().to_dict()
        
        # Profit distribution (percentiles)
        profit_distribution = {
            'p10': np.percentile(returns, 10),
            'p25': np.percentile(returns, 25),
            'p50': np.percentile(returns, 50),
            'p75': np.percentile(returns, 75),
            'p90': np.percentile(returns, 90),
            'p95': np.percentile(returns, 95),
            'p99': np.percentile(returns, 99)
        }
        
        return TradeDistributionMetrics(
            avg_win_size=avg_win_size,
            avg_loss_size=avg_loss_size,
            largest_win=largest_win,
            largest_loss=largest_loss,
            win_loss_ratio=win_loss_ratio,
            consecutive_wins_max=consecutive_wins_max,
            consecutive_losses_max=consecutive_losses_max,
            avg_trade_duration=avg_duration,
            median_trade_duration=median_duration,
            trade_count_by_hour=trade_count_by_hour,
            trade_count_by_day=trade_count_by_day,
            profit_distribution=profit_distribution
        )
    
    def _calculate_temporal_analysis(self, df: pd.DataFrame) -> TemporalAnalysis:
        """Calculate time-based performance analysis"""
        if df.empty:
            return self._empty_temporal_analysis()
        
        try:
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Add weekday column for later use
            df['weekday'] = df['date'].dt.day_name()
            
            # Monthly returns
            df['month'] = df['date'].dt.to_period('M')
            monthly_returns = df.groupby('month')['pnl'].sum().to_dict()
            monthly_returns = {str(k): v for k, v in monthly_returns.items()}
            
            # Daily returns
            daily_returns = df.groupby(df['date'].dt.date)['pnl'].sum().to_dict()
            daily_returns = {str(k): v for k, v in daily_returns.items()}
            
            # Rolling metrics (30-day windows)
            df_daily = df.groupby(df['date'].dt.date)['pnl'].sum().reset_index()
            df_daily.columns = ['date', 'daily_pnl']
            
            rolling_sharpe_30d = []
            rolling_max_drawdown_30d = []
            
            if len(df_daily) >= 30:
                for i in range(29, len(df_daily)):
                    window_returns = df_daily['daily_pnl'].iloc[i-29:i+1].values
                    window_mean = np.mean(window_returns)
                    window_std = np.std(window_returns, ddof=1)
                    
                    # Rolling Sharpe
                    if window_std > 0:
                        rolling_sharpe = (window_mean * np.sqrt(252) - self.risk_free_rate) / (window_std * np.sqrt(252))
                        rolling_sharpe_30d.append(rolling_sharpe)
                    else:
                        rolling_sharpe_30d.append(0)
                    
                    # Rolling Max Drawdown
                    cumulative = np.cumsum(window_returns)
                    peak = cumulative[0]
                    max_dd = 0
                    for val in cumulative:
                        if val > peak:
                            peak = val
                        else:
                            dd = (peak - val) / abs(peak) if peak != 0 else 0
                            max_dd = max(max_dd, dd)
                    rolling_max_drawdown_30d.append(max_dd * 100)
            
            # Performance by month
            df['month_name'] = df['date'].dt.month_name()
            performance_by_month = {}
            for month in df['month_name'].unique():
                month_data = df[df['month_name'] == month]
                performance_by_month[month] = {
                    'total_pnl': month_data['pnl'].sum(),
                    'avg_pnl': month_data['pnl'].mean(),
                    'win_rate': (month_data['pnl'] > 0).mean() * 100,
                    'trade_count': len(month_data)
                }
            
            # Performance by weekday
            performance_by_weekday = {}
            for day in df['weekday'].unique():
                day_data = df[df['weekday'] == day]
                performance_by_weekday[day] = {
                    'total_pnl': day_data['pnl'].sum(),
                    'avg_pnl': day_data['pnl'].mean(),
                    'win_rate': (day_data['pnl'] > 0).mean() * 100,
                    'trade_count': len(day_data)
                }
            
            # Best and worst months
            if monthly_returns:
                best_month = max(monthly_returns.items(), key=lambda x: x[1])
                worst_month = min(monthly_returns.items(), key=lambda x: x[1])
            else:
                best_month = worst_month = ("N/A", 0)
            
            return TemporalAnalysis(
                monthly_returns=monthly_returns,
                daily_returns=daily_returns,
                rolling_sharpe_30d=rolling_sharpe_30d,
                rolling_max_drawdown_30d=rolling_max_drawdown_30d,
                performance_by_month=performance_by_month,
                performance_by_weekday=performance_by_weekday,
                best_month=best_month,
                worst_month=worst_month
            )
            
        except Exception as e:
            print(f"ERROR in temporal analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._empty_temporal_analysis()
    
    def _calculate_statistical_significance(self, df: pd.DataFrame) -> StatisticalSignificance:
        """Calculate statistical significance tests"""
        if df.empty or len(df) < 2:
            return self._empty_statistical_significance()
        
        returns = df['pnl'].values
        
        # T-test against zero (are returns significantly different from zero?)
        t_statistic, t_test_pvalue = stats.ttest_1samp(returns, 0)
        
        # Normality test
        if len(returns) >= 8:  # Minimum for Shapiro-Wilk
            _, normality_test_pvalue = stats.shapiro(returns)
        else:
            normality_test_pvalue = 1.0
        
        # Runs test for randomness
        runs_test_pvalue = self._runs_test(returns > np.median(returns))
        
        # Autocorrelation (lag 1)
        if len(returns) > 1:
            autocorr_lag1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr_lag1):
                autocorr_lag1 = 0
        else:
            autocorr_lag1 = 0
        
        # Information Ratio (return/volatility of excess returns)
        excess_returns = returns - np.mean(returns)
        if np.std(excess_returns, ddof=1) > 0:
            information_ratio = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
        else:
            information_ratio = 0
        
        # Batting Average (% of positive periods)
        batting_average = np.mean(returns > 0) * 100
        
        return StatisticalSignificance(
            t_test_pvalue=t_test_pvalue,
            t_test_statistic=t_statistic,
            normality_test_pvalue=normality_test_pvalue,
            runs_test_pvalue=runs_test_pvalue,
            autocorrelation_lag1=autocorr_lag1,
            information_ratio=information_ratio,
            batting_average=batting_average
        )
    
    def _max_consecutive(self, condition_array: np.ndarray) -> int:
        """Calculate maximum consecutive True values"""
        if len(condition_array) == 0:
            return 0
        
        max_count = 0
        current_count = 0
        
        for condition in condition_array:
            if condition:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
                
        return max_count
    
    def _runs_test(self, binary_sequence: np.ndarray) -> float:
        """
        Runs test for randomness
        Tests if a binary sequence is random
        """
        if len(binary_sequence) < 2:
            return 1.0
        
        # Count runs
        runs = 1
        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] != binary_sequence[i-1]:
                runs += 1
        
        # Count 0s and 1s
        n1 = np.sum(binary_sequence)
        n0 = len(binary_sequence) - n1
        
        if n1 == 0 or n0 == 0:
            return 1.0
        
        # Expected runs and variance
        expected_runs = (2 * n0 * n1) / (n0 + n1) + 1
        variance_runs = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / ((n0 + n1) ** 2 * (n0 + n1 - 1))
        
        if variance_runs <= 0:
            return 1.0
        
        # Z-score and p-value
        z_score = (runs - expected_runs) / np.sqrt(variance_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return p_value
    
    def _generate_summary(self, risk_metrics: RiskMetrics, 
                         distribution_metrics: TradeDistributionMetrics,
                         temporal_analysis: TemporalAnalysis,
                         statistical_significance: StatisticalSignificance) -> Dict[str, Any]:
        """Generate summary insights"""
        
        # Risk assessment
        risk_level = "High"
        if risk_metrics.sharpe_ratio > 1.5:
            risk_level = "Low"
        elif risk_metrics.sharpe_ratio > 0.5:
            risk_level = "Medium"
        
        # Strategy quality assessment
        quality_score = 0
        if risk_metrics.sharpe_ratio > 1.0:
            quality_score += 2
        elif risk_metrics.sharpe_ratio > 0.5:
            quality_score += 1
            
        if risk_metrics.max_drawdown < 10:
            quality_score += 2
        elif risk_metrics.max_drawdown < 20:
            quality_score += 1
            
        if distribution_metrics.win_loss_ratio > 1.5:
            quality_score += 2
        elif distribution_metrics.win_loss_ratio > 1.0:
            quality_score += 1
            
        if statistical_significance.t_test_pvalue < 0.05:
            quality_score += 1
        
        quality_rating = "Excellent" if quality_score >= 6 else \
                        "Good" if quality_score >= 4 else \
                        "Fair" if quality_score >= 2 else "Poor"
        
        return {
            'risk_level': risk_level,
            'quality_rating': quality_rating,
            'quality_score': quality_score,
            'key_strengths': self._identify_strengths(risk_metrics, distribution_metrics, statistical_significance),
            'key_weaknesses': self._identify_weaknesses(risk_metrics, distribution_metrics, statistical_significance),
            'statistical_confidence': 'High' if statistical_significance.t_test_pvalue < 0.01 else
                                    'Medium' if statistical_significance.t_test_pvalue < 0.05 else 'Low'
        }
    
    def _identify_strengths(self, risk_metrics: RiskMetrics, 
                           distribution_metrics: TradeDistributionMetrics,
                           statistical_significance: StatisticalSignificance) -> List[str]:
        """Identify strategy strengths"""
        strengths = []
        
        if risk_metrics.sharpe_ratio > 1.5:
            strengths.append("Excellent risk-adjusted returns")
        elif risk_metrics.sharpe_ratio > 1.0:
            strengths.append("Good risk-adjusted returns")
            
        if risk_metrics.max_drawdown < 5:
            strengths.append("Very low drawdown")
        elif risk_metrics.max_drawdown < 10:
            strengths.append("Low drawdown")
            
        if distribution_metrics.win_loss_ratio > 2.0:
            strengths.append("Excellent win/loss ratio")
        elif distribution_metrics.win_loss_ratio > 1.5:
            strengths.append("Good win/loss ratio")
            
        if risk_metrics.profit_factor > 2.0:
            strengths.append("High profit factor")
            
        if statistical_significance.t_test_pvalue < 0.01:
            strengths.append("Statistically significant results")
            
        return strengths
    
    def _identify_weaknesses(self, risk_metrics: RiskMetrics, 
                            distribution_metrics: TradeDistributionMetrics,
                            statistical_significance: StatisticalSignificance) -> List[str]:
        """Identify strategy weaknesses"""
        weaknesses = []
        
        if risk_metrics.sharpe_ratio < 0:
            weaknesses.append("Negative risk-adjusted returns")
        elif risk_metrics.sharpe_ratio < 0.5:
            weaknesses.append("Poor risk-adjusted returns")
            
        if risk_metrics.max_drawdown > 20:
            weaknesses.append("High drawdown risk")
        elif risk_metrics.max_drawdown > 10:
            weaknesses.append("Moderate drawdown risk")
            
        if distribution_metrics.win_loss_ratio < 1.0:
            weaknesses.append("Average losses exceed average wins")
            
        if distribution_metrics.consecutive_losses_max > 10:
            weaknesses.append("Long losing streaks")
            
        if statistical_significance.t_test_pvalue > 0.1:
            weaknesses.append("Results may not be statistically significant")
            
        if abs(statistical_significance.autocorrelation_lag1) > 0.3:
            weaknesses.append("Returns show autocorrelation (non-random)")
            
        return weaknesses
    
    # Empty result methods for edge cases
    def _empty_analysis(self) -> Dict[str, Any]:
        return {
            'risk_metrics': self._empty_risk_metrics().__dict__,
            'distribution_metrics': self._empty_distribution_metrics().__dict__,
            'temporal_analysis': self._empty_temporal_analysis().__dict__,
            'statistical_significance': self._empty_statistical_significance().__dict__,
            'summary': {'quality_rating': 'No Data', 'risk_level': 'Unknown'}
        }
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _empty_distribution_metrics(self) -> TradeDistributionMetrics:
        return TradeDistributionMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {})
    
    def _empty_temporal_analysis(self) -> TemporalAnalysis:
        return TemporalAnalysis({}, {}, [], [], {}, {}, ("N/A", 0), ("N/A", 0))
    
    def _empty_statistical_significance(self) -> StatisticalSignificance:
        return StatisticalSignificance(1.0, 0, 1.0, 1.0, 0, 0, 0)