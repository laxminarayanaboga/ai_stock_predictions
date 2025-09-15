import sys
from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import time as dtime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.scalping.strategies.common_strategies import REGISTRY
from src.scalping.strategies.base import Strategy
from src.simulator.intraday_core import IntradaySimulator, TradeEntry


def load_5s_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['datetime_ist'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
    df['time_ist'] = df['datetime_ist'].dt.time
    df['date_str'] = df['datetime_ist'].dt.strftime('%Y-%m-%d')
    return df


def aggregate_1m(df5: pd.DataFrame) -> pd.DataFrame:
    df5 = df5.copy()
    df5['dt'] = pd.to_datetime(df5['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
    ohlc = df5.set_index('dt').resample('1min', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    ohlc['timestamp'] = ohlc['dt'].dt.tz_convert('UTC').astype('int64') // 10**9
    ohlc['datetime_ist'] = ohlc['dt']
    ohlc['time_ist'] = ohlc['datetime_ist'].dt.time
    ohlc['date_str'] = ohlc['datetime_ist'].dt.strftime('%Y-%m-%d')
    return ohlc[['timestamp','open','high','low','close','volume','datetime_ist','time_ist','date_str']]


def to_trade_entry(order, sl_pct_default: float, tp_pct_default: float) -> TradeEntry:
    t = order.time.time()
    sl = order.sl_pct if order.sl_pct is not None else sl_pct_default
    tp = order.tp_pct if order.tp_pct is not None else tp_pct_default
    return TradeEntry(entry_time=t, entry_price=order.price, direction=order.side,
                      stop_loss_pct=sl, take_profit_pct=tp, position_size=order.qty)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--outroot', default='src/scalping/results')
    parser.add_argument('--qty', type=int, default=100)
    parser.add_argument('--max-per-day', type=int, default=5, help='Max trades per day per strategy variant')
    args = parser.parse_args()

    df5 = load_5s_csv(args.csv)
    df1 = aggregate_1m(df5)
    windows = [(dtime(9,20), dtime(10,30)), (dtime(15,0), dtime(15,20))]

    grids = {
        'orb': [
            {'orb_start': dtime(9,15), 'orb_end': dtime(9,30), 'windows': windows, 'min_range_pct': m, 'break_buffer_pct': bbuf,
             'sl_pct': 0.003, 'tp_pct': 0.006, 'qty': args.qty, 'require_next_confirm': conf, 'min_volume_percentile': v}
            for m in [0.0020, 0.0025, 0.0035]
            for bbuf in [0.0003, 0.0005]
            for conf in [True, False]
            for v in [0.4, 0.6]
        ],  # 3*2*2*2 = 24
        'ema_pullback': [
            {'windows': windows, 'ema_window': ew, 'slope_lookback': 6, 'min_slope_pct': s, 'sl_pct': 0.003, 'tp_pct': 0.006, 'qty': args.qty}
            for ew in [9, 13, 20]
            for s in [0.0003, 0.0006]
        ],  # 3*2 = 6
        'vwap_reversion': [
            {'windows': [(dtime(12,0), dtime(14,30))], 'band_pct': b, 'sl_mult': sm, 'tp_pct': 0.004, 'qty': args.qty}
            for b in [0.0015, 0.0020, 0.0025]
            for sm in [1.5, 2.0]
        ],  # 3*2 = 6
        'rsi_reversal': [
            {'windows': windows, 'period': lb, 'oversold': os, 'overbought': ob, 'sl_pct': 0.003, 'tp_pct': 0.006, 'qty': args.qty}
            for lb in [14, 21]
            for (os, ob) in [(25, 75), (30, 70)]
        ],  # 2*2 = 4
        'macd_cross': [
            {'windows': windows, 'sl_pct': 0.003, 'tp_pct': 0.006, 'qty': args.qty}
        ],  # 1
        'boll_reversion': [
            {'windows': windows, 'period': 20, 'mult': mult, 'sl_pct': 0.003, 'tp_pct': 0.006, 'qty': args.qty}
            for mult in [2.0, 2.5]
        ],  # 2
    }

    out_root = Path(args.outroot) / args.symbol.replace(':','_')
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    simulator = IntradaySimulator()

    for strat_name, param_list in grids.items():
        StratCls = REGISTRY[strat_name]
        for params in param_list:
            strat: Strategy = StratCls(strat_name, **params)
            # Keep (date_str, TradeResult) pairs to preserve exact day mapping
            all_trades = []
            # Parameter ID for folder naming
            params_str = json.dumps(params, default=str, sort_keys=True)
            param_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            variant_dir = out_root / strat_name / param_hash
            variant_dir.mkdir(parents=True, exist_ok=True)
            for d, day1 in df1.groupby('date_str'):
                day5 = df5[df5['date_str'] == d]
                if day1.empty or day5.empty:
                    continue
                sres = strat.generate(day1)
                # Consider multiple orders per day; simulate sequentially without overlap
                # Sort orders by time to ensure chronological processing
                orders = sorted(sres.orders, key=lambda o: o.time)
                trades_today = 0
                last_exit_time = None
                day5_exec = day5[['timestamp','open','high','low','close','volume','datetime_ist','time_ist','date_str']]
                for o in orders:
                    if trades_today >= args.max_per_day:
                        break
                    te = to_trade_entry(o, params.get('sl_pct', 0.003), params.get('tp_pct', 0.006))
                    # Skip if overlapping with previous exited trade
                    if last_exit_time is not None and te.entry_time <= last_exit_time:
                        continue
                    tr = simulator.simulate_trade(te, day5_exec)
                    if tr:
                        all_trades.append((d, tr))
                        trades_today += 1
                        last_exit_time = tr.exit_time

            total_trades = len(all_trades)
            results_only = [r for (_, r) in all_trades]
            wins = [r for r in results_only if r.pnl > 0]
            losses = [r for r in results_only if r.pnl < 0]
            gross_profit = sum(r.pnl for r in wins)
            gross_loss = -sum(r.pnl for r in losses)
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            total_pnl = gross_profit - gross_loss
            win_rate = (len(wins) / total_trades * 100.0) if total_trades else 0.0

            # Advanced analytics
            pnls = np.array([r.pnl for r in results_only], dtype=float)
            avg_trade_pnl = float(np.mean(pnls)) if total_trades else 0.0
            pnl_std = float(np.std(pnls, ddof=1)) if total_trades > 1 else 0.0
            sharpe = float(avg_trade_pnl / pnl_std) if pnl_std > 0 else 0.0  # per-trade Sharpe (not annualized)
            # Equity curve and max drawdown
            if total_trades:
                equity = np.cumsum(pnls)
                running_max = np.maximum.accumulate(equity)
                drawdowns = running_max - equity
                max_drawdown = float(np.max(drawdowns)) if drawdowns.size else 0.0
            else:
                max_drawdown = 0.0
            avg_win = float(np.mean([r.pnl for r in wins])) if wins else 0.0
            avg_loss = float(np.mean([abs(r.pnl) for r in losses])) if losses else 0.0
            rr_ratio = float(avg_win / avg_loss) if avg_loss > 0 else float('inf') if avg_win > 0 else 0.0
            expectancy = float(avg_trade_pnl)
            median_trade_pnl = float(np.median(pnls)) if total_trades else 0.0
            largest_win = float(np.max(pnls)) if total_trades else 0.0
            largest_loss = float(np.min(pnls)) if total_trades else 0.0

            # Write per-variant artifacts
            if total_trades:
                tdf = pd.DataFrame([
                    {
                        'date': d,
                        'entry_time': r.entry_time.strftime('%H:%M:%S'),
                        'exit_time': r.exit_time.strftime('%H:%M:%S'),
                        'direction': r.direction,
                        'entry_price': r.entry_price,
                        'exit_price': r.exit_price,
                        'qty': r.position_size,
                        'pnl': r.pnl,
                        'exit_reason': r.exit_reason,
                        'duration_min': r.duration_minutes,
                    }
                    for (d, r) in all_trades
                ])
                tdf.to_csv(variant_dir / 'trades.csv', index=False)
                tdf['cum_pnl'] = tdf['pnl'].cumsum()
                tdf[['date','pnl','cum_pnl']].to_csv(variant_dir / 'equity_curve.csv', index=False)
            with open(variant_dir / 'metrics.json', 'w') as f:
                json.dump({
                    'strategy': strat_name,
                    'params': params,
                    'param_hash': param_hash,
                    'total_trades': total_trades,
                    'win_rate_pct': round(win_rate, 2),
                    'profit_factor': float('inf') if profit_factor == float('inf') else round(profit_factor, 3),
                    'total_pnl': round(total_pnl, 2),
                    'gross_profit': round(gross_profit, 2),
                    'gross_loss': round(gross_loss, 2),
                    'avg_trade_pnl': round(avg_trade_pnl, 2),
                    'median_trade_pnl': round(median_trade_pnl, 2),
                    'pnl_std': round(pnl_std, 2),
                    'sharpe_per_trade': round(sharpe, 3),
                    'max_drawdown': round(max_drawdown, 2),
                    'avg_win': round(avg_win, 2),
                    'avg_loss': round(avg_loss, 2),
                    'rr_ratio': float('inf') if rr_ratio == float('inf') else round(rr_ratio, 3),
                    'expectancy': round(expectancy, 2),
                    'largest_win': round(largest_win, 2),
                    'largest_loss': round(largest_loss, 2),
                }, f, indent=2, default=str)

            summary_rows.append({
                'strategy': strat_name,
                'params': json.dumps(params, default=str),
                'param_hash': param_hash,
                'output_dir': str(variant_dir),
                'total_trades': total_trades,
                'win_rate_pct': round(win_rate, 2),
                'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 'inf',
                'total_pnl': round(total_pnl, 2),
                'gross_profit': round(gross_profit, 2),
                'gross_loss': round(gross_loss, 2),
                'expectancy': round(expectancy, 2),
                'sharpe_per_trade': round(sharpe, 3),
                'max_drawdown': round(max_drawdown, 2),
            })

    summary_df = pd.DataFrame(summary_rows)
    out_csv = out_root / 'summary.csv'
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv} ({len(summary_df)} rows)")


if __name__ == '__main__':
    main()
