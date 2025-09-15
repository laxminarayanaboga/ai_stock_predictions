import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import time as dtime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.simulator.intraday_core import IntradaySimulator, TradeEntry


def load_5s_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['datetime_ist'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
    df['time_ist'] = df['datetime_ist'].dt.time
    df['date_str'] = df['datetime_ist'].dt.strftime('%Y-%m-%d')
    return df


def compute_vwap_rolling(df5: pd.DataFrame) -> pd.Series:
    pv = df5['close'] * df5['volume']
    cum_pv = pv.cumsum()
    cum_vol = df5['volume'].cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_vol
    return vwap.bfill()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--outdir', default='src/scalping/results')
    parser.add_argument('--qty', type=int, default=100)
    # micro-scalper params
    parser.add_argument('--band-pct', type=float, default=0.0012, help='VWAP deviation threshold (e.g., 0.0012 = 0.12%)')
    parser.add_argument('--tp-pct', type=float, default=0.0008, help='Take-profit percentage (used in tp_sl mode)')
    parser.add_argument('--sl-pct', type=float, default=0.0020, help='Stop-loss percentage')
    parser.add_argument('--cooldown-sec', type=int, default=90, help='Cooldown seconds after exit before next entry')
    parser.add_argument('--max-per-day', type=int, default=20, help='Max trades per day')
    parser.add_argument('--start', default='09:20:00')
    parser.add_argument('--end', default='15:10:00')
    parser.add_argument('--entry-mode', choices=['touch','reversion'], default='reversion', help='Signal mode')
    parser.add_argument('--exit-mode', choices=['tp_sl','vwap_close'], default='vwap_close', help='Exit logic')
    parser.add_argument('--reversion-eps', type=float, default=0.00005, help='Minimum reversion towards VWAP to trigger entry in reversion mode')
    # regime controls (optional)
    parser.add_argument('--pause-start', default=None, help='Optional pause start time (e.g., 12:00:00) to skip trading')
    parser.add_argument('--pause-end', default=None, help='Optional pause end time (e.g., 13:15:00) to resume trading')
    parser.add_argument('--trend-filter', choices=['none','flat'], default='none', help='Optional trend regime filter')
    parser.add_argument('--ema-secs', type=int, default=300, help='EMA window in seconds for trend filter (default 5 minutes)')
    parser.add_argument('--slope-threshold', type=float, default=0.0002, help='Max abs EMA slope per minute (as pct) for flat regime')
    # adaptive band (optional)
    parser.add_argument('--adaptive-band', action='store_true', help='Use adaptive VWAP band based on rolling dev std')
    parser.add_argument('--band-k', type=float, default=2.0, help='Multiplier for rolling stdev when adaptive band is enabled')
    parser.add_argument('--band-window-secs', type=int, default=600, help='Rolling window in seconds for band stdev (default 10 minutes)')
    parser.add_argument('--min-band-pct', type=float, default=0.0006, help='Minimum band pct when adaptive band is enabled')
    args = parser.parse_args()

    df5 = load_5s_csv(args.csv)
    out_root = Path(args.outdir) / args.symbol.replace(':','_') / 'micro_scalper_5s'
    out_root.mkdir(parents=True, exist_ok=True)

    simulator = IntradaySimulator()
    start_t = dtime.fromisoformat(args.start)
    end_t = dtime.fromisoformat(args.end)

    summary_rows = []

    for d, day in df5.groupby('date_str'):
        day = day[(day['time_ist'] >= start_t) & (day['time_ist'] <= end_t)].copy()
        if day.empty:
            continue
        # rolling VWAP (since day starts filtered, compute from first bar)
        day['vwap'] = compute_vwap_rolling(day)
        # deviation from VWAP
        day['dev'] = np.where(day['vwap'] > 0, (day['close'] / day['vwap']) - 1.0, 0.0)

        # optional lunch pause window
        if args.pause_start and args.pause_end:
            ps = dtime.fromisoformat(args.pause_start)
            pe = dtime.fromisoformat(args.pause_end)
            day['in_pause'] = (day['time_ist'] >= ps) & (day['time_ist'] <= pe)
        else:
            day['in_pause'] = False

        # optional trend filter via EMA slope on 5s bars (approx per-minute slope)
        if args.trend_filter != 'none':
            ema_span = max(1, int(args.ema_secs // 5))
            day['ema'] = day['close'].ewm(span=ema_span, adjust=False).mean()
            # compute slope over 12 bars (~1 minute)
            lag = 12
            day['ema_shift'] = day['ema'].shift(lag)
            # per-minute pct slope
            day['ema_slope_pm'] = (day['ema'] - day['ema_shift']) / day['ema_shift']
            if args.trend_filter == 'flat':
                day['regime_ok'] = day['ema_slope_pm'].abs() <= args.slope_threshold
            else:
                day['regime_ok'] = True
        else:
            day['regime_ok'] = True

        # optional adaptive band
        if args.adaptive_band:
            win = max(1, int(args.band_window_secs // 5))
            dev_std = day['dev'].rolling(window=win, min_periods=win//2).std().bfill()
            day['band_dyn'] = np.maximum(args.min_band_pct, args.band_k * dev_std)
        else:
            day['band_dyn'] = args.band_pct
        trades = []
        results = []
        last_exit_time = None
        count = 0

        prev_dev = None
        prev_band = None
        for idx, row in day.iterrows():
            t = row['time_ist']
            if row['in_pause']:
                prev_dev = row['dev']
                prev_band = row['band_dyn']
                continue
            if not row['regime_ok']:
                prev_dev = row['dev']
                prev_band = row['band_dyn']
                continue
            if last_exit_time is not None and (pd.Timestamp.combine(pd.to_datetime('today').date(), t) - pd.Timestamp.combine(pd.to_datetime('today').date(), last_exit_time)).seconds < args.cooldown_sec:
                continue
            price = float(row['close'])
            vw = float(row['vwap'])
            dev = float(row['dev'])
            band_here = float(row['band_dyn'])
            direction = None
            if args.entry_mode == 'touch':
                if dev <= -band_here:
                    direction = 'LONG'
                elif dev >= band_here:
                    direction = 'SHORT'
            else:  # reversion mode: require previous bar beyond band and current dev magnitude decreasing by eps
                if prev_dev is not None and prev_band is not None:
                    if prev_dev <= -prev_band and (dev - prev_dev) > args.reversion_eps:
                        direction = 'LONG'
                    elif prev_dev >= prev_band and (prev_dev - dev) > args.reversion_eps:
                        direction = 'SHORT'
            if direction is None:
                prev_dev = dev
                prev_band = band_here
                continue

            if args.exit_mode == 'tp_sl':
                te = TradeEntry(entry_time=t, entry_price=price, direction=direction,
                                stop_loss_pct=args.sl_pct, take_profit_pct=args.tp_pct, position_size=args.qty)
                tr = simulator.simulate_trade(te, day[['timestamp','open','high','low','close','volume','datetime_ist','time_ist','date_str']])
                if tr:
                    results.append(tr)
                    last_exit_time = tr.exit_time
                    count += 1
            else:  # vwap_close: exit when close crosses back to VWAP (or SL on close)
                entry_price = price
                entry_time = t
                dir_mult = 1 if direction == 'LONG' else -1
                sl_price = entry_price * (1 - args.sl_pct) if direction == 'LONG' else entry_price * (1 + args.sl_pct)
                # iterate forward
                after = day[day['time_ist'] > t]
                exit_price = None
                exit_time = None
                exit_reason = None
                for _, r in after.iterrows():
                    rc = float(r['close'])
                    rvw = float(r['vwap'])
                    rt = r['time_ist']
                    # SL on close
                    if direction == 'LONG' and rc <= sl_price:
                        exit_price, exit_time, exit_reason = sl_price, rt, 'STOP_LOSS'
                        break
                    if direction == 'SHORT' and rc >= sl_price:
                        exit_price, exit_time, exit_reason = sl_price, rt, 'STOP_LOSS'
                        break
                    # vwap cross on close
                    if direction == 'LONG' and rc >= rvw:
                        exit_price, exit_time, exit_reason = rc, rt, 'VWAP_RETOUCH'
                        break
                    if direction == 'SHORT' and rc <= rvw:
                        exit_price, exit_time, exit_reason = rc, rt, 'VWAP_RETOUCH'
                        break
                if exit_price is None:
                    # end of window
                    last = day.iloc[-1]
                    exit_price = float(last['close'])
                    exit_time = last['time_ist']
                    exit_reason = 'END_OF_WINDOW'

                # PnL with charges
                from src.simulator.pnl_calculator import get_npl
                if direction == 'LONG':
                    pnl = get_npl(entry_price, exit_price, args.qty)
                else:
                    pnl = get_npl(exit_price, entry_price, args.qty)
                duration = ((pd.Timestamp.combine(pd.to_datetime('today').date(), exit_time) - pd.Timestamp.combine(pd.to_datetime('today').date(), entry_time)).seconds) // 60
                results.append(type('R', (), {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'direction': direction,
                    'position_size': args.qty,
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'duration_minutes': duration,
                }))
                last_exit_time = exit_time
                count += 1
            if count >= args.max_per_day:
                break
            prev_dev = dev
            prev_band = band_here

        total_trades = len(results)
        wins = [r for r in results if r.pnl > 0]
        losses = [r for r in results if r.pnl < 0]
        gross_profit = sum(r.pnl for r in wins)
        gross_loss = -sum(r.pnl for r in losses)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
        total_pnl = gross_profit - gross_loss

        # write per-day trades
        day_dir = out_root / d
        day_dir.mkdir(parents=True, exist_ok=True)
        if results:
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
                } for r in results
            ])
            tdf['cum_pnl'] = tdf['pnl'].cumsum()
            tdf.to_csv(day_dir / 'trades.csv', index=False)
            tdf[['date','pnl','cum_pnl']].to_csv(day_dir / 'equity_curve.csv', index=False)

        summary_rows.append({
            'date': d,
            'total_trades': total_trades,
            'win_rate_pct': round((len(wins) / total_trades * 100.0) if total_trades else 0.0, 2),
            'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 'inf',
            'total_pnl': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
        })

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        out_csv = out_root / 'summary.csv'
        sdf.to_csv(out_csv, index=False)
        print(f"Saved micro-scalper summary: {out_csv} ({len(sdf)} days)")
    else:
        print("No trades generated by micro-scalper.")


if __name__ == '__main__':
    main()
