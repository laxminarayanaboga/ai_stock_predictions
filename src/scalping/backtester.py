from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd
import numpy as np

from src.simulator.intraday_core import IntradaySimulator, TradeEntry


@dataclass
class ScalpingConfig:
    symbol: str
    sl_pct: float = 0.003
    tp_pct: float = 0.006
    qty: int = 100
    orb_window_min: int = 5
    ema_window: int = 20
    session_start: time = time(9, 15)
    session_end: time = time(15, 15)
    trade_windows: Optional[list] = None
    orb_min_range_pct: float = 0.002
    orb_break_buffer_pct: float = 0.0003
    ema_min_slope_pct: float = 0.0003
    ema_slope_lookback: int = 5
    max_trades_per_day: int = 1
    enable_vwap_band: bool = True
    vwap_band_pct: float = 0.0015
    vwap_stop_mult: float = 1.5
    atr_period: int = 14
    atr_k_sl: Optional[float] = None
    atr_k_tp: Optional[float] = None
    orb_require_next_bar_confirm: bool = True
    min_volume_percentile: float = 0.5
    execute_on_5s: bool = False


def _load_5s_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['datetime_ist'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
    df['time_ist'] = df['datetime_ist'].dt.time
    df['date_str'] = df['datetime_ist'].dt.strftime('%Y-%m-%d')
    return df


def _aggregate_to_1min(df5s: pd.DataFrame) -> pd.DataFrame:
    df = df5s.copy().set_index('datetime_ist')
    ohlc = df[['open','high','low','close','volume']].resample('1min').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna().reset_index()
    ohlc['time_ist'] = ohlc['datetime_ist'].dt.time
    ohlc['date_str'] = ohlc['datetime_ist'].dt.strftime('%Y-%m-%d')
    ohlc['timestamp'] = (ohlc['datetime_ist'].astype('int64') // 10**9).astype(int)
    return ohlc[['timestamp','open','high','low','close','volume','datetime_ist','time_ist','date_str']]


def _prepare_5s(df5s: pd.DataFrame) -> pd.DataFrame:
    df = df5s.copy()
    return df[['timestamp','open','high','low','close','volume','datetime_ist','time_ist','date_str']]


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    c = df['close']
    prev_c = c.shift(1)
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - prev_c).abs(),
        (df['low'] - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _generate_orb_trades(day_df_1m: pd.DataFrame, cfg: ScalpingConfig) -> List[TradeEntry]:
    start_time = cfg.session_start
    orb_end = (datetime.combine(datetime.utcnow().date(), start_time) + pd.Timedelta(minutes=cfg.orb_window_min)).time()
    day_df = day_df_1m[(day_df_1m['time_ist'] >= start_time) & (day_df_1m['time_ist'] <= cfg.session_end)]
    if day_df.empty:
        return []
    window = day_df[day_df['time_ist'] < orb_end]
    if len(window) < 2:
        return []
    orb_high = window['high'].max()
    orb_low = window['low'].min()
    ref_price = window.iloc[0]['close'] if not window.empty else day_df.iloc[0]['close']
    if (orb_high - orb_low) / max(1e-9, ref_price) < cfg.orb_min_range_pct:
        return []

    post = day_df[day_df['time_ist'] >= orb_end].copy()
    vol_thresh = post['volume'].quantile(cfg.min_volume_percentile) if len(post) else 0
    post['atr'] = _atr(day_df, cfg.atr_period).reindex(post.index)

    def _in_window(t: time) -> bool:
        if not cfg.trade_windows:
            return True
        return any(w[0] <= t <= w[1] for w in cfg.trade_windows)

    trades: List[TradeEntry] = []
    for pos, (idx, bar) in enumerate(post.iterrows()):
        if not _in_window(bar['time_ist']) or bar['volume'] < vol_thresh:
            continue
        if bar['close'] > orb_high * (1.0 + cfg.orb_break_buffer_pct):
            if cfg.orb_require_next_bar_confirm and (pos + 1) < len(post):
                nb = post.iloc[pos + 1]
                if nb['close'] <= orb_high * (1.0 + cfg.orb_break_buffer_pct):
                    continue
            sl_pct = cfg.sl_pct
            tp_pct = cfg.tp_pct
            if cfg.atr_k_sl and not pd.isna(bar['atr']):
                sl_pct = max(sl_pct, cfg.atr_k_sl * (bar['atr'] / max(1e-9, bar['close'])))
            if cfg.atr_k_tp and not pd.isna(bar['atr']):
                tp_pct = max(tp_pct, cfg.atr_k_tp * (bar['atr'] / max(1e-9, bar['close'])))
            trades.append(TradeEntry(bar['time_ist'], bar['close'], 'LONG', sl_pct, tp_pct, cfg.qty))
            break
    for pos, (idx, bar) in enumerate(post.iterrows()):
        if not _in_window(bar['time_ist']) or bar['volume'] < vol_thresh:
            continue
        if bar['close'] < orb_low * (1.0 - cfg.orb_break_buffer_pct):
            if cfg.orb_require_next_bar_confirm and (pos + 1) < len(post):
                nb = post.iloc[pos + 1]
                if nb['close'] >= orb_low * (1.0 - cfg.orb_break_buffer_pct):
                    continue
            sl_pct = cfg.sl_pct
            tp_pct = cfg.tp_pct
            if cfg.atr_k_sl and not pd.isna(bar['atr']):
                sl_pct = max(sl_pct, cfg.atr_k_sl * (bar['atr'] / max(1e-9, bar['close'])))
            if cfg.atr_k_tp and not pd.isna(bar['atr']):
                tp_pct = max(tp_pct, cfg.atr_k_tp * (bar['atr'] / max(1e-9, bar['close'])))
            trades.append(TradeEntry(bar['time_ist'], bar['close'], 'SHORT', sl_pct, tp_pct, cfg.qty))
            break
    return trades


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _generate_ema_pullback_trades(day_df_1m: pd.DataFrame, cfg: ScalpingConfig) -> List[TradeEntry]:
    df = day_df_1m[(day_df_1m['time_ist'] >= cfg.session_start) & (day_df_1m['time_ist'] <= cfg.session_end)].copy()
    if len(df) < cfg.ema_window + 5:
        return []
    df['ema'] = _ema(df['close'], cfg.ema_window)

    def _in_window(t: time) -> bool:
        if not cfg.trade_windows:
            return True
        return any(w[0] <= t <= w[1] for w in cfg.trade_windows)

    for i in range(cfg.ema_window + max(2, cfg.ema_slope_lookback), len(df)):
        prev = df.iloc[i-1]
        bar = df.iloc[i]
        if not _in_window(bar['time_ist']):
            continue
        ema_back = df.iloc[i - cfg.ema_slope_lookback]['ema']
        ref_price = df.iloc[i - cfg.ema_slope_lookback]['close']
        slope_pct = (prev['ema'] - ema_back) / max(1e-9, ref_price)
        if slope_pct > cfg.ema_min_slope_pct and prev['close'] > prev['ema'] and bar['low'] <= prev['ema'] <= bar['high'] and bar['close'] > prev['close']:
            return [TradeEntry(bar['time_ist'], bar['close'], 'LONG', cfg.sl_pct, cfg.tp_pct, cfg.qty)]
        if slope_pct < -cfg.ema_min_slope_pct and prev['close'] < prev['ema'] and bar['low'] <= prev['ema'] <= bar['high'] and bar['close'] < prev['close']:
            return [TradeEntry(bar['time_ist'], bar['close'], 'SHORT', cfg.sl_pct, cfg.tp_pct, cfg.qty)]
    return []


def _compute_vwap(day_df_1m: pd.DataFrame) -> pd.Series:
    pv = (day_df_1m['close'] * day_df_1m['volume']).cumsum()
    vv = day_df_1m['volume'].cumsum().replace(0, np.nan)
    return pv / vv


def _generate_vwap_band_trades(day_df_1m: pd.DataFrame, cfg: ScalpingConfig) -> List[TradeEntry]:
    if not cfg.enable_vwap_band:
        return []
    df = day_df_1m[(day_df_1m['time_ist'] >= cfg.session_start) & (day_df_1m['time_ist'] <= cfg.session_end)].copy()
    if df.empty:
        return []
    df['vwap'] = _compute_vwap(df)
    mid_start, mid_end = time(11, 0), time(14, 30)

    def _in_window(t: time) -> bool:
        in_mid = (mid_start <= t <= mid_end)
        if not in_mid:
            return False
        if not cfg.trade_windows:
            return True
        return any(w[0] <= t <= w[1] for w in cfg.trade_windows)

    band = cfg.vwap_band_pct
    sl_pct = band * cfg.vwap_stop_mult
    for i in range(1, len(df)):
        bar = df.iloc[i]
        if not _in_window(bar['time_ist']) or pd.isna(bar['vwap']) or bar['vwap'] <= 0:
            continue
        dev = (bar['close'] - bar['vwap']) / bar['vwap']
        if dev <= -band:
            return [TradeEntry(bar['time_ist'], bar['close'], 'LONG', max(sl_pct, cfg.sl_pct), max(cfg.tp_pct, band), cfg.qty)]
        if dev >= band:
            return [TradeEntry(bar['time_ist'], bar['close'], 'SHORT', max(sl_pct, cfg.sl_pct), max(cfg.tp_pct, band), cfg.qty)]
    return []


def run_scalping_backtest(csv_5s_path: str, cfg: ScalpingConfig, output_root: Optional[str] = None) -> Dict:
    df5s = _load_5s_csv(csv_5s_path)
    df1m = _aggregate_to_1min(df5s)
    dates = sorted(df1m['date_str'].unique())
    sim = IntradaySimulator(market_close_time=cfg.session_end)

    all_trades = []
    total_pnl = 0.0
    for d in dates:
        day_1m = df1m[df1m['date_str'] == d].copy()
        day_5s = df5s[df5s['date_str'] == d].copy()
        exec_df = _prepare_5s(day_5s) if cfg.execute_on_5s else day_1m
        if exec_df.empty or day_1m.empty:
            continue
        trades: List[TradeEntry] = []
        trades += _generate_orb_trades(day_1m, cfg)
        if len(trades) < cfg.max_trades_per_day:
            trades += _generate_ema_pullback_trades(day_1m, cfg)
        if len(trades) < cfg.max_trades_per_day:
            trades += _generate_vwap_band_trades(day_1m, cfg)
        trades = trades[: cfg.max_trades_per_day]
        for t in trades:
            res = sim.simulate_trade(t, exec_df)
            if not res:
                continue
            total_pnl += res.pnl
            all_trades.append({
                'date': d,
                'entry_time': res.entry_time.strftime('%H:%M:%S'),
                'exit_time': res.exit_time.strftime('%H:%M:%S'),
                'direction': res.direction,
                'entry_price': res.entry_price,
                'exit_price': res.exit_price,
                'qty': res.position_size,
                'pnl': res.pnl,
                'exit_reason': res.exit_reason,
                'duration_min': res.duration_minutes
            })

    wins = sum(1 for r in all_trades if r['pnl'] > 0)
    total_trades = len(all_trades)
    gross_profit = sum(r['pnl'] for r in all_trades if r['pnl'] > 0)
    gross_loss = -sum(r['pnl'] for r in all_trades if r['pnl'] <= 0)
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
    results = {
        'symbol': cfg.symbol,
        'total_trades': total_trades,
        'win_rate_pct': (wins / total_trades * 100.0) if total_trades else 0.0,
        'total_pnl': total_pnl,
        'profit_factor': pf,
        'sl_pct': cfg.sl_pct,
        'tp_pct': cfg.tp_pct,
        'qty': cfg.qty,
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(output_root or 'src/scalping/results') / cfg.symbol.replace(':','_') / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(out_dir / 'trades.csv', index=False)
        trades_df['cum_pnl'] = trades_df['pnl'].cumsum()
        trades_df[['date','pnl','cum_pnl']].to_csv(out_dir / 'equity_curve.csv', index=False)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


def main():
    import argparse
    from src.data.historical_data_cache_management import ensure_5s_csv
    parser = argparse.ArgumentParser(description='Run 5s scalping backtest')
    parser.add_argument('--symbol', default='NSE:RELIANCE-EQ')
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--qty', type=int, default=100)
    parser.add_argument('--sl', type=float, default=0.003)
    parser.add_argument('--tp', type=float, default=0.006)
    args = parser.parse_args()
    csv_path = ensure_5s_csv(args.symbol, args.start, args.end)
    cfg = ScalpingConfig(symbol=args.symbol, qty=args.qty, sl_pct=args.sl, tp_pct=args.tp)
    res = run_scalping_backtest(csv_path, cfg)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
