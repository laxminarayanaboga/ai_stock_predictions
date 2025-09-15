import sys
from pathlib import Path
from datetime import time as dtime
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.historical_data_cache_management import convert_cached_interval_to_csv
from src.scalping.backtester import run_scalping_backtest, ScalpingConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='NSE:RELIANCE-EQ')
    parser.add_argument('--root', default='cache_raw_data_all_intraday')
    parser.add_argument('--qty', type=int, default=100)
    parser.add_argument('--sl', type=float, default=0.003)
    parser.add_argument('--tp', type=float, default=0.006)
    args = parser.parse_args()

    out_csv = f"data/raw/{args.symbol.replace(':','_')}_5s_cache.csv"
    convert_cached_interval_to_csv(args.symbol, '5S', out_csv, market_filter=True)

    windows = [(dtime(9,20), dtime(10,30)), (dtime(15,0), dtime(15,20))]
    cfg = ScalpingConfig(
        symbol=args.symbol,
        qty=args.qty,
        sl_pct=args.sl,
        tp_pct=args.tp,
        trade_windows=windows,
        max_trades_per_day=1,
        orb_min_range_pct=0.003,
        orb_break_buffer_pct=0.0005,
        ema_min_slope_pct=0.0005,
        ema_slope_lookback=6,
        enable_vwap_band=True,
        vwap_band_pct=0.002,
        vwap_stop_mult=1.5,
        execute_on_5s=True,
        atr_period=14,
        atr_k_sl=0.8,
        atr_k_tp=1.6,
        orb_require_next_bar_confirm=True,
        min_volume_percentile=0.5
    )
    res = run_scalping_backtest(out_csv, cfg)
    print("\nSummary:\n", res)


if __name__ == '__main__':
    main()
