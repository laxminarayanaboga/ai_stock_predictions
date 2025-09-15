import sys
from pathlib import Path
import itertools
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.scalping.scripts.run_micro_scalper_5s import main as run_single


def run_grid(
    csv_path: str,
    symbol: str,
    outdir: str,
    qty: int,
    use_adaptive: bool = False,
    band_list = None,
    entry_list = None,
    exit_list = None,
    cooldown_list = None,
    max_per_day_list = None,
    band_k_list = None,
    band_window_secs: int = 600,
    min_band_pct: float = 0.0006,
    trend_filters = None,
    ema_secs: int = 300,
    slope_threshold: float = 0.0002,
    pause_start: str | None = None,
    pause_end: str | None = None,
):
    # Defaults
    bands = band_list or [0.0008, 0.0012, 0.0016]
    entries = entry_list or ['reversion', 'touch']
    exits = exit_list or ['vwap_close', 'tp_sl']
    cooldowns = cooldown_list or [60, 120]
    max_per_day = max_per_day_list or [20, 30]
    trend_filters = trend_filters or ['none', 'flat']

    if use_adaptive:
        band_k_list = band_k_list or [1.5, 2.0, 2.5]
        combos = list(itertools.product(band_k_list, entries, exits, cooldowns, max_per_day, trend_filters))
        print(f"Running micro-scalper adaptive grid with {len(combos)} combos...")
    else:
        combos = list(itertools.product(bands, entries, exits, cooldowns, max_per_day, trend_filters))
        print(f"Running micro-scalper grid with {len(combos)} combos...")

    summary_rows = []
    for combo in combos:
        if use_adaptive:
            (band_k, entry, exit_mode, cd, mpd, trend_f) = combo
        else:
            (band, entry, exit_mode, cd, mpd, trend_f) = combo
        # Build args to call the single-run main
        args = [
            '--csv', csv_path,
            '--symbol', symbol,
            '--outdir', outdir,
            '--qty', str(qty),
            '--entry-mode', entry,
            '--exit-mode', exit_mode,
            '--cooldown-sec', str(cd),
            '--max-per-day', str(mpd),
            '--trend-filter', trend_f,
            '--ema-secs', str(ema_secs),
            '--slope-threshold', str(slope_threshold),
        ]
        if pause_start and pause_end:
            args += ['--pause-start', pause_start, '--pause-end', pause_end]
        if use_adaptive:
            args += [
                '--adaptive-band',
                '--band-k', str(band_k),
                '--band-window-secs', str(band_window_secs),
                '--min-band-pct', str(min_band_pct),
            ]
        else:
            args += ['--band-pct', str(band)]
        # Slightly adapt TP/SL for tp_sl exit to be tighter when band smaller
        if exit_mode == 'tp_sl':
            # when adaptive, use min_band_pct heuristic
            th_band = min_band_pct if use_adaptive else band
            tp = 0.0005 if th_band <= 0.0012 else 0.0008
            sl = 0.0010 if th_band <= 0.0012 else 0.0016
            args += ['--tp-pct', str(tp), '--sl-pct', str(sl)]

        # Monkey-patch sys.argv for reuse of single main()
        old_argv = sys.argv
        try:
            sys.argv = ['run_micro_scalper_5s.py'] + args
            run_single()
        finally:
            sys.argv = old_argv

        # After run, read per-run summary
        run_dir = Path(outdir) / symbol.replace(':','_') / 'micro_scalper_5s'
        sum_csv = run_dir / 'summary.csv'
        if sum_csv.exists():
            sdf = pd.read_csv(sum_csv)
            # aggregate over the days
            total_trades = int(sdf['total_trades'].sum())
            gross_profit = float(sdf['gross_profit'].sum())
            gross_loss = float(sdf['gross_loss'].sum())
            total_pnl = float(sdf['total_pnl'].sum())
            pf = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
            row = {
                'entry_mode': entry,
                'exit_mode': exit_mode,
                'cooldown_sec': cd,
                'max_per_day': mpd,
                'trend_filter': trend_f,
                'ema_secs': ema_secs,
                'slope_threshold': slope_threshold,
                'pause_start': pause_start or '',
                'pause_end': pause_end or '',
                'total_trades': total_trades,
                'profit_factor': pf if pf != float('inf') else 'inf',
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
            }
            if use_adaptive:
                row.update({
                    'adaptive': True,
                    'band_k': band_k,
                    'band_window_secs': band_window_secs,
                    'min_band_pct': min_band_pct,
                })
            else:
                row.update({
                    'adaptive': False,
                    'band_pct': band,
                })
            summary_rows.append(row)

    if summary_rows:
        out = Path(outdir) / symbol.replace(':','_') / 'micro_scalper_5s_grid_summary.csv'
        df = pd.DataFrame(summary_rows)
        df.to_csv(out, index=False)
        print(f"Saved grid summary: {out} ({len(df)} combos)")
    else:
        print("No grid results captured.")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--symbol', required=True)
    p.add_argument('--outdir', default='src/scalping/results')
    p.add_argument('--qty', type=int, default=100)
    # optional advanced sweep controls
    p.add_argument('--use-adaptive', action='store_true', help='Enable adaptive band sweep instead of fixed bands')
    p.add_argument('--pause-start', default=None)
    p.add_argument('--pause-end', default=None)
    p.add_argument('--ema-secs', type=int, default=300)
    p.add_argument('--slope-threshold', type=float, default=0.0002)
    p.add_argument('--band-window-secs', type=int, default=600)
    p.add_argument('--min-band-pct', type=float, default=0.0006)
    args = p.parse_args()
    run_grid(
        csv_path=args.csv,
        symbol=args.symbol,
        outdir=args.outdir,
        qty=args.qty,
        use_adaptive=args.use_adaptive,
        pause_start=args.pause_start,
        pause_end=args.pause_end,
        ema_secs=args.ema_secs,
        slope_threshold=args.slope_threshold,
        band_window_secs=args.band_window_secs,
        min_band_pct=args.min_band_pct,
    )


if __name__ == '__main__':
    main()
