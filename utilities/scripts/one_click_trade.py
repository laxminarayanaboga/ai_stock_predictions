"""
One-click trade runner

What it does (safe by default: dry-run):
- Ensures Fyers access token is valid (refresh via refresh_token if needed)
- Updates raw data (5min + daily) for target symbols since last available timestamp
- Runs v9 model live inference to produce next-day OHLC prediction for each symbol
- Applies selected strategy to the prediction to decide whether to trade
- Computes bracket order distances from predicted OHLC and places orders (or prints in dry-run)

Usage examples:
  python -m utilities.scripts.one_click_trade --dry-run
  python -m utilities.scripts.one_click_trade --no-dry-run --confirm

Notes:
- Symbols are limited to 4 and mapped to strategies in SYMBOL_PLAN below.
- Requires models/versions/v9 checkpoints and artifacts present (already in repo).
- Uses data/raw/{SYMBOL}_NSE_{5min|daily}_20171001_to_*.csv as the consolidated source; this script appends new candles if available.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports
from api.fyers_session_management import check_access_token_expiry_and_get_fyers_session
from api.fyers_purchase_management import (
    place_intraday_buy_bracket_order,
    place_intraday_sell_bracket_order,
)
from src.data.data_fetch import fetch_historical_raw_data
from utilities.date_utilities import get_epoch_timestamp_from_datetime_ist_string

# v9 model imports
from models.versions.v9.data_pipeline import (
    DataPipeline,
    FeatureConfig,
    Scalers,
    get_default_feature_columns,
)
from models.versions.v9.model import LSTMAttnNextDayOHLC, ModelConfig
import yaml
import torch

# Strategies
from src.simulator.signal_generators import create_signal_generators, TradingSignal


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

# Symbol plan maps symbol -> (strategy_id, pretty_name)
SYMBOL_PLAN: Dict[str, Tuple[str, str]] = {
    # symbol: (strategy_key, label)
    'TATAMOTORS': ('tech_confirm_standard_target_based', 'Technical Confirmation Standard'),
    'NTPC': ('tech_confirm_ultra_aggressive', 'Technical Confirmation Ultra'),
    'TCS': ('tech_confirm_ultra_aggressive', 'Technical Confirmation Ultra'),
    'RELIANCE': ('tech_confirm_ultra_aggressive', 'Technical Confirmation Ultra'),
}

# Capital per trade for position sizing (used to estimate quantity)
DEFAULT_CAPITAL_PER_TRADE = 100000.0  # INR

# Data file templates
RAW_DIR = ROOT / 'data' / 'raw'
FIVE_MIN_TEMPLATE = '{sym}_NSE_5min_20171001_to_20250912.csv'  # existing baseline
DAILY_TEMPLATE = '{sym}_NSE_daily_20171001_to_20250912.csv'

# Fyers symbol mapper
def to_fyers_symbol(sym_simple: str) -> str:
    return f'NSE:{sym_simple}-EQ'


# -------------------------------------------------------------
# Utilities: Data update
# -------------------------------------------------------------

def _load_existing_csv(sym: str, timeframe: str) -> Tuple[pd.DataFrame, Path]:
    """Load existing consolidated CSV from data/raw.
    timeframe: '5min' or 'daily'
    Returns (df, path). If not exists, raises FileNotFoundError.
    """
    fname = (FIVE_MIN_TEMPLATE if timeframe == '5min' else DAILY_TEMPLATE).format(sym=sym)
    fpath = RAW_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Missing raw data file: {fpath}")
    df = pd.read_csv(fpath)
    if 'timestamp' not in df.columns:
        raise ValueError(f"Unexpected schema in {fpath}, missing 'timestamp'")
    return df, fpath


def _append_new_candles(df_existing: pd.DataFrame, new: List[List[float]]) -> pd.DataFrame:
    if not new:
        return df_existing
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df_new = pd.DataFrame(new, columns=cols)
    # Concatenate and drop duplicates by timestamp
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df_all


def update_raw_data_for_symbol(sym: str, verbose: bool = True) -> Dict[str, int]:
    """Fetch and append candles after the last timestamp in existing raw CSVs.
    Returns counts dict: {'5min_added': n1, 'daily_added': n2}
    """
    counts = {'5min_added': 0, 'daily_added': 0}

    # 5min
    try:
        df5, path5 = _load_existing_csv(sym, '5min')
        last_ts = int(df5['timestamp'].max())
        # Fetch from next minute after last to today 21:00 IST
        start_dt_ist = datetime.fromtimestamp(last_ts) + timedelta(minutes=1)
        start_str_ist = start_dt_ist.strftime('%Y-%m-%d %H:%M:%S')
        start_epoch = get_epoch_timestamp_from_datetime_ist_string(start_str_ist)
        end_epoch = get_epoch_timestamp_from_datetime_ist_string(datetime.now().strftime('%Y-%m-%d') + ' 21:00:00')
        if end_epoch > start_epoch:
            resp = fetch_historical_raw_data(to_fyers_symbol(sym), '5', start_epoch, end_epoch)
            if resp and resp.get('s') == 'ok' and resp.get('candles'):
                before = len(df5)
                df5u = _append_new_candles(df5, resp['candles'])
                counts['5min_added'] = len(df5u) - before
                if counts['5min_added'] > 0:
                    df5u.to_csv(path5, index=False)
                    if verbose:
                        print(f"[{sym}] 5min: appended {counts['5min_added']} rows → {path5}")
            else:
                if verbose:
                    print(f"[{sym}] 5min: no new candles returned")
        else:
            if verbose:
                print(f"[{sym}] 5min: up-to-date (last_ts={last_ts})")
    except FileNotFoundError as e:
        print(str(e))

    # daily
    try:
        dfd, pathd = _load_existing_csv(sym, 'daily')
        last_day_ts = int(dfd['timestamp'].max())
        # Next day 09:00 IST to today 21:00 IST
        last_day = datetime.fromtimestamp(last_day_ts) + timedelta(days=1)
        start_epoch = get_epoch_timestamp_from_datetime_ist_string(last_day.strftime('%Y-%m-%d') + ' 09:00:00')
        end_epoch = get_epoch_timestamp_from_datetime_ist_string(datetime.now().strftime('%Y-%m-%d') + ' 21:00:00')
        if end_epoch > start_epoch:
            resp = fetch_historical_raw_data(to_fyers_symbol(sym), '1D', start_epoch, end_epoch)
            if resp and resp.get('s') == 'ok' and resp.get('candles'):
                before = len(dfd)
                dfdu = _append_new_candles(dfd, resp['candles'])
                counts['daily_added'] = len(dfdu) - before
                if counts['daily_added'] > 0:
                    dfdu.to_csv(pathd, index=False)
                    if verbose:
                        print(f"[{sym}] daily: appended {counts['daily_added']} rows → {pathd}")
            else:
                if verbose:
                    print(f"[{sym}] daily: no new candles returned")
        else:
            if verbose:
                print(f"[{sym}] daily: up-to-date (last_ts={last_day_ts})")
    except FileNotFoundError as e:
        print(str(e))

    return counts


# -------------------------------------------------------------
# Inference: v9 next-day OHLC
# -------------------------------------------------------------

def _load_v9_cfg() -> Dict:
    cfg_path = ROOT / 'models' / 'versions' / 'v9' / 'config.yaml'
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class V9Prediction:
    symbol: str
    prediction_date: str
    predicted: Dict[str, float]
    confidence: float
    last_actual: Dict[str, float]
    model_version: str


def predict_next_day_v9(sym: str) -> Optional[V9Prediction]:
    """Produce next-day absolute OHLC prediction for the given symbol using v9 ensemble.
    Returns None if no prediction could be generated.
    """
    cfg = _load_v9_cfg()

    # Use consolidated raw CSVs
    daily_csv = ROOT / 'data' / 'raw' / DAILY_TEMPLATE.format(sym=sym)
    id_csv = ROOT / 'data' / 'raw' / FIVE_MIN_TEMPLATE.format(sym=sym)

    feat_cfg = FeatureConfig(
        use_intraday_aggregates=cfg['features']['use_intraday_aggregates'],
        use_calendar_features=cfg['features']['use_calendar_features'],
        use_regime_features=cfg['features']['use_regime_features'],
        rsi_period=int(cfg['features']['rsi_period']),
        atr_period=int(cfg['features']['atr_period']),
        ma_windows=tuple(cfg['features']['ma_windows']),
    )
    pipe = DataPipeline(str(daily_csv), str(id_csv), feat_cfg)
    ddf = pipe.load_daily()
    idf = pipe.load_intraday()
    id_agg = pipe.compute_intraday_aggregates(idf) if idf is not None else None
    df = pipe.engineer_features(ddf, id_agg)
    df = pipe.make_targets(df)
    df = df.sort_values('date').reset_index(drop=True)

    # Prepare features
    # Discover folds/checkpoints
    ckpt_dir = ROOT / cfg['output']['checkpoints_dir']
    art_dir = ROOT / cfg['output']['artifacts_dir']
    run_name = f"{cfg['output']['run_name']}_{sym}"
    fold_idxs: List[int] = []
    for i in range(0, 16):
        if (ckpt_dir / f"{run_name}_fold{i}.pt").exists():
            fold_idxs.append(i)
    if not fold_idxs:
        print(f"No checkpoints found for {run_name} in {ckpt_dir}")
        return None

    device = (
        torch.device('cuda') if torch.cuda.is_available() else (
            torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else torch.device('cpu')
        )
    )

    # Build features per fold, respecting saved feature list + lookback
    preds_accum = []
    probs_accum = []
    last_dates = []

    for fi in fold_idxs:
        state = torch.load(ckpt_dir / f"{run_name}_fold{fi}.pt", map_location='cpu')
        feature_cols = state.get('feature_cols')
        if feature_cols is None:
            feature_cols = get_default_feature_columns(df)
        lookback = int(state.get('lookback', cfg['sequence']['lookback']))

        req_cols = feature_cols + ['open_rel', 'dh_rel', 'dl_rel', 'dc_rel', 'dir_label', 'open', 'high', 'low', 'close']
        req_cols = [c for c in req_cols if c in df.columns]
        df_clean = df.dropna(subset=req_cols).reset_index(drop=True)
        if len(df_clean) < lookback + 1:
            print(f"Insufficient data after cleaning for fold {fi}")
            continue

        X, y, y_aux, dates, base_close = DataPipeline.sequence_dataset(df_clean, feature_cols, lookback)

        # Load scaler for this fold
        scaler = Scalers()
        scaler_path = art_dir / f"{run_name}_fold{fi}_scaler.json"
        if not scaler_path.exists():
            print(f"Missing scaler for fold {fi}: {scaler_path}")
            continue
        scaler.load(str(scaler_path))
        X_s = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        # Temperature (optional)
        T = 1.0
        tpath = art_dir / f"{run_name}_fold{fi}_temperature.json"
        if tpath.exists():
            try:
                with open(tpath, 'r') as tf:
                    T = float(json.load(tf).get('T', 1.0))
            except Exception:
                T = 1.0

        model = LSTMAttnNextDayOHLC(ModelConfig(
            input_size=X.shape[-1],
            hidden_size=cfg['model']['hidden_size'],
            num_layers=cfg['model']['num_layers'],
            dropout=cfg['model']['dropout'],
            attn_dim=cfg['model'].get('attn_dim', 64),
        ))
        model.load_state_dict(state['state_dict'])
        model.to(device)
        model.eval()
        with torch.no_grad():
            y_reg, y_dir = model(torch.tensor(X_s, dtype=torch.float32, device=device))
            y_reg = y_reg.cpu().numpy()
            probs = torch.softmax(y_dir / max(T, 1e-3), dim=-1).cpu().numpy()

        preds_accum.append(y_reg)
        probs_accum.append(probs)
        last_dates = dates  # keep aligned

    if not preds_accum:
        return None

    y_reg_mean = np.mean(preds_accum, axis=0)
    dir_prob_mean = np.mean(probs_accum, axis=0)

    # Use the last available base date -> prediction date next day
    i = len(last_dates) - 1
    base_date = pd.to_datetime(last_dates[i])
    pred_date = (base_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # atr_pct for clamping
    df_idx = df.set_index(pd.to_datetime(df['date']))
    row = df_idx.loc[pd.to_datetime(base_date)] if pd.to_datetime(base_date) in df_idx.index else None
    atr_pct = float(row.get('atr', np.nan) / (row.get('close', np.nan) + 1e-8)) if row is not None and 'atr' in df_idx.columns else None

    o, h, l, c = DataPipeline.postprocess_ohlc_from_rel(
        float(df_idx.loc[pd.to_datetime(base_date)]['close']),
        float(y_reg_mean[i, 0]), float(y_reg_mean[i, 1]), float(y_reg_mean[i, 2]), float(y_reg_mean[i, 3]),
        atr_pct, clamp_mult=float(cfg.get('postprocess', {}).get('clamp_atr_mult', 3.5))
    )
    conf = float(np.max(dir_prob_mean[i]))

    # last actual = last day's OHLC
    la = df_idx.loc[pd.to_datetime(base_date)][['open', 'high', 'low', 'close']]
    last_actual = {k.capitalize(): float(la[k]) for k in ['open', 'high', 'low', 'close']}

    return V9Prediction(
        symbol=sym,
        prediction_date=pred_date,
        predicted={'Open': float(o), 'High': float(h), 'Low': float(l), 'Close': float(c)},
        confidence=conf,
        last_actual=last_actual,
        model_version=f"v9_lstm_attn_ohlc_{sym}"
    )


# -------------------------------------------------------------
# Strategy decision + order placement
# -------------------------------------------------------------

def _build_market_data_for_strategy(sym: str) -> pd.DataFrame:
    """Load 5min raw data and build market_data DataFrame with date_str and time_ist as expected by generators."""
    fpath = RAW_DIR / FIVE_MIN_TEMPLATE.format(sym=sym)
    df = pd.read_csv(fpath)
    df['datetime_ist'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
    df['time_ist'] = df['datetime_ist'].dt.time
    df['date_str'] = df['datetime_ist'].dt.strftime('%Y-%m-%d')
    # Filter to market hours similar to strategy_runner
    mask = (df['time_ist'] >= pd.to_datetime('09:15', format='%H:%M').time()) & (df['time_ist'] <= pd.to_datetime('15:15', format='%H:%M').time())
    return df[mask].copy()


def decide_and_maybe_place_order(
    sym: str,
    strategy_key: str,
    prediction: V9Prediction,
    dry_run: bool = True,
    capital_per_trade: float = DEFAULT_CAPITAL_PER_TRADE,
    prefer_actual_open: bool = False,
) -> Dict[str, any]:
    """Apply strategy to prediction and optionally place an order.
    Returns an execution record dict.
    """
    # Prepare predictions dict expected by generators
    pred_map = {
        prediction.prediction_date: {
            'symbol': sym,
            'prediction_date': prediction.prediction_date,
            'last_actual': prediction.last_actual,
            'predicted': prediction.predicted,
            'confidence': prediction.confidence,
            'model_version': prediction.model_version,
        }
    }

    market_data = _build_market_data_for_strategy(sym)

    # Build generators and parameter sets mapping to strategy_id
    gens = create_signal_generators()

    # Map strategy key to (generator_key, params)
    strat_map = {
        'tech_confirm_standard_target_based': ('tech_confirm_standard', 'target_based'),
        'tech_confirm_ultra_aggressive': ('tech_confirm_ultra', 'aggressive'),
    }
    if strategy_key not in strat_map:
        raise ValueError(f"Unsupported strategy_key: {strategy_key}")
    gen_key, param_profile = strat_map[strategy_key]
    gen = gens[gen_key]

    # Generate signal for the prediction date
    date_str = prediction.prediction_date
    signal: Optional[TradingSignal] = gen.generate_signal(date_str, pred_map, market_data)
    if signal is None or not gen.should_trade(signal):
        return {
            'symbol': sym,
            'strategy_key': strategy_key,
            'decision': 'NO_TRADE',
            'reason': 'no_signal_or_below_threshold',
            'prediction': prediction.__dict__,
        }

    # Determine entry price preference order:
    # 1) signal-provided entry
    # 2) actual opening price from first 5-min candle of the day
    # 3) model-predicted Open
    actual_open = None
    try:
        day_rows = market_data[market_data['date_str'] == date_str]
        if not day_rows.empty:
            # Prefer the 'open' of the first bar; fall back to 'close' if needed
            first_row = day_rows.iloc[0]
            if 'open' in first_row and not pd.isna(first_row['open']):
                actual_open = float(first_row['open'])
            elif 'close' in first_row and not pd.isna(first_row['close']):
                actual_open = float(first_row['close'])
    except Exception:
        actual_open = None

    if prefer_actual_open and actual_open is not None:
        entry_price = actual_open
    else:
        entry_price = signal.entry_price or actual_open or prediction.predicted['Open']
    qty = max(1, int(capital_per_trade // entry_price))

    # Compute bracket distances from predicted OHLC
    o = prediction.predicted['Open']
    h = prediction.predicted['High']
    l = prediction.predicted['Low']
    if signal.signal_type == 'BUY':
        stop_loss = max(0.05, o - l)  # absolute distance
        take_profit = max(0.05, h - o)
    else:  # SELL
        stop_loss = max(0.05, h - o)
        take_profit = max(0.05, o - l)

    fy_symbol = to_fyers_symbol(sym)
    side = 'BUY' if signal.signal_type == 'BUY' else 'SELL'

    order_payload = {
        'symbol': fy_symbol,
        'qty': qty,
        'productType': 'BO',
        'side': side,
        'entry_ref_price': entry_price,
        'actual_open': actual_open,
        'stop_loss': round(stop_loss, 2),
        'take_profit': round(take_profit, 2),
        'confidence': signal.confidence,
        'signal_strength': signal.signal_strength,
        'strategy_key': strategy_key,
        'prediction_date': date_str,
    }

    # Place the order (dry-run by default)
    placed = None
    if not dry_run:
        if signal.signal_type == 'BUY':
            placed = place_intraday_buy_bracket_order(fy_symbol, qty, stop_loss, take_profit)
        else:
            placed = place_intraday_sell_bracket_order(fy_symbol, qty, stop_loss, take_profit)

    return {
        'symbol': sym,
        'strategy_key': strategy_key,
        'decision': 'ORDER_PLACED' if placed is not None else ('DRY_RUN' if dry_run else 'PLACEMENT_ATTEMPTED'),
        'order': order_payload,
        'prediction': prediction.__dict__,
    }


# -------------------------------------------------------------
# Main Orchestration
# -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='One-click Fyers trading (v9 + strategies)')
    ap.add_argument('--symbols', type=str, default=','.join(SYMBOL_PLAN.keys()), help='Comma-separated symbols')
    ap.add_argument('--no-dry-run', action='store_true', help='Actually place orders (default is dry-run)')
    ap.add_argument('--confirm', action='store_true', help='Require explicit confirmation before placing orders')
    ap.add_argument('--prefer-actual-open', action='store_true', help='Use actual market open as entry when available (overrides signal entry)')
    ap.add_argument('--wait-until-open', action='store_true', help='If run before market open, wait until 09:16 IST then proceed')
    ap.add_argument('--open-time', type=str, default='09:16', help='HH:MM IST to begin decisions (default 09:16)')
    ap.add_argument('--wait-timeout-mins', type=int, default=30, help='Maximum minutes to wait for open (default 30)')
    ap.add_argument('--capital', type=float, default=DEFAULT_CAPITAL_PER_TRADE, help='Capital per trade (INR)')
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    dry_run = not args.no_dry_run

    print('=== Step 1: Ensure Fyers session (refresh token if needed) ===')
    try:
        _ = check_access_token_expiry_and_get_fyers_session()
        print('Fyers session is valid.')
    except Exception as e:
        print(f'Failed to ensure Fyers session: {e}')
        return 2

    print('\n=== Step 2: Update data since 2025-09-12 (if any) ===')
    updates: Dict[str, Dict[str, int]] = {}
    for sym in symbols:
        updates[sym] = update_raw_data_for_symbol(sym)
    print('Data update summary:', updates)

    # Optional wait until market open (IST)
    if args.wait_until_open:
        ist = ZoneInfo('Asia/Kolkata')
        now_ist = datetime.now(tz=ist)
        # Weekend guard: if Saturday(5) or Sunday(6), don't wait indefinitely
        if now_ist.weekday() >= 5:
            print('Weekend detected (IST). Skipping wait-until-open.')
        else:
            try:
                hh, mm = [int(x) for x in args.open_time.split(':')]
                target_dt = now_ist.replace(hour=hh, minute=mm, second=0, microsecond=0)
                if now_ist > target_dt:
                    print(f"Current IST time {now_ist.strftime('%H:%M')} is past open target {args.open_time}. Not waiting.")
                else:
                    print(f"\n=== Waiting until {args.open_time} IST for first 5-min candle ===")
                    deadline = now_ist + timedelta(minutes=args.wait_timeout_mins)
                    while datetime.now(tz=ist) < target_dt and datetime.now(tz=ist) < deadline:
                        remaining = (target_dt - datetime.now(tz=ist)).total_seconds()
                        sleep_for = max(5, min(30, int(remaining)))
                        try:
                            import time as _time
                            _time.sleep(sleep_for)
                        except Exception:
                            break
                    # Re-update data to capture the first bars after waiting
                    print('Re-updating data after wait...')
                    updates_after_wait: Dict[str, Dict[str, int]] = {}
                    for sym in symbols:
                        updates_after_wait[sym] = update_raw_data_for_symbol(sym, verbose=False)
                    print('Post-wait data update summary:', updates_after_wait)
            except Exception as e:
                print(f"Wait-until-open encountered an issue: {e}. Proceeding without waiting.")

    print('\n=== Step 3: Run v9 next-day predictions ===')
    preds: Dict[str, Optional[V9Prediction]] = {}
    for sym in symbols:
        preds[sym] = predict_next_day_v9(sym)
        if preds[sym]:
            print(f"{sym}: predicted {preds[sym].predicted} for {preds[sym].prediction_date} (conf={preds[sym].confidence:.2f})")
        else:
            print(f"{sym}: prediction unavailable")

    print('\n=== Step 4: Strategy decisions and order placements ===')
    results = []
    for sym in symbols:
        if sym not in SYMBOL_PLAN:
            print(f"Skipping {sym}: not in SYMBOL_PLAN")
            continue
        pred = preds.get(sym)
        if not pred:
            results.append({'symbol': sym, 'decision': 'NO_TRADE', 'reason': 'no_prediction'})
            continue
        strategy_key = SYMBOL_PLAN[sym][0]
        rec = decide_and_maybe_place_order(
            sym,
            strategy_key,
            pred,
            dry_run=dry_run,
            capital_per_trade=args.capital,
            prefer_actual_open=args.prefer_actual_open,
        )
        results.append(rec)
        print(f"{sym}: {rec['decision']} → {rec.get('order', {})}")

    # Save run log
    logs_dir = ROOT / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = logs_dir / f'trade_run_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': ts,
            'symbols': symbols,
            'dry_run': dry_run,
            'updates': updates,
            'results': results,
        }, f, indent=2)
    print(f"\nSaved run log → {out_path}")

    if not dry_run and not args.confirm:
        print("WARNING: You ran with --no-dry-run but without --confirm. Consider using --confirm for safety next time.")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
