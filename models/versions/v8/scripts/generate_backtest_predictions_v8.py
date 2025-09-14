import os
import sys
import json
from pathlib import Path
from typing import Dict
import argparse

import numpy as np
import pandas as pd
import torch
import yaml

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.versions.v8.data_pipeline import DataPipeline, FeatureConfig, Scalers, get_default_feature_columns
from models.versions.v8.model import LSTMNextDayOHLC, ModelConfig


def _load_cfg():
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def _load_daily_csv(path: str) -> pd.DataFrame:
    tmp_cfg = FeatureConfig()
    pipe = DataPipeline(path, None, tmp_cfg)
    return pipe.load_daily()


def _load_intraday_csv(path: str) -> pd.DataFrame:
    tmp_cfg = FeatureConfig(use_intraday_aggregates=True)
    pipe = DataPipeline('', path, tmp_cfg)
    return pipe.load_intraday()


def resolve_paths_for_symbol(cfg, symbol: str):
    base_dir = cfg['data']['base_dir']
    train_daily = os.path.join(base_dir, symbol, 'daily', f"{symbol}_NSE_2017-10-01_to_2024-12-31.csv")
    train_id = os.path.join(base_dir, symbol, '5min', f"{symbol}_NSE_2017-10-01_to_2024-12-31.csv")
    return train_daily, train_id


def generate_backtest_predictions_v8(symbol: str, start_date: str, end_date: str, out_path: str) -> Dict:
    cfg = _load_cfg()
    symbol = symbol.upper()

    train_daily, train_id = resolve_paths_for_symbol(cfg, symbol)
    bt_daily = f'data/backtest/{symbol}/daily/{symbol}_NSE_{start_date}_to_{end_date}.csv'
    bt_id = f'data/backtest/{symbol}/5min/{symbol}_NSE_{start_date}_to_{end_date}.csv'

    # Load and combine daily
    ddf_train = _load_daily_csv(train_daily)
    ddf_bt = _load_daily_csv(bt_daily)
    ddf = pd.concat([ddf_train, ddf_bt], ignore_index=True).sort_values('date').reset_index(drop=True)

    # Load and combine intraday
    idf_train = _load_intraday_csv(train_id) if os.path.exists(train_id) else None
    idf_bt = _load_intraday_csv(bt_id) if os.path.exists(bt_id) else None
    if idf_train is not None and idf_bt is not None:
        idf = pd.concat([idf_train, idf_bt], ignore_index=True)
    else:
        idf = idf_train if idf_train is not None else idf_bt

    feat_cfg = FeatureConfig(
        use_intraday_aggregates=cfg['features']['use_intraday_aggregates'],
        use_calendar_features=cfg['features']['use_calendar_features'],
        use_regime_features=cfg['features']['use_regime_features'],
        rsi_period=int(cfg['features']['rsi_period']),
        atr_period=int(cfg['features']['atr_period']),
        ma_windows=tuple(cfg['features']['ma_windows'])
    )
    pipe = DataPipeline('', '', feat_cfg)

    id_agg = pipe.compute_intraday_aggregates(idf) if idf is not None else None
    df = pipe.engineer_features(ddf, id_agg)
    df = pipe.make_targets(df)

    # Load checkpoint and scaler for this symbol
    run_name = f"{cfg['output']['run_name']}_{symbol}"
    ckpt_dir = cfg['output']['checkpoints_dir']
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt') and f.startswith(run_name)]
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found for {run_name}')
    ckpts.sort()
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    ckpt = torch.load(ckpt_path, map_location='cpu')

    lookback = int(ckpt['lookback'])
    feature_cols = ckpt.get('feature_cols')
    if feature_cols is None:
        feature_cols = get_default_feature_columns(df)

    scaler = Scalers()
    art_dir = cfg['output']['artifacts_dir']
    scalers = [f for f in os.listdir(art_dir) if f.startswith(run_name) and f.endswith('_scaler.json')]
    if not scalers:
        raise FileNotFoundError(f'No scaler artifact found for {run_name}')
    scalers.sort()
    scaler.load(os.path.join(art_dir, scalers[-1]))

    # Temperature (optional)
    temp_files = [f for f in os.listdir(art_dir) if f.startswith(run_name) and f.endswith('_temperature.json')]
    T = 1.0
    if temp_files:
        temp_files.sort()
        try:
            with open(os.path.join(art_dir, temp_files[-1]), 'r') as tf:
                T = float(json.load(tf).get('T', 1.0))
        except Exception:
            T = 1.0

    # Prepare sequences
    df = df.sort_values('date').reset_index(drop=True)
    req_cols = feature_cols + ['open_rel', 'dh_rel', 'dl_rel', 'dc_rel', 'dir_label', 'open', 'high', 'low', 'close']
    df = df.dropna(subset=[c for c in req_cols if c in df.columns]).reset_index(drop=True)

    X, y, y_aux, dates, base_close = pipe.sequence_dataset(df, feature_cols, lookback)
    X_s = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    model = LSTMNextDayOHLC(ModelConfig(input_size=X.shape[-1], hidden_size=cfg['model']['hidden_size'], num_layers=cfg['model']['num_layers'], dropout=cfg['model']['dropout']))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    with torch.no_grad():
        y_reg, y_dir = model(torch.tensor(X_s, dtype=torch.float32))
        y_reg = y_reg.numpy()
        logits = y_dir.numpy()
        probs = torch.softmax(torch.tensor(logits) / max(T, 1e-3), dim=-1).numpy()

    atr_pct = None
    if 'atr' in df.columns:
        atr_series = (df['atr'] / (df['close'] + 1e-8)).values.astype(np.float32)
        atr_pct = atr_series[lookback:len(df)-0-1]

    clamp_mult = cfg.get('postprocess', {}).get('clamp_atr_mult', 3.5)
    ohlc_map = df.set_index('date')[['open', 'high', 'low', 'close']].to_dict('index')

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    out: Dict[str, dict] = {}
    from models.versions.v8.data_pipeline import DataPipeline as DP
    n = y_reg.shape[0]
    for i in range(n):
        base_date = pd.to_datetime(dates[i])
        pred_date = base_date + pd.Timedelta(days=1)
        if not (start_dt <= pred_date <= end_dt):
            continue
        ap = float(atr_pct[i]) if isinstance(atr_pct, np.ndarray) else None
        o, h, l, c = DP.postprocess_ohlc_from_rel(float(base_close[i]), float(y_reg[i,0]), float(y_reg[i,1]), float(y_reg[i,2]), float(y_reg[i,3]), ap, clamp_mult)
        conf = float(np.max(probs[i]))
        la = ohlc_map.get(base_date)
        if la is None:
            continue
        out[pred_date.strftime('%Y-%m-%d')] = {
            'symbol': symbol,
            'prediction_date': pred_date.strftime('%Y-%m-%d'),
            'last_actual': {
                'Open': float(la['open']),
                'High': float(la['high']),
                'Low': float(la['low']),
                'Close': float(la['close'])
            },
            'predicted': {
                'Open': float(o),
                'High': float(h),
                'Low': float(l),
                'Close': float(c)
            },
            'confidence': conf,
            'model_version': run_name
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} backtest predictions for {symbol} to {out_path}")
    return out


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--start', default='2025-01-01')
    p.add_argument('--end', default='2025-05-31')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    sym = args.symbol.upper()
    start = args.start
    end = args.end
    if args.out is None:
        out = f"data/predictions/v8_backtest_{sym}_{start.replace('-', '')}_to_{end.replace('-', '')}.json"
    else:
        out = args.out
    generate_backtest_predictions_v8(sym, start, end, out)


if __name__ == '__main__':
    cli()
