import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure project root is on sys.path for absolute imports
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.versions.v7.data_pipeline import DataPipeline, FeatureConfig, Scalers, get_default_feature_columns
from models.versions.v7.model import LSTMNextDayOHLC, ModelConfig

import yaml


def _load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def _load_daily_csv(path: str) -> pd.DataFrame:
    # Use DataPipeline.load_daily to normalize
    tmp_cfg = FeatureConfig()
    pipe = DataPipeline(path, None, tmp_cfg)
    return pipe.load_daily()


def _load_intraday_csv(path: str) -> pd.DataFrame:
    tmp_cfg = FeatureConfig(use_intraday_aggregates=True)
    pipe = DataPipeline('', path, tmp_cfg)
    return pipe.load_intraday()


def generate_backtest_predictions_v7(
    start_date: str = '2025-01-01',
    end_date: str = '2025-05-31',
    out_path: str = 'data/predictions/backtest_predictions_v7_20250101_to_20250531.json'
) -> Dict:
    cfg = _load_cfg()

    # Paths
    train_daily = cfg['data']['daily_path']
    train_id = cfg['data']['intraday_5min_path']
    bt_daily = 'data/backtest/RELIANCE/daily/RELIANCE_NSE_2025-01-01_to_2025-05-31.csv'
    bt_id = 'data/backtest/RELIANCE/5min/RELIANCE_NSE_2025-01-01_to_2025-05-31.csv'

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

    # Build pipeline with feature config
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

    # Load checkpoint and scaler
    ckpt_dir = cfg['output']['checkpoints_dir']
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt') and f.startswith(cfg['output']['run_name'])]
    if not ckpts:
        raise FileNotFoundError('No checkpoints found for v7')
    ckpts.sort()
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    ckpt = torch.load(ckpt_path, map_location='cpu')

    lookback = int(ckpt['lookback'])
    feature_cols = ckpt.get('feature_cols')
    if feature_cols is None:
        feature_cols = get_default_feature_columns(df)

    scaler = Scalers()
    art_dir = cfg['output']['artifacts_dir']
    scalers = [f for f in os.listdir(art_dir) if f.startswith(cfg['output']['run_name']) and f.endswith('_scaler.json')]
    if not scalers:
        raise FileNotFoundError('No scaler artifact found for v7')
    scalers.sort()
    scaler.load(os.path.join(art_dir, scalers[-1]))

    # Temperature (optional)
    temp_files = [f for f in os.listdir(art_dir) if f.startswith(cfg['output']['run_name']) and f.endswith('_temperature.json')]
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

    # Scale features
    X_s = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Run model in batch
    model = LSTMNextDayOHLC(ModelConfig(input_size=X.shape[-1], hidden_size=cfg['model']['hidden_size'], num_layers=cfg['model']['num_layers'], dropout=cfg['model']['dropout']))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    with torch.no_grad():
        y_reg, y_dir = model(torch.tensor(X_s, dtype=torch.float32))
        y_reg = y_reg.numpy()
        logits = y_dir.numpy()
        probs = torch.softmax(torch.tensor(logits) / max(T, 1e-3), dim=-1).numpy()

    # ATR pct per sequence index i
    atr_pct = None
    if 'atr' in df.columns:
        atr_series = (df['atr'] / (df['close'] + 1e-8)).values.astype(np.float32)
        atr_pct = atr_series[lookback:len(df)-0-1]  # aligns with sequence indices

    clamp_mult = cfg.get('postprocess', {}).get('clamp_atr_mult', 3.5)

    # Build mapping for last actual by date
    ohlc_map = df.set_index('date')[['open', 'high', 'low', 'close']].to_dict('index')

    # Date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    out: Dict[str, dict] = {}
    from models.versions.v7.data_pipeline import DataPipeline as DP
    n = y_reg.shape[0]
    for i in range(n):
        base_date = pd.to_datetime(dates[i])
        pred_date = base_date + pd.Timedelta(days=1)
        if not (start_dt <= pred_date <= end_dt):
            continue
        ap = float(atr_pct[i]) if isinstance(atr_pct, np.ndarray) else None
        o, h, l, c = DP.postprocess_ohlc_from_rel(float(base_close[i]), float(y_reg[i,0]), float(y_reg[i,1]), float(y_reg[i,2]), float(y_reg[i,3]), ap, clamp_mult)
        conf = float(np.max(probs[i])) * 100.0
        # last actual is base_date OHLC
        la = ohlc_map.get(base_date)
        if la is None:
            continue
        out[pred_date.strftime('%Y-%m-%d')] = {
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
            'model_version': cfg['output']['run_name']
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} backtest predictions to {out_path}")
    return out


if __name__ == '__main__':
    generate_backtest_predictions_v7()
