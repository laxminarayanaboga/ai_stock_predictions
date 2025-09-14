import os
import json
import numpy as np
import pandas as pd
import torch

from models.versions.v7.data_pipeline import DataPipeline, FeatureConfig, Scalers, get_default_feature_columns
from models.versions.v7.model import LSTMNextDayOHLC, ModelConfig

import yaml


def load_latest_checkpoint(checkpoints_dir: str, run_name: str):
    files = [f for f in os.listdir(checkpoints_dir) if f.startswith(run_name) and f.endswith('.pt')]
    if not files:
        raise FileNotFoundError('No checkpoints found')
    files.sort()
    return os.path.join(checkpoints_dir, files[-1])


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    daily_csv = cfg['data']['daily_path']
    intraday_csv = cfg['data']['intraday_5min_path']

    feat_cfg = FeatureConfig(
        use_intraday_aggregates=cfg['features']['use_intraday_aggregates'],
        use_calendar_features=cfg['features']['use_calendar_features'],
        use_regime_features=cfg['features']['use_regime_features'],
        rsi_period=int(cfg['features']['rsi_period']),
        atr_period=int(cfg['features']['atr_period']),
        ma_windows=tuple(cfg['features']['ma_windows'])
    )

    pipe = DataPipeline(daily_csv, intraday_csv, feat_cfg)
    ddf = pipe.load_daily()
    idf = pipe.load_intraday()
    id_agg = pipe.compute_intraday_aggregates(idf) if idf is not None else None
    df = pipe.engineer_features(ddf, id_agg)
    df = pipe.make_targets(df)

    feature_cols = get_default_feature_columns(df)
    df = df.dropna(subset=feature_cols + ['close']).reset_index(drop=True)

    ckpt_path = load_latest_checkpoint(cfg['output']['checkpoints_dir'], cfg['output']['run_name'])
    ckpt = torch.load(ckpt_path, map_location='cpu')
    lookback = ckpt['lookback']

    scaler = Scalers()
    scaler_path_candidates = [f for f in os.listdir(cfg['output']['artifacts_dir']) if f.startswith(cfg['output']['run_name']) and f.endswith('_scaler.json')]
    if not scaler_path_candidates:
        raise FileNotFoundError('No scaler artifact found')
    scaler_path_candidates.sort()
    scaler.load(os.path.join(cfg['output']['artifacts_dir'], scaler_path_candidates[-1]))

    # Build last window
    X_all = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X_all)
    window = X_scaled[-lookback:]
    base_close = df['close'].iloc[-1]
    xb = torch.tensor(window[None, ...], dtype=torch.float32)

    model = LSTMNextDayOHLC(ModelConfig(input_size=window.shape[-1]))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    with torch.no_grad():
        y_reg, y_dir = model(xb)
        y_reg = y_reg.numpy()[0]
        # ATR percentage from last row (if present)
        atr_pct = None
        if 'atr' in df.columns and df['atr'].iloc[-1] > 0:
            atr_pct = float(df['atr'].iloc[-1] / (df['close'].iloc[-1] + 1e-8))
        clamp_mult = cfg.get('postprocess', {}).get('clamp_atr_mult', 3.5)
        from models.versions.v7.data_pipeline import DataPipeline as DP
        open_pred, high_pred, low_pred, close_pred = DP.postprocess_ohlc_from_rel(base_close, y_reg[0], y_reg[1], y_reg[2], y_reg[3], atr_pct, clamp_mult)
        # Load temperature if present and apply calibration
        T = 1.0
        temp_files = [f for f in os.listdir(cfg['output']['artifacts_dir']) if f.startswith(cfg['output']['run_name']) and f.endswith('_temperature.json')]
        if temp_files:
            temp_files.sort()
            try:
                with open(os.path.join(cfg['output']['artifacts_dir'], temp_files[-1]), 'r') as tf:
                    T = float(json.load(tf).get('T', 1.0))
            except Exception:
                T = 1.0
        logits = torch.tensor(y_dir.numpy()[0]) / max(T, 1e-3)
        dir_prob = torch.softmax(logits, dim=-1).numpy().tolist()

    conf_th = cfg.get('postprocess', {}).get('confidence_threshold', 0.7)
    confidence = max(dir_prob)
    action_allowed = confidence >= conf_th

    pred = {
        'open': float(open_pred),
        'high': float(high_pred),
        'low': float(low_pred),
        'close': float(close_pred),
        'direction_probs': {'bear': dir_prob[0], 'bull': dir_prob[1]},
        'confidence': confidence,
        'action_allowed': action_allowed
    }

    os.makedirs(cfg['output']['predictions_dir'], exist_ok=True)
    out_path = os.path.join(cfg['output']['predictions_dir'], f"{cfg['output']['run_name']}_latest.json")
    with open(out_path, 'w') as f:
        json.dump(pred, f, indent=2)
    print(json.dumps(pred, indent=2))


if __name__ == '__main__':
    main()
