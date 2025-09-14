import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch

_CUR = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_CUR, '..', '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.versions.v9.data_pipeline import DataPipeline, FeatureConfig, Scalers, get_default_feature_columns
from models.versions.v9.model import LSTMAttnNextDayOHLC, ModelConfig
import yaml


def resolve_symbol_paths(cfg, symbol: str):
    base_dir = cfg['data']['base_dir']
    daily_csv = os.path.join(base_dir, symbol, 'daily', f"{symbol}_NSE_2017-10-01_to_2024-12-31.csv")
    intraday_csv = os.path.join(base_dir, symbol, '5min', f"{symbol}_NSE_2017-10-01_to_2024-12-31.csv")
    return daily_csv, intraday_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--folds', type=str, default=None, help='Comma-separated fold indices to ensemble; default: all found')
    args = parser.parse_args()
    symbol = args.symbol.upper()

    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    daily_csv, intraday_csv = resolve_symbol_paths(cfg, symbol)
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
    df = df.dropna(subset=feature_cols + ['open_rel', 'dh_rel', 'dl_rel', 'dc_rel', 'dir_label']).reset_index(drop=True)

    lookback = int(cfg['sequence']['lookback'])

    # gather all folds
    ckpt_dir = cfg['output']['checkpoints_dir']
    run_name = f"{cfg['output']['run_name']}_{symbol}"
    fold_idxs = []
    if args.folds:
        fold_idxs = [int(i) for i in args.folds.split(',') if i.strip().isdigit()]
    else:
        # discover by files
        for i in range(0, 12):
            p = os.path.join(ckpt_dir, f"{run_name}_fold{i}.pt")
            if os.path.exists(p):
                fold_idxs.append(i)
    if not fold_idxs:
        raise SystemExit(f"No checkpoints found for {run_name} in {ckpt_dir}")

    preds = []
    dir_probs = []

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # last available window
    X_all = df[feature_cols].values.astype(np.float32)
    X_seq = []
    dates = []
    for i in range(lookback, len(df)):
        X_seq.append(X_all[i - lookback:i])
        dates.append(pd.to_datetime(df['date'].iloc[i]))
    X_seq = np.stack(X_seq)

    for fi in fold_idxs:
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_fold{fi}.pt")
        if not os.path.exists(ckpt_path):
            continue
        state = torch.load(ckpt_path, map_location='cpu')
        saved_cols = state.get('feature_cols', feature_cols)
        saved_lookback = state.get('lookback', lookback)
        if saved_cols != feature_cols:
            print(f"Warning: feature columns mismatch for fold {fi}; attempting to align common columns")
            common = [c for c in saved_cols if c in feature_cols]
            if len(common) < 10:
                print(f"Skipping fold {fi} due to insufficient common features")
                continue
            # rebuild sequences for common features
            X_all_c = df[common].values.astype(np.float32)
            X_seq_c = []
            for i in range(saved_lookback, len(df)):
                X_seq_c.append(X_all_c[i - saved_lookback:i])
            X_in = np.stack(X_seq_c)
        else:
            X_in = X_seq
            saved_lookback = lookback

        model = LSTMAttnNextDayOHLC(ModelConfig(
            input_size=len(saved_cols),
            hidden_size=cfg['model']['hidden_size'],
            num_layers=cfg['model']['num_layers'],
            dropout=cfg['model']['dropout'],
            attn_dim=cfg['model'].get('attn_dim', 64),
        ))
        model.load_state_dict(state['state_dict'])
        model.to(device)
        model.eval()

        with torch.no_grad():
            Xb = torch.tensor(X_in).to(device)
            y_reg, y_dir = model(Xb)
            preds.append(y_reg.cpu().numpy())
            probs = torch.softmax(y_dir, dim=-1).cpu().numpy()
            dir_probs.append(probs)

    if not preds:
        raise SystemExit("No predictions could be generated; no valid checkpoints loaded")

    y_reg_mean = np.mean(preds, axis=0)
    dir_prob_mean = np.mean(dir_probs, axis=0)

    out = []
    for i, d in enumerate(dates):
        out.append({
            'date': d.strftime('%Y-%m-%d'),
            'open_rel': float(y_reg_mean[i, 0]),
            'dh_rel': float(y_reg_mean[i, 1]),
            'dl_rel': float(y_reg_mean[i, 2]),
            'dc_rel': float(y_reg_mean[i, 3]),
            'dir_prob_up': float(dir_prob_mean[i, 1]),
            'dir_prob_down': float(dir_prob_mean[i, 0]),
        })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out or cfg['output']['predictions_dir']
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"v9_predictions_{symbol}_{ts}.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote predictions to {out_path}")


if __name__ == '__main__':
    main()
