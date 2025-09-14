import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
import torch

_CUR = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_CUR, '..', '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.versions.v9.data_pipeline import DataPipeline, FeatureConfig, Scalers, get_default_feature_columns
from models.versions.v9.model import LSTMAttnNextDayOHLC, ModelConfig


def _load_cfg() -> Dict:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--start', default='2025-01-01', help='YYYY-MM-DD subset start for emission (prediction date)')
    parser.add_argument('--end', default='2025-05-31', help='YYYY-MM-DD subset end for emission (prediction date)')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    symbol = args.symbol.upper()

    cfg = _load_cfg()

    # Resolve training paths and corresponding backtest paths
    train_daily, train_id = resolve_paths_for_symbol(cfg, symbol)
    bt_daily = f"data/backtest/{symbol}/daily/{symbol}_NSE_{args.start}_to_{args.end}.csv"
    bt_id = f"data/backtest/{symbol}/5min/{symbol}_NSE_{args.start}_to_{args.end}.csv"

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

    # Feature engineering
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
    df = df.sort_values('date').reset_index(drop=True)

    # Prepare for inference
    ckpt_dir = cfg['output']['checkpoints_dir']
    art_dir = cfg['output']['artifacts_dir']
    run_name = f"{cfg['output']['run_name']}_{symbol}"

    # Discover folds
    fold_idxs: List[int] = []
    for i in range(0, 16):
        if os.path.exists(os.path.join(ckpt_dir, f"{run_name}_fold{i}.pt")):
            fold_idxs.append(i)
    if not fold_idxs:
        raise SystemExit(f"No checkpoints found for {run_name} in {ckpt_dir}")

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Map for last actual OHLC by date
    ohlc_map = df.set_index('date')[['open', 'high', 'low', 'close']].to_dict('index')

    # Aggregate predictions across folds keyed by prediction date
    agg: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for fi in fold_idxs:
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_fold{fi}.pt")
        state = torch.load(ckpt_path, map_location='cpu')
        feature_cols = state.get('feature_cols')
        if feature_cols is None:
            feature_cols = get_default_feature_columns(df)
        lookback = int(state.get('lookback', cfg['sequence']['lookback']))

        # Build sequences using saved feature columns
        req_cols = feature_cols + ['open_rel', 'dh_rel', 'dl_rel', 'dc_rel', 'dir_label', 'open', 'high', 'low', 'close']
        req_cols = [c for c in req_cols if c in df.columns]
        df_clean = df.dropna(subset=req_cols).reset_index(drop=True)

        X, y, y_aux, dates, base_close = pipe.sequence_dataset(df_clean, feature_cols, lookback)

        # Load scaler for this fold
        scaler_path = os.path.join(art_dir, f"{run_name}_fold{fi}_scaler.json")
        scaler = Scalers()
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler for fold {fi}: {scaler_path}")
        scaler.load(scaler_path)
        X_s = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        # Temperature (optional)
        T = 1.0
        tpath = os.path.join(art_dir, f"{run_name}_fold{fi}_temperature.json")
        if os.path.exists(tpath):
            try:
                with open(tpath, 'r') as tf:
                    T = float(json.load(tf).get('T', 1.0))
            except Exception:
                T = 1.0

        # Run model
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

        # ATR percent for clamping (aligned to sequence indices by date lookup)
        df_clean_idx = df_clean.set_index(pd.to_datetime(df_clean['date']))
        for i in range(y_reg.shape[0]):
            base_date = pd.to_datetime(dates[i])
            pred_date = base_date + pd.Timedelta(days=1)
            # Filter to requested window (prediction date)
            if not (pd.to_datetime(args.start) <= pred_date <= pd.to_datetime(args.end)):
                continue

            row = df_clean_idx.loc[base_date] if base_date in df_clean_idx.index else None
            if row is None:
                continue
            atr_pct = float(row.get('atr', np.nan) / (row.get('close', np.nan) + 1e-8)) if 'atr' in df_clean_idx.columns else None

            # Post-process absolute OHLC
            o, h, l, c = DataPipeline.postprocess_ohlc_from_rel(
                float(base_close[i]),
                float(y_reg[i, 0]), float(y_reg[i, 1]), float(y_reg[i, 2]), float(y_reg[i, 3]),
                atr_pct,
                clamp_mult=float(cfg.get('postprocess', {}).get('clamp_atr_mult', 3.5))
            )
            conf = float(np.max(probs[i]))

            key = pred_date.strftime('%Y-%m-%d')
            if key not in agg:
                agg[key] = {'Open': 0.0, 'High': 0.0, 'Low': 0.0, 'Close': 0.0, 'confidence': 0.0}
                counts[key] = 0
            agg[key]['Open'] += o
            agg[key]['High'] += h
            agg[key]['Low'] += l
            agg[key]['Close'] += c
            agg[key]['confidence'] += conf
            counts[key] += 1

    # Build final output dict keyed by prediction date
    out: Dict[str, dict] = {}
    for key, vals in agg.items():
        n = max(counts.get(key, 1), 1)
        # Last actual from previous day
        base_date = (pd.to_datetime(key) - pd.Timedelta(days=1)).to_pydatetime()
        la = ohlc_map.get(pd.to_datetime(base_date))
        if la is None:
            # Try without time component (already date-only keys)
            la = ohlc_map.get(pd.to_datetime(base_date).normalize())
        if la is None:
            continue
        out[key] = {
            'symbol': symbol,
            'prediction_date': key,
            'last_actual': {
                'Open': float(la['open']),
                'High': float(la['high']),
                'Low': float(la['low']),
                'Close': float(la['close'])
            },
            'predicted': {
                'Open': float(vals['Open'] / n),
                'High': float(vals['High'] / n),
                'Low': float(vals['Low'] / n),
                'Close': float(vals['Close'] / n)
            },
            'confidence': float(vals['confidence'] / n),
            'model_version': run_name
        }

    # Write output
    out_dir = args.out or cfg['output']['predictions_dir']
    os.makedirs(out_dir, exist_ok=True)
    start_tag = args.start.replace('-', '')
    end_tag = args.end.replace('-', '')
    out_path = os.path.join(out_dir, f"v9_backtest_{symbol}_{start_tag}_to_{end_tag}.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} backtest predictions for {symbol} to {out_path}")


if __name__ == '__main__':
    main()
