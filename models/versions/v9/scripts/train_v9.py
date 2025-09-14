import os
import sys
import json
import argparse
from time import time
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_CUR = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_CUR, '..', '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.versions.v9.data_pipeline import DataPipeline, FeatureConfig, Scalers, get_default_feature_columns
from models.versions.v9.model import LSTMAttnNextDayOHLC, ModelConfig

import yaml


def huber_loss(pred, target, delta=1.0):
    err = pred - target
    abs_err = torch.abs(err)
    quadratic = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
    linear = abs_err - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)
        pt = torch.exp(-ce_loss)
        focal = (self.alpha * (1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


def train_one_fold(model, optim, scheduler, train_loader, val_loader, device, cfg_train):
    best_val = float('inf')
    best_state = None
    patience = int(cfg_train.get('early_stopping', {}).get('patience', 10)) if cfg_train.get('early_stopping', {}).get('enabled', False) else None
    min_delta = float(cfg_train.get('early_stopping', {}).get('min_delta', 0.0))
    wait = 0

    # Direction loss
    dloss_cfg = cfg_train.get('direction_loss', {})
    class_weights = cfg_train.get('class_weights', None)
    if dloss_cfg.get('type', 'ce') == 'focal':
        ce_like = FocalLoss(alpha=float(dloss_cfg.get('alpha', 0.25)), gamma=float(dloss_cfg.get('gamma', 2.0)))
        cw = None
    else:
        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        ce_like = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    start_t = time()
    for epoch in range(cfg_train['epochs']):
        model.train()
        running = []
        for xb, yb, yaux in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yaux = yaux.to(device)
            optim.zero_grad()
            y_reg, y_dir = model(xb)
            loss_reg = huber_loss(y_reg, yb, delta=cfg_train.get('huber_delta', 1.0)).mean()
            loss_dir = ce_like(y_dir, yaux)
            loss = cfg_train['loss_weights']['regression'] * loss_reg + cfg_train['loss_weights'].get('direction', 0.0) * loss_dir
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg_train.get('grad_clip', 1.0))
            optim.step()
            running.append(loss.item())
        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            vlosses = []
            for xb, yb, yaux in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_reg, y_dir = model(xb)
                vloss = huber_loss(y_reg, yb, delta=cfg_train.get('huber_delta', 1.0)).mean().item()
                vlosses.append(vloss)
            val_mean = float(np.mean(vlosses)) if vlosses else float('inf')
        if (epoch + 1) % 1 == 0:
            elapsed = time() - start_t
            print(f"Epoch {epoch+1}/{cfg_train['epochs']} - train_loss={np.mean(running):.6f} val_loss={val_mean:.6f} elapsed={elapsed:.1f}s")
        if val_mean < best_val - min_delta:
            best_val = val_mean
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if patience is not None and wait >= patience:
                print(f"Early stopping at epoch {epoch+1} (best_val={best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def resolve_symbol_paths(cfg, symbol: str):
    base_dir = cfg['data']['base_dir']
    daily_csv = os.path.join(base_dir, symbol, 'daily', f"{symbol}_NSE_2017-10-01_to_2024-12-31.csv")
    intraday_csv = os.path.join(base_dir, symbol, '5min', f"{symbol}_NSE_2017-10-01_to_2024-12-31.csv")
    return daily_csv, intraday_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True, help='Symbol to train (e.g., RELIANCE, TCS, NTPC, TATAMOTORS)')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs for quick runs')
    parser.add_argument('--max-folds', type=int, default=None, help='Train only the first N folds')
    parser.add_argument('--no-calibration', action='store_true', help='Disable temperature scaling calibration step')
    args = parser.parse_args()
    symbol = args.symbol.upper()

    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg['train']['epochs'] = int(args.epochs)
    if args.no_calibration:
        cfg['train'].setdefault('calibration', {})
        cfg['train']['calibration']['temperature_scaling'] = False

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
    X, y, y_aux, dates, base_close = pipe.sequence_dataset(df, feature_cols, lookback)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    splits = []
    if len(dates) > 0:
        dser = pd.Series(dates)
        start_date = dser.iloc[0]
        end_date = dser.iloc[-1]
        cursor = start_date + pd.DateOffset(years=cfg['walk_forward']['train_years'])
        while cursor + pd.DateOffset(months=cfg['walk_forward']['val_months']) <= end_date:
            train_end = cursor
            val_end = cursor + pd.DateOffset(months=cfg['walk_forward']['val_months'])
            train_idx = dser.index[dser <= train_end]
            val_idx = dser.index[(dser > train_end) & (dser <= val_end)]
            if len(train_idx) > 100 and len(val_idx) > 20:
                splits.append((train_idx, val_idx))
            cursor = cursor + pd.DateOffset(months=cfg['walk_forward']['step_months'])

    os.makedirs(cfg['output']['checkpoints_dir'], exist_ok=True)
    os.makedirs(cfg['output']['artifacts_dir'], exist_ok=True)

    run_name = f"{cfg['output']['run_name']}_{symbol}"

    if args.max_folds is not None and args.max_folds > 0:
        splits = splits[: args.max_folds]

    print(f"Starting walk-forward for {symbol} with {len(splits)} folds...")
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFold {fold_i+1}/{len(splits)} ➜ train={len(train_idx)} val={len(val_idx)}")
        X_train = X[train_idx]
        y_train = y[train_idx]
        yaux_train = y_aux[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        yaux_val = y_aux[val_idx]

        scaler = Scalers()
        scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
        X_train_s = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_s = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train), torch.tensor(yaux_train)), batch_size=cfg['train']['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val_s), torch.tensor(y_val), torch.tensor(yaux_val)), batch_size=cfg['train']['batch_size'], shuffle=False)

        model = LSTMAttnNextDayOHLC(ModelConfig(
            input_size=X.shape[-1],
            hidden_size=cfg['model']['hidden_size'],
            num_layers=cfg['model']['num_layers'],
            dropout=cfg['model']['dropout'],
            attn_dim=cfg['model'].get('attn_dim', 64),
        )).to(device)

        optim = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
        scheduler = None
        if cfg['train'].get('scheduler', {}).get('type') == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                T_max=int(cfg['train']['scheduler'].get('T_max', cfg['train']['epochs'])),
                eta_min=float(cfg['train']['scheduler'].get('eta_min', 1e-5)),
            )

        best_val = train_one_fold(model, optim, scheduler, train_loader, val_loader, device, cfg['train'])
        print(f"Fold {fold_i} best_val={best_val:.6f}")

        ckpt_path = os.path.join(cfg['output']['checkpoints_dir'], f"{run_name}_fold{fold_i}.pt")
        torch.save({'state_dict': model.state_dict(), 'feature_cols': feature_cols, 'lookback': lookback}, ckpt_path)
        scaler_path = os.path.join(cfg['output']['artifacts_dir'], f"{run_name}_fold{fold_i}_scaler.json")
        scaler.save(scaler_path)

        if cfg['train'].get('calibration', {}).get('temperature_scaling', False):
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val_s).to(device)
                _, y_dir_logits = model(Xv)
                logits = y_dir_logits.cpu()
                labels = torch.tensor(yaux_val)
            T = torch.ones(1, requires_grad=True)
            cal_lr = float(cfg['train']['calibration'].get('lr', 0.01))
            cal_iters = int(cfg['train']['calibration'].get('max_iters', 300))
            optimizer_T = torch.optim.LBFGS([T], lr=cal_lr, max_iter=cal_iters)
            ce = nn.CrossEntropyLoss()

            def closure():
                optimizer_T.zero_grad()
                scaled = logits / T.clamp(min=1e-3)
                loss = ce(scaled, labels)
                loss.backward()
                return loss

            optimizer_T.step(closure)
            T_val = float(T.detach().clamp(min=1e-3).item())
            tpath = os.path.join(cfg['output']['artifacts_dir'], f"{run_name}_fold{fold_i}_temperature.json")
            with open(tpath, 'w') as f:
                json.dump({'T': T_val}, f)
            print(f"Fold {fold_i} temperature T={T_val:.3f}")

    print("Training complete.")


if __name__ == '__main__':
    main()
