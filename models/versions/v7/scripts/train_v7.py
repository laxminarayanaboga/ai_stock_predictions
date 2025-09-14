import os
import json
import math
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error

from models.versions.v7.data_pipeline import DataPipeline, FeatureConfig, Scalers, get_default_feature_columns
from models.versions.v7.model import LSTMNextDayOHLC, ModelConfig

import yaml
from time import time


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


def make_wf_splits(dates: list[pd.Timestamp], train_years=3, val_months=6, step_months=3):
    # Generate (train_idx, val_idx) boundaries over time
    # dates assumed sorted
    splits = []
    dser = pd.Series(dates)
    start_date = dser.iloc[0]
    end_date = dser.iloc[-1]
    cursor = start_date + pd.DateOffset(years=train_years)
    while cursor + pd.DateOffset(months=val_months) <= end_date:
        train_end = cursor
        val_end = cursor + pd.DateOffset(months=val_months)
        train_idx = dser.index[dser <= train_end]
        val_idx = dser.index[(dser > train_end) & (dser <= val_end)]
        if len(train_idx) > 100 and len(val_idx) > 20:
            splits.append((train_idx, val_idx))
        cursor = cursor + pd.DateOffset(months=step_months)
    return splits


def train_one_fold(model, optim, train_loader, val_loader, device, cfg_train):
    best_val = float('inf')
    best_state = None
    # Direction loss
    dloss_cfg = cfg_train.get('direction_loss', {})
    if dloss_cfg.get('type', 'ce') == 'focal':
        ce_like = FocalLoss(alpha=float(dloss_cfg.get('alpha', 0.25)), gamma=float(dloss_cfg.get('gamma', 2.0)))
    else:
        ce_like = nn.CrossEntropyLoss(label_smoothing=0.05)

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
        # Validation
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
        if val_mean < best_val:
            best_val = val_mean
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        # Early stopping patience ~10 epochs
        if epoch > 10 and len(vlosses) > 0 and val_mean > best_val * 1.01:
            # simple stop if no improvement margin
            pass
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def main():
    # Load config
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

    # Feature columns
    feature_cols = get_default_feature_columns(df)

    # Drop rows with NAs
    df = df.dropna(subset=feature_cols + ['open_rel', 'dh_rel', 'dl_rel', 'dc_rel', 'dir_label']).reset_index(drop=True)

    # Build sequences
    lookback = int(cfg['sequence']['lookback'])
    X, y, y_aux, dates, base_close = pipe.sequence_dataset(df, feature_cols, lookback)

    # Scaler fit on entire feature matrix BEFORE sequence? Prefer fit on training in each fold.
    # We'll scale per fold to avoid leakage.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits = make_wf_splits(dates, cfg['walk_forward']['train_years'], cfg['walk_forward']['val_months'], cfg['walk_forward']['step_months'])

    os.makedirs(cfg['output']['checkpoints_dir'], exist_ok=True)
    os.makedirs(cfg['output']['artifacts_dir'], exist_ok=True)

    oos_records = []
    print(f"Starting walk-forward with {len(splits)} folds...")
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFold {fold_i+1}/{len(splits)} ➜ train={len(train_idx)} val={len(val_idx)}")
        X_train = X[train_idx]
        y_train = y[train_idx]
        yaux_train = y_aux[train_idx]
        bc_train = base_close[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        yaux_val = y_aux[val_idx]
        bc_val = base_close[val_idx]

        # Fit scaler on training features only
        # Flatten time dimension for scaling
        scaler = Scalers()
        scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
        X_train_s = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_s = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train), torch.tensor(yaux_train)), batch_size=cfg['train']['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val_s), torch.tensor(y_val), torch.tensor(yaux_val)), batch_size=cfg['train']['batch_size'], shuffle=False)

        model = LSTMNextDayOHLC(ModelConfig(input_size=X.shape[-1], hidden_size=cfg['model']['hidden_size'], num_layers=cfg['model']['num_layers'], dropout=cfg['model']['dropout'])).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

        best_val = train_one_fold(model, optim, train_loader, val_loader, device, cfg['train'])
        print(f"Fold {fold_i} best_val={best_val:.6f}")

        # Save fold checkpoint and scaler
        ckpt_path = os.path.join(cfg['output']['checkpoints_dir'], f"{cfg['output']['run_name']}_fold{fold_i}.pt")
        torch.save({'state_dict': model.state_dict(), 'feature_cols': feature_cols, 'lookback': lookback}, ckpt_path)
        scaler_path = os.path.join(cfg['output']['artifacts_dir'], f"{cfg['output']['run_name']}_fold{fold_i}_scaler.json")
        scaler.save(scaler_path)

        # OOS predictions on val for metrics
        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_val_s).to(device)
            y_reg, y_dir = model(Xv)
            preds = y_reg.cpu().numpy()
            truth = y_val
            # Reconstruct absolute OHLC using base_close and ATR-based clamping
            atr_pct = None
            if 'atr' in df.columns:
                # atr percentage vs close for the same indices
                atr_pct = (df['atr'].values[lookback:][val_idx] / (df['close'].values[lookback:][val_idx] + 1e-8)).astype(np.float32)
            clamp_mult = cfg.get('postprocess', {}).get('clamp_atr_mult', 3.5)
            open_pred = np.zeros(len(preds), dtype=np.float32)
            high_pred = np.zeros(len(preds), dtype=np.float32)
            low_pred = np.zeros(len(preds), dtype=np.float32)
            close_pred = np.zeros(len(preds), dtype=np.float32)
            open_true = np.zeros(len(preds), dtype=np.float32)
            high_true = np.zeros(len(preds), dtype=np.float32)
            low_true = np.zeros(len(preds), dtype=np.float32)
            close_true = np.zeros(len(preds), dtype=np.float32)
            for i in range(len(preds)):
                ap = atr_pct[i] if isinstance(atr_pct, np.ndarray) else None
                o, h, l, c = DataPipeline.postprocess_ohlc_from_rel(bc_val[i], preds[i, 0], preds[i, 1], preds[i, 2], preds[i, 3], ap, clamp_mult)
                open_pred[i], high_pred[i], low_pred[i], close_pred[i] = o, h, l, c
                o2, h2, l2, c2 = DataPipeline.postprocess_ohlc_from_rel(bc_val[i], truth[i, 0], truth[i, 1], truth[i, 2], truth[i, 3], ap, clamp_mult)
                open_true[i], high_true[i], low_true[i], close_true[i] = o2, h2, l2, c2

            mae_open = float(np.mean(np.abs(open_pred - open_true)))
            mae_high = float(np.mean(np.abs(high_pred - high_true)))
            mae_low = float(np.mean(np.abs(low_pred - low_true)))
            mae_close = float(np.mean(np.abs(close_pred - close_true)))

            # Confidence-binned report (by direction head)
            # Optionally calibrate direction probs with temperature scaling using val set
            probs_raw = torch.softmax(y_dir, dim=-1).cpu().numpy()
            probs = probs_raw
            conf = probs.max(axis=1)
            pred_dir = (probs[:, 1] >= 0.5).astype(int)
            true_dir = yaux_val
            # Confidence bin edges from config (fallback to defaults)
            bins_cfg = cfg.get('postprocess', {}).get('confidence_bins', None)
            if bins_cfg is None:
                bins = [(0.0, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]
            else:
                # Ensure list of tuples
                bins = [tuple(map(float, b)) for b in bins_cfg]
            rows = []
            for lo, hi in bins:
                idx = np.where((conf >= lo) & (conf < hi))[0]
                if len(idx) == 0:
                    rows.append({
                        'bin_low': lo, 'bin_high': hi, 'count': 0,
                        'dir_acc': np.nan,
                        'mae_open': np.nan, 'mae_high': np.nan, 'mae_low': np.nan, 'mae_close': np.nan
                    })
                    continue
                acc = float(np.mean((pred_dir[idx] == true_dir[idx]).astype(float)))
                rows.append({
                    'bin_low': lo, 'bin_high': hi, 'count': int(len(idx)),
                    'dir_acc': acc,
                    'mae_open': float(np.mean(np.abs(open_pred[idx] - open_true[idx]))),
                    'mae_high': float(np.mean(np.abs(high_pred[idx] - high_true[idx]))),
                    'mae_low': float(np.mean(np.abs(low_pred[idx] - low_true[idx]))),
                    'mae_close': float(np.mean(np.abs(close_pred[idx] - close_true[idx])))
                })
            # Save CSV per fold
            conf_path = os.path.join(cfg['output']['artifacts_dir'], f"{cfg['output']['run_name']}_fold{fold_i}_confidence_report.csv")
            import pandas as _pd
            _pd.DataFrame(rows).to_csv(conf_path, index=False)
            # Print quick high-confidence summary
            hi_idx = np.where(conf >= 0.8)[0]
            if len(hi_idx) > 0:
                hi_acc = float(np.mean((pred_dir[hi_idx] == true_dir[hi_idx]).astype(float)))
                print(f"Fold {fold_i} high-confidence (>=0.8) N={len(hi_idx)} dir_acc={hi_acc:.3f} mae_close={float(np.mean(np.abs(close_pred[hi_idx]-close_true[hi_idx]))):.2f}")

        # Temperature scaling calibration (optimize on val logits vs true labels)
        if cfg['train'].get('calibration', {}).get('temperature_scaling', False):
            # We need raw logits on val set
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
            # Save temperature parameter
            tpath = os.path.join(cfg['output']['artifacts_dir'], f"{cfg['output']['run_name']}_fold{fold_i}_temperature.json")
            with open(tpath, 'w') as f:
                json.dump({'T': T_val}, f)
            print(f"Fold {fold_i} temperature T={T_val:.3f}")

        rec = {
            'fold': fold_i,
            'best_val_loss': best_val,
            'mae_open': mae_open,
            'mae_high': mae_high,
            'mae_low': mae_low,
            'mae_close': mae_close,
        }
        oos_records.append(rec)
        print(f"Fold {fold_i} OOS MAE: open={mae_open:.3f} high={mae_high:.3f} low={mae_low:.3f} close={mae_close:.3f}")

    # Save summary
    summary_path = os.path.join(cfg['output']['artifacts_dir'], f"{cfg['output']['run_name']}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({'oos': oos_records}, f, indent=2)


if __name__ == '__main__':
    main()
