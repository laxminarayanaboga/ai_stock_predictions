import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import talib
except Exception:
    talib = None


@dataclass
class FeatureConfig:
    use_intraday_aggregates: bool = True
    use_calendar_features: bool = True
    use_regime_features: bool = True
    rsi_period: int = 14
    atr_period: int = 14
    ma_windows: Tuple[int, ...] = (5, 10, 20, 50, 200)


class Scalers:
    def __init__(self):
        self.feature_scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        self.feature_scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.feature_scaler.transform(X)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'feature_scaler_mean': self.feature_scaler.mean_.tolist(),
            'feature_scaler_scale': self.feature_scaler.scale_.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(state, f)

    def load(self, path: str):
        with open(path, 'r') as f:
            state = json.load(f)
        self.feature_scaler.mean_ = np.array(state['feature_scaler_mean'])
        self.feature_scaler.scale_ = np.array(state['feature_scaler_scale'])
        self.feature_scaler.var_ = self.feature_scaler.scale_ ** 2


class DataPipeline:
    def __init__(self, daily_csv: str, intraday_csv: str | None, cfg: FeatureConfig):
        self.daily_csv = daily_csv
        self.intraday_csv = intraday_csv
        self.cfg = cfg

    def load_daily(self) -> pd.DataFrame:
        df = pd.read_csv(self.daily_csv)
        cols_lower = {c.lower(): c for c in df.columns}
        if 'timestamp' in cols_lower:
            ts_col = cols_lower['timestamp']
            date = pd.to_datetime(df[ts_col], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.floor('D').dt.tz_localize(None)
        elif 'date' in cols_lower:
            ts_col = cols_lower['date']
            date = pd.to_datetime(df[ts_col])
        else:
            ts_col = df.columns[0]
            date = pd.to_datetime(df[ts_col], errors='coerce')
        rename_map = {}
        for name in ['open', 'high', 'low', 'close', 'volume']:
            if name in cols_lower:
                rename_map[cols_lower[name]] = name
            elif name.capitalize() in df.columns:
                rename_map[name.capitalize()] = name
        df = df.rename(columns=rename_map)
        if isinstance(date, pd.Series):
            if hasattr(date.dt, 'tz'):
                try:
                    df['date'] = date.dt.tz_localize(None)
                except Exception:
                    df['date'] = pd.to_datetime(date)
            else:
                df['date'] = pd.to_datetime(date)
        else:
            df['date'] = pd.to_datetime(date)
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)
        return df

    def load_intraday(self) -> pd.DataFrame | None:
        if not self.cfg.use_intraday_aggregates or not self.intraday_csv or not os.path.exists(self.intraday_csv):
            return None
        idf = pd.read_csv(self.intraday_csv)
        cols_lower = {c.lower(): c for c in idf.columns}
        if 'timestamp' in cols_lower:
            ts_col = cols_lower['timestamp']
            ts = pd.to_datetime(idf[ts_col], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
        else:
            ts_col = idf.columns[0]
            ts = pd.to_datetime(idf[ts_col], utc=True)
        idf['ts'] = ts
        idf['date'] = idf['ts'].dt.floor('D').dt.tz_localize(None)
        idf = idf.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
        })
        return idf

    def compute_intraday_aggregates(self, idf: pd.DataFrame) -> pd.DataFrame:
        if idf is None:
            return pd.DataFrame()
        grp = idf.groupby('date')
        feats = grp.apply(lambda g: pd.Series({
            'id_ret_mean': (np.log(g['close']).diff()).mean(skipna=True),
            'id_ret_std': (np.log(g['close']).diff()).std(skipna=True),
            'id_ret_max': (np.log(g['close']).diff()).max(skipna=True),
            'id_ret_min': (np.log(g['close']).diff()).min(skipna=True),
            'id_first_hour_range': (g.iloc[:min(12, len(g))]['high'].max() - g.iloc[:min(12, len(g))]['low'].min()) / (g['close'].iloc[0] if len(g) > 0 else np.nan),
            'id_last_hour_ret': (g['close'].iloc[-1] / g['close'].iloc[max(len(g)-12, 0)] - 1) if len(g) > 12 else np.nan,
            'id_vwap_dev_close': ((g['close'].iloc[-1] - (g['close'] * g['volume']).sum() / max(g['volume'].sum(), 1)) / ((g['close'] * g['volume']).sum() / max(g['volume'].sum(), 1))) if g['volume'].sum() > 0 else 0.0,
            'id_first_hour_vol_ratio': g['volume'].iloc[:min(12, len(g))].sum() / max(g['volume'].sum(), 1)
        }), include_groups=False)
        feats.index = pd.to_datetime(feats.index)
        feats = feats.reset_index().rename(columns={'index': 'date'})
        feats = feats.sort_values('date')
        feats['date'] = feats['date'] + pd.Timedelta(days=1)
        return feats

    def engineer_features(self, ddf: pd.DataFrame, id_agg: pd.DataFrame | None) -> pd.DataFrame:
        df = ddf.copy()
        df['prev_close'] = df['close'].shift(1)
        df['ret1'] = np.log(df['close'] / df['prev_close'])
        df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
        df['range'] = (df['high'] - df['low']) / df['close']
        df['prev_dir'] = (df['close'].shift(1) > df['open'].shift(1)).astype(float)
        df['gap_sign'] = np.sign(df['gap']).astype(float)

        for w in self.cfg.ma_windows:
            df[f'ma{w}'] = df['close'].rolling(w).mean()
            df[f'ma{w}_slope'] = df[f'ma{w}'].diff()

        if talib is not None:
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.cfg.rsi_period)
            df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=self.cfg.atr_period)
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
        else:
            delta = df['close'].diff()
            gain = (delta.clip(lower=0)).rolling(self.cfg.rsi_period).mean()
            loss = (-delta.clip(upper=0)).rolling(self.cfg.rsi_period).mean()
            rs = gain / (loss + 1e-8)
            df['rsi'] = 100 - (100 / (1 + rs))
            tr1 = (df['high'] - df['low']).abs()
            tr2 = (df['high'] - df['prev_close']).abs()
            tr3 = (df['low'] - df['prev_close']).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = tr.rolling(self.cfg.atr_period).mean()
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

        for p in [3, 8, 21]:
            if talib is not None:
                df[f'rsi_{p}'] = talib.RSI(df['close'].values, timeperiod=p)
            else:
                delta = df['close'].diff()
                gain = (delta.clip(lower=0)).rolling(p).mean()
                loss = (-delta.clip(upper=0)).rolling(p).mean()
                rs = gain / (loss + 1e-8)
                df[f'rsi_{p}'] = 100 - (100 / (1 + rs))
        df['rsi_3_8_diff'] = df['rsi_3'] - df['rsi_8']
        df['rsi_8_21_diff'] = df['rsi_8'] - df['rsi_21']
        df['macd_hist_slope'] = df['macd_hist'].diff()

        df['ret5'] = df['ret1'].rolling(5).sum()
        df['ret10'] = df['ret1'].rolling(10).sum()
        df['ret20'] = df['ret1'].rolling(20).sum()
        df['rv10'] = df['ret1'].rolling(10).std()
        df['rv20'] = df['ret1'].rolling(20).std()

        df['vol_ma5'] = df['volume'].rolling(5).mean()
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['vol_ratio_5_20'] = df['vol_ma5'] / (df['vol_ma20'] + 1e-8)

        if self.cfg.use_regime_features:
            df['ma200_slope_pos'] = (df['ma200_slope'] > 0).astype(float)
            df['atr_pct'] = df['atr'] / (df['close'] + 1e-8)
            df['atr_percentile_1y'] = df['atr_pct'].rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x.dropna())>0 else np.nan, raw=False)

        if self.cfg.use_calendar_features:
            df['dow'] = pd.to_datetime(df['date']).dt.weekday
            df['month'] = pd.to_datetime(df['date']).dt.month
            for d in range(1, 6):
                df[f'dow_{d}'] = (df['dow'] == d).astype(float)
            for m in [1, 3, 6, 9, 12]:
                df[f'month_{m}'] = (df['month'] == m).astype(float)

        if id_agg is not None and len(id_agg) > 0:
            id_agg2 = id_agg.copy()
            id_agg2['date'] = pd.to_datetime(id_agg2['date'])
            df = df.merge(id_agg2, on='date', how='left')

        df = df.sort_values('date').reset_index(drop=True)
        return df

    @staticmethod
    def postprocess_ohlc_from_rel(base_close: float, open_rel: float, dh: float, dl: float, dc: float, atr_pct: float | None, clamp_mult: float = 3.5) -> Tuple[float, float, float, float]:
        if atr_pct is not None and not (np.isnan(atr_pct) or np.isinf(atr_pct)):
            cap = max(atr_pct * clamp_mult, 0.01)
            dh = float(np.clip(dh, -cap, cap))
            dl = float(np.clip(dl, -cap, cap))

        o = base_close * (1.0 + float(open_rel))
        h = o * (1.0 + float(dh))
        l = o * (1.0 + float(dl))
        c = o * (1.0 + float(dc))

        l = min(l, o, c, h)
        h = max(h, o, c, l)
        if l > h:
            mid = (o + c) / 2.0
            span = max(abs(h - l), 1e-6)
            h = mid + span / 2.0
            l = mid - span / 2.0
        return float(o), float(h), float(l), float(c)

    @staticmethod
    def make_targets(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['open_next'] = df['open'].shift(-1)
        df['high_next'] = df['high'].shift(-1)
        df['low_next'] = df['low'].shift(-1)
        df['close_next'] = df['close'].shift(-1)
        df['dh_rel'] = (df['high_next'] - df['open_next']) / df['open_next']
        df['dl_rel'] = (df['low_next'] - df['open_next']) / df['open_next']
        df['dc_rel'] = (df['close_next'] - df['open_next']) / df['open_next']
        df['open_rel'] = (df['open_next'] - df['close']) / df['close']
        df['dir_label'] = (df['close_next'] > df['open_next']).astype(int)
        return df

    @staticmethod
    def sequence_dataset(df: pd.DataFrame, feature_cols: List[str], lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], np.ndarray]:
        X, y, y_aux, dates, base_close = [], [], [], [], []
        values = df[feature_cols].values.astype(np.float32)
        for i in range(lookback, len(df) - 1):
            if df[['open_rel', 'dh_rel', 'dl_rel', 'dc_rel']].iloc[i].isna().any():
                continue
            X.append(values[i - lookback:i])
            y.append(df[['open_rel', 'dh_rel', 'dl_rel', 'dc_rel']].iloc[i].values.astype(np.float32))
            y_aux.append(df['dir_label'].iloc[i].astype(np.int64))
            dates.append(pd.to_datetime(df['date'].iloc[i]))
            base_close.append(df['close'].iloc[i])
        return np.stack(X), np.stack(y), np.array(y_aux), dates, np.array(base_close, dtype=np.float32)


def get_default_feature_columns(df: pd.DataFrame) -> List[str]:
    base_cols = [
        'open', 'high', 'low', 'close', 'volume', 'prev_close', 'ret1', 'gap', 'range',
        'ret5', 'ret10', 'ret20', 'rv10', 'rv20',
        'rsi', 'atr', 'macd', 'macd_signal', 'macd_hist',
        'vol_ma5', 'vol_ma20', 'vol_ratio_5_20',
        'ma5', 'ma10', 'ma20', 'ma50', 'ma200', 'ma5_slope', 'ma10_slope', 'ma20_slope', 'ma50_slope', 'ma200_slope',
    ]
    for c in ['prev_dir', 'gap_sign', 'rsi_3', 'rsi_8', 'rsi_21', 'rsi_3_8_diff', 'rsi_8_21_diff', 'macd_hist_slope']:
        if c in df.columns:
            base_cols.append(c)
    if 'ma200_slope_pos' in df.columns:
        base_cols += ['ma200_slope_pos', 'atr_pct', 'atr_percentile_1y']
    for c in ['id_ret_mean', 'id_ret_std', 'id_ret_max', 'id_ret_min', 'id_first_hour_range', 'id_last_hour_ret', 'id_vwap_dev_close', 'id_first_hour_vol_ratio']:
        if c in df.columns:
            base_cols.append(c)
    for c in ['dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'month_1', 'month_3', 'month_6', 'month_9', 'month_12']:
        if c in df.columns:
            base_cols.append(c)
    base_cols = [c for c in base_cols if c in df.columns]
    return base_cols
