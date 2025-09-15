from __future__ import annotations

import pandas as pd
import numpy as np

try:
    import ta  # type: ignore
    _HAS_TA = True
except Exception:
    ta = None  # type: ignore
    _HAS_TA = False


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    pv = df['close'] * df['volume']
    cum_pv = pv.cumsum()
    cum_vol = df['volume'].cumsum().replace(0, np.nan)
    out = (cum_pv / cum_vol)
    return out.bfill()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if _HAS_TA:
        return ta.momentum.RSIIndicator(series, window=period).rsi()
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist
