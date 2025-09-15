from __future__ import annotations

from typing import List, Dict, Any, Tuple
import pandas as pd

from .base import Strategy, StrategyResult, Order
from .indicators import ema, atr, vwap, rsi, macd


def _within_windows(df: pd.DataFrame, windows: List[Tuple]) -> pd.DataFrame:
    if not windows:
        return df
    mask = False
    for start, end in windows:
        mask = mask | ((df['time_ist'] >= start) & (df['time_ist'] <= end))
    return df[mask]


class ORBStrategy(Strategy):
    def generate(self, df_1m: pd.DataFrame) -> StrategyResult:
        p = self.params
        orb_start = p.get('orb_start')
        orb_end = p.get('orb_end')
        windows = p.get('windows')
        min_range = p.get('min_range_pct', 0.002)
        buf = p.get('break_buffer_pct', 0.0005)
        sl = p.get('sl_pct', 0.003)
        tp = p.get('tp_pct', 0.006)
        qty = p.get('qty', 100)
        confirm = p.get('require_next_confirm', True)
        vol_pct = p.get('min_volume_percentile', 0.5)

        day = df_1m.copy()
        pre = day[(day['time_ist'] >= orb_start) & (day['time_ist'] <= orb_end)]
        if len(pre) < 2:
            return StrategyResult(self.name, p, [])
        orb_high = pre['high'].max()
        orb_low = pre['low'].min()
        if (orb_high / max(1e-9, orb_low) - 1.0) < min_range:
            return StrategyResult(self.name, p, [])
        post = day[day['time_ist'] > orb_end].copy()
        vol_thresh = post['volume'].quantile(vol_pct) if len(post) else 0
        post = _within_windows(post, windows)
        if post.empty:
            return StrategyResult(self.name, p, [])
        orders: List[Order] = []
        post = post.reset_index(drop=True)
        for i, bar in post.iterrows():
            if bar['volume'] < vol_thresh:
                continue
            if bar['close'] > orb_high * (1 + buf):
                if confirm and i + 1 < len(post):
                    nb = post.iloc[i + 1]
                    if nb['close'] <= orb_high * (1 + buf):
                        continue
                orders.append(Order(bar['datetime_ist'], float(bar['close']), 'LONG', qty, sl, tp))
                break
            if bar['close'] < orb_low * (1 - buf):
                if confirm and i + 1 < len(post):
                    nb = post.iloc[i + 1]
                    if nb['close'] >= orb_low * (1 - buf):
                        continue
                orders.append(Order(bar['datetime_ist'], float(bar['close']), 'SHORT', qty, sl, tp))
                break
        return StrategyResult(self.name, p, orders)


class EMAPullbackStrategy(Strategy):
    def generate(self, df_1m: pd.DataFrame) -> StrategyResult:
        p = self.params
        windows = p.get('windows')
        ema_win = p.get('ema_window', 20)
        slope_lb = p.get('slope_lookback', 5)
        min_slope = p.get('min_slope_pct', 0.0005)
        sl = p.get('sl_pct', 0.003)
        tp = p.get('tp_pct', 0.006)
        qty = p.get('qty', 100)
        day = _within_windows(df_1m.copy(), windows)
        if day.empty:
            return StrategyResult(self.name, p, [])
        day['ema'] = ema(day['close'], ema_win)
        day['ema_slope'] = day['ema'] - day['ema'].shift(slope_lb)
        orders: List[Order] = []
        for i in range(1, len(day)):
            row = day.iloc[i]
            prev = day.iloc[i-1]
            if row['ema_slope'] > prev['close'] * min_slope and prev['close'] > prev['ema'] and row['close'] > row['ema'] and row['low'] <= row['ema']:
                orders.append(Order(row['datetime_ist'], float(row['close']), 'LONG', qty, sl, tp))
            if row['ema_slope'] < -prev['close'] * min_slope and prev['close'] < prev['ema'] and row['close'] < row['ema'] and row['high'] >= row['ema']:
                orders.append(Order(row['datetime_ist'], float(row['close']), 'SHORT', qty, sl, tp))
        return StrategyResult(self.name, p, orders)


class VWAPReversionStrategy(Strategy):
    def generate(self, df_1m: pd.DataFrame) -> StrategyResult:
        p = self.params
        windows = p.get('windows')
        band = p.get('band_pct', 0.002)
        sl_mult = p.get('sl_mult', 1.5)
        tp = p.get('tp_pct', 0.004)
        qty = p.get('qty', 100)
        day = _within_windows(df_1m.copy(), windows)
        if day.empty:
            return StrategyResult(self.name, p, [])
        day['vwap'] = vwap(day)
        orders: List[Order] = []
        for i in range(1, len(day)):
            row = day.iloc[i]
            if row['close'] < row['vwap'] * (1 - band):
                sl = band * sl_mult
                orders.append(Order(row['datetime_ist'], float(row['close']), 'LONG', qty, sl, tp))
            if row['close'] > row['vwap'] * (1 + band):
                sl = band * sl_mult
                orders.append(Order(row['datetime_ist'], float(row['close']), 'SHORT', qty, sl, tp))
        return StrategyResult(self.name, p, orders)


class RSIReversalStrategy(Strategy):
    def generate(self, df_1m: pd.DataFrame) -> StrategyResult:
        p = self.params
        windows = p.get('windows')
        lb = p.get('period', 14)
        os_th = p.get('oversold', 25)
        ob_th = p.get('overbought', 75)
        sl = p.get('sl_pct', 0.003)
        tp = p.get('tp_pct', 0.006)
        qty = p.get('qty', 100)
        day = _within_windows(df_1m.copy(), windows)
        if day.empty:
            return StrategyResult(self.name, p, [])
        day['rsi'] = rsi(day['close'], lb)
        orders: List[Order] = []
        for i in range(1, len(day)):
            row = day.iloc[i]
            if row['rsi'] < os_th:
                orders.append(Order(row['datetime_ist'], float(row['close']), 'LONG', qty, sl, tp))
            if row['rsi'] > ob_th:
                orders.append(Order(row['datetime_ist'], float(row['close']), 'SHORT', qty, sl, tp))
        return StrategyResult(self.name, p, orders)


class MACDCrossStrategy(Strategy):
    def generate(self, df_1m: pd.DataFrame) -> StrategyResult:
        p = self.params
        windows = p.get('windows')
        sl = p.get('sl_pct', 0.003)
        tp = p.get('tp_pct', 0.006)
        qty = p.get('qty', 100)
        day = _within_windows(df_1m.copy(), windows)
        if day.empty:
            return StrategyResult(self.name, p, [])
        macd_line, signal_line, hist = macd(day['close'])
        orders: List[Order] = []
        cross_up = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line > signal_line)
        cross_dn = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line < signal_line)
        for i in range(1, len(day)):
            if cross_up.iloc[i]:
                row = day.iloc[i]
                orders.append(Order(row['datetime_ist'], float(row['close']), 'LONG', qty, sl, tp))
                break
            if cross_dn.iloc[i]:
                row = day.iloc[i]
                orders.append(Order(row['datetime_ist'], float(row['close']), 'SHORT', qty, sl, tp))
                break
        return StrategyResult(self.name, p, orders)


class BollingerReversionStrategy(Strategy):
    def generate(self, df_1m: pd.DataFrame) -> StrategyResult:
        p = self.params
        windows = p.get('windows')
        lb = p.get('period', 20)
        mult = p.get('mult', 2.0)
        sl = p.get('sl_pct', 0.003)
        tp = p.get('tp_pct', 0.006)
        qty = p.get('qty', 100)
        day = _within_windows(df_1m.copy(), windows)
        if day.empty:
            return StrategyResult(self.name, p, [])
        ma = day['close'].rolling(lb).mean()
        sd = day['close'].rolling(lb).std(ddof=0)
        upper = ma + mult * sd
        lower = ma - mult * sd
        orders: List[Order] = []
        for i in range(lb, len(day)):
            row = day.iloc[i]
            if row['close'] < lower.iloc[i]:
                orders.append(Order(row['datetime_ist'], float(row['close']), 'LONG', qty, sl, tp))
            if row['close'] > upper.iloc[i]:
                orders.append(Order(row['datetime_ist'], float(row['close']), 'SHORT', qty, sl, tp))
        return StrategyResult(self.name, p, orders)


REGISTRY = {
    'orb': ORBStrategy,
    'ema_pullback': EMAPullbackStrategy,
    'vwap_reversion': VWAPReversionStrategy,
    'rsi_reversal': RSIReversalStrategy,
    'macd_cross': MACDCrossStrategy,
    'boll_reversion': BollingerReversionStrategy,
}
