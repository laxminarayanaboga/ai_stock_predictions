"""
Simple data split for TATAMOTORS, TCS, NTPC from raw to training_data and backtest.
- Input: data/raw/{SYMBOL}_NSE_{timeframe}_{20171001_to_20250912}.csv
- Outputs:
  - data/training_data/{SYMBOL}/{timeframe}/{SYMBOL}_NSE_2017-10-01_to_2024-12-31.csv
  - data/backtest/{SYMBOL}/{timeframe}/{SYMBOL}_NSE_2025-01-01_to_2025-05-31.csv

Assumptions:
- timestamp column is epoch seconds (UTC). We'll filter by date boundaries inclusive.
- Columns: timestamp,open,high,low,close,volume
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

SYMBOLS = ["TATAMOTORS", "TCS", "NTPC"]
TIMEFRAMES = {
    "daily": "daily",
    "5min": "5min",
}
RAW_DIR = Path("data/raw")
TRAIN_DIR = Path("data/training_data")
BACKTEST_DIR = Path("data/backtest")

TRAIN_START = pd.Timestamp("2017-10-01", tz="UTC")
TRAIN_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
BT_START = pd.Timestamp("2025-01-01", tz="UTC")
BT_END = pd.Timestamp("2025-05-31 23:59:59", tz="UTC")


def ensure_dirs(symbol: str):
    for base in (TRAIN_DIR, BACKTEST_DIR):
        for tf in TIMEFRAMES.keys():
            (base / symbol / tf).mkdir(parents=True, exist_ok=True)


def read_raw(symbol: str, timeframe: str) -> pd.DataFrame:
    # Find file like SYMBOL_NSE_{timeframe}_20171001_to_20250912.csv
    pattern = f"{symbol}_NSE_{timeframe}_20171001_to_20250912.csv"
    fpath = RAW_DIR / pattern
    if not fpath.exists():
        raise FileNotFoundError(f"Missing raw file: {fpath}")
    df = pd.read_csv(fpath)
    if "timestamp" not in df.columns:
        raise ValueError(f"timestamp column missing in {fpath}")
    # Convert epoch seconds to UTC timestamp index
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.index = ts
    return df


def slice_save(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, out_path: Path):
    ddf = df.loc[(df.index >= start) & (df.index <= end)].copy()
    # Keep original columns order
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ddf.to_csv(out_path, index=False)
    return len(ddf)


def process_symbol(symbol: str):
    ensure_dirs(symbol)
    results = {}
    for tf in TIMEFRAMES.keys():
        df = read_raw(symbol, tf)
        # Training split
        train_out = TRAIN_DIR / symbol / tf / f"{symbol}_NSE_2017-10-01_to_2024-12-31.csv"
        n_train = slice_save(df, TRAIN_START, TRAIN_END, train_out)
        # Backtest split
        bt_out = BACKTEST_DIR / symbol / tf / f"{symbol}_NSE_2025-01-01_to_2025-05-31.csv"
        n_bt = slice_save(df, BT_START, BT_END, bt_out)
        results[tf] = {"train_rows": n_train, "backtest_rows": n_bt}
    return results


def main():
    summary = {}
    for sym in SYMBOLS:
        summary[sym] = process_symbol(sym)
    # Print compact summary
    for sym, res in summary.items():
        print(f"{sym}:")
        for tf, info in res.items():
            print(f"  {tf}: train_rows={info['train_rows']}, backtest_rows={info['backtest_rows']}")


if __name__ == "__main__":
    main()
