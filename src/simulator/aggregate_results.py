"""
Aggregate StrategyRunner outputs across multiple symbol runs.

Features:
- Discover latest run per symbol (or accept explicit run dirs)
- Merge core numeric metrics with advanced analytics per strategy
- Produce overall leaderboards and per-symbol top strategies
- Save CSV and JSON summaries under simulation_results/aggregate_summary_<ts>

Usage examples:
  python src/simulator/aggregate_results.py --latest-per-symbol \
    --symbols RELIANCE TCS NTPC TATAMOTORS

  python src/simulator/aggregate_results.py \
    --run-dirs src/simulator/results/run_RELIANCE_20250915_005121 \
              src/simulator/results/run_TCS_20250915_005436
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


RESULTS_ROOT_DEFAULT = Path("src/simulator/results")
OUTPUT_ROOT_DEFAULT = Path("simulation_results")


RUN_DIR_PATTERN = re.compile(r"^run(?:_(?P<symbol>[A-Z0-9]+))?_(?P<ts>\d{8}_\d{6})$")


ADV_KEYS_CANDIDATES = [
    # Common performance analytics keys (only included if present)
    "sharpe_ratio",
    "sortino_ratio",
    "profit_factor",
    "max_drawdown",
    "max_drawdown_pct",
    "expectancy",
    "mean_pnl",
    "std_pnl",
    "avg_win",
    "avg_loss",
    "best_trade",
    "worst_trade",
    "win_streak_max",
    "loss_streak_max",
    "volatility",
    "kelly_fraction",
    "risk_of_ruin",
    "ulcer_index",
]


@dataclass
class RunInfo:
    symbol: Optional[str]
    timestamp: str
    path: Path


def parse_run_dir(dir_path: Path) -> Optional[RunInfo]:
    name = dir_path.name
    m = RUN_DIR_PATTERN.match(name)
    if not m:
        return None
    symbol = m.group("symbol")
    ts = m.group("ts")
    return RunInfo(symbol=symbol, timestamp=ts, path=dir_path)


def discover_runs_latest_per_symbol(results_root: Path, symbols: Optional[List[str]] = None) -> List[RunInfo]:
    runs: Dict[str, RunInfo] = {}
    for d in results_root.glob("run_*_*"):
        if not d.is_dir():
            continue
        info = parse_run_dir(d)
        if not info:
            continue
        sym = info.symbol or "UNKNOWN"
        if symbols and sym not in symbols:
            continue
        prev = runs.get(sym)
        if not prev or info.timestamp > prev.timestamp:
            runs[sym] = info
    return list(runs.values())


def read_comparison_numeric(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "strategy_comparison_numeric.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing comparison file: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def load_advanced_metrics(run_dir: Path, strategy_id: str) -> Dict:
    adv_path = run_dir / f"{strategy_id}_advanced_metrics.json"
    if adv_path.exists():
        try:
            return json.load(open(adv_path, "r"))
        except Exception:
            return {}
    return {}


def flatten_advanced_metrics(adv: Dict) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    # Many analyzers return a dict at top-level or under a key like 'summary'/'metrics'
    candidates = [adv]
    for key in ("summary", "metrics", "analysis"):
        if isinstance(adv.get(key), dict):
            candidates.append(adv[key])

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        for k in ADV_KEYS_CANDIDATES:
            if k in cand and isinstance(cand[k], (int, float)):
                flat[k] = cand[k]
    return flat


def aggregate_runs(run_infos: List[RunInfo], out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []

    for info in run_infos:
        symbol = info.symbol or "UNKNOWN"
        df = read_comparison_numeric(info.path)
        for _, row in df.iterrows():
            strategy_id = str(row.get("Strategy ID"))
            adv = load_advanced_metrics(info.path, strategy_id)
            adv_flat = flatten_advanced_metrics(adv)

            out_row = {
                "Symbol": symbol,
                "RunDir": str(info.path),
                "RunTimestamp": info.timestamp,
                # Core metrics from numeric comparison
                "Strategy ID": strategy_id,
                "Strategy Name": row.get("Strategy Name"),
                "Total Signals": row.get("Total Signals"),
                "Executed Trades": row.get("Executed Trades"),
                "Execution Rate %": row.get("Execution Rate %"),
                "Win Rate %": row.get("Win Rate %"),
                "Total PnL": row.get("Total PnL ₹"),
                "Avg PnL/Trade": row.get("Avg PnL/Trade ₹"),
                "Avg PnL/Signal": row.get("Avg PnL/Signal ₹"),
            }
            out_row.update(adv_flat)
            rows.append(out_row)

    agg_df = pd.DataFrame(rows)
    # Sort overall by Total PnL desc
    if not agg_df.empty:
        agg_df = agg_df.sort_values(["Total PnL", "Win Rate %", "Executed Trades"], ascending=[False, False, False])

    # Per-symbol top strategies (best by Total PnL)
    if not agg_df.empty:
        top_per_symbol = agg_df.sort_values("Total PnL", ascending=False).groupby("Symbol", as_index=False).head(5)
    else:
        top_per_symbol = pd.DataFrame()

    # Overall top strategies
    top_overall = agg_df.head(20) if not agg_df.empty else pd.DataFrame()

    # Save CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(out_dir / "aggregated_strategies_numeric_and_advanced.csv", index=False)
    top_per_symbol.to_csv(out_dir / "top_strategies_per_symbol.csv", index=False)
    top_overall.to_csv(out_dir / "top_strategies_overall.csv", index=False)

    # Save JSON summary
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs": [
            {
                "symbol": ri.symbol,
                "timestamp": ri.timestamp,
                "path": str(ri.path),
            }
            for ri in run_infos
        ],
        "overall_top": top_overall.head(10).to_dict(orient="records"),
        "per_symbol_top": {
            sym: grp.sort_values("Total PnL", ascending=False).head(5).to_dict(orient="records")
            for sym, grp in (agg_df.groupby("Symbol") if not agg_df.empty else [])
        },
    }
    with open(out_dir / "aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return agg_df, top_per_symbol, top_overall


def main():
    parser = argparse.ArgumentParser(description="Aggregate StrategyRunner results across symbols")
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT_DEFAULT),
        help="Root folder containing run_*_* directories",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        help="Explicit list of run directories to include (overrides discovery when provided)",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional symbol filter (e.g., RELIANCE TCS NTPC TATAMOTORS)",
    )
    parser.add_argument(
        "--latest-per-symbol",
        action="store_true",
        help="When set, discover the latest run per symbol in results-root",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output directory; defaults to simulation_results/aggregate_summary_<timestamp>",
    )

    args = parser.parse_args()

    results_root = Path(args.results_root)
    if args.run_dirs:
        run_infos: List[RunInfo] = []
        for p in args.run_dirs:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"Run dir does not exist: {path}")
            info = parse_run_dir(path)
            if not info:
                # Fallback: allow manual symbol via folder naming heuristics
                info = RunInfo(symbol=None, timestamp=path.name.split("_")[-2] if "_" in path.name else "", path=path)
            run_infos.append(info)
    else:
        # Discovery path
        run_infos = discover_runs_latest_per_symbol(results_root, symbols=args.symbols)

    if not run_infos:
        raise SystemExit("No run directories found to aggregate.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else OUTPUT_ROOT_DEFAULT / f"aggregate_summary_{timestamp}"

    agg_df, top_per_symbol, top_overall = aggregate_runs(run_infos, out_dir)

    print(f"\n✅ Aggregation complete. Outputs written to: {out_dir}")
    print(f"   - aggregated_strategies_numeric_and_advanced.csv  (rows: {len(agg_df)})")
    print(f"   - top_strategies_per_symbol.csv                   (rows: {len(top_per_symbol)})")
    print(f"   - top_strategies_overall.csv                      (rows: {len(top_overall)})")
    print(f"   - aggregate_summary.json")


if __name__ == "__main__":
    main()
