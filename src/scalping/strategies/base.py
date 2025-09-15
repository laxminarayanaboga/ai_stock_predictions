from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List
import pandas as pd


@dataclass
class Order:
    time: pd.Timestamp
    price: float
    side: str  # 'LONG' or 'SHORT'
    qty: int
    sl_pct: float
    tp_pct: float


@dataclass
class StrategyResult:
    name: str
    params: Dict[str, Any]
    orders: List[Order] = field(default_factory=list)


class Strategy:
    def __init__(self, name: str, **params: Any) -> None:
        self.name = name
        self.params = params

    def generate(self, df_1m: pd.DataFrame) -> StrategyResult:
        raise NotImplementedError
