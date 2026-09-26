"""测试公共夹具：小型合成市场（4 只资产 × 30 个交易日），确定性。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys_path = Path(__file__).resolve().parents[1]


class TinyMarket:
    """构造一个完全可控的小市场，用于回测/因子正确性验证。"""

    def __init__(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        codes = ["A", "B", "C", "D"]
        rng = np.random.default_rng(7)
        rets = pd.DataFrame(rng.normal(0.001, 0.01, size=(30, 4)), index=dates, columns=codes)
        close = 100 * (1 + rets).cumprod()
        open_ = close.shift(1) * (1 + rng.normal(0, 0.002, size=(30, 4)))
        open_.iloc[0] = 100.0
        volume = pd.DataFrame(1_000_000.0, index=dates, columns=codes)
        volume.iloc[10, 1] = 0.0  # B 第 11 天停牌
        self.dates = dates
        self.codes = codes
        self.close_adj = close
        self.open_adj = open_
        self.high_adj = close * 1.01
        self.low_adj = close * 0.99
        self.close_raw = close.copy()
        self.volume = volume
        self.amount = volume * close
        self.adj_factor = pd.DataFrame(1.0, index=dates, columns=codes)
        self.suspended = volume <= 0
        self.benchmark_close = pd.Series(100 * (1 + 0.0002) ** np.arange(30), index=dates)
        self.fields = {"open": open_, "high": self.high_adj, "low": self.low_adj,
                       "close": close, "volume": volume}
        self.quality = {"aggregate": {}}

    def daily_returns(self):
        return self.close_adj / self.close_adj.shift(1) - 1


@pytest.fixture()
def tiny_market():
    return TinyMarket()


@pytest.fixture()
def bt_cfg():
    return {
        "name": "test",
        "sample": {"start": "2024-01-01", "end": "2024-12-31"},
        "signal": {"type": "factor", "factor": "x", "preprocess": []},
        "rebalance": {"freq": "every_h", "h": 5},
        "portfolio": {"top_n": 2, "weighting": "equal_weight"},
        "cost": {"commission_buy": 0.001, "commission_sell": 0.002, "slippage": 0.0},
        "execution": {"price": "open_adj", "timing": "t+1 open"},
        "data_handling": {"suspend": "skip_and_log", "missing_price": "skip_and_log",
                          "delisted_or_frozen": "freeze_last_price"},
        "benchmark": "000300",
    }
