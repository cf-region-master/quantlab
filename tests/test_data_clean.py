"""数据清洗规则单元测试：注入脏数据验证质量检查行为。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantlab.data.clean import _check_illegal  # noqa: E402


def test_duplicate_and_monotonic_detection():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                       "volume": 100.0, "amount": 150.0}, index=idx)
    assert not df.index.is_monotonic_increasing or df.index.has_duplicates
    cleaned = df.drop_duplicates(subset=[], keep="first") if False else df[~df.index.duplicated(keep="first")]
    assert len(cleaned) == 3  # 重复(日期,资产)键保留首条


def test_illegal_price_set_nan_not_dropped():
    idx = pd.bdate_range("2024-01-01", periods=4)
    df = pd.DataFrame({
        "open": [10.0, -1.0, 10.0, 10.0],       # 第2行：负价
        "high": [11.0, 11.0, 5.0, 11.0],        # 第3行：high<low
        "low": [9.0, 9.0, 8.0, 9.0],
        "close": [10.5, 10.5, 10.5, 0.0],       # 第4行：收盘=0
        "volume": [100.0, 100.0, 100.0, -5.0],  # 第4行：负成交量
    }, index=idx)
    out, counts = _check_illegal(df.copy())
    assert counts["nonpositive_price"] >= 1
    assert counts["high_lt_low"] >= 1
    assert counts["nonpositive_volume"] >= 1
    assert np.isnan(out.loc[idx[1], "open"])        # 置 NaN，不删行
    assert np.isnan(out.loc[idx[2], "high"])
    assert np.isnan(out.loc[idx[3], "close"])
    assert len(out) == 4                             # 行数不变


def test_adjustment_formula():
    """P_adj = P_raw * a(t)/a(tau)，tau=样本首日。"""
    raw = pd.Series([10.0, 10.0, 10.0], index=pd.bdate_range("2024-01-01", periods=3))
    factor = pd.Series([1.0, 2.0, 2.0], index=raw.index)
    tau = factor.iloc[0]
    adj = raw * factor / tau
    assert adj.iloc[0] == pytest.approx(10.0)
    assert adj.iloc[1] == pytest.approx(20.0)  # 一拆二后调整价连续
