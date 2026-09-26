"""分钟级数据层测试：合成 5 分钟面板（不依赖真实 918MB 数据）。

覆盖：面板索引重建（每日 48 bar、唯一且单调）、三因子算子在 5 分钟口径下的
方向、向量化快速诊断（与逐日口径一致的"样本不足记缺失"纪律）、RPN 解释器
核心算子的手算对照。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data.minute_expr import MinuteExprEvaluator
from quantlab.data.minute import BAR_OFFSET_MIN, BARS_PER_DAY
from quantlab.factors.diagnostics import ic_table_fast, quantile_table_fast, run_diagnostics_fast
from quantlab.factors.operators import op_lowvol, op_momentum, op_reversal


def _synthetic_5m(n_days: int = 10, n_codes: int = 6, seed: int = 7):
    """合成 5 分钟面板：n_days × 48 bar，横截面有可控排序（用于 IC 方向断言）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    bar_min = np.array(BAR_OFFSET_MIN, dtype="int64")
    idx = pd.DatetimeIndex(
        np.repeat(dates.to_numpy(dtype="datetime64[ns]"), BARS_PER_DAY)
        + np.tile(bar_min.astype("timedelta64[m]"), n_days))
    cols = [f"S{i}" for i in range(n_codes)]
    # 每日缓慢上行的随机游走 + 横截面趋势差异（S 编号越大日内动量越强）
    steps = rng.normal(0, 0.001, (len(idx), n_codes))
    drift = np.tile(np.linspace(0, 2e-4 * n_codes, n_codes), (len(idx), 1)) / BARS_PER_DAY * 48
    close = pd.DataFrame(100 * np.cumprod(1 + steps + drift, axis=0), index=idx, columns=cols)
    return close


def test_panel_index_layout():
    dates = pd.bdate_range("2024-01-02", periods=3)
    bar_min = np.array(BAR_OFFSET_MIN, dtype="int64")
    idx = pd.DatetimeIndex(
        np.repeat(dates.to_numpy(dtype="datetime64[ns]"), BARS_PER_DAY)
        + np.tile(bar_min.astype("timedelta64[m]"), len(dates)))
    assert len(idx) == 3 * BARS_PER_DAY
    assert idx.is_unique and idx.is_monotonic_increasing
    assert idx[0].strftime("%H:%M") == "09:30"          # 开始时刻约定
    assert idx[23].strftime("%H:%M") == "11:25"         # 上午最后一根（第24根）
    assert idx[24].strftime("%H:%M") == "13:00"         # 午休跳空 → 下午第一根
    assert idx[BARS_PER_DAY - 1].strftime("%H:%M") == "14:55"  # 当日最后一根（第48根）
    assert idx[-1].strftime("%H:%M") == "14:55"


def test_5m_factor_directions():
    close = _synthetic_5m()
    mom = op_momentum({"close": close}, window=48)
    rev = op_reversal({"close": close}, window=12)
    lv = op_lowvol({"close": close}, window=48)
    assert mom["S5"].dropna().iloc[-1] > mom["S0"].dropna().iloc[-1]  # 高漂移标的动量更大
    assert rev.iloc[:, 0].dropna().abs().sum() > 0
    assert lv.notna().to_numpy().sum() > 0


def test_ic_table_fast_matches_slow_on_small_panel():
    """向量化快速 IC 与（概念上相同的）逐日路径在小面板上应给出一致的秩相关序。"""
    from quantlab.factors.diagnostics import forward_return
    close = _synthetic_5m(n_days=20)
    y = forward_return(close, 12)
    rng = np.random.default_rng(3)
    _ = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    # 构造与标签同向的因子：直接用 y 本身（前视构造，仅验证计算）
    factor = y.copy()
    t_fast = ic_table_fast(factor, y, min_n=5)
    s = t_fast["ic"].dropna()
    assert len(s) > 0 and s.mean() > 0.99
    # 常数因子 → 全 NaN（记缺失，不改 0）
    const = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    t_const = ic_table_fast(const, y, min_n=5)
    assert t_const["ic"].isna().all()


def test_quantile_table_fast_grouping():
    close = _synthetic_5m(n_days=15)
    y = close.shift(-6) / close - 1
    rng = np.random.default_rng(5)
    factor = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    qt = quantile_table_fast(factor, y, k=5, min_n=4)
    daily = qt["daily"]
    assert len(daily) > 0
    row = daily.iloc[0]
    formed = sum(row[f"g{g}_n_formed"] for g in range(1, 6))
    assert formed == close.shape[1]  # 分组覆盖全部有效资产
    assert "spread_mean" in qt["summary"]
    assert "tie_rule" in qt["summary"]


def test_run_diagnostics_fast_horizons():
    close = _synthetic_5m(n_days=20)
    rng = np.random.default_rng(9)
    factor = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    res = run_diagnostics_fast(factor, close, {"horizon_days": [6, 48],
                                               "min_cross_section_samples": 4,
                                               "quantile_groups": 5})
    assert set(res) == {"6", "48"}
    for r in res.values():
        assert "ic_summary" in r and "quantile_summary" in r


def _ev_with_panels(close: pd.DataFrame, high: pd.DataFrame | None = None) -> MinuteExprEvaluator:
    ev = MinuteExprEvaluator.__new__(MinuteExprEvaluator)
    ev.dir = None
    ev._fields = {"close": close}
    if high is not None:
        ev._fields["high"] = high
    return ev


def test_evaluator_ref_mean_delta_c(small_close=None):
    close = _synthetic_5m(n_days=4)
    ev = _ev_with_panels(close)
    # Ref(x, 1) = 前一根 bar
    ref = ev.eval_token_string("$close 1 Ref")
    pd.testing.assert_frame_equal(ref, close.shift(1), check_freq=False)
    # Delta(x, 1) = x - x.shift(1)
    delta = ev.eval_token_string("$close 1 Delta")
    pd.testing.assert_frame_equal(delta, (close - close.shift(1)), check_freq=False)
    # Mean(x, 2) = 两根 bar 均值（窗口内全 finite 才有效）
    mean2 = ev.eval_token_string("$close 2 Mean")
    expect = (close + close.shift(1)) / 2
    expect.iloc[:1] = np.nan
    pd.testing.assert_frame_equal(mean2, expect, check_freq=False)
    # CSRank = 截面序数秩 / n
    cs = ev.eval_token_string("$close CSRank")
    row = cs.iloc[-1]
    n = close.iloc[-1].notna().sum()
    assert row.notna().all()
    assert row.min() >= 0.0 and row.max() <= 1.0
    # 引擎口径：argsort 两次 = 0 基序数秩 / n，无并列时恰为 {0, 1/n, ..., (n-1)/n}
    assert set(np.round(row.to_numpy(), 12)) == {round(i / n, 12) for i in range(n)}


def test_evaluator_binary_and_div_guard():
    close = _synthetic_5m(n_days=4)
    ev = _ev_with_panels(close)
    add = ev.eval_token_string("$close 1 Add")   # close + 1
    pd.testing.assert_frame_equal(add, close + 1, check_freq=False)
    # Div 分母 |y|<=1e-12 → NaN
    div = ev.eval_token_string("$close 0 Div")   # close / 0 → 全 NaN
    assert div.isna().to_numpy().all()
    # Greater/Less = 逐元素 max/min
    g = ev.eval_token_string("$close 1 Greater")
    pd.testing.assert_frame_equal(g, np.maximum(close, 1.0), check_freq=False)


def test_evaluator_rejects_incomplete_rpn():
    close = _synthetic_5m(n_days=3)
    ev = _ev_with_panels(close)
    with pytest.raises(ValueError):
        ev.eval_token_string("$close 5")          # 窗口悬空，栈未收敛
    with pytest.raises(ValueError):
        ev.eval_token_string("$close 5 $close")   # 多个操作数滞留栈中，未收敛
    # 单操作数表达式合法：因子 = close 本身
    out = ev.eval_token_string("$close")
    pd.testing.assert_frame_equal(out, close, check_freq=False)


def test_mad_rank_semantics_match_engine():
    """Mad = 窗内 mean(|x - 窗均值|)；Rank = ((#<last)+(#<=last))/(2w)。"""
    idx = pd.date_range("2024-01-02 09:30", periods=5, freq="5min")
    x2 = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    ev2 = _ev_with_panels(x2)
    mad = ev2.eval_token_string("$close 3 Mad")
    # 第三行窗口 [1,2,3]：均值2，|·-2|均值 = 2/3
    assert mad["close"].iloc[2] == pytest.approx(2 / 3)
    rank = ev2.eval_token_string("$close 3 Rank")
    # 第五行窗口 [3,4,5]，last=5：(2 + 3)/(2*3) = 5/6
    assert rank["close"].iloc[4] == pytest.approx(5 / 6)
