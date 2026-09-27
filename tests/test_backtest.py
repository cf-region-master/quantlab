"""回测引擎正确性测试：成本核算、停牌处理、净值一致性、无前视、确定性。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.checks import run_checks
from quantlab.backtest.engine import formation_dates, run_backtest
from quantlab.backtest.metrics import compute_metrics


def _signal_flat(market, value=1.0):
    return pd.DataFrame(value, index=market.dates, columns=market.codes)


def _price_market(open_values, close_values):
    dates = pd.bdate_range("2024-01-01", periods=len(close_values))
    codes = ["A"]
    market = type("RiskMarket", (), {})()
    market.dates = dates
    market.codes = codes
    market.open_adj = pd.DataFrame({"A": open_values}, index=dates)
    market.close_adj = pd.DataFrame({"A": close_values}, index=dates)
    market.volume = pd.DataFrame(1_000_000.0, index=dates, columns=codes)
    market.suspended = market.volume <= 0
    market.benchmark_close = pd.Series(100.0, index=dates)
    return market


def _risk_cfg(bt_cfg, **risk):
    return {
        **bt_cfg,
        "rebalance": {"freq": "every_h", "h": 2},
        "portfolio": {"top_n": 1, "weighting": "equal_weight",
                      "target_exposure": 1.0, "lot_size": 0},
        "risk": {"reentry_days": 20, **risk},
    }


def test_formation_dates_weekly_monthly():
    idx = pd.bdate_range("2024-01-01", periods=60)  # 12 个完整 ISO 周，跨 3 个月
    weekly = formation_dates(idx, "weekly")
    monthly = formation_dates(idx, "monthly")
    pos = {d: i for i, d in enumerate(idx)}
    # 全部形成日都存在次一交易日（引擎保证可执行）
    assert all(pos[d] + 1 < len(idx) for d in weekly + monthly)
    # 最后一个 ISO 周 / 最后一个月份（其样本内末月日=样本末日，无次日可执行）被剔除
    n_weeks = len({(d.isocalendar().year, d.isocalendar().week) for d in idx})
    n_months = len({(d.year, d.month) for d in idx})
    assert len(weekly) == n_weeks - 1
    assert len(monthly) == n_months - 1
    assert all(d in set(idx) for d in weekly)


def test_cost_math_exact(tiny_market, bt_cfg):
    """单次调仓的成本 = c_b*买入额 + c_s*卖出额（按实际成交金额）。"""
    sig = _signal_flat(tiny_market)
    res = run_backtest(sig, tiny_market, bt_cfg, name="t")
    cb = bt_cfg["cost"]["commission_buy"]
    cs = bt_cfg["cost"]["commission_sell"]
    for t in res.trades:
        expect = cb * t["amount"] if t["side"] == "buy" else cs * t["amount"]
        assert t["cost"] == pytest.approx(expect, rel=1e-12)
    ledger = res.cost_series.sum()
    assert ledger == pytest.approx(sum(t["cost"] for t in res.trades), rel=1e-9)


def test_suspended_no_trade_no_fee(tiny_market, bt_cfg):
    """停牌资产不可成交：不产生交易，不扣费，卖出顺延并留痕。
    every_h=5 → 形成日 idx[4]/idx[9]/...；B 在 idx[10] 停牌（执行日）。"""
    dates = tiny_market.dates
    sig = pd.DataFrame(1.0, index=dates, columns=tiny_market.codes)
    sig["B"] = 2.0            # idx[4] 形成日：B 排第一 → 次日买入
    sig.loc[dates[9], "B"] = 0.0  # idx[9] 形成日：B 掉出目标 → 次日（停牌）应卖出但不可成交
    res = run_backtest(sig, tiny_market, bt_cfg, name="t")
    exec1, exec2 = str(dates[5].date()), str(dates[10].date())
    bought = [t for t in res.trades if t["date"] == exec1 and t["code"] == "B" and t["side"] == "buy"]
    assert bought, "B 应在首个执行日买入"
    assert not [t for t in res.trades if t["date"] == exec2 and t["code"] == "B"], "停牌日不得成交"
    deferred = [e for e in res.events if e.get("code") == "B" and e["type"] == "sell_deferred"]
    assert deferred, "卖出顺延必须留痕"
    # 停牌日不得产生 B 的任何费用。
    # 注意：同一天其它资产会按【目标权重】重平衡而正常扣费（这是有意的语义），
    # 因此只断言 B 的费用为 0，而不是整天费用为 0。
    assert not [t for t in res.trades if t["date"] == exec2 and t["code"] == "B" and t["cost"] != 0], \
        "停牌资产不得产生费用"
    # 顺延的卖出意味着 B 在停牌日之后仍应在持仓中
    hh = {h["date"]: h for h in res.holdings_history}
    assert "B" in hh[exec2]["weights"], "卖出被顺延，B 应仍在持仓"


def test_nav_return_consistency_and_checks(tiny_market, bt_cfg):
    sig = _signal_flat(tiny_market)
    res = run_backtest(sig, tiny_market, bt_cfg, name="t")
    gross = run_backtest(sig, tiny_market, bt_cfg, name="t", disable_cost=True)
    res.nav_gross = gross.nav
    rf, A = 0.015, 243
    res.metrics = compute_metrics(res, rf, A)
    checks = run_checks(res, tiny_market, nav_gross=gross.nav)
    assert checks["nav_vs_daily_return"]["pass"]
    assert checks["cost_ledger"]["pass"]
    assert checks["weights_cash_conservation"]["pass"]
    assert checks["no_lookahead"]["pass"]
    assert checks["all_pass"]


def test_net_le_gross_when_costs_positive(tiny_market, bt_cfg):
    sig = _signal_flat(tiny_market)
    net = run_backtest(sig, tiny_market, bt_cfg, name="t")
    gross = run_backtest(sig, tiny_market, bt_cfg, name="t", disable_cost=True)
    # 有成本的最终净值不应高于零成本（成交路径相同）
    assert net.nav.iloc[-1] <= gross.nav.iloc[-1] + 1e-12


def test_deterministic_rerun(tiny_market, bt_cfg):
    sig = _signal_flat(tiny_market)
    a = run_backtest(sig, tiny_market, bt_cfg, name="t")
    b = run_backtest(sig, tiny_market, bt_cfg, name="t")
    assert a.nav.equals(b.nav)
    assert a.trades == b.trades


def test_metrics_formulas(tiny_market, bt_cfg):
    sig = _signal_flat(tiny_market)
    res = run_backtest(sig, tiny_market, bt_cfg, name="t")
    rf, A = 0.02, 243
    m = compute_metrics(res, rf, A)["net"]
    nav = res.nav
    N = len(nav) - 1
    assert m["cumulative_return"] == pytest.approx(nav.iloc[-1] / nav.iloc[0] - 1, rel=1e-12)
    assert m["annualized_return"] == pytest.approx((nav.iloc[-1] / nav.iloc[0]) ** (A / N) - 1, rel=1e-12)
    ret = (nav / nav.shift(1) - 1).iloc[1:]
    assert m["annualized_vol"] == pytest.approx(np.sqrt(A) * ret.std(ddof=1), rel=1e-12)
    runmax = nav.cummax()
    assert m["max_drawdown"] == pytest.approx(float((1 - nav / runmax).max()), rel=1e-12)


def test_stamp_duty_sell_only_and_gate(tiny_market, bt_cfg):
    """印花税只对卖出单边计费：买入 cost=cb·B；卖出 cost=(cs+stamp)·S；检查门禁应 PASS。"""
    bt_cfg["cost"]["stamp_duty_sell"] = 0.001  # 便于断言的测试税率
    sig = _signal_flat(tiny_market)
    res = run_backtest(sig, tiny_market, bt_cfg, name="stamp")
    gross = run_backtest(sig, tiny_market, bt_cfg, name="stamp", disable_cost=True)
    res.nav_gross = gross.nav
    res.metrics = compute_metrics(res, 0.015, 243)
    checks = run_checks(res, tiny_market, nav_gross=gross.nav)
    assert checks["cost_ledger"]["pass"]
    for t in res.trades:
        if t["side"] == "buy":
            assert t["stamp_duty"] == 0.0
            assert t["cost"] == pytest.approx(bt_cfg["cost"]["commission_buy"] * t["amount"], rel=1e-12)
        else:
            assert t["stamp_duty"] == pytest.approx(0.001 * t["amount"], rel=1e-12)
            assert t["cost"] == pytest.approx((bt_cfg["cost"]["commission_sell"] + 0.001) * t["amount"],
                                              rel=1e-12)
    # 净值分解：stamp_duty_total 只来自卖出
    assert res.metrics["cost"]["stamp_duty_total"] == pytest.approx(
        sum(t["stamp_duty"] for t in res.trades), rel=1e-9)
    # 有印花税的最终净值 ≤ 无印花税版本
    assert res.nav.iloc[-1] <= gross.nav.iloc[-1] + 1e-12


# ---------------- 三段边界语义（Phase：AlphaGen 自定义划分 FAILED 修复） ----------------
def test_segment_split_end_boundary_on_weekend():
    """train_end 落在周末/节假日：应回落到 ≤ 该日期的最后一个交易日，而非下一个交易日。

    历史 bug：结束类边界用 searchsorted-left 映射到下一交易日，与 valid_start 撞位，
    顺序校验误判拒绝 —— 按自然月末/年末划分的合理输入全部 FAILED。
    """
    from quantlab.lab.gp_engine import segment_split
    dates = pd.bdate_range("2023-01-02", periods=601)  # 2023-01-02..2025-04-21
    seg = segment_split(dates, purge=5, bounds={
        "train_start": "2023-01-02", "train_end": "2024-06-30",   # 6-30 是周日
        "valid_start": "2024-07-01", "valid_end": "2024-12-31",
        "test_start": "2025-01-01"})
    assert len(seg["train"]) and len(seg["valid"]) and len(seg["test"])
    assert seg["train"][-1] < seg["valid"][0] < seg["valid"][-1] < seg["test"][0]


def test_segment_split_adjacent_trading_day_bounds():
    """train_end=交易日 X、valid_start=次一交易日：必须成立（连续三段的最自然写法）。"""
    from quantlab.lab.gp_engine import segment_split
    dates = pd.bdate_range("2023-01-02", periods=601)
    seg = segment_split(dates, purge=5, bounds={
        "train_end": str(dates[358].date()), "valid_start": str(dates[359].date()),
        "test_start": str(dates[480].date())})
    # train=[0,359) 再 purge 末 5 天 → 最后一天是 dates[353]；valid 从 dates[359] 起
    assert seg["train"][-1] == dates[353]
    assert seg["valid"][0] == dates[359]


def test_segment_split_partial_bounds_still_work():
    from quantlab.lab.gp_engine import segment_split
    dates = pd.bdate_range("2023-01-02", periods=601)
    seg = segment_split(dates, purge=5,
                        bounds={"valid_start": "2024-01-01", "test_start": "2024-10-01"})
    assert len(seg["train"]) and len(seg["valid"]) and len(seg["test"])


# ---------------- 行业中性（组合层分散化约束） ----------------
def test_select_top_industry_cap():
    """行业上限：单行业不得超 cap；无标签资产计入“未知”组同样受限。"""
    from quantlab.backtest.engine import select_top
    sig = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0, "F": 0.5})
    ind = pd.Series({"A": "银行", "B": "银行", "C": "银行", "D": "医药", "E": "医药"})
    # cap=1：每行业最多 1 只 → 银行取 A，医药取 D，然后 B? 不行——B 银行超限 → E? 医药超限 → F(未知)
    picked = select_top(sig, None, 4, industry_row=ind, max_per_industry=1)
    # 三组（银行/医药/未知）各取 1 只 → top4 只有 3 只（组已耗尽，不足额不硬凑）
    assert picked == ["A", "D", "F"]
    # cap=2：银行 A、B，医药 D、E
    picked2 = select_top(sig, None, 4, industry_row=ind, max_per_industry=2)
    assert set(picked2[:4]) == {"A", "B", "D", "E"}
    # F 无标签 → "未知"组同样受限
    picked3 = select_top(sig, None, 6, industry_row=ind, max_per_industry=1)
    assert picked3 == ["A", "D", "F"]


def test_run_backtest_industry_neutral_flags(tiny_market, bt_cfg):
    """启用行业中性后 config 留痕（max_per_industry），且检查仍全部通过。"""
    bt_cfg["portfolio"]["max_per_industry"] = 1
    sig = _signal_flat(tiny_market)
    res = run_backtest(sig, tiny_market, bt_cfg, name="ind")
    assert res.config["portfolio"]["max_per_industry"] == 1
    assert res.config["portfolio"]["top_n"] == 2


# ---------------- 波动率目标（只降不升） ----------------
def test_vol_target_reduces_realized_vol(tiny_market, bt_cfg):
    """高波动市场 + 低波动目标：开启后已实现波动应低于未开启，且向目标靠拢。"""
    # 放大市场波动（把 TinyMarket 的日收益放大 8 倍 → 年化波动极高）
    scale = 8.0
    base = tiny_market.close_adj
    noisy = base.iloc[0] * (1 + (base / base.shift(1) - 1).fillna(0) * scale).cumprod()
    tiny_market.close_adj = noisy
    tiny_market.open_adj = tiny_market.open_adj.iloc[0] * noisy / noisy.iloc[0]
    tiny_market.high_adj = tiny_market.high_adj.iloc[0] * noisy / noisy.iloc[0]
    tiny_market.low_adj = tiny_market.low_adj.iloc[0] * noisy / noisy.iloc[0]
    tiny_market.suspended = tiny_market.volume <= 0

    sig = _signal_flat(tiny_market)
    bt_cfg["trading_days_per_year"] = 243
    plain = run_backtest(sig, tiny_market, bt_cfg, name="plain")

    vt_cfg = {**bt_cfg, "portfolio": {**bt_cfg["portfolio"], "vol_target": 0.10,
                                      "vol_window": 30}}
    vt = run_backtest(sig, tiny_market, vt_cfg, name="vt")
    A = 243
    vol_plain = float((plain.nav / plain.nav.shift(1) - 1).std(ddof=1) * np.sqrt(A))
    vol_vt = float((vt.nav / vt.nav.shift(1) - 1).std(ddof=1) * np.sqrt(A))
    assert vol_vt < vol_plain                     # 波动被压低
    assert vol_vt < vol_plain * 0.9               # 压低幅度显著
    # 仓位缩放留痕
    assert vt.config["portfolio"]["vol_target"] == 0.10


def test_benchmark_metrics_math(tiny_market, bt_cfg):
    """基准相对指标：对构造序列验证 Beta/超额/IR 的闭式性质。"""
    from quantlab.backtest.metrics import _benchmark_metrics

    sig = _signal_flat(tiny_market)
    res = run_backtest(sig, tiny_market, bt_cfg, name="t")
    idx = res.nav.index
    t = np.arange(len(idx))
    # 基准：确定性变动的日收益（非常数 → 方差 > 0，Beta 可估）
    rb = 0.001 * (1.0 + 0.5 * np.sin(t / 4.0))
    bench = pd.Series(np.cumprod(1.0 + rb), index=idx)

    # 恒定收益基准（方差 0）→ Beta 无定义（NaN）
    const_bench = pd.Series(np.cumprod(np.full(len(idx), 1.001)), index=idx)
    m0 = _benchmark_metrics(res.nav, const_bench, 0.0, 243)
    assert not np.isfinite(m0["beta"])

    # 策略与基准完全相同：Beta=1、超额 0、日胜率 0（相等不算赢）、TE=0、IR 无定义
    m1 = _benchmark_metrics(bench, bench, 0.0, 243)
    assert m1["beta"] == pytest.approx(1.0, rel=1e-9)
    assert m1["excess_annual"] == pytest.approx(0.0, abs=1e-9)
    assert m1["daily_win_rate"] == pytest.approx(0.0)
    assert m1["tracking_error_annual"] == pytest.approx(0.0, abs=1e-12)
    assert not np.isfinite(m1["information_ratio"])

    # 策略收益 = 基准收益 + 确定性小额活跃收益（均值>0 且非常数）→ IR>0、TE>0
    rp = rb + 0.0002 + 0.0003 * np.sin(t / 2.0)
    strat = pd.Series(np.cumprod(1.0 + rp), index=idx)
    m2 = _benchmark_metrics(strat, bench, 0.0, 243)
    assert np.isfinite(m2["beta"]) and m2["beta"] > 0
    assert m2["information_ratio"] > 0
    assert m2["excess_annual"] > 0
    assert m2["tracking_error_annual"] > 0


def test_benchmark_metrics_none_when_missing(tiny_market, bt_cfg):
    from quantlab.backtest.metrics import _benchmark_metrics

    sig = _signal_flat(tiny_market)
    res = run_backtest(sig, tiny_market, bt_cfg, name="t")
    assert _benchmark_metrics(res.nav, None, 0.0, 243) is None
    short = pd.Series([1.0, 1.01], index=res.nav.index[:2])
    assert _benchmark_metrics(res.nav, short, 0.0, 243) is None


@pytest.mark.parametrize("risk_key,risk_value,close_values,reason", [
    ("stop_loss_pct", 0.10, [10, 10, 8, 8, 8, 8], "stop_loss"),
    ("take_profit_pct", 0.10, [10, 10, 12, 12, 12, 12], "take_profit"),
    ("ma_windows", [2], [10, 10, 9, 9, 9, 9], "close_below_ma2"),
])
def test_risk_rules_force_full_exit_next_open(bt_cfg, risk_key, risk_value,
                                               close_values, reason):
    # 形成日 idx[1]，idx[2] 开盘买入；idx[2] 收盘触发，idx[3] 开盘全仓卖出。
    market = _price_market([10, 10, 10, 8, 8, 8], close_values)
    cfg = _risk_cfg(bt_cfg, **{risk_key: risk_value})
    sig = _signal_flat(market)
    res = run_backtest(sig, market, cfg, name="risk")

    buy = next(t for t in res.trades if t["side"] == "buy")
    risk_sell = next(t for t in res.trades if t.get("kind") == "risk_exit")
    assert buy["date"] == str(market.dates[2].date())
    assert risk_sell["date"] == str(market.dates[3].date())
    assert risk_sell["shares"] == pytest.approx(buy["shares"])
    assert reason in risk_sell["reasons"]
    assert res.risk_summary["n_triggers"] == 1
    assert res.risk_summary["n_forced_sells"] == 1

    checks = run_checks(res, market, signal=sig)
    assert checks["no_lookahead"]["pass"]
    assert checks["all_pass"]
