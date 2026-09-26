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
