"""绩效指标（Project1 分析模块，扣费后为主）。

公式：
  R_t = V_t/V_{t-1} - 1;  e_t = R_t - r_f_t（rf 为同频率日无风险收益）
  累计收益 = V_N/V_0 - 1;  年化收益 = (V_N/V_0)^(A/N) - 1
  年化波动 = sqrt(A)*std(R_t, ddof=1);  年化 Sharpe = sqrt(A)*mean(e_t)/std(e_t, ddof=1)
  最大回撤 MDD = max_t(1 - V_t/max_{s<=t}V_s)
  Turnover_t = (B_t+S_t)/V^-_t（引擎内计算）；TotalCost = Σ Cost_t（货币单位）
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _core_metrics(nav: pd.Series, rf_annual: float, A: int) -> dict[str, Any]:
    v0 = float(nav.iloc[0])
    vn = float(nav.iloc[-1])
    ret = nav / nav.shift(1) - 1
    ret = ret.iloc[1:]
    N = len(ret)
    cum = vn / v0 - 1
    ann = (vn / v0) ** (A / N) - 1 if N > 0 else np.nan
    vol = float(np.sqrt(A) * ret.std(ddof=1)) if N > 1 else np.nan
    rf_d = rf_annual / A
    e = ret - rf_d
    if N > 1 and e.std(ddof=1) > 0:
        sharpe = float(np.sqrt(A) * e.mean() / e.std(ddof=1))
    else:
        sharpe = np.nan
    runmax = nav.cummax()
    dd = 1 - nav / runmax
    mdd = float(dd.max()) if len(dd) else np.nan
    mdd_date = str(dd.idxmax().date()) if len(dd) and np.isfinite(dd.max()) else None
    return {
        "v0": v0, "v_final": vn, "n_days": N,
        "cumulative_return": float(cum),
        "annualized_return": float(ann),
        "annualized_vol": vol,
        "annualized_sharpe": sharpe,
        "max_drawdown": mdd,
        "max_drawdown_date": mdd_date,
        "calmar": float(ann / mdd) if mdd and mdd > 0 else np.nan,
        "daily_return_mean": float(ret.mean()) if N else np.nan,
    }


def compute_metrics(result, rf_annual: float, A: int) -> dict[str, Any]:
    net = _core_metrics(result.nav, rf_annual, A)
    gross = _core_metrics(result.nav_gross, rf_annual, A) if result.nav_gross is not None else {}

    total_cost = float(result.cost_series.sum()) if len(result.cost_series) else 0.0
    turnover_sum = float(result.turnover.sum()) if len(result.turnover) else 0.0
    holds = [h["n_holdings"] for h in result.holdings_history]
    max_w = 0.0
    for h in result.holdings_history:
        if h["weights"]:
            max_w = max(max_w, max(h["weights"].values()))
    buys = sum(1 for t in result.trades if t["side"] == "buy")
    sells = sum(1 for t in result.trades if t["side"] == "sell")

    return {
        "net": net,
        "gross": gross,
        "gross_vs_net": {
            "annualized_return_diff": (net["annualized_return"] - gross.get("annualized_return", np.nan)),
            "note": "毛/净为同一成交路径（决策不依赖成本），差异全部来自交易成本",
        },
        "cost": {
            "total_cost_currency": total_cost,
            "total_cost_pct_of_avg_nav": total_cost / float(result.nav.mean()),
            "commission_buy": result.config["cost"]["commission_buy"],
            "commission_sell": result.config["cost"]["commission_sell"],
            "stamp_duty_sell": float(result.config["cost"].get("stamp_duty_sell", 0.0)),
            # 成本分项合计（从逐笔账本汇总；毛跑时全为 0）
            "commission_total": float(sum(t.get("commission", t["cost"]) for t in result.trades)),
            "stamp_duty_total": float(sum(t.get("stamp_duty", 0.0) for t in result.trades)),
            "slippage": result.config["cost"].get("slippage", 0.0),
        },
        "turnover": {
            "sum": turnover_sum,
            "mean_per_rebalance": float(result.turnover.mean()) if len(result.turnover) else 0.0,
            # n_formation_dates 为日历上的全部形成日；n_rebalances 为实际发生成交的调仓次数。
            # 原实现用 len(formation_dates) 充当 n_rebalances（203 vs 实际 179），
            # 与 mean_per_rebalance（按换手记录求均值）自相矛盾。
            "n_formation_dates": len(result.formation_dates),
            "n_rebalances": int(len(result.turnover)),
        },
        "portfolio": {
            "avg_holdings": float(np.mean(holds)) if holds else 0.0,
            "max_single_weight": float(max_w),
            "n_buy_trades": buys,
            "n_sell_trades": sells,
            "n_trades": len(result.trades),
            "n_events": len(result.events),
        },
    }
