"""回测结果基本检查（Project1：输入与配置 / 收益与净值 / 交易成本 / 异常与缺失 / 重复运行）。

设计原则（修正原实现的"自证式检查"问题）：
  - 门禁项必须能与被检查对象【独立】重算，不能是恒等式或硬编码 True：
      * 收益与净值：用【对外发布的 daily_return】反推净值，再用累计乘积独立核对 metrics
      * 交易成本：用配置中的费率【重新计算】每笔费用，与账本逐笔比对
      * 无前视：用【信号面板独立重算】每个形成日的目标持仓，与成交记录比对
      * 现金：直接断言非负（原实现 cap 未含手续费，823/969 天现金为负）
  - 纯计数/留痕类（events、output_hash、毛净归因）不带 pass 字段，不参与门禁。
  - all_pass 由 finalize_checks 统一计算：只统计显式带 pass 的项，
    缺 pass 字段【不再默认通过】（原实现 v.get("pass", True) 使漏加的检查静默通过）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


def finalize_checks(checks: dict[str, Any]) -> dict[str, Any]:
    """统一计算 all_pass：只统计显式声明 pass 的门禁项。必须在所有检查都写入后调用。"""
    gates = {k: bool(v["pass"]) for k, v in checks.items()
             if isinstance(v, dict) and "pass" in v}
    checks["all_pass"] = bool(gates) and all(gates.values())
    checks["gates"] = gates
    return checks


def run_checks(result, market, nav_gross: pd.Series | None = None,
               signal: pd.DataFrame | None = None,
               pool_mask: pd.DataFrame | None = None) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    cfg_cost = (result.config or {}).get("cost", {}) or {}
    # 毛跑（费率置零）时实际扣费为 0，但决策用的费率仍是配置值
    gross_run = bool((result.config or {}).get("_disable_cost"))
    cb = 0.0 if gross_run else float(cfg_cost.get("commission_buy", 0.0))
    cs = 0.0 if gross_run else float(cfg_cost.get("commission_sell", 0.0))

    # 1) 收益与净值一致：用对外发布的 daily_return 反推净值
    nav = result.nav
    ret = result.daily_return
    if len(nav) > 1:
        rebuilt = nav.iloc[0] * (1.0 + ret.iloc[1:].fillna(0.0)).cumprod()
        diff = float((rebuilt - nav.iloc[1:]).abs().max())
    else:
        diff = 0.0
    checks["nav_vs_daily_return"] = {
        "pass": bool(diff < 1e-8), "max_abs_diff": diff, "tolerance": 1e-8,
        "note": "以发布的 daily_return 连乘反推净值，与 nav 比对",
    }

    # 2) 累计收益独立核对：三条来源互证（nav 首尾 / daily_return 连乘 / metrics 输出）
    cum_from_nav = float(nav.iloc[-1] / nav.iloc[0] - 1) if len(nav) > 1 else 0.0
    cum_from_ret = float((1.0 + ret.iloc[1:].fillna(0.0)).prod() - 1) if len(nav) > 1 else 0.0
    cum_from_metrics = float((result.metrics or {}).get("net", {}).get("cumulative_return", np.nan))
    d_ret = abs(cum_from_nav - cum_from_ret)
    d_met = abs(cum_from_nav - cum_from_metrics) if np.isfinite(cum_from_metrics) else np.nan
    ok2 = bool(d_ret < 1e-8 and (not np.isfinite(d_met) or d_met < 1e-8))
    checks["cumulative_return_consistent"] = {
        "pass": ok2,
        "cumulative_return": cum_from_nav,
        "from_daily_return_product": cum_from_ret,
        "from_metrics": None if not np.isfinite(cum_from_metrics) else cum_from_metrics,
        "abs_diff_vs_daily_return": d_ret,
        "abs_diff_vs_metrics": None if not np.isfinite(d_met) else d_met,
        "note": "三条独立来源互证：nav 首尾、daily_return 连乘、metrics 输出",
    }

    # 3) 交易成本：用【配置费率】重算每笔费用，与账本逐笔比对；并校验收盘价与成交量有效
    ledger_sum = float(result.cost_series.sum()) if len(result.cost_series) else 0.0
    trade_sum = float(sum(t["cost"] for t in result.trades))
    fee_bad, exec_bad = [], []
    open_px, vol = market.open_adj, market.volume
    for t in result.trades:
        expect = (cb if t["side"] == "buy" else cs) * float(t["amount"])
        # 金额为货币单位（可达 1e5~1e6），必须用【相对】容差
        if abs(float(t["cost"]) - expect) > 1e-9 * max(1.0, abs(float(t["amount"]))):
            fee_bad.append({"date": t["date"], "code": t["code"], "side": t["side"],
                            "recorded": float(t["cost"]), "recomputed": expect})
        d = pd.Timestamp(t["date"])
        code = t["code"]
        if code not in open_px.columns or d not in open_px.index:
            exec_bad.append(t)
            continue
        px = open_px.at[d, code]
        v = vol.at[d, code] if code in vol.columns else np.nan
        if not (np.isfinite(px) and px > 0 and np.isfinite(v) and v > 0):
            exec_bad.append(t)
    ledger_tol = 1e-9 * max(1.0, abs(ledger_sum), abs(trade_sum))
    checks["cost_ledger"] = {
        "pass": bool(abs(ledger_sum - trade_sum) < ledger_tol and not fee_bad and not exec_bad),
        "ledger_sum": ledger_sum, "trade_sum": trade_sum,
        "abs_diff": abs(ledger_sum - trade_sum), "rel_tolerance": 1e-9,
        "n_fee_mismatch": len(fee_bad), "fee_mismatch_examples": fee_bad[:5],
        "n_invalid_execution": len(exec_bad), "invalid_execution_examples": exec_bad[:5],
        "note": "每笔费用按配置费率 cb/cs 重算后比对（相对容差）；未成交不扣费；毛跑时费率为 0",
    }

    # 4) 权重与现金守恒：invested + cash = nav
    nav_map = {str(d.date()): float(v) for d, v in result.nav.items()}
    worst = 0.0
    for h in result.holdings_history:
        nav_v = nav_map.get(h["date"])
        if not nav_v:
            continue
        worst = max(worst, abs(h["invested"] + h["cash"] - nav_v) / abs(nav_v))
    checks["weights_cash_conservation"] = {
        "pass": bool(worst < 1e-6), "max_relative_residual": worst,
        "note": "invested + cash 与 nav 的一致性（逐日）",
    }

    # 5) 现金非负：原实现下单金额未预留手续费，导致现金为负（无息融资）
    cash_min = float(min((h["cash"] for h in result.holdings_history), default=0.0))
    n_neg = int(sum(1 for h in result.holdings_history if h["cash"] < -1e-12))
    checks["cash_non_negative"] = {
        "pass": bool(cash_min >= -1e-9), "min_cash": cash_min,
        "n_days_negative": n_neg, "n_days": len(result.holdings_history),
        "note": "下单预算须为手续费预留额度，不得使现金为负",
    }

    # 6) 无前视：成交日 = 形成日的次一交易日；且成交构成须与【独立重算】的目标一致    fdates = set(result.formation_dates)
    dates = list(market.dates)
    pos = {d: i for i, d in enumerate(dates)}
    bad_schedule, bad_composition, negative_position = [], [], []
    log_by_exec = {r["exec_date"]: r for r in (result.rebalance_log or [])}
    for lg in (result.rebalance_log or []):
        fi, ei = pos.get(pd.Timestamp(lg["formation_date"])), pos.get(pd.Timestamp(lg["exec_date"]))
        if fi is None or ei is None or ei != fi + 1:
            bad_schedule.append(lg)
    # 独立重放成交序列重建持仓：
    #   (a) 买入的标的必须在当日形成日的目标集合内
    #   (b) 卖出数量不得超过当时持仓（不允许卖空 / 幽灵卖出）
    held: dict[str, float] = {}
    for t in result.trades:
        code, q = t["code"], float(t["shares"])
        lg = log_by_exec.get(t["date"])
        if t["side"] == "buy":
            if lg is None or code not in set(lg["targets"]):
                bad_composition.append({"date": t["date"], "code": code, "side": "buy",
                                        "in_formation_targets": False})
            held[code] = held.get(code, 0.0) + q
        else:
            have = held.get(code, 0.0)
            if q > have + 1e-6:
                negative_position.append({"date": t["date"], "code": code,
                                          "sold": q, "held_before": have})
            held[code] = have - q
    # 独立重算：直接由信号面板重建每个形成日的目标集合，与引擎记录比对
    mismatch_targets = []
    if signal is not None:
        from .engine import select_top
        top_n = int((result.config or {}).get("portfolio", {}).get("top_n", 10))
        for lg in (result.rebalance_log or []):
            fd = pd.Timestamp(lg["formation_date"])
            pool_row = None
            if pool_mask is not None and fd in pool_mask.index:
                pool_row = pool_mask.loc[fd]
            expect = sorted(select_top(signal.loc[fd], pool_row, top_n)) if fd in signal.index else []
            if sorted(lg["targets"]) != expect:
                mismatch_targets.append({"formation_date": lg["formation_date"],
                                         "engine": sorted(lg["targets"]), "recomputed": expect})
    checks["no_lookahead"] = {
        "pass": bool(not bad_schedule and not bad_composition and not negative_position
                     and not mismatch_targets),
        "n_rebalances": len(result.rebalance_log or []),
        "n_bad_schedule": len(bad_schedule), "bad_schedule_examples": bad_schedule[:3],
        "n_bad_composition": len(bad_composition), "bad_composition_examples": bad_composition[:5],
        "n_negative_position": len(negative_position),
        "negative_position_examples": negative_position[:5],
        "n_target_mismatch": len(mismatch_targets), "target_mismatch_examples": mismatch_targets[:3],
        "note": ("成交日必须为形成日的次一交易日；买入标的须在当日目标集合内（该集合由 signal "
                 "面板独立重算比对，含股票池）；卖出数量不得超过重放持仓（不允许卖空）"),
        "signal_independently_recomputed": signal is not None,
    }

    # 6b) 整手约束：买入股数必须为 lot_size 的整数倍（A 股 100 股一手）
    lot = int(((result.config or {}).get("portfolio", {}) or {}).get("lot_size", 0) or 0)
    bad_lot = []
    if lot > 0:
        for t in result.trades:
            if t["side"] != "buy":
                continue
            q = float(t["shares"])
            if abs(q / lot - round(q / lot)) > 1e-6:
                bad_lot.append({"date": t["date"], "code": t["code"], "shares": q, "lot": lot})
    checks["lot_size_conformance"] = {
        "pass": bool(lot <= 0 or not bad_lot),
        "lot_size": lot, "n_violations": len(bad_lot), "examples": bad_lot[:5],
        "note": ("买入股数须为整手倍数（lot_size=0 表示不建模整手约束）"),
    }

    # 7) 异常与缺失留痕（信息项，不设门禁）
    checks["events_logged"] = {
        "n_events": len(result.events), "sample": result.events[:5],
        "note": "停牌/缺价导致的递延与跳过均留痕；纯计数项，不参与 all_pass",
    }

    # 8) 重复运行确定性（信息项：哈希供跨运行比对）
    payload = json.dumps({"nav": [float(v) for v in result.nav],
                          "trades": result.trades}, sort_keys=True)
    checks["output_hash"] = {"sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}

    return finalize_checks(checks)
