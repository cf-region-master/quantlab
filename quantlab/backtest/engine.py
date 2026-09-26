"""回测引擎（Project1 回测模块 · 组合执行版）。

口径（全部在结果与报告中显式声明）：
  - 信号：t 日收盘的因子/信号值（预处理后，分数越高越优）
  - 目标组合：t 日收盘在【股票池】内按分数取 top_n，按 weighting 规则分配目标权重，
    目标仓位合计 = target_exposure（默认 0.98，留现金缓冲；缓冲是常数，不随费率变化）
  - 目标股数：shares_i = nav_prev × w_i / (price_i × (1 + fee_rate_i))，再按整手向下取整
    —— 费率在【权重→股数】换算时计入，因此下单决策不依赖累计现金；
       target_exposure 的现金缓冲吸收执行价漂移，保证毛/净两条路径的成交序列一致
  - 成交：t+1 交易日开盘价；先卖后买（卖出释放资金）；停牌/缺价当日不可成交，顺延不补并留痕
  - 费用：Cost = c_b·B + c_s·S，按实际成交金额；未成交不扣费
  - 换手：Turnover_t = (B_t + S_t) / V_prev_t，V_prev_t 为交易前组合总值（调整价名义金额）
  - 估值：收盘价逐日盯市；无价日按最后有效收盘估值（冻结）并记事件
  - 毛收益对照：同一成交路径下将费率置零（决策用的费率仍取配置值，故成交序列一致）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def formation_dates(index: pd.DatetimeIndex, freq: str, h: int | None = None) -> list[pd.Timestamp]:
    """调仓形成日：weekly=每周最后交易日；monthly=每月最后交易日；every_h=每 h 个交易日。"""
    idx = list(index)
    if freq == "weekly":
        keys: dict[tuple, pd.Timestamp] = {}
        for d in idx:
            keys[(d.isocalendar().year, d.isocalendar().week)] = d
        dates = sorted(keys.values())
    elif freq == "monthly":
        keys = {}
        for d in idx:
            keys[(d.year, d.month)] = d
        dates = sorted(keys.values())
    elif freq == "every_h":
        if not h or h <= 0:
            raise ValueError("every_h 需要正整数 h")
        dates = idx[h - 1::h]
    else:
        raise ValueError(f"未知调仓频率: {freq}")
    # 形成日次日必须仍是交易日（否则无法执行）
    pos = {d: i for i, d in enumerate(idx)}
    return [d for d in dates if pos[d] + 1 < len(idx)]


def select_top(signal_row: pd.Series, pool_row: pd.Series | None, top_n: int,
               industry_row: pd.Series | None = None,
               max_per_industry: int | None = None) -> list[str]:
    """目标持仓：股票池内按分数降序取 top_n，并列时按资产代码升序（确定性）。

    行业中性（可选）：industry_row 提供当日各资产的行业标签，max_per_industry
    为单行业持仓上限 —— 贪心选取时超限跳过。无标签的资产归入 "未知" 组，
    同样受限（不留后门）；行业数据缺失（industry_row=None）则退化为普通选股。
    """
    s = signal_row.dropna()
    s = s[np.isfinite(s)]
    if pool_row is not None:
        allowed = pool_row.reindex(s.index).fillna(False).astype(bool)
        s = s[allowed]
    if s.empty:
        return []
    order = sorted(s.index, key=lambda c: (-s[c], str(c)))
    if industry_row is None or not max_per_industry or max_per_industry <= 0:
        return order[:top_n]
    counts: dict[str, int] = {}
    picked: list[str] = []
    for c in order:
        lab = industry_row.get(c)
        g = str(lab) if (lab is not None and lab == lab and str(lab) != "") else "未知"
        if counts.get(g, 0) >= max_per_industry:
            continue
        counts[g] = counts.get(g, 0) + 1
        picked.append(c)
        if len(picked) >= top_n:
            break
    return picked


def target_weights(selected: list[str], scores: pd.Series, weighting: str,
                   exposure: float) -> dict[str, float]:
    """把选中的标的与分数转成目标权重（合计 = exposure）。

    weighting:
      equal_weight 等权
      signal_prop  按分数（平移为正后）比例分配
      rank_prop    按名次线性分配（名次越高权重越大）
    """
    n = len(selected)
    if n == 0:
        return {}
    if weighting == "equal_weight":
        raw = {c: 1.0 for c in selected}
    elif weighting == "signal_prop":
        v = scores.reindex(selected).astype(float)
        lo = float(v.min())
        shifted = (v - lo) + 1e-9          # 平移为正，保留相对差距
        tot = float(shifted.sum())
        raw = {c: (float(shifted[c]) / tot if tot > 0 else 1.0 / n) for c in selected}
    elif weighting == "rank_prop":
        # 分数降序 → 名次权重 n, n-1, ..., 1
        ranked = sorted(selected, key=lambda c: (-float(scores[c]), str(c)))
        w = {c: float(n - i) for i, c in enumerate(ranked)}
        tot = float(sum(w.values()))
        raw = {c: w[c] / tot for c in selected}
    else:
        raise ValueError(f"未知权重规则: {weighting}")
    tot = float(sum(raw.values())) or 1.0
    return {c: raw[c] / tot * exposure for c in selected}


@dataclass
class BacktestResult:
    name: str
    config: dict[str, Any]
    nav: pd.Series
    daily_return: pd.Series
    trades: list[dict]
    holdings_history: list[dict]
    turnover: pd.Series
    cost_series: pd.Series
    events: list[dict]
    formation_dates: list[pd.Timestamp]
    rebalance_log: list[dict] = field(default_factory=list)  # 每次调仓的形成日/成交日/目标
    nav_gross: pd.Series | None = None
    benchmark_nav: pd.Series | None = None
    metrics: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "config": self.config,
            "nav": {"index": [str(d.date()) for d in self.nav.index],
                    "values": [float(v) for v in self.nav]},
            "daily_return": {"index": [str(d.date()) for d in self.daily_return.index],
                             "values": [None if not np.isfinite(v) else float(v)
                                        for v in self.daily_return]},
            "trades": self.trades,
            "turnover": {"index": [str(d.date()) for d in self.turnover.index],
                         "values": [float(v) for v in self.turnover]},
            "cost_series": {"index": [str(d.date()) for d in self.cost_series.index],
                            "values": [float(v) for v in self.cost_series]},
            "events": self.events[:500],
            "n_events": len(self.events),
            "metrics": self.metrics,
            "checks": self.checks,
            "formation_dates": [str(d.date()) for d in self.formation_dates],
            "n_rebalances_logged": len(self.rebalance_log),
        }
        if self.nav_gross is not None:
            out["nav_gross"] = {"index": [str(d.date()) for d in self.nav_gross.index],
                                "values": [float(v) for v in self.nav_gross]}
        if self.benchmark_nav is not None:
            out["benchmark"] = {"index": [str(d.date()) for d in self.benchmark_nav.index],
                                "values": [float(v) for v in self.benchmark_nav]}
        return out


def run_backtest(signal: pd.DataFrame, market, bt_cfg: dict, name: str = "bt",
                 disable_cost: bool = False, pool_mask: pd.DataFrame | None = None,
                 industry_labels: pd.DataFrame | None = None
                 ) -> BacktestResult:
    """执行一次模拟成交。

    disable_cost=True 时【只将对费用置零】，目标股数仍按配置费率换算，
    因此毛/净两次运行的成交序列完全一致（可宣称"同一成交路径归因"）。
    """
    cost_cfg = bt_cfg["cost"]
    cb_cfg = float(cost_cfg["commission_buy"])
    cs_cfg = float(cost_cfg["commission_sell"])
    # 印花税：卖出单边（A股 2023-08-28 起 0.05%，此前 0.1%；时间变动未建模，见 README 局限）
    stamp_cfg = float(cost_cfg.get("stamp_duty_sell", 0.0))
    slip = 0.0 if disable_cost else float(cost_cfg.get("slippage", 0.0))
    # 决策用费率（始终取配置值，保证毛/净决策一致）与实际扣费用率（毛跑时为 0）
    cb_dec, cs_dec = cb_cfg, cs_cfg
    cb_chg, cs_chg = (0.0, 0.0) if disable_cost else (cb_cfg, cs_cfg)
    stamp_chg = 0.0 if disable_cost else stamp_cfg

    port = bt_cfg["portfolio"]
    top_n = int(port["top_n"])
    max_per_industry = port.get("max_per_industry")
    weighting = str(port.get("weighting", "equal_weight"))
    exposure = float(port.get("target_exposure", 0.98))
    lot = int(port.get("lot_size", 100) or 0)
    freq = bt_cfg["rebalance"]["freq"]
    h = bt_cfg["rebalance"].get("h")

    open_px, close_px, vol, susp = market.open_adj, market.close_adj, market.volume, market.suspended
    # 成交/估值日历必须限定在 bt_cfg["sample"] 声明的区间内。
    # 历史 bug：这里直接用 market.dates（全样本），使 sample 完全不生效 ——
    #   1) 请求 [2023-01-01, 2025-12-31] 实际却从 2022-01-04 开始模拟；
    #   2) 面板覆盖不到的早期形成日 selected=[]，等于把组合整个清仓持有现金，
    #      让净值曲线前段失真（实测 25% 的调仓目标是空的）。
    dates = market.dates
    sample = bt_cfg.get("sample") or {}
    if sample.get("start"):
        dates = dates[dates >= pd.Timestamp(sample["start"])]
    if sample.get("end"):
        dates = dates[dates <= pd.Timestamp(sample["end"])]
    if len(dates) < 2:
        raise ValueError(
            f"回测区间 {sample.get('start')} ~ {sample.get('end')} 内不足 2 个交易日，无法模拟")
    fdates = formation_dates(dates, freq, h)
    fset = set(fdates)

    cash = float(bt_cfg.get("initial_cash", 1_000_000.0))
    shares: dict[str, float] = {}
    last_close: dict[str, float] = {}
    nav_rows, trades, events, hold_rows = [], [], [], []
    turnover_rows, cost_rows, rebalance_rows = [], [], []

    def mark_to_market(d) -> float:
        v = cash
        for code, sh in shares.items():
            px = close_px.at[d, code] if code in close_px.columns else np.nan
            if np.isfinite(px):
                last_close[code] = float(px)
            v += sh * last_close.get(code, np.nan)
        return v

    for i, d in enumerate(dates):
        # ---- 当日可执行的调仓：昨日（形成日 t）收盘信号 → 今日开盘成交 ----
        if i > 0 and dates[i - 1] in fset:
            t = dates[i - 1]
            exec_day = d
            nav_prev = cash + sum(sh * last_close.get(c, np.nan) for c, sh in shares.items())

            pool_row = None
            if pool_mask is not None and t in pool_mask.index:
                pool_row = pool_mask.loc[t]
            ind_row = None
            if industry_labels is not None and t in industry_labels.index:
                ind_row = industry_labels.loc[t]
            selected = (select_top(signal.loc[t], pool_row, top_n,
                                   industry_row=ind_row,
                                   max_per_industry=max_per_industry)
                        if t in signal.index else [])
            w = target_weights(selected, signal.loc[t], weighting, exposure) if selected else {}
            rebalance_rows.append({"formation_date": str(t.date()),
                                   "exec_date": str(exec_day.date()),
                                   "targets": list(selected),
                                   "weights": {c: round(float(v), 8) for c, v in w.items()}})

            def tradable(code: str) -> bool:
                if code not in open_px.columns:
                    return False
                px = open_px.at[exec_day, code]
                is_susp = bool(susp.at[exec_day, code]) if code in susp.columns else True
                return bool(np.isfinite(px) and px > 0 and not is_susp)

            # ---------- 1) 计算目标股数（费率在权重→股数换算时计入） ----------
            target: dict[str, float] = {}
            for code, wi in w.items():
                if not tradable(code):
                    continue
                px = float(open_px.at[exec_day, code])
                notional = nav_prev * wi
                sh = notional / (px * (1.0 + cb_dec))     # 费率内计：花掉的是 notional
                if lot > 0:
                    sh = float(np.floor(sh / lot) * lot)  # 整手向下取整
                if sh > 0:
                    target[code] = sh

            # ---------- 2) 先卖：把持仓调至目标股数 ----------
            # 含两类：目标外的清仓，以及目标内的【减仓】（真正的按目标权重重平衡）。
            # 只清仓不减仓会导致买入缺钱 → 触发按现金缩量 → 成交序列依赖手续费，
            # 毛/净两条路径因此分叉。
            sell_amount_total = 0.0
            for code in list(shares.keys()):
                tgt = target.get(code, 0.0)
                cur = shares[code]
                if cur <= tgt + 1e-9:
                    continue
                if not tradable(code):
                    events.append({"date": str(exec_day.date()), "type": "sell_deferred",
                                   "code": code, "reason": "suspended_or_missing_open"})
                    continue
                qty = cur - tgt
                px = float(open_px.at[exec_day, code])
                px_exec = px * (1 - slip)
                amount = qty * px_exec
                commission = cs_chg * amount
                stamp = stamp_chg * amount
                fee = commission + stamp
                cash += amount - fee
                sell_amount_total += amount
                trades.append({"date": str(exec_day.date()), "code": code, "side": "sell",
                               "shares": float(qty), "price": px_exec,
                               "amount": float(amount), "cost": float(fee),
                               "commission": float(commission), "stamp_duty": float(stamp)})
                cost_rows.append({"date": exec_day, "cost": fee})
                if tgt <= 1e-9:
                    del shares[code]
                else:
                    shares[code] = cur - qty

            # ---------- 3) 再买：按目标股数买入，逐笔模拟真实现金流 ----------
            buy_amount_total = 0.0
            need = {}
            for code, tgt in target.items():
                cur = shares.get(code, 0.0)
                if tgt > cur + 1e-12:
                    need[code] = tgt - cur
            # 预估所需现金（含手续费），不足则按比例缩量并留痕
            need_cash = 0.0
            for code, qty in need.items():
                px = float(open_px.at[exec_day, code]) * (1 + slip)
                need_cash += qty * px * (1.0 + cb_chg)
            scale = 1.0
            if need_cash > cash and need_cash > 0:
                scale = max(0.0, cash / need_cash)
                events.append({"date": str(exec_day.date()), "type": "buy_scaled",
                               "reason": "insufficient_cash", "scale": round(float(scale), 6),
                               "need_cash": float(need_cash), "cash": float(cash)})
            for code, qty in need.items():
                if scale <= 0:
                    events.append({"date": str(exec_day.date()), "type": "buy_skipped",
                                   "code": code, "reason": "insufficient_cash"})
                    continue
                q = qty * scale
                if lot > 0:
                    q = float(np.floor(q / lot) * lot)
                if q <= 0:
                    events.append({"date": str(exec_day.date()), "type": "buy_skipped",
                                   "code": code, "reason": "below_one_lot"})
                    continue
                px = float(open_px.at[exec_day, code])
                px_exec = px * (1 + slip)
                amount = q * px_exec
                commission = cb_chg * amount
                fee = commission  # 买入无印花税
                cash -= amount + fee
                shares[code] = shares.get(code, 0.0) + q
                last_close.setdefault(code, px)
                buy_amount_total += amount
                trades.append({"date": str(exec_day.date()), "code": code, "side": "buy",
                               "shares": float(q), "price": px_exec,
                               "amount": float(amount), "cost": float(fee),
                               "commission": float(commission), "stamp_duty": 0.0})
                cost_rows.append({"date": exec_day, "cost": fee})

            if buy_amount_total + sell_amount_total > 0:
                v_prev = nav_prev if nav_prev > 0 else np.nan
                turnover_rows.append({"date": exec_day,
                                      "turnover": (buy_amount_total + sell_amount_total) / v_prev})

        # ---- 逐日盯市 ----
        nav = mark_to_market(d)
        if not np.isfinite(nav):
            nav = cash
            events.append({"date": str(d.date()), "type": "nav_fallback_cash"})
        nav_rows.append({"date": d, "nav": nav})
        hold_rows.append({
            "date": str(d.date()),
            "n_holdings": len(shares),
            "invested": float(sum(sh * last_close.get(c, 0.0) for c, sh in shares.items())),
            "cash": float(cash),
            "weights": {c: float(sh * last_close.get(c, 0.0) / nav)
                        for c, sh in shares.items() if nav > 0},
        })

    nav_s = pd.DataFrame(nav_rows).set_index("date")["nav"]
    ret_s = nav_s / nav_s.shift(1) - 1
    turnover_s = (pd.DataFrame(turnover_rows).set_index("date")["turnover"]
                  if turnover_rows else pd.Series(dtype="float64"))
    cost_s = (pd.DataFrame(cost_rows).set_index("date")["cost"]
              if cost_rows else pd.Series(dtype="float64"))

    bench = market.benchmark_close.reindex(nav_s.index)
    # 基准归一到策略起始净值（同为货币口径），便于同轴对比；
    # 仅做比例缩放，不改变收益率形状
    bench_nav = (bench / bench.iloc[0] * float(nav_s.iloc[0])
                 if len(bench) and np.isfinite(bench.iloc[0]) else None)

    result = BacktestResult(
        name=name, config=json.loads(json.dumps(
            {**bt_cfg, "_disable_cost": bool(disable_cost)}, default=str)),
        nav=nav_s, daily_return=ret_s, trades=trades,
        holdings_history=hold_rows, turnover=turnover_s, cost_series=cost_s,
        events=events, formation_dates=fdates, rebalance_log=rebalance_rows,
        benchmark_nav=bench_nav,
    )
    return result
