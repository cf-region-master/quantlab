"""分层净值：**因子计算的附属统计量**（随因子计算自动产出，不需要单独触发）。

定位与纪律
----------
这是"这个因子值分层后长什么样"的**描述性附属统计量**，因此刻意做成轻量实现：

  - 评分 = 因子预处理后的取值；按【截面】升序切成 K 层，第 1 层最低、第 K 层最高
  - 每层是一个**等权组合**，每 h 个交易日调仓一次（h 取该因子的持有期）
  - 期内买入持有（权重自然漂移），期界按真实权重变动计算换手并**扣双边费率**
  - 停牌/缺价按最后有效收盘价冻结估值（与回测引擎同一约定，不制造价格）

明确不建模（因此**不宣称可执行收益**）：
  整手约束、t+1 开盘成交、现金约束、涨跌停 —— 这些在 `/backtests` 的完整回测里建模。
  本统计量的用途是"快速看清因子的分层单调性与方向"，不是替代回测。

相比"各组平均 h 日收益柱状图"的关键改进：柱状图用的是**重叠标签的日均收益**，
不复利、不可执行、也无法跨层直接比较；这里是**分层组合的复利净值曲线**。
"""
from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd

from ..backtest.engine import formation_dates

CONVENTION = ("等权分层组合、每 h 个交易日调仓、期内买入持有、按真实权重变动扣双边费率；"
              "未建模整手约束/次日开盘成交/现金约束（可执行版本见回测页）")


@contextlib.contextmanager
def _quiet():
    """屏蔽「全 NaN 成员」触发的 RuntimeWarning（errstate 挡不住 nanmean 的 warn）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _max_drawdown(nav: np.ndarray) -> float:
    if nav.size == 0:
        return np.nan
    peak = np.maximum.accumulate(nav)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = nav / peak - 1.0
    return float(np.nanmin(dd)) if np.isfinite(dd).any() else np.nan


def _core(nav: pd.Series, A: int, rf: float) -> dict:
    v = nav.to_numpy(dtype="float64")
    v = v[np.isfinite(v)]
    if v.size < 2 or v[0] <= 0:
        return {"total_return": None, "annualized_return": None,
                "annualized_vol": None, "sharpe": None, "max_drawdown": None}
    daily = v[1:] / v[:-1] - 1.0
    n = daily.size
    total = float(v[-1] / v[0] - 1.0)
    ann = float((v[-1] / v[0]) ** (A / n) - 1.0) if n > 0 else None
    vol = float(np.nanstd(daily, ddof=1) * np.sqrt(A)) if n > 1 else None
    sharpe = (float((ann - rf) / vol)
              if (ann is not None and vol and np.isfinite(vol) and vol > 1e-12) else None)
    return {"total_return": total, "annualized_return": ann,
            "annualized_vol": vol, "sharpe": sharpe, "max_drawdown": _max_drawdown(v)}


def _period_rel(px: np.ndarray, i0: int, i1: int, cols: np.ndarray) -> np.ndarray:
    """期内组合相对价值（等权、买入持有）= 成员相对价格的平均。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        base = px[i0][cols]
        mat = px[i0:i1 + 1][:, cols] / base
    with _quiet(), np.errstate(all="ignore"):
        return np.nanmean(mat, axis=1)


def _drift(px: np.ndarray, i0: int, i1: int, cols: np.ndarray) -> dict[int, float]:
    """把期内等权持仓按价格漂移到期末，返回归一化权重。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        base, end = px[i0][cols], px[i1][cols]
        growth = np.where(np.isfinite(base) & (base > 0) & np.isfinite(end),
                          end / base, np.nan)
    gw = np.where(np.isfinite(growth), growth, 1.0)
    tot = float(np.sum(gw)) or 1.0
    return {int(j): float(w / tot) for j, w in zip(cols, gw)}


def layered_nav(panel: pd.DataFrame, market, *, groups: int = 5, h: int = 5,
                pool_mask: pd.DataFrame | None = None,
                commission_buy: float = 0.0, commission_sell: float = 0.0,
                trading_days: int = 252, rf_annual: float = 0.0) -> dict:
    """计算 K 层净值曲线。panel = 因子（预处理后）评分宽表 date×code。"""
    groups = max(2, int(groups))
    h = max(1, int(h))
    idx = panel.index
    if len(idx) < h + 2:
        raise ValueError(f"区间内只有 {len(idx)} 个交易日，不足以做 h={h} 的分层")

    # 估值价：停牌/缺价按最后有效收盘冻结（与回测引擎同口径）
    px = market.close_adj.reindex(index=idx, columns=panel.columns).ffill()
    px_arr = px.to_numpy(dtype="float64")

    arr = panel.to_numpy(dtype="float64")
    ok = np.isfinite(arr) & np.isfinite(px_arr)
    if pool_mask is not None:
        aligned = pool_mask.reindex(index=idx, columns=panel.columns)
        ok = ok & aligned.fillna(False).to_numpy(dtype=bool)

    fdates = formation_dates(idx, "every_h", h)
    if len(fdates) < 2:
        raise ValueError(f"区间内只有 {len(fdates)} 个调仓日，不足以做分层净值")
    pos = {d: i for i, d in enumerate(idx)}

    # 每个调仓日的层归属 {层号: 成员列位置}（升序切分：第 1 层因子值最低）
    membership: list[tuple[pd.Timestamp, dict[int, np.ndarray]]] = []
    for t in fdates:
        i = pos[t]
        cols = np.where(ok[i])[0]
        if cols.size < groups:
            membership.append((t, {}))
            continue
        order = cols[np.argsort(arr[i][cols], kind="stable")]
        membership.append((t, {k: b for k, b in enumerate(np.array_split(order, groups), 1)
                              if b.size}))

    cost_rate = float(commission_buy) + float(commission_sell)
    out_layers: dict[str, dict] = {}

    for k in range(1, groups + 1):
        nav = np.full(len(idx), np.nan)
        cur, drift, holds = 1.0, None, []
        turnover_sum, n_reb = 0.0, 0

        for si, (t, mem) in enumerate(membership):
            i0 = pos[t]
            i1 = (pos[membership[si + 1][0]] if si + 1 < len(membership) else len(idx) - 1)
            if i1 <= i0:
                continue
            members = mem.get(k)

            if members is None or members.size == 0:
                # 样本不足：沿用上一期持仓（不清仓，也不计换手）；无持仓则持平
                if drift:
                    cols = np.array(sorted(drift), dtype=int)
                    rel = _period_rel(px_arr, i0, i1, cols)
                    nav[i0:i1 + 1] = cur * rel
                    drift = _drift(px_arr, i0, i1, cols)
                    holds.append(int(cols.size))
                else:
                    nav[i0:i1 + 1] = cur
                cur = float(nav[i1]) if np.isfinite(nav[i1]) else cur
                continue

            w_new = {int(j): 1.0 / members.size for j in members}
            if drift is None:
                turnover = 1.0                                  # 建仓
            else:
                keys = set(drift) | set(w_new)
                turnover = 0.5 * sum(abs(w_new.get(j, 0.0) - drift.get(j, 0.0)) for j in keys)
            turnover_sum += turnover
            n_reb += 1
            cur *= max(0.0, 1.0 - turnover * cost_rate)         # 调仓日一次性扣费

            rel = _period_rel(px_arr, i0, i1, members)
            nav[i0:i1 + 1] = cur * rel
            drift = _drift(px_arr, i0, i1, members)
            holds.append(int(members.size))
            cur = float(nav[i1]) if np.isfinite(nav[i1]) else cur

        nav_s = pd.Series(nav, index=idx)
        base = nav_s.dropna()
        if len(base) and base.iloc[0] > 0:
            nav_s = nav_s / base.iloc[0]                        # 首日归一
        m = _core(nav_s, int(trading_days), float(rf_annual))
        out_layers[str(k)] = {
            "nav": [None if not np.isfinite(v) else round(float(v), 6) for v in nav_s],
            **{kk: (None if vv is None or not np.isfinite(vv) else round(float(vv), 6))
               for kk, vv in m.items()},
            "turnover_sum": round(float(turnover_sum), 4),
            "n_rebalances": int(n_reb),
            "avg_holdings": round(float(np.mean(holds)), 2) if holds else 0.0,
        }

    hi, lo = out_layers.get(str(groups)), out_layers.get("1")
    spread: dict = {}
    if hi and lo:
        hs = [v for v in hi["nav"] if v is not None]
        ls = [v for v in lo["nav"] if v is not None]
        if hs and ls and ls[-1]:
            spread = {"final_high": hs[-1], "final_low": ls[-1],
                      "final_ratio": round(hs[-1] / ls[-1], 6)}
            if hi["annualized_return"] is not None and lo["annualized_return"] is not None:
                spread["high_minus_low_annualized"] = round(
                    hi["annualized_return"] - lo["annualized_return"], 6)

    return {"h": h, "groups": groups,
            "index": [str(d.date()) for d in idx],
            "layers": out_layers,
            "long_short_spread": spread,
            "cost": {"commission_buy": float(commission_buy),
                     "commission_sell": float(commission_sell)},
            "convention": CONVENTION}
