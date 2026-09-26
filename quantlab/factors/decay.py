"""IC 衰减分析：多持有期 IC 曲线与半衰期。

回答两个实战问题：
  - 这个因子的信息衰减多快？（决定持有期/调仓频率的上限）
  - 最优持有期在哪？（|IC| 最大的 h）

半衰期定义（文档化口径）：以 h=1（或曲线中最小的 h）的 |IC| 为基准，
|IC(h)| 首次跌破基准一半的最小 h；全程未跌破则为 None（信息持续期长）。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 15, 20, 40)


def ic_decay_curve(factor: pd.DataFrame, close_adj: pd.DataFrame,
                   horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                   min_n: int = 10) -> dict[str, Any]:
    """各持有期的平均 RankIC（向量化快速路径）。"""
    from .diagnostics import ic_table_fast

    out = {}
    for h in horizons:
        y = close_adj.shift(-int(h)) / close_adj - 1
        t = ic_table_fast(factor, y, min_n)
        s = t["rank_ic"].dropna()
        out[int(h)] = {"rank_ic_mean": (float(s.mean()) if len(s) else np.nan),
                       "n_obs": int(len(s))}
    return out


def decay_summary(curve: dict[int, dict]) -> dict[str, Any]:
    """半衰期与最优持有期（描述性）。

    half_life：|IC(h)| 首次 <= |IC(h0)|/2 的最小 h（h0=曲线最小持有期）；
    未跌破 → None。best_h：|IC| 最大的 h。
    """
    hs = sorted(int(h) for h in curve)
    if not hs:
        return {"half_life": None, "best_h": None}
    base = abs(curve[hs[0]]["rank_ic_mean"])
    if not np.isfinite(base) or base < 1e-6:
        return {"half_life": None, "best_h": max(hs, key=lambda h: abs(
            curve[h]["rank_ic_mean"]) if np.isfinite(curve[h]["rank_ic_mean"]) else 0.0),
            "note": "h0 的 |IC| 过小，半衰期无意义"}
    half = None
    for h in hs:
        v = curve[h]["rank_ic_mean"]
        if np.isfinite(v) and abs(v) <= base / 2.0:
            half = h
            break
    best = max(hs, key=lambda h: abs(curve[h]["rank_ic_mean"])
               if np.isfinite(curve[h]["rank_ic_mean"]) else 0.0)
    return {"half_life": half, "best_h": int(best), "base_h": hs[0],
            "base_rank_ic": round(float(base), 6),
            "rebalance_hint": (f"半衰期 ≈ {half} 个交易日：调仓频率不宜快于该周期"
                               if half else "持有期内信息未见减半：可维持当前调仓频率")}


def full_decay_report(factor: pd.DataFrame, close_adj: pd.DataFrame,
                      horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                      min_n: int = 10) -> dict[str, Any]:
    curve = ic_decay_curve(factor, close_adj, horizons, min_n)
    curve_json = {str(h): {"rank_ic_mean": (round(v["rank_ic_mean"], 6)
                                            if np.isfinite(v["rank_ic_mean"]) else None),
                           "n_obs": v["n_obs"]} for h, v in curve.items()}
    return {"curve": curve_json, **decay_summary(curve)}
