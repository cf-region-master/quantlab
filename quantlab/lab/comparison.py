"""组合方式对照实验：同一组因子，逐一应用各组合模型，同口径对照 RankIC。

把课程要求的"对照实验"自动化：只改变【组合模型】这一个因素，其余（因子集、
预处理、区间、标签口径）保持一致。输出各模型的 RankIC 均值 / NW 修正 t /
观测数，并落盘 JSON 供报告引用。

默认模型族为轻量组合（等权/IC/滚动 ICIR/均值方差/正交化），秒级完成；
linear/tree 的 walk-forward 拟合较慢，需显式传 models 才会参与。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_config
from ..factors.diagnostics import run_diagnostics
from ..factors.diagnostics import newey_west_tstat

DEFAULT_MODELS = ("equal_weight", "ic_weight", "ic_weight_rolling",
                  "ic_meanvar", "ortho_ic_weight_rolling")


def compare_combinations(factor_ids: list[int], start: str, end: str,
                         horizon: int = 5, models: list[str] | None = None,
                         min_cross_section: int = 10,
                         include_backtest: bool = False,
                         rebalance_freqs: list[str] | None = None,
                         pool: dict | None = None) -> dict[str, Any]:
    """返回 {model: {"rank_ic_mean","t_naive","t_nw","n_obs"}} 与实验元信息。

    include_backtest=True 时对每个模型追加一次真实模拟成交（周度 top10 等权，
    配置费率），输出年化/Sharpe/MDD/换手/成本 —— 把"组合方式更好"从 IC 层
    验证到净值层。回测在内存执行，不落 BacktestRun（对照是实验，不是资产）。
    """
    from ..factors.user_code import compute_user_factor  # noqa: F401 —— 确保 combine 路径可用
    from ..storage import store
    from ..storage.db import Factor

    cfg = load_config()
    mk = store.market()
    models = list(models or DEFAULT_MODELS)

    # 对照纪律：各模型的因子 spec/池保持一致（取第一个因子的 spec 作代表）
    s = store.get_session()
    try:
        f0 = s.get(Factor, int(factor_ids[0]))
        spec = f0.preprocess_spec if f0 else None
        names = {f.id: f.name for f in s.query(Factor).all()}
    finally:
        s.close()

    out: dict[str, Any] = {}
    for m in models:
        sig = SimpleNamespace(
            components=[{"factor_id": int(i), "weight": 1.0 / len(factor_ids)}
                        for i in factor_ids],
            model_type=m,
            model_params={"label_horizon": horizon, "window": 252, "min_periods": 60,
                          "ridge": 0.5},
            start_date=pd.Timestamp(start), end_date=pd.Timestamp(end),
            preprocess=[],
        )
        try:
            panel = store.signal_panel(sig, mk, start, end)
            diag = run_diagnostics(panel, mk.close_adj,
                                   {"horizon_days": [horizon],
                                    "min_cross_section_samples": min_cross_section,
                                    "quantile_groups": int(cfg.factors["diagnosis"]
                                                           .get("quantile_groups", 5))})
            summ = diag[str(horizon)]["ic_summary"]["rank_ic"]
            entry = {"rank_ic_mean": round(summ["mean"], 6),
                     "t_naive": round(summ["t_stat"], 3),
                     "t_nw": round(summ["t_stat_nw"], 3)
                     if summ.get("t_stat_nw") is not None
                     and np.isfinite(summ["t_stat_nw"]) else None,
                     "n_obs": summ["n_obs"]}
            if include_backtest:
                freqs = rebalance_freqs or ["weekly"]
                for fq in freqs:
                    try:
                        bt = _quick_backtest(panel, mk, start, end,
                                             rebalance_freq=fq, pool=pool)
                        for k, v in bt.items():
                            entry[f"bt_{k}_{fq}"] = v   # 统一后缀：bt_<k>_<freq>
                    except Exception as e:  # noqa: BLE001
                        entry[f"bt_error_{fq}"] = f"{type(e).__name__}: {e}"
            out[m] = entry
        except Exception as e:  # noqa: BLE001 —— 单模型失败记录原因，不中断对照
            out[m] = {"error": f"{type(e).__name__}: {e}"}

    # 跨模型稳健性聚合：方向一致比例 + NW 显著模型计数（|t_nw|≥1.96 记显著）
    valid = [v for v in out.values() if isinstance(v, dict) and v.get("rank_ic_mean") is not None]
    n_pos = sum(1 for v in valid if v["rank_ic_mean"] > 0)
    n_neg = sum(1 for v in valid if v["rank_ic_mean"] < 0)
    majority = "正" if n_pos >= n_neg else "负"
    consistent = n_pos if majority == "正" else n_neg
    robustness = {
        "n_models": len(valid),
        "majority_direction": majority if valid else None,
        "direction_consistency": round(consistent / len(valid), 3) if valid else None,
        "n_sig_nw": sum(1 for v in valid
                        if v.get("t_nw") is not None and abs(v["t_nw"]) >= 1.96),
        "note": "方向一致比例=与多数方向相同的模型占比；t_nw 为 Newey-West 修正 t",
    }
    return {"factor_ids": factor_ids,
            "factor_names": [names.get(i, str(i)) for i in factor_ids],
            "start": start, "end": end, "horizon": horizon,
            "pool": pool,
            "robustness": robustness,
            "env": cfg.snapshot().get("data", {}).get("universe", {}).get("name"),
            "models": out,
            "note": ("对照纪律：同因子集/同预处理/同区间/同标签，只改组合模型；"
                     "t_nw 为 Newey-West 修正 t（lag=h-1，重叠标签纪律）；"
                     "均为描述性统计")}


def _quick_backtest(panel: pd.DataFrame, mk, start: str, end: str,
                    rebalance_freq: str = "weekly",
                    pool: dict | None = None) -> dict[str, float]:
    """内存版组合回测（top10 等权，配置费率），返回关键净指标。"""
    from ..backtest.engine import run_backtest
    from ..config import load_config

    cfg = load_config()
    bt_cfg = dict(cfg.backtest)
    bt_cfg["rebalance"] = {**bt_cfg["rebalance"], "freq": rebalance_freq}
    bt_cfg["portfolio"] = {**bt_cfg["portfolio"], "top_n": 10, "weighting": "equal_weight"}
    bt_cfg["sample"] = {"start": start, "end": end}
    bt_cfg["trading_days_per_year"] = int(cfg.data["trading_days_per_year"])
    pool_mask = None
    if pool:
        from ..data.universe import resolve_pool
        pool_mask, _ = resolve_pool(mk, pool)
    res = run_backtest(panel, mk, bt_cfg, name="compare", pool_mask=pool_mask)
    from ..backtest.metrics import compute_metrics
    res.metrics = compute_metrics(res, float(cfg.data["risk_free_annual"]),
                                  int(cfg.data["trading_days_per_year"]))
    net = res.metrics.get("net", {}) if res.metrics else {}
    return {"annualized_return": net.get("annualized_return"),
            "sharpe": net.get("annualized_sharpe"),
            "max_drawdown": net.get("max_drawdown"),
            "total_cost": float(res.cost_series.sum()) if len(res.cost_series) else 0.0,
            "turnover_sum": float(res.turnover.sum()) if len(res.turnover) else 0.0}
