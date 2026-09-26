"""因子诊断（Project1 分析模块）：IC / Rank IC / 覆盖率 / 分组收益。

公式约定（与课程要求一致）：
  IC(h)_t      = Corr_{i∈S_t}( f_{i,t} , y_{i,t}^{(h)} )          Pearson
  RankIC(h)_t  = Corr_{i∈S_t}( rank(f_{i,t}), rank(y_{i,t}^{(h)}) )  Spearman（并列平均秩）
  y_{i,t}^{(h)}= C_{i,t+h}/C_{i,t} - 1（后复权收盘价）
  R_{k,t}      = 组内等权平均收益（对有效标签）；Spread_t = R_{K,t} - R_{1,t}

纪律：
  - 事先设定最低样本数 min_cross_section_samples；常数或样本不足当日记缺失（NaN），不改成 0
  - 标签缺失时报告原分组人数与有效人数，不重新分组
  - 重叠标签的分组收益是描述性结果，不复利、不当作策略净值
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps


def forward_return(close_adj: pd.DataFrame, h: int) -> pd.DataFrame:
    """形成日 t、持有 h 个交易日的标签：C(t+h)/C(t)-1。样本末端不足 h 日记缺失。"""
    return close_adj.shift(-h) / close_adj - 1


def _cross_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """截面相关系数。method ∈ pearson | spearman | kendall（对齐参考实现的 ic_method）。"""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    xv, yv = x[m], y[m]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:  # 常数序列记缺失，不改成 0
        return np.nan
    if method == "pearson":
        return float(np.corrcoef(xv, yv)[0, 1])
    if method == "kendall":
        return float(sps.kendalltau(xv, yv).statistic)
    if method == "spearman":
        xv = pd.Series(xv).rank(method="average").to_numpy()
        yv = pd.Series(yv).rank(method="average").to_numpy()
        return float(np.corrcoef(xv, yv)[0, 1])
    raise KeyError(f"未知 IC 口径: {method}")


def ic_table(factor: pd.DataFrame, y: pd.DataFrame, min_n: int,
             ic_method: str = "pearson") -> pd.DataFrame:
    """逐日 IC / RankIC / 有效资产数 / 覆盖率。

    `ic` 列按 ic_method 计算（pearson | spearman | kendall），`rank_ic` 列固定为
    Spearman 秩相关。两者都留：ic_method 决定报告里的"主 IC 口径"，另一列作为对照。
    """
    farr, yarr = factor.to_numpy(dtype="float64"), y.to_numpy(dtype="float64")
    rows = []
    for i in range(farr.shape[0]):
        x, yv = farr[i], yarr[i]
        n = int((np.isfinite(x) & np.isfinite(yv)).sum())
        ic = _cross_corr(x, yv, ic_method)
        ric = _cross_corr(x, yv, "spearman")
        if n < min_n:
            ic = ric = np.nan
        rows.append({"date": factor.index[i], "ic": ic, "rank_ic": ric, "n_valid": n,
                     "coverage": n / max(1, factor.shape[1])})
    return pd.DataFrame(rows).set_index("date")


def newey_west_tstat(series: pd.Series, lag: int) -> float:
    """Newey-West HAC 修正的 t 统计量（Bartlett 核）。

    动机：h 期标签的 IC 序列天然自相关（相邻两天的标签共享 t+1..t+h 的价格），
    普通 t 检验低估标准误、显著性虚高。课程材料明确要求"重叠标签会影响显著性
    判断"—— 本函数给出 Honest 版本：lag 取 h-1（相邻 IC 的标签重叠天数）。
    """
    x = pd.Series(series).dropna().to_numpy(dtype="float64")
    n = len(x)
    if n < 5:
        return float("nan")
    mu = float(x.mean())
    e = x - mu
    s0 = float((e * e).sum()) / n
    var = s0
    L = min(int(lag), n - 1)
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        cov_l = float((e[l:] * e[:-l]).sum()) / n
        var += 2.0 * w * cov_l
    if var <= 0:
        return float("nan")
    se = float(np.sqrt(var / n))
    return mu / se if se > 0 else float("nan")


def ic_summary(table: pd.DataFrame, nw_lag: int | None = None) -> dict[str, Any]:
    """IC 时间序列的均值/标准差/ICIR/t 统计量（描述性）。

    nw_lag：Newey-West 修正的最大滞后（传入 h-1 或 h 可修正重叠标签的自相关）。
    给出 t_stat（普通）与 t_stat_nw（HAC 修正）两个口径，诚实并列。
    """
    out = {}
    for col in ("ic", "rank_ic"):
        s = table[col].dropna()
        n = len(s)
        out[col] = {
            "n_obs": n,
            "mean": float(s.mean()) if n else np.nan,
            "std": float(s.std(ddof=1)) if n > 1 else np.nan,
            "t_stat": float(s.mean() / (s.std(ddof=1) / np.sqrt(n))) if n > 2 else np.nan,
            "icir": float(s.mean() / s.std(ddof=1)) if n > 1 and s.std(ddof=1) > 0 else np.nan,
            "win_rate": float((s > 0).mean()) if n else np.nan,
            "t_stat_nw": (newey_west_tstat(s, nw_lag)
                          if nw_lag and n > 2 else None),
        }
    out["coverage_mean"] = float(table["coverage"].mean())
    out["n_valid_mean"] = float(table["n_valid"].mean())
    out["nw_lag"] = nw_lag
    return out


def quantile_table(factor: pd.DataFrame, y: pd.DataFrame, k: int, min_n: int) -> dict[str, Any]:
    """形成日按因子升序分 K 组（并列平均秩定序，次序键=资产代码保证确定性）。

    逐日返回：各组 n_formed（按因子分组人数）与 n_valid（组内标签有效人数，不重新分组）。
    样本不足规则：形成日有效因子数 < max(min_n, k) 时整日记缺失（跳过），
    与 IC 路径共用 min_cross_section_samples，避免"事先设最低样本要求"只作用于 IC。
    """
    farr, yarr = factor.to_numpy(dtype="float64"), y.to_numpy(dtype="float64")
    codes = np.array(factor.columns)
    need = max(int(min_n), int(k))
    daily_rows = []
    n_skipped_insufficient = 0
    for i in range(farr.shape[0]):
        x, yv = farr[i], yarr[i]
        m = np.isfinite(x)
        if m.sum() < need:  # 少于组数或低于最低样本要求 -> 记缺失，不改 0、不重新分组
            n_skipped_insufficient += 1
            continue
        idx = np.where(m)[0]
        # 稳定排序：因子秩升序，并列时按资产代码
        order = sorted(idx, key=lambda j: (x[j], str(codes[j])))
        bins = np.array_split(np.array(order), k)
        row = {"date": factor.index[i]}
        for g, members in enumerate(bins, start=1):
            ys = yv[members]
            valid = np.isfinite(ys)
            row[f"g{g}_n_formed"] = len(members)
            row[f"g{g}_n_valid"] = int(valid.sum())
            row[f"g{g}_ret"] = float(np.nanmean(ys[valid])) if valid.sum() >= 1 else np.nan
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows).set_index("date") if daily_rows else pd.DataFrame()

    group_mean = {g: (daily[f"g{g}_ret"].mean() if f"g{g}_ret" in daily else np.nan)
                  for g in range(1, k + 1)}
    spread = (daily[f"g{k}_ret"] - daily["g1_ret"]) if f"g{k}_ret" in daily and "g1_ret" in daily else pd.Series(dtype="float64")
    gm = np.array([group_mean[g] for g in range(1, k + 1)], dtype="float64")
    mono = float(sps.spearmanr(np.arange(1, k + 1), gm).statistic) if np.isfinite(gm).all() else np.nan
    summary = {
        "group_mean_return": {f"g{g}": (None if not np.isfinite(group_mean[g]) else float(group_mean[g]))
                              for g in range(1, k + 1)},
        "spread_mean": float(spread.mean()) if len(spread) else np.nan,
        "spread_annualized_note": "日均价差，未扣费，不等于可执行多空收益",
        "monotonicity_spearman": mono,
        "n_days": int(len(daily)),
        "n_days_skipped_insufficient": int(n_skipped_insufficient),
        "min_cross_section_samples": int(min_n),
        "form_and_valid_example": {
            "first_date_formed": {f"g{g}": {"n_formed": int(daily.iloc[0][f"g{g}_n_formed"]),
                                            "n_valid": int(daily.iloc[0][f"g{g}_n_valid"])}
                                  for g in range(1, k + 1)} if len(daily) else {},
        },
    }
    return {"daily": daily, "summary": summary}


def run_diagnostics(factor_values: pd.DataFrame, close_adj: pd.DataFrame,
                    diag_cfg: dict) -> dict[str, Any]:
    """对单个因子值面板执行完整诊断，返回 {horizon: {...}}。

    diag_cfg:
      horizon_days               [5, 20]       持有期 h
      min_cross_section_samples  10            最低截面样本
      quantile_groups            5             分层组数 K
      ic_method                  'pearson'     IC 相关系数口径（pearson|spearman|kendall）
    """
    min_n = int(diag_cfg.get("min_cross_section_samples", 10))
    k = int(diag_cfg.get("quantile_groups", 5))
    ic_method = str(diag_cfg.get("ic_method", "pearson"))
    results: dict[str, Any] = {}
    for h in diag_cfg.get("horizon_days", [5, 20]):
        y = forward_return(close_adj, h)
        table = ic_table(factor_values, y, min_n, ic_method)
        qt = quantile_table(factor_values, y, k, min_n)
        results[str(h)] = {
            "horizon": h,
            "ic_method": ic_method,
            "quantile_groups": k,
            "ic_table": table,
            "ic_summary": ic_summary(table, nw_lag=max(1, int(h) - 1)),
            "quantile_summary": qt["summary"],
            "quantile_daily": qt["daily"],
        }
    return results


def diagnostics_json(results: dict[str, Any]) -> dict[str, Any]:
    """可 JSON 化的诊断结果（不含逐日明细大表）。"""
    out = {}
    for h, r in results.items():
        t = r["ic_table"]
        out[h] = {
            "horizon": r["horizon"],
            "ic_summary": r["ic_summary"],
            "quantile_summary": r["quantile_summary"],
            "ic_series": {"index": [str(d.date()) for d in t.index],
                          "ic": [None if not np.isfinite(v) else float(v) for v in t["ic"]],
                          "rank_ic": [None if not np.isfinite(v) else float(v) for v in t["rank_ic"]],
                          "n_valid": [int(v) for v in t["n_valid"]],
                          "coverage": [float(v) for v in t["coverage"]]},
        }
    return out


# ---------------- 分钟级大面板的向量化诊断（与上方口径一致，行为等价） ----------------
def _vec_pearson(x: np.ndarray, y: np.ndarray, mask: np.ndarray, min_n: int) -> np.ndarray:
    """逐行 NaN 感知 Pearson 相关（向量化）。常数序列 → NaN。"""
    n = mask.sum(axis=1).astype("float64")
    xm = np.where(mask, x, 0.0)
    ym = np.where(mask, y, 0.0)
    sx = xm.sum(axis=1)
    sy = ym.sum(axis=1)
    sxx = (xm * xm).sum(axis=1)
    syy = (ym * ym).sum(axis=1)
    sxy = (xm * ym).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        ic = cov / np.sqrt(vx * vy)
    bad = (n < max(3, min_n)) | (vx <= 1e-12) | (vy <= 1e-12) | ~np.isfinite(ic)
    ic[bad] = np.nan
    return ic


def ic_table_fast(factor: pd.DataFrame, y: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """大面板逐日 IC/RankIC/有效资产数/覆盖率（向量化，结果口径与 ic_table 一致）。"""
    f = factor.to_numpy(dtype="float64")
    v = y.to_numpy(dtype="float64")
    mask = np.isfinite(f) & np.isfinite(v)
    ic = _vec_pearson(f, v, mask, min_n)
    # Spearman：平均秩变换后同样本秩上的 Pearson
    fr = factor.rank(axis=1, method="average").to_numpy(dtype="float64")
    yr = y.rank(axis=1, method="average").to_numpy(dtype="float64")
    ric = _vec_pearson(fr, yr, mask, min_n)
    n = mask.sum(axis=1)
    with np.errstate(invalid="ignore"):
        cov = n / factor.shape[1]
    return pd.DataFrame({"ic": ic, "rank_ic": ric, "n_valid": n, "coverage": cov},
                        index=factor.index)


def quantile_table_fast(factor: pd.DataFrame, y: pd.DataFrame, k: int, min_n: int) -> dict:
    """大面板 K 组分组（平均秩百分位落组，K 等宽）+ 组内等权收益与价差（向量化）。"""
    farr = factor.to_numpy(dtype="float64")
    yarr = y.to_numpy(dtype="float64")
    fm = np.isfinite(farr)
    # 平均秩百分位 → 组号 1..K（并列按平均秩落组，规则预先确定）
    rpct = factor.rank(axis=1, method="average", pct=True).to_numpy(dtype="float64")
    grp = np.ceil(rpct * k).astype("float64")
    grp[(rpct <= 0) | ~fm] = np.nan
    grp[grp > k] = k

    rows_idx = []
    g_ret = np.full((len(factor), k), np.nan)
    n_formed = np.zeros((len(factor), k), dtype="int64")
    n_valid = np.zeros((len(factor), k), dtype="int64")
    for g in range(1, k + 1):
        gm = (grp == g) & np.isfinite(yarr)
        n_formed[:, g - 1] = ((grp == g) & fm).sum(axis=1)
        n_valid[:, g - 1] = gm.sum(axis=1)
        with np.errstate(invalid="ignore"):
            s = np.where(gm, yarr, 0.0).sum(axis=1)
            g_ret[:, g - 1] = np.where(n_valid[:, g - 1] > 0, s / np.maximum(n_valid[:, g - 1], 1), np.nan)
    keep = n_formed.sum(axis=1) > 0
    daily = pd.DataFrame(
        {f"g{g}_n_formed": n_formed[keep, g - 1] for g in range(1, k + 1)}
        | {f"g{g}_n_valid": n_valid[keep, g - 1] for g in range(1, k + 1)}
        | {f"g{g}_ret": g_ret[keep, g - 1] for g in range(1, k + 1)},
        index=factor.index[keep])
    gm_mean = {g: float(daily[f"g{g}_ret"].mean()) for g in range(1, k + 1)}
    spread = daily[f"g{k}_ret"] - daily["g1_ret"]
    gvec = np.array([gm_mean[g] for g in range(1, k + 1)])
    mono = float(sps.spearmanr(np.arange(1, k + 1), gvec).statistic) if np.isfinite(gvec).all() else np.nan
    first = daily.iloc[0] if len(daily) else None
    summary = {
        "group_mean_return": {f"g{g}": (None if not np.isfinite(gm_mean[g]) else gm_mean[g])
                              for g in range(1, k + 1)},
        "spread_mean": float(spread.mean()) if len(spread) else np.nan,
        "spread_annualized_note": "bar 均价差，未扣费，不等于可执行多空收益",
        "monotonicity_spearman": mono,
        "n_days": int(len(daily)),
        "form_and_valid_example": {
            "first_date_formed": {f"g{g}": {"n_formed": int(first[f"g{g}_n_formed"]),
                                            "n_valid": int(first[f"g{g}_n_valid"])}
                                  for g in range(1, k + 1)} if first is not None else {},
        },
        "tie_rule": "并列值按平均秩百分位等宽落组（预先确定）",
    }
    return {"daily": daily, "summary": summary}


def run_diagnostics_fast(factor_values: pd.DataFrame, close_adj: pd.DataFrame,
                         diag_cfg: dict) -> dict:
    """分钟级大面板诊断入口：horizon 单位=bar。"""
    min_n = int(diag_cfg.get("min_cross_section_samples", 10))
    k = int(diag_cfg.get("quantile_groups", 5))
    results = {}
    for h in diag_cfg.get("horizon_days", [48]):
        y = forward_return(close_adj, int(h))
        table = ic_table_fast(factor_values, y, min_n)
        qt = quantile_table_fast(factor_values, y, k, min_n)
        results[str(h)] = {
            "horizon": int(h),
            "ic_table": table,
            "ic_summary": ic_summary(table, nw_lag=max(1, int(h) - 1)),
            "quantile_summary": qt["summary"],
            "quantile_daily": qt["daily"],
        }
        del y
        import gc; gc.collect()
    return results


