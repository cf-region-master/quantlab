"""因子组合器：相关性 / 去冗余 / walk-forward IC 权重 / IC 均值-方差凸组合。

设计动机（回答"怎么更好地组合因子"）：
  1. **相关性感知**：高相关因子等权相加 = 把同一风险放大 N 倍。组合前先看
     因子面板的池化相关（pooled correlation，逐 (日期,资产) 格子合并计算），
     贪心剔除 |ρ| 超阈值中分数较低者。
  2. **无前视的 IC 加权**：经典 ic_weight 用【全样本】RankIC 均值定权重再在同
     样本上回测 —— 样本内偏误。本模块的滚动权重在日期 t 只使用 t-h 之前的
     IC 信息（IC(t) 本身要看 t+h 的价格），天然 walk-forward。
  3. **IC 均值-方差凸组合**（Grinold-Kahn 式）：w(t) ∝ (Σ(t)+λI)⁻¹ μ(t)，
     μ=各因子滚动 IC 均值、Σ=因子间滚动 IC 协方差 —— 分散化收益直接进权重。

权重矩阵约定：index=日期, columns=因子键；每行权重已归一（Σ|w|=1，保留符号），
信号 = Σ_f w_f(t) · z_f(t)（逐行加权）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------- 相关性与去冗余 ----------------
def pooled_corr(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """因子两两的池化 Pearson 相关：把面板拉长成 (日期,资产) 样本后合并计算。

    比起"逐日截面相关的均值"，池化相关对离群更敏感但计算一次到位；
    两者在单调变换下排序一致，这里取前者（对"信息是否重复"更直接）。
    """
    keys = list(panels)
    cols = {}
    for k in keys:
        s = panels[k].stack(future_stack=True)
        cols[k] = s
    long = pd.DataFrame(cols)
    return long.corr(min_periods=200)


def redundancy_prune(corr: pd.DataFrame, scores: dict[str, float],
                     max_abs_corr: float = 0.8) -> dict:
    """贪心去冗余：按 |分数|（如 RankIC）降序保留，与任一已保留因子 |ρ|≥阈值 的淘汰。

    返回 {"keep": [...], "drop": [{"key","vs","corr"}...]} —— 页面可直接展示
    "建议剔除 X（与 Y 相关 0.93）"。
    """
    order = sorted(corr.columns, key=lambda k: -abs(scores.get(k, 0.0)))
    keep: list[str] = []
    drops: list[dict] = []
    for k in order:
        clash = None
        worst = 0.0
        signed = 0.0
        for j in keep:
            c = float(corr.loc[k, j]) if (k in corr.columns and j in corr.columns) else 0.0
            if abs(c) > worst:
                worst, clash, signed = abs(c), j, c
        if clash is not None and worst >= max_abs_corr:
            drops.append({"key": k, "vs": clash, "corr": round(signed, 4)})
        else:
            keep.append(k)
    return {"keep": keep, "drop": drops, "threshold": max_abs_corr}


# ---------------- 逐日 IC 序列（向量化） ----------------
def daily_ic_series(panel: pd.DataFrame, close_adj: pd.DataFrame, h: int,
                    min_n: int = 10, method: str = "spearman") -> pd.Series:
    """逐日截面 IC（默认 RankIC=平均秩 Spearman），样本不足/常数记 NaN。

    用于滚动权重估计 —— 与诊断模块同一口径，但只返回一列，供 rolling。
    """
    from .diagnostics import ic_table_fast

    t = ic_table_fast(panel, close_adj.shift(-h) / close_adj - 1, min_n)
    return t["rank_ic"] if method == "spearman" else t["ic"]


def _normalize_rows(w: pd.DataFrame) -> pd.DataFrame:
    den = w.abs().sum(axis=1)
    out = w.div(den.where(den > 1e-12, np.nan), axis=0)
    return out


# ---------------- walk-forward 权重 ----------------
def rolling_icir_weights(panels: dict[str, pd.DataFrame], close_adj: pd.DataFrame,
                         h: int = 5, window: int = 252, min_periods: int = 60,
                         min_n: int = 10) -> pd.DataFrame:
    """ICIR 滚动权重：w_f(t) ∝ mean(IC_f[·< t-h]) / std(IC_f[·< t-h])。

    无前视关键点：IC(t) 的标签用到 t+h 的价格，因此构造权重时先把 IC 序列
    shift(h)（t 行携带的是截至 t-h 的信息），再做 expanding/滚动统计。
    窗口不足 min_periods 时用 expanding（样本少的早期也有合理权重）。
    """
    ics = pd.concat({k: daily_ic_series(v, close_adj, h, min_n) for k, v in panels.items()},
                    axis=1).shift(h)  # ← 前视屏蔽
    mu = ics.rolling(window, min_periods=min_periods).mean().combine_first(
        ics.expanding(min_periods=min_periods).mean())
    sd = ics.rolling(window, min_periods=min_periods).std(ddof=1).combine_first(
        ics.expanding(min_periods=min_periods).std(ddof=1))
    w = mu / sd.where(sd > 1e-12, np.nan)
    return _normalize_rows(w.reindex(close_adj.index))


def rolling_inversevol_weights(panels: dict[str, pd.DataFrame], close_adj: pd.DataFrame,
                               h: int = 5, window: int = 252, min_periods: int = 60,
                               min_n: int = 10) -> pd.DataFrame:
    """滚动 IC 逆波动率权重：w_f(t) ∝ 1 / std(IC_f[·< t-h])。

    ICIR（mean/std）在 IC 均值穿零附近会剧烈翻符号、权重抖动大；
    逆波动率是它的稳健变体：只按"信号稳定度"分配，均值方向交给多因子
    分散化与前视屏蔽后的样本外表现。前视屏蔽与 ICIR 完全一致（IC 序列 shift(h)）。
    """
    ics = pd.concat({k: daily_ic_series(v, close_adj, h, min_n) for k, v in panels.items()},
                    axis=1).shift(h)  # ← 前视屏蔽
    sd = ics.rolling(window, min_periods=min_periods).std(ddof=1).combine_first(
        ics.expanding(min_periods=min_periods).std(ddof=1))
    w = 1.0 / sd.where(sd > 1e-12, np.nan)
    return _normalize_rows(w.reindex(close_adj.index))


def rolling_ic_meanvar_weights(panels: dict[str, pd.DataFrame], close_adj: pd.DataFrame,
                               h: int = 5, window: int = 252, min_periods: int = 60,
                               ridge: float = 0.5, min_n: int = 10) -> pd.DataFrame:
    """IC 均值-方差凸组合：w(t) ∝ (Σ+λ·mean(diag(Σ))·I)⁻¹ μ（逐日滚动，walk-forward）。

    μ/Σ 在日期 t 只用 t-h 之前的 IC 样本（同 rolling_icir_weights 的 shift(h) 纪律）。
    λ 相对化（乘 Σ 对角均值），使 ridge 与因子量纲无关；样本不足的日期退化为
    ICIR 权重（μ 对角近似）。闭式解、无迭代，日频全样本毫秒级。
    """
    ics = pd.concat({k: daily_ic_series(v, close_adj, h, min_n) for k, v in panels.items()},
                    axis=1).shift(h)
    keys = list(panels)
    mu = ics.rolling(window, min_periods=min_periods).mean().combine_first(
        ics.expanding(min_periods=min_periods).mean())
    cov = ics.rolling(window, min_periods=min_periods).cov().combine_first(
        ics.expanding(min_periods=min_periods).cov())

    # 退化回退：ICIR 权重（Σ 难逆时的对角近似）
    sd = ics.rolling(window, min_periods=min_periods).std(ddof=1).combine_first(
        ics.expanding(min_periods=min_periods).std(ddof=1))
    w_icir = mu / sd.where(sd > 1e-12, np.nan)

    rows = ics.index
    out = np.full((len(rows), len(keys)), np.nan)
    mu_arr = mu.to_numpy(dtype="float64")
    for i in range(len(rows)):
        c = cov.loc[rows[i]]
        if not np.isfinite(c.to_numpy(dtype="float64")).all():
            continue
        m = mu_arr[i]
        if not np.isfinite(m).all():
            continue
        sigma = c.to_numpy(dtype="float64")
        lam = ridge * float(np.mean(np.diag(sigma))) if np.mean(np.diag(sigma)) > 0 else ridge
        try:
            w = np.linalg.solve(sigma + lam * np.eye(len(keys)), m)
        except np.linalg.LinAlgError:
            continue
        out[i] = w
    w_df = pd.DataFrame(out, index=rows, columns=keys)
    # 样本不足的日期用 ICIR 兜底（避免全 NaN 开头）
    w_df = w_df.combine_first(w_icir)
    return _normalize_rows(w_df.reindex(close_adj.index))


def apply_weights(panels: dict[str, pd.DataFrame], weights: pd.DataFrame) -> pd.DataFrame:
    """逐行加权合成信号：signal(t) = Σ_f w_f(t)·z_f(t)。权重缺失的日期记缺失。"""
    keys = list(panels)
    base = None
    for k in keys:
        wk = weights[k].reindex(panels[k].index).ffill()
        term = panels[k].mul(wk, axis=0)
        base = term if base is None else base.add(term, fill_value=np.nan)
    if base is None:
        raise ValueError("没有任何因子面板")
    base.columns.name = "code"
    return base


def combination_gain(single_ics: dict[str, float], combined_ic: float) -> dict:
    """组合增益摘要：组合 IC 相对最强单因子的提升比例（诚实展示为描述性统计）。"""
    if not single_ics or combined_ic is None or not np.isfinite(combined_ic):
        return {"note": "样本不足，无法计算组合增益"}
    best_key = max(single_ics, key=lambda k: abs(single_ics[k]))
    best = single_ics[best_key]
    gain = (abs(combined_ic) / abs(best) - 1.0) if abs(best) > 1e-12 else np.nan
    sign_flip = bool(best * combined_ic < 0)
    note = "|组合IC| 相对最强单因子的提升比例；描述性统计，未做显著性检验"
    if sign_flip:
        note += "；⚠️ 组合 IC 方向与最强单因子相反（分散化/翻正效应），此时增益比例不适用，请以方向与显著性为准"
    return {"best_single_key": best_key, "best_single_ic": round(best, 6),
            "combined_ic": round(float(combined_ic), 6),
            "gain_vs_best_single": None if not np.isfinite(gain) else round(gain, 4),
            "sign_flip": sign_flip, "note": note}


# ---------------- Schmidt 正交化（增量信息分解） ----------------
def orthogonalize(panels: dict[str, pd.DataFrame],
                  order: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """按给定顺序做截面 Gram-Schmidt：每个因子对【前面因子的残差基】回归取残差。

    语义：order 越靠前保留越多原始信息（第一个因子原样保留）；排后面的因子
    只保留前面解释不掉的【增量信息】—— 正交残差的 IC 即增量 IC。
    NaN 处理：投影计算用 0 填充，残差回写原 NaN 掩码（近似口径，已文档化）。
    返回 (正交化面板 dict, 实际顺序)。
    """
    keys = [k for k in order if k in panels]
    if not keys:
        raise ValueError("orthogonalize: 没有可用面板")
    ref = panels[keys[0]]
    long = pd.concat([panels[k].stack(future_stack=True) for k in keys], axis=1)
    long.columns = keys
    arr = long.to_numpy(dtype="float64")
    nan_mask = ~np.isfinite(arr)
    filled = np.where(nan_mask, 0.0, arr)

    basis = np.zeros((arr.shape[0], 0))
    out = np.full_like(arr, np.nan)
    for i, k in enumerate(keys):
        v = filled[:, i]
        r = v - (basis @ (basis.T @ v) if basis.shape[1] else 0.0)
        r[nan_mask[:, i]] = np.nan
        out[:, i] = r
        rr = np.where(nan_mask[:, i], 0.0, r)
        norm = float(np.linalg.norm(rr))
        if norm > 1e-12:
            q = rr / norm
            q = q - basis @ (basis.T @ q)
            qn = float(np.linalg.norm(q))
            if qn > 1e-12:
                basis = np.hstack([basis, (q / qn)[:, None]])

    parts = []
    for i in range(len(keys)):
        res = pd.Series(out[:, i], index=long.index).unstack()
        parts.append(res.reindex(index=ref.index, columns=ref.columns))
    return {k: parts[i] for i, k in enumerate(keys)}, keys


def ortho_incremental_ic(panels: dict[str, pd.DataFrame], close_adj: pd.DataFrame,
                         order: list[str], h: int = 5, min_n: int = 10) -> dict[str, float]:
    """各因子的【增量 RankIC】（正交残差的 IC）：组合中每个因子新贡献了多少信息。"""
    ortho, keys = orthogonalize(panels, order)
    out = {}
    for k in keys:
        ics = daily_ic_series(ortho[k], close_adj, h, min_n).dropna()
        out[k] = float(ics.mean()) if len(ics) else float("nan")
    return out
