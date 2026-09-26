"""模型类信号的组合器：线性模型 / 决策树，walk-forward 拟合。

前视纪律（这是本模块存在的唯一理由）：
  用因子组合成信号时，模型的**训练集必须严格早于预测时点**。若用全样本拟合再"预测"历史，
  就是把未来信息喂给了过去 —— 这是因子研究里最常见的隐性前视。

本模块的做法：
  1. 逐 refit 点滚动：在 r 处重新拟合一次，只用于预测 [r, r+refit_days)
  2. 训练集窗口 = [r - train_window, r - label_horizon - purge)
     —— 标签 y(t) = C(t+label_h)/C(t)-1 会用到 t+label_h 的价格，
        因此训练样本必须截止到 r - label_h，标签才在预测起点之前完全可观测；
        purge 是额外的前后缓冲，防止相邻样本的标签窗口粘连。
  3. 输出的信号面板**全部是样本外预测**，可直接喂给回测与诊断。

参数（存在 Signal.model_params）：
  refit_days    多久重新拟合一次（交易日）
  train_window  用之前多久的数据拟合（交易日）
  purge         标签缓冲（交易日，默认 5）
  label_horizon 训练标签的持有期（交易日，默认 20）
  model         ridge | tree | gbdt（默认 ridge）
  alpha         ridge 的正则强度（默认 1.0）
  max_depth     tree 的最大深度（默认 4）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from .preprocess import apply_preprocess


def model_params_of(sig) -> dict:
    mp = dict(sig.model_params or {})
    # model_type=linear → ridge，tree → tree；model_params['model'] 可显式覆盖
    default_model = "ridge" if getattr(sig, "model_type", "") == "linear" else "tree"
    return {
        "model": str(mp.get("model", default_model)),
        "refit_days": max(1, int(mp.get("refit_days", 20))),
        "train_window": max(20, int(mp.get("train_window", 252))),
        "purge": max(0, int(mp.get("purge", 5))),
        "label_horizon": max(1, int(mp.get("label_horizon", 20))),
        "alpha": float(mp.get("alpha", 1.0)),
        "max_depth": int(mp.get("max_depth", 4)),
    }


def _make_model(p: dict):
    if p["model"] == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=p["alpha"])
    if p["model"] == "tree":
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(max_depth=p["max_depth"], random_state=7)
    if p["model"] == "gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_depth=p["max_depth"], random_state=7)
    raise ValueError(f"未知模型: {p['model']}（可选 ridge | tree | gbdt）")


def fit_walk_forward(sig, market, start=None, end=None) -> pd.DataFrame:
    """返回样本外信号面板（date × code）。"""
    from ..storage.store import factor_values

    cfg = load_config()
    diag = cfg.factors["diagnosis"]
    p = model_params_of(sig)
    comps = list(sig.components or [])
    if not comps:
        raise ValueError("信号没有任何因子分量")

    # 特征：每个因子做截面 winsorize + zscore（与等权/IC 加权保持同一预处理口径）
    feats = {}
    for c in comps:
        fid = int(c["factor_id"])
        v = factor_values(fid)
        feats[fid] = apply_preprocess(v, ["winsorize", "zscore"], diag)

    # 对齐到统一的面板
    idx = None
    cols = None
    for v in feats.values():
        idx = v.index if idx is None else idx.intersection(v.index)
        cols = v.columns if cols is None else cols.union(v.columns)
    feats = {k: v.reindex(index=idx, columns=cols) for k, v in feats.items()}

    close = market.close_adj.reindex(index=idx, columns=cols)
    y_panel = close.shift(-p["label_horizon"]) / close - 1

    dates = pd.DatetimeIndex(idx)
    pred = pd.DataFrame(np.nan, index=dates, columns=cols)
    log = []

    # 拟合点：从 start 开始每 refit_days 一个
    d0 = pd.Timestamp(start) if start is not None else dates[0]
    d1 = pd.Timestamp(end) if end is not None else dates[-1]
    refit_points = [d for d in dates if d0 <= d <= d1][:: p["refit_days"]]

    for r in refit_points:
        train_end = r - pd.Timedelta(days=0)          # 按交易日切，不用自然日
        pos_r = dates.get_loc(r)
        lo_pos = max(0, pos_r - p["train_window"])
        hi_pos = pos_r - p["label_horizon"] - p["purge"]   # 标签须在 r 之前完全可观测
        if hi_pos - lo_pos < 30:
            log.append(f"{r.date()}: 训练样本不足（{max(0, hi_pos - lo_pos)} 行），跳过该拟合点")
            continue
        tr_dates = dates[lo_pos:hi_pos]

        # 组装训练集（只保留特征与标签都有效的格子）
        Xs, ys = [], []
        for k, v in feats.items():
            Xs.append(v.reindex(index=tr_dates).to_numpy(dtype="float64"))
        X = np.stack(Xs, axis=-1)                       # (T, N, F)
        Y = y_panel.reindex(index=tr_dates).to_numpy(dtype="float64")
        ok = np.isfinite(Y) & np.isfinite(X).all(axis=-1)
        if ok.sum() < 100:
            log.append(f"{r.date()}: 有效训练样本仅 {int(ok.sum())} 条，跳过")
            continue
        Xf = X[ok]
        Yf = Y[ok]
        try:
            m = _make_model(p)
            m.fit(Xf, Yf)
        except Exception as e:  # noqa: BLE001
            log.append(f"{r.date()}: 拟合失败 {type(e).__name__}: {e}")
            continue

        # 预测下一段（样本外）
        seg = [d for d in dates if r <= d < r + pd.Timedelta(days=p["refit_days"] * 1)]
        # 用位置切片更稳妥
        seg_pos = list(range(pos_r, min(pos_r + p["refit_days"], len(dates))))
        seg_dates = dates[seg_pos]
        Xp = np.stack([v.reindex(index=seg_dates).to_numpy(dtype="float64")
                       for v in feats.values()], axis=-1)
        flat = Xp.reshape(-1, Xp.shape[-1])
        good = np.isfinite(flat).all(axis=1)
        out = np.full(flat.shape[0], np.nan)
        if good.any():
            out[good] = m.predict(flat[good])
        pred.loc[seg_dates, :] = out.reshape(len(seg_dates), len(cols))

    pred.attrs["wf_log"] = log
    pred.attrs["model_params"] = p
    pred.columns.name = "code"
    # 只返回请求区间内的样本外预测
    return pred.loc[(pred.index >= d0) & (pred.index <= d1)]
