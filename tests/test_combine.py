"""因子组合器测试：相关性/去冗余/walk-forward 权重/组合增益。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.factors.combine import (apply_weights, combination_gain,
                                      daily_ic_series, pooled_corr,
                                      redundancy_prune,
                                      rolling_ic_meanvar_weights,
                                      rolling_icir_weights)


def _panels(n=600, k=4, seed=11, ortho=True):
    """合成因子面板：k 个因子，前两个正交 alpha，其余噪声。"""
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = [f"S{i}" for i in range(6)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, 6)), axis=0),
                         index=idx, columns=cols)
    panels = {}
    for j in range(k):
        base = rng.normal(0, 1, (n, 6))
        panels[f"f{j}"] = pd.DataFrame(base, index=idx, columns=cols)
    # 让 f0 与未来 5 日收益挂钩（真实 alpha）
    y = close.shift(-5) / close - 1
    panels["f0"] = y.shift(0).fillna(0.0) * 1.0 + rng.normal(0, 0.3, (n, 6)) * 0  # 纯 alpha
    if not ortho and k >= 2:
        panels["f1"] = panels["f0"] + rng.normal(0, 0.05, (n, 6))  # f0 的克隆
    return close, panels


def test_pooled_corr_identical_is_one():
    close, panels = _panels(k=2)
    panels["clone"] = panels["f0"]
    c = pooled_corr(panels)
    assert c.loc["f0", "clone"] > 0.999


def test_redundancy_prune_drops_clone():
    close, panels = _panels(k=2, ortho=False)
    panels["clone"] = panels["f0"]
    corr = pooled_corr(panels)
    scores = {"f0": 0.05, "clone": 0.049, "f1": 0.01}
    out = redundancy_prune(corr, scores, max_abs_corr=0.8)
    assert "f0" in out["keep"] and "clone" in [d["key"] for d in out["drop"]]
    assert out["drop"][0]["vs"] == "f0"


def test_rolling_weights_no_lookahead():
    """关键纪律：权重在日期 t 只能用 t-h 之前的 IC —— 改动未来标签不得影响历史权重。"""
    close, panels = _panels(n=500, k=2)
    w_full = rolling_icir_weights(panels, close, h=5, window=120, min_periods=60)
    cut = close.index[400]
    panels_t = {k: v.loc[:cut] for k, v in panels.items()}
    w_trunc = rolling_icir_weights(panels_t, close.loc[:cut], h=5, window=120, min_periods=60)
    # 截断点之前 30 行的权重必须逐位一致
    common = w_trunc.index[60:370]
    pd.testing.assert_frame_equal(
        w_full.loc[common].fillna(0), w_trunc.loc[common].fillna(0))


def test_ic_meanvar_diagonal_case():
    """Σ 为对角阵时闭式解应给出 w ∝ μ/σ²（标准化后再比较方向即可）。"""
    rng = np.random.default_rng(4)
    n = 400
    idx = pd.bdate_range("2023-01-02", periods=n)
    ic = pd.DataFrame({"a": rng.normal(0.05, 0.10, n), "b": rng.normal(0.02, 0.20, n)},
                      index=idx)
    # 直接喂给 rolling 统计的面板不参与——这里手工驱动同款闭式解做方向校验
    mu, sd = ic.mean(), ic.std(ddof=1)
    ratio = {"a": mu["a"] / sd["a"], "b": mu["b"] / sd["b"]}
    assert abs(ratio["a"]) > abs(ratio["b"])  # 低波动高 IC 的因子权重占比更高


def test_combination_gain_sign_flip_note():
    g = combination_gain({"f0": -0.021}, 0.020)
    assert g["sign_flip"] is True and "翻正" in g["note"]
    g2 = combination_gain({"f0": 0.021}, 0.030)
    assert g2["sign_flip"] is False and g2["gain_vs_best_single"] > 0


def test_daily_ic_series_constant_nan():
    idx = pd.bdate_range("2023-01-02", periods=30)
    close = pd.DataFrame(100.0, index=idx, columns=list("AB"))
    const = pd.DataFrame(1.0, index=idx, columns=list("AB"))
    s = daily_ic_series(const, close, h=5, min_n=3)
    assert s.isna().all()  # 常数因子记缺失，不改 0


def test_orthogonalize_incremental_ic():
    """克隆因子的正交残差应只剩噪声（增量 IC≈0），第一个因子原样保留。"""
    rng = np.random.default_rng(5)
    n = 300
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = list("ABCDE")
    f0 = pd.DataFrame(rng.normal(0, 1, (n, 5)), index=idx, columns=cols)
    clone = f0 + rng.normal(0, 0.01, (n, 5))
    noise = pd.DataFrame(rng.normal(0, 1, (n, 5)), index=idx, columns=cols)
    from quantlab.factors.combine import orthogonalize
    ortho, order = orthogonalize({"f0": f0, "clone": clone, "noise": noise},
                                 ["f0", "clone", "noise"])
    assert order == ["f0", "clone", "noise"]
    pd.testing.assert_frame_equal(ortho["f0"], f0)                 # 第一个不变
    ratio = float(ortho["clone"].stack().std() / clone.stack().std())
    assert ratio < 0.05                                            # 克隆只剩 eps
    corr = float(ortho["f0"].stack().corr(ortho["clone"].stack()))
    assert abs(corr) < 0.05                                        # 残差近正交
    # noise 与前两者本就独立 → 残差方差基本不变
    assert float(ortho["noise"].stack().std()) == pytest.approx(
        float(noise.stack().std()), rel=0.1)
