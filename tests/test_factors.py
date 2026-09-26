"""因子与诊断正确性测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.factors.diagnostics import forward_return, ic_summary, ic_table, quantile_table
from quantlab.factors.operators import op_lowvol, op_momentum, op_reversal
from quantlab.factors.preprocess import winsorize_cs, zscore_cs


def _flat_market(n_days=40, n_codes=5, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"S{i}" for i in range(n_codes)]
    close = pd.DataFrame(100 * (1 + rng.normal(0.001, 0.02, (n_days, n_codes))).cumprod(axis=0),
                         index=idx, columns=cols)
    return close


def test_momentum_exact_value():
    close = pd.DataFrame({"A": [100.0, 110.0, 120.0, 130.0, 121.0]})
    out = op_momentum({"close": close}, window=4)
    assert np.isnan(out.iloc[:4, 0]).all()
    assert out.iloc[4, 0] == pytest.approx(121.0 / 100.0 - 1)


def test_reversal_sign():
    close = pd.DataFrame({"A": [100.0, 90.0, 95.0, 92.0, 91.0]})
    out = op_reversal({"close": close}, window=4)
    # 近期下跌 → 反转因子为正（统一方向：分高=预期收益高）
    assert out.iloc[4, 0] == pytest.approx(-(91.0 / 100.0 - 1))
    assert out.iloc[4, 0] > 0


def test_lowvol_direction():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=60)
    calm = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, 0.001, 60))}, index=idx)
    wild = pd.DataFrame({"B": 100 * np.cumprod(1 + rng.normal(0, 0.05, 60))}, index=idx)
    close = calm.join(wild)
    out = op_lowvol({"close": close}, window=20)
    # 低波动资产得分（取负号后）应高于高波动资产
    assert out.iloc[-1, 0] > out.iloc[-1, 1]


def test_winsorize_and_zscore():
    idx = pd.bdate_range("2024-01-01", periods=3)
    df = pd.DataFrame([[1.0, 2.0, 3.0, 100.0]] * 3, index=idx,
                      columns=list("abcd"))
    # 参数名对齐参考契约：percentile 用 alpha（旧名 pct 仍可用）
    w = winsorize_cs(df, alpha=0.2)
    assert w.iloc[0, 3] < 100.0  # 极端值被压缩
    assert winsorize_cs(df, pct=0.2).equals(w)
    z = zscore_cs(df, min_n=4)
    assert np.isfinite(z.to_numpy()).all()
    assert z.iloc[0].mean() == pytest.approx(0.0, abs=1e-12)
    # 样本不足 → 全 NaN（记缺失，不改成 0）
    z2 = zscore_cs(df, min_n=10)
    assert np.isnan(z2.to_numpy()).all()
    # MAD / 3sigma 也走同一套边界逻辑
    from quantlab.factors.preprocess import mad_cs, sigma_cs
    assert mad_cs(df, k=3.0).iloc[0, 3] <= 100.0
    assert sigma_cs(df, k=3.0).iloc[0, 3] <= 100.0


def test_ic_constant_factor_is_nan():
    idx = pd.bdate_range("2024-01-01", periods=30)
    cols = list("ABCDE")
    rng = np.random.default_rng(1)
    close = pd.DataFrame(100 * (1 + rng.normal(0.001, 0.01, (30, 5))).cumprod(axis=0),
                         index=idx, columns=cols)
    factor = pd.DataFrame(1.0, index=idx, columns=cols)  # 常数因子
    y = forward_return(close, 5)
    t = ic_table(factor, y, min_n=3)
    assert t["ic"].isna().all() and t["rank_ic"].isna().all()  # 记缺失，不改成 0


def test_ic_recovers_sorted_signal():
    """构造与未来收益完全同序的因子 → IC 应接近 1。"""
    n = 40
    idx = pd.bdate_range("2024-01-01", periods=n)
    cols = list("ABCDE")
    rng = np.random.default_rng(2)
    close = pd.DataFrame(100 * (1 + rng.normal(0.001, 0.02, (n, 5))).cumprod(axis=0),
                         index=idx, columns=cols)
    y = forward_return(close, 5)
    # 直接用未来收益自身作为因子（前视构造，仅验证 IC 计算正确性）
    factor = y.copy()
    t = ic_table(factor, y, min_n=3)
    s = t["ic"].dropna()
    assert len(s) > 0 and s.mean() > 0.99


def test_quantile_groups_counts_and_spread():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2024-01-01", periods=50)
    cols = [f"S{i}" for i in range(10)]
    close = pd.DataFrame(100 * (1 + rng.normal(0.001, 0.02, (50, 10))).cumprod(axis=0),
                         index=idx, columns=cols)
    y = forward_return(close, 1)
    factor = pd.DataFrame(rng.normal(size=(50, 10)), index=idx, columns=cols)
    qt = quantile_table(factor, y, k=5, min_n=3)
    daily = qt["daily"]
    row = daily.iloc[0]
    formed = sum(row[f"g{g}_n_formed"] for g in range(1, 6))
    assert formed == 10  # 分组覆盖全部有效资产
    assert all(1 <= row[f"g{g}_n_formed"] <= 3 for g in range(1, 6))
    assert "spread_mean" in qt["summary"]


# ---------------- 分层净值（因子计算的附属统计量） ----------------
class _FakeMarket:
    """layered_nav 只用到 close_adj。"""

    def __init__(self, close: pd.DataFrame):
        self.close_adj = close


def test_layered_nav_monotone_when_factor_predicts():
    """因子 = 未来收益的完美预测：第 K 层年化应严格高于第 1 层，且净值单调递增。"""
    from quantlab.factors.layers import layered_nav

    rng = np.random.default_rng(11)
    n_days, n_codes = 120, 20
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_codes)]
    rets = rng.normal(0.0, 0.01, (n_days, n_codes))
    # 让"资产 i 的漂移"递增：因子值取资产编号，则高分资产长期跑赢
    drift = np.linspace(-0.002, 0.002, n_codes)
    close = pd.DataFrame(100 * (1 + rets + drift).cumprod(axis=0), index=idx, columns=cols)
    factor = pd.DataFrame(np.tile(np.arange(n_codes, dtype=float), (n_days, 1)),
                          index=idx, columns=cols)

    out = layered_nav(factor, _FakeMarket(close), groups=5, h=5,
                      commission_buy=0.0, commission_sell=0.0, trading_days=243)
    anns = [out["layers"][str(k)]["annualized_return"] for k in range(1, 6)]
    assert all(a is not None for a in anns)
    assert anns == sorted(anns), f"分层年化应单调递增，实际 {anns}"
    assert anns[-1] > anns[0]
    assert len(out["index"]) == n_days
    assert all(len(out["layers"][str(k)]["nav"]) == n_days for k in range(1, 6))
    # 第一个调仓日之前没有组合，净值如实为空缺；首个有效值归一为 1.0
    nav1 = out["layers"]["1"]["nav"]
    first_valid = next(i for i, v in enumerate(nav1) if v is not None)
    assert first_valid == 4, f"h=5 时首个调仓日应在第 5 个交易日（下标 4），实际 {first_valid}"
    assert nav1[first_valid] == pytest.approx(1.0, abs=1e-6)


def test_layered_nav_charges_cost_and_reduces_return():
    """扣费必须真的降低净值：同参数下收费的期末净值 < 免费的。"""
    from quantlab.factors.layers import layered_nav

    rng = np.random.default_rng(12)
    idx = pd.bdate_range("2024-01-01", periods=120)
    cols = [f"S{i:02d}" for i in range(20)]
    close = pd.DataFrame(100 * (1 + rng.normal(0.0, 0.01, (120, 20))).cumprod(axis=0),
                         index=idx, columns=cols)
    factor = pd.DataFrame(rng.normal(size=(120, 20)), index=idx, columns=cols)
    mk = _FakeMarket(close)
    free = layered_nav(factor, mk, groups=5, h=5, commission_buy=0.0, commission_sell=0.0)
    paid = layered_nav(factor, mk, groups=5, h=5, commission_buy=0.0015, commission_sell=0.0015)
    f_end = free["layers"]["5"]["nav"][-1]
    p_end = paid["layers"]["5"]["nav"][-1]
    assert paid["layers"]["5"]["turnover_sum"] > 0
    assert p_end < f_end, (f_end, p_end)


def test_layered_nav_rejects_short_window():
    from quantlab.factors.layers import layered_nav

    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    factor = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
    with pytest.raises(ValueError):
        layered_nav(factor, _FakeMarket(close), groups=5, h=20)


# ---------------- 复权口径（hfq / qfq） ----------------
def _stub_market(adj: pd.DataFrame, close_adj: pd.DataFrame):
    """不落盘地构造一个只管复权口径的 MarketData。

    真实对象从 {f}_raw.parquet + adj_factor 派生调整价；这里直接把给定的
    `close_adj` 当作"原始价"，配上传入的复权因子，用来验证派生逻辑本身。
    """
    from quantlab.data.clean import MarketData, PRICE_FIELDS

    md = object.__new__(MarketData)
    md.adj_factor = adj
    md._raw = {f: close_adj for f in PRICE_FIELDS}
    md.vwap_raw = close_adj
    md.volume = close_adj * 0 + 1_000_000.0      # 量类字段与口径无关，给个占位
    md.amount = close_adj * 1_000_000.0
    md._tau_cache = {}
    md._price_cache = {}
    md._view_cache = {}
    return md


def test_basis_scale_is_per_stock_constant():
    """核心命题：P_qfq = P_hfq · k，k = a(τ首)/a(τ末) 与 t 无关；hfq 无需缩放。"""
    idx = pd.bdate_range("2024-01-01", periods=6)
    cols = ["A", "B", "C"]
    # A 从未除权(a 恒为1)，B 中途除权翻倍，C 持续上升
    adj = pd.DataFrame({"A": [1.0] * 6, "B": [1, 1, 2, 2, 2, 2], "C": [1, 1.1, 1.2, 1.3, 1.5, 2.0]},
                       index=idx, columns=cols)
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    md = _stub_market(adj, close)

    assert md.basis_scale("hfq") is None          # 落盘的就是后复权，不用缩放
    k = md.basis_scale("qfq")
    assert k["A"] == pytest.approx(1.0)           # 从未除权 -> k = 1
    assert k["B"] == pytest.approx(1 / 2)         # a(τ首)/a(τ末) = 1/2
    assert k["C"] == pytest.approx(1 / 2.0)

    hfq, qfq = md.price("close", "hfq"), md.price("close", "qfq")
    ratio = hfq / qfq
    for c in cols:                                # 比值逐日恒等于 1/k
        assert np.allclose(ratio[c].to_numpy(), 1.0 / k[c])
    # 收益率完全不受口径影响
    assert np.allclose(hfq.pct_change().dropna().to_numpy(),
                       qfq.pct_change().dropna().to_numpy())


def test_basis_changes_price_level_factor_but_not_ratio_factor():
    """价格水平型因子截面排序会变；比值型因子（动量）完全不变。"""
    idx = pd.bdate_range("2024-01-01", periods=30)
    cols = [f"S{i}" for i in range(8)]
    rng = np.random.default_rng(7)
    base = pd.DataFrame(100 * (1 + rng.normal(0, 0.01, (30, 8))).cumprod(axis=0),
                        index=idx, columns=cols)
    # 每只股票不同的复权因子路径 -> 不同的 k
    factors = pd.DataFrame({c: np.linspace(1.0, 1.0 + 0.1 * (i + 1), 30)
                            for i, c in enumerate(cols)}, index=idx)
    md = _stub_market(factors, base)

    hfq, qfq = md.fields_for("hfq")["close"], md.fields_for("qfq")["close"]
    lh, lq = hfq.iloc[-1], qfq.iloc[-1]
    assert not np.allclose(lh.rank().to_numpy(), lq.rank().to_numpy())  # 水平型：排序变

    mom_h = hfq / hfq.shift(5) - 1
    mom_q = qfq / qfq.shift(5) - 1
    assert np.allclose(mom_h.dropna().to_numpy(), mom_q.dropna().to_numpy())  # 比值型：不变


def test_basis_unknown_rejected():
    idx = pd.bdate_range("2024-01-01", periods=3)
    md = _stub_market(pd.DataFrame(1.0, index=idx, columns=["A"]),
                      pd.DataFrame(1.0, index=idx, columns=["A"]))
    with pytest.raises(ValueError):
        md.basis_scale("bad")
    with pytest.raises(KeyError):           # 复权口径只对价格类字段有意义
        md.price("volume", "qfq")


# ---------------- 挖掘任务：因子计算参数 + 加权组合 ----------------
def test_task_spec_normalizes_and_fills_defaults():
    """任务里的 spec 走与因子页同一套 normalize（旧名可用、缺省补齐）。"""
    from quantlab.storage.store import task_spec

    sp = task_spec({"spec": {"outlier_method": "mad", "outlier_k": 2.5,
                             "ic_method": "spearman"}})
    assert sp["outlier_method"] == "MAD"      # 旧名 mad -> MAD
    assert sp["k"] == 2.5                     # 旧名 outlier_k -> k
    assert sp["ic_method"] == "spearman"
    assert sp["adjust"] == "hfq"              # 缺省补齐
    assert task_spec({})["quantile"] == 5     # 完全没有 spec 也能用


def test_weighted_model_keeps_explicit_weights_not_equal():
    """挖掘任务的因子池组合：权重来自算法，必须保留符号且不能退化成等权。"""
    from types import SimpleNamespace

    from quantlab.storage.store import signal_component_weights

    sig = SimpleNamespace(model_type="weighted",
                          components=[{"factor_id": 1, "weight": -1.0},
                                      {"factor_id": 2, "weight": 0.25}])
    w = signal_component_weights(sig)
    assert w[1] == pytest.approx(-1.0 / 1.25)   # 负权重（算法学到反向）必须保留
    assert w[2] == pytest.approx(0.25 / 1.25)
    assert w[1] != w[2]                          # 不是等权

    eq = SimpleNamespace(model_type="equal_weight",
                         components=[{"factor_id": 1, "weight": 1.0},
                                     {"factor_id": 2, "weight": 1.0}])
    assert signal_component_weights(eq) == {1: 0.5, 2: 0.5}


def test_migration_adds_missing_columns_idempotently():
    """create_all 不给已有表加列；迁移必须能补列且可重复执行。"""
    from sqlalchemy import inspect

    from quantlab.storage.db import _MIGRATIONS, _migrate, engine, init_db

    init_db()
    init_db()                                  # 幂等：第二次不应再产生 DDL
    assert _migrate() == []
    insp = inspect(engine)
    for table, cols in _MIGRATIONS.items():
        have = {c["name"] for c in insp.get_columns(table)}
        assert set(cols) <= have, f"{table} 缺少 {set(cols) - have}"


# ---------------- 挖掘任务的 train/valid/test 防泄漏纪律 ----------------
def test_segment_split_purges_label_overlap():
    """段末 h 天的标签会读到下一段的价格，必须被 purge 掉。

    这是防泄漏纪律的核心断言：train 最后一根 bar 的 h 期标签，其价格下标
    必须**严格小于** valid 第一根 bar 的下标；valid→test 同理。
    """
    from quantlab.lab.gp_engine import segment_split

    idx = pd.bdate_range("2023-01-03", periods=200)
    pos = {d: i for i, d in enumerate(idx)}
    for h in (1, 5, 10, 20):
        seg = segment_split(idx, purge=h, train_ratio=0.6, valid_ratio=0.2)
        tr, va, te = seg["train"], seg["valid"], seg["test"]
        assert len(tr) and len(va) and len(te)
        # train 末 + h 必须 < valid 首（中间留出 h 天 embargo）
        assert pos[tr[-1]] + h < pos[va[0]], f"h={h}: train 标签跨入 valid（泄漏）"
        assert pos[va[-1]] + h < pos[te[0]], f"h={h}: valid 标签跨入 test（泄漏）"
        # 空档长度正好是 h
        assert pos[va[0]] - pos[tr[-1]] - 1 == h
        assert pos[te[0]] - pos[va[-1]] - 1 == h


def test_segment_split_ratios_respected():
    from quantlab.lab.gp_engine import segment_split

    idx = pd.bdate_range("2023-01-03", periods=100)
    seg = segment_split(idx, purge=0, train_ratio=0.7, valid_ratio=0.15)
    assert len(seg["train"]) == 70
    assert len(seg["valid"]) == 15
    assert len(seg["test"]) == 15
    # 测试段至少留 1 天，不能因为比例过大而被吞掉
    seg2 = segment_split(idx, purge=0, train_ratio=0.8, valid_ratio=0.4)
    assert len(seg2["test"]) >= 1


def test_segment_split_explicit_bounds():
    """显式边界模式：用户自己定三个区间；缺的边界按相邻项推；非法组合必须报错。"""
    from quantlab.lab.gp_engine import segment_split

    idx = pd.bdate_range("2023-01-03", periods=100)
    pos = {d: i for i, d in enumerate(idx)}
    h = 5
    D = lambda i: str(idx[i].date())          # noqa: E731

    # 六个都填
    s = segment_split(idx, purge=h, bounds={
        "train_start": D(0), "train_end": D(39),
        "valid_start": D(50), "valid_end": D(69),
        "test_start": D(80), "test_end": D(99)})
    assert pos[s["train"][-1]] == 39 - h      # 段末 purge 掉了 h 天
    assert pos[s["valid"][0]] == 50
    assert pos[s["valid"][-1]] == 69 - h
    assert pos[s["test"][0]] == 80 and pos[s["test"][-1]] == 99

    # 只给两个切分点也成立
    s2 = segment_split(idx, purge=h, bounds={"valid_start": D(40), "test_start": D(70)})
    assert pos[s2["train"][-1]] + h < pos[s2["valid"][0]]
    assert pos[s2["valid"][-1]] + h < pos[s2["test"][0]]
    assert pos[s2["test"][-1]] == 99

    # 只给 train_end：valid/test 边界按比例兜底
    s3 = segment_split(idx, purge=h, bounds={"train_end": D(45)})
    assert pos[s3["train"][-1]] + h < pos[s3["valid"][0]]

    # 非法：train 与 valid 重叠
    with pytest.raises(ValueError):
        segment_split(idx, purge=h, bounds={"train_end": D(60), "valid_start": D(50)})
    # 非法：purge 后某段为空
    with pytest.raises(ValueError):
        segment_split(idx, purge=50, bounds={"valid_start": D(30), "test_start": D(40)})


def test_alphagen_segments_purge_matches_gp():
    """两个引擎的防泄漏规则必须一致（AlphaGen 曾经漏了 purge）。"""
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from quantlab.lab.alphagen_engine import AlphaGenEngine, AlphaGenParams
    from quantlab.lab.gp_engine import segment_split

    idx = pd.bdate_range("2023-01-03", periods=200)

    class _Mkt:
        dates = idx

    h = 5
    eng = AlphaGenEngine(_Mkt(), AlphaGenParams(label_horizon=h,
                                                train_ratio=0.6, valid_ratio=0.2))
    seg = eng._segments()
    gp = segment_split(idx, purge=h, train_ratio=0.6, valid_ratio=0.2)
    assert seg["train"][0] == str(gp["train"][0].date())
    assert seg["train"][1] == str(gp["train"][-1].date())
    assert seg["valid"][0] == str(gp["valid"][0].date())
    assert seg["valid"][1] == str(gp["valid"][-1].date())
    assert seg["test"][0] == str(gp["test"][0].date())
    assert seg["test"][1] == str(gp["test"][-1].date())




