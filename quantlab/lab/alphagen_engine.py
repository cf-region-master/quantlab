"""AlphaGen（日频 · CPU）挖掘引擎 —— 接入实验室的 MiningEngine 协议。

来源：`alphagen_5m_code_20260925` 的 RL 引擎，**只改数据层**：
  - 5 分钟 48 bar/日 → 日频 1 bar/日（见 `alphagen/daily_cache.py`）
  - 其余（表达式 DSL / 因子池 / MaskablePPO / 三段隔离）原样复用
  - 设备改为 CPU：原版 `train.py` 本来就是 `--device auto`，无 GPU 时自动回落 CPU

与 GP 引擎的差异：
  - GP 用符号回归（锦标赛 + 子树交叉变异），AlphaGen 用 RL（GRU 逐 token 生成表达式，
    奖励 = 因子池的 ensemble IC）
  - 二者共用同一套治理：train/valid/test 三段隔离、候选同时报告三段 IC、不按 test 择优

候选采纳：AlphaGen 的表达式树会被**转译成等价的 Python 源码**（`def factor(fields)`），
因此可直接走已有的**沙箱化用户因子**路径入库，不需要为它单开一条求值链路。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT

CACHE_DIR = ROOT / "data" / "store" / "alphagen_cache"


# ---------------- 参数 ----------------
@dataclass
class AlphaGenParams:
    steps: int = 8192              # PPO 总步数（CPU 约每分钟 2~4 万步）
    n_envs: int = 4
    rollout_steps: int = 128
    batch_size: int = 128
    pool_capacity: int = 10
    max_expr_length: int = 8
    label_horizon: int = 5
    d_model: int = 32
    n_layers: int = 1
    cell: str = "gru"
    seed: int = 0
    ridge: float = 1e-3
    novelty_coef: float = 0.0
    variant: str = "baseline"      # baseline | counterfactual | novelty | counterfactual_novelty
    train_ratio: float = 0.6       # 训练段占比（段末按 label_horizon purge）
    valid_ratio: float = 0.2       # 验证段占比；测试段取剩余
    split_bounds: dict | None = None   # 显式三段边界（给了就不用比例）

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class AlphaGenResult:
    candidates: list[dict] = field(default_factory=list)
    curve: list[dict] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ---------------- 表达式 → Python 源码 ----------------
# 表达式里的 feature 节点存的是 FeatureType 整数索引（见 alphagen/expressions.py）
_FEATURE_BY_INDEX = {0: "open", 1: "close", 2: "high", 3: "low", 4: "volume", 5: "vwap"}
_FEATURE_BY_NAME = {"open": "open", "close": "close", "high": "high", "low": "low",
                    "volume": "volume", "turnover_rate": "turnover", "vwap": "vwap"}
_UNARY = {
    "Abs": lambda a: f"({a}).abs()",
    "Sign": lambda a: f"np.sign({a})",
    "Log": lambda a: f"(np.sign({a}) * np.log1p(({a}).abs()))",
    "CSRank": lambda a: f"({a}).rank(axis=1, pct=True)",
}
_BINARY = {
    "Add": lambda a, b: f"({a} + {b})",
    "Sub": lambda a, b: f"({a} - {b})",
    "Mul": lambda a, b: f"({a} * {b})",
    "Div": lambda a, b: f"(({a}) / ({b}).where(({b}).abs() > 1e-12, np.nan))",
    "Greater": lambda a, b: f"(({a}) > ({b})).astype(float)",
    "Less": lambda a, b: f"(({a}) < ({b})).astype(float)",
}
_ROLL = {
    "Ref": lambda a, w: f"({a}).shift({w})",
    "Mean": lambda a, w: f"({a}).rolling({w}, min_periods={w}).mean()",
    "Sum": lambda a, w: f"({a}).rolling({w}, min_periods={w}).sum()",
    "Std": lambda a, w: f"({a}).rolling({w}, min_periods=max(2, {w} // 2)).std()",
    "Var": lambda a, w: f"({a}).rolling({w}, min_periods=max(2, {w} // 2)).var()",
    "Max": lambda a, w: f"({a}).rolling({w}, min_periods={w}).max()",
    "Min": lambda a, w: f"({a}).rolling({w}, min_periods={w}).min()",
    "Med": lambda a, w: f"({a}).rolling({w}, min_periods={w}).median()",
    "Mad": lambda a, w: f"(({a}) - ({a}).rolling({w}, min_periods={w}).median()).abs()"
                        f".rolling({w}, min_periods=max(2, {w} // 2)).mean()",
    "Rank": lambda a, w: f"({a}).rolling({w}, min_periods={w}).rank(pct=True)",
    "Delta": lambda a, w: f"(({a}) - ({a}).shift({w}))",
    "WMA": lambda a, w: f"({a}).rolling({w}, min_periods={w}).apply("
                        f"lambda v: float(np.dot(v, np.arange(1, {w} + 1)) / ({w} * ({w} + 1) / 2)), raw=True)",
    "EMA": lambda a, w: f"({a}).ewm(span={w}, adjust=False).mean()",
    "Cov": lambda a, b, w: f"({a}).rolling({w}, min_periods=max(2, {w} // 2)).cov({b})",
    "Corr": lambda a, b, w: f"({a}).rolling({w}, min_periods=max(2, {w} // 2)).corr({b})",
}


def expr_to_python(expr) -> str:
    """把 AlphaGen 的 Expr 树转成等价的中缀 Python 表达式（作用在宽表上）。"""
    name = getattr(expr, "name", None)
    args = getattr(expr, "args", ())
    value = getattr(expr, "value", None)
    if name == "feature":
        field = _FEATURE_BY_INDEX.get(value)
        if field is None:
            field = _FEATURE_BY_NAME.get(str(value).lower())
        if field is None:
            raise ValueError(f"未知字段 {value!r}")
        return f'fields["{field}"]'
    if name == "const":
        return repr(float(value))
    if name in _UNARY and len(args) == 1:
        return _UNARY[name](expr_to_python(args[0]))
    if name in _BINARY and len(args) == 2:
        return _BINARY[name](expr_to_python(args[0]), expr_to_python(args[1]))
    if name in _ROLL:
        w = int(value or 1)
        if name in ("Cov", "Corr"):
            if len(args) != 2:
                raise ValueError(f"{name} 需要两个参数")
            return _ROLL[name](expr_to_python(args[0]), expr_to_python(args[1]), w)
        if len(args) != 1:
            raise ValueError(f"{name} 需要一个参数")
        return _ROLL[name](expr_to_python(args[0]), w)
    raise ValueError(f"暂不支持的算子 {name}（args={len(args)}）")


def expr_to_factor_code(expr, name: str = "alphagen") -> str:
    """生成可直接提交到"手写因子"的完整源码。"""
    body = expr_to_python(expr)
    return ("def factor(fields):\n"
            f'    """AlphaGen-RL 挖掘候选（{name}）。字段为 date×code 调整价宽表。"""\n'
            f"    return {body}\n")


# ---------------- 引擎 ----------------
class AlphaGenEngine:
    """实现 MiningEngine 协议的 AlphaGen 适配器（CPU · 日频）。"""

    engine_id = "alphagen_daily_cpu"
    display_name = "AlphaGen-RL（日频 · CPU）"
    supported_objectives = ["factor_mining"]
    resource_class = "cpu"

    def __init__(self, market, params: AlphaGenParams, cache_dir=None):
        self.market = market
        self.p = params
        # 缓存目录按「口径 + 区间 + 池」分目录（market 已是 scoped_view），
        # 否则换一次区间/池就要重建整份缓存
        self.cache_dir = cache_dir
        self._cancel = False
        self.result = AlphaGenResult()

    def cancel(self):
        self._cancel = True

    def _segments(self):
        """训练/验证/测试三段的日期区间，段末按 label_horizon 做 purge。

        支持两种模式：比例（train_ratio/valid_ratio）或**显式边界**（split_bounds）。
        两者都复用 `gp_engine.segment_split`，保证与 GP 的切分与防泄漏规则逐日一致。

        ⚠️ 这里曾经漏了 purge：train 段最后一根 bar 的 h 期标签用的是 t+h 的价格，
        那个价格落在 valid 段内 —— 等于训练时偷看了验证段。GP 那边一直有 purge，
        AlphaGen 这条没跟上，两个引擎的防泄漏纪律不一致，已按同样规则修正。
        """
        from .gp_engine import segment_split

        h = int(self.p.label_horizon)
        seg = segment_split(
            self.market.dates, purge=h,
            train_ratio=getattr(self.p, "train_ratio", 0.6),
            valid_ratio=getattr(self.p, "valid_ratio", 0.2),
            bounds=getattr(self.p, "split_bounds", None))
        return {k: (str(v.min().date()), str(v.max().date())) for k, v in seg.items()}

    def _segment_report(self, seg: dict) -> dict:
        """把三段区间整理成页面可直接展示的形状（起止 + 交易日数）。"""
        out = {}
        for k, (a, b) in seg.items():
            m = (self.market.dates >= pd.Timestamp(a)) & (self.market.dates <= pd.Timestamp(b))
            out[k] = {"start": a, "end": b, "n_bars": int(m.sum())}
        out["purge_days"] = int(self.p.label_horizon)
        out["note"] = (f"段末按持有期 {int(self.p.label_horizon)} 天 purge，"
                       f"两段之间留出同长度的 embargo 空档（该空档不计入任何段）")
        return out

    def run(self, report=None) -> AlphaGenResult:
        import torch
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback

        from .alphagen.data import MarketData as AgMarket
        from .alphagen.daily_cache import build_daily_cache
        from .alphagen.env import BatchedAlphaVecEnv
        from .alphagen.policy import RecurrentExtractor
        from .alphagen.pool import AlphaPool, Calculator

        p = self.p
        t0 = time.time()
        seg = self._segments()
        cache = build_daily_cache(self.market, self.cache_dir or CACHE_DIR)
        dev = torch.device("cpu")          # 无 GPU：显式 CPU，不静默失败
        m = AgMarket.load(cache, dev, p.label_horizon, False,
                          seg["train"][0], seg["train"][1],
                          seg["valid"][0], seg["valid"][1],
                          seg["test"][0], seg["test"][1])
        self.result.log.append(f"[data] {m.summary()}")
        self.result.log.append(f"[data] 缓存 {cache}；设备 cpu")
        if report:
            report(progress=0.02, stage="data", log_lines=self.result.log[-2:])

        calc = Calculator(m, "train", 512, "fp32")
        valid_calc = calc.view("validation")
        test_calc = calc.view("test")
        counterfactual = "counterfactual" in p.variant
        novelty = "novelty" in p.variant
        pool = AlphaPool(p.pool_capacity, calc,
                         "counterfactual" if counterfactual else "baseline",
                         p.novelty_coef if novelty else 0.0, p.ridge, 0.995, 2)
        env = BatchedAlphaVecEnv(pool, p.n_envs, p.max_expr_length)

        eng = self

        class _CB(BaseCallback):
            def _on_step(self):
                if eng._cancel:
                    return False
                n = int(self.num_timesteps)
                if n % max(1, p.rollout_steps) == 0:
                    msg = (f"[gen] step={n} pool={len(pool.entries)} "
                           f"elapsed={time.time()-t0:.0f}s")
                    eng.result.log.append(msg)
                    eng.result.curve.append({"generation": n, "pool_size": len(pool.entries)})
                    if report:
                        report(progress=min(0.98, 0.02 + 0.96 * n / max(1, p.steps)),
                               stage="training", log_lines=[msg])
                return True

        model = MaskablePPO(
            "MlpPolicy", env, seed=p.seed, device=dev, gamma=1.0, ent_coef=0.01,
            n_steps=p.rollout_steps, batch_size=p.batch_size, verbose=0,
            policy_kwargs={"features_extractor_class": RecurrentExtractor,
                           "features_extractor_kwargs": {"d_model": p.d_model,
                                                         "n_layers": p.n_layers,
                                                         "cell": p.cell}})
        model.learn(total_timesteps=p.steps, callback=_CB(), progress_bar=False)

        # 候选：池内因子。**逐候选**计算 train/valid/test 的 RankIC（诚实展示，不按 test 择优）
        from .gp_engine import rankic_mean
        from ..factors.user_code import compute_user_factor

        mk = self.market
        cands = []
        # 池内因子的线性组合权重（AlphaPool 在训练中学到的，已归一化）：
        # 一起带出去，采纳后可以直接把"因子池"落成一个带真实权重的策略。
        try:
            w_arr = [float(w) for w in np.asarray(pool.weights, dtype="float64").ravel()]
        except Exception:  # noqa: BLE001
            w_arr = []
        for i_entry, entry in enumerate(list(pool.entries)):
            expr = entry.expr
            try:
                code = expr_to_factor_code(expr, str(expr)[:40])
                values = compute_user_factor(code, mk.fields)
            except Exception as e:  # noqa: BLE001 —— 单个候选失败不影响其余
                self.result.log.append(f"[skip] {str(expr)[:50]} -> {type(e).__name__}: {e}")
                continue
            row = {"expression": str(expr), "code": code, "size": int(expr.length),
                   "pool_weight": (round(w_arr[i_entry], 8)
                                   if i_entry < len(w_arr) else None)}
            for tag in ("train", "valid", "test"):
                lo, hi = seg[tag]
                idx = [d for d in values.index
                       if lo <= str(d.date()) <= hi]
                try:
                    row[f"{tag}_ic"] = round(
                        float(rankic_mean(values, mk.close_adj, idx, p.label_horizon)), 6)
                except Exception:  # noqa: BLE001
                    row[f"{tag}_ic"] = None
            cands.append(row)
        self.result.candidates = cands
        try:
            self.result.summary = {
                "pool_size": len(pool.entries),
                "train": pool.metrics(),
                "validation": pool.metrics(valid_calc),
                "test": pool.metrics(test_calc),
                "data": m.summary(),
                "segments": self._segment_report(seg),
                "steps": p.steps, "device": "cpu",
                "elapsed_seconds": round(time.time() - t0, 1),
            }
        except Exception as e:  # noqa: BLE001
            self.result.summary = {"error": f"{type(e).__name__}: {e}"}
        self.result.log.append(f"[done] 候选 {len(cands)} 条，用时 {time.time()-t0:.1f}s")
        if report:
            report(progress=1.0, stage="done", log_lines=[self.result.log[-1]])
        return self.result
