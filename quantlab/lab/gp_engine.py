"""算法实验室 · GP 遗传编程挖掘引擎（factor_mining）。

设计（与产品设计方案 5.3 引擎插件协议一致）：
  - 表达式：字段(close/open/high/low/volume) + 常数 + 白名单算子（滚动/截面）
  - 适应度：训练段 RankIC 均值（固定持有期 h）；表达式 token 数超限罚 -1
  - 泛化纪律：train/valid/test 三段区间隔离，候选同时报告三段 IC（诚实展示，不择优）
  - 确定性：seed 固定 → 同参数重跑候选一致
  - 进度回调：generation/总代数、最优适应度、日志（对接 AlgoTask 状态机）
AlphaGen-RL 日频/分钟频引擎以同一协议适配（容器化运行），见 README 的引擎接入说明。
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable

import numpy as np
import pandas as pd

WINDOWS = (3, 5, 10, 20, 60)
FIELDS = ("close", "open", "high", "low", "volume")
CONSTANTS = (1.0, 0.5, 2.0, -1.0)
MAX_DEPTH = 4
MAX_TOKENS = 20


# ---------------- 表达式树 ----------------
def rand_leaf(rng: random.Random) -> tuple:
    if rng.random() < 0.75:
        return ("field", rng.choice(FIELDS))
    return ("const", rng.choice(CONSTANTS))


def rand_expr(rng: random.Random, depth: int | None = None) -> tuple:
    d = rng.randint(1, MAX_DEPTH) if depth is None else depth
    if d <= 0:
        return rand_leaf(rng)
    kind = rng.random()
    w = rng.choice(WINDOWS)
    if kind < 0.30:  # 一元滚动
        return (rng.choice(["ref", "mean", "std", "max", "min", "delta"]),
                rand_expr(rng, d - 1), ("const", float(w)))
    if kind < 0.45:  # 截面算子
        return ("csrank", rand_expr(rng, d - 1))
    if kind < 0.60:  # 一元
        return (rng.choice(["neg", "abs", "log"]), rand_expr(rng, d - 1))
    # 二元
    return (rng.choice(["add", "sub", "mul", "div"]), rand_expr(rng, d - 1), rand_expr(rng, d - 1))


def expr_tokens(e: tuple) -> int:
    if not isinstance(e, tuple):
        return 1
    return 1 + sum(expr_tokens(a) for a in e[1:])


def expr_infix(e: tuple) -> str:
    op = e[0]
    if op == "field":
        return e[1]
    if op == "const":
        return str(e[1])
    if op in ("ref", "mean", "std", "max", "min", "delta"):
        return f"{op}({expr_infix(e[1])}, {int(e[2][1])})"
    if op == "csrank":
        return f"cs_rank({expr_infix(e[1])})"
    if op in ("neg", "abs", "log"):
        return f"{op}({expr_infix(e[1])})"
    a, b = expr_infix(e[1]), expr_infix(e[2])
    return f"({a} {op} {b})"


def expr_key(e: tuple) -> str:
    return hashlib.sha256(expr_infix(e).encode("utf-8")).hexdigest()[:16]


def collect_subtrees(e: tuple, acc: list | None = None) -> list[tuple]:
    acc = acc if acc is not None else []
    acc.append(e)
    for a in e[1:]:
        if isinstance(a, tuple) and a and isinstance(a[0], str):
            if a[0] in ("field",):
                continue
            collect_subtrees(a, acc)
    return acc


def replace_random_subtree(e: tuple, repl: tuple, rng: random.Random) -> tuple:
    if not isinstance(e, tuple) or e[0] in ("field", "const"):
        return repl
    subs = collect_subtrees(e)
    target = rng.choice([s for s in subs if s[0] not in ("field", "const")] or [e])
    return _replace(e, target, repl, rng)


def _replace(e: tuple, target: tuple, repl: tuple, rng: random.Random) -> tuple:
    if e == target:
        return repl
    if not isinstance(e, tuple):
        return e
    args = []
    for a in e[1:]:
        if isinstance(a, tuple) and isinstance(a[0], str) and a[0] not in ("field", "const"):
            args.append(_replace(a, target, repl, rng))
        else:
            args.append(a)
    return (e[0], *args)


def mutate(e: tuple, rng: random.Random) -> tuple:
    r = rng.random()
    if r < 0.4:  # 子树替换
        return replace_random_subtree(e, rand_expr(rng, rng.randint(1, 2)), rng)
    if r < 0.7:  # 窗口扰动
        return _mutate_window(e, rng)
    return rand_expr(rng, rng.randint(1, MAX_DEPTH)) if rng.random() < 0.2 else _point_mutate(e, rng)


def _mutate_window(e: tuple, rng: random.Random) -> tuple:
    if not isinstance(e, tuple):
        return e
    args = []
    for a in e[1:]:
        if isinstance(a, tuple) and a[0] == "const" and a[1] in WINDOWS:
            args.append(("const", float(rng.choice(WINDOWS))))
        elif isinstance(a, tuple) and isinstance(a[0], str) and a[0] not in ("field", "const"):
            args.append(_mutate_window(a, rng))
        else:
            args.append(a)
    return (e[0], *args)


def _point_mutate(e: tuple, rng: random.Random) -> tuple:
    if not isinstance(e, tuple):
        return rand_leaf(rng)
    if e[0] == "field" and rng.random() < 0.3:
        return ("field", rng.choice(FIELDS))
    if e[0] == "const":
        return ("const", rng.choice(CONSTANTS))
    args = []
    for a in e[1:]:
        if isinstance(a, tuple) and isinstance(a[0], str) and a[0] not in ("field", "const"):
            args.append(_point_mutate(a, rng))
        else:
            args.append(a)
    return (e[0], *args)


def crossover(a: tuple, b: tuple, rng: random.Random) -> tuple:
    sa = rng.choice([s for s in collect_subtrees(a) if s[0] not in ("field", "const")] or [a])
    sb = rng.choice(collect_subtrees(b))
    return _replace(a, sa, sb, rng)


# ---------------- 求值（全部向量化，NaN 传播） ----------------
def _safe_div(a, b):
    return a / b.where(b.abs() > 1e-12, np.nan) if isinstance(b, pd.DataFrame) else a / b


def eval_expr(e: tuple, fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    op = e[0]
    if op == "field":
        return fields[e[1]]
    if op == "const":
        return None  # 常数按标量处理
    if op in ("ref", "mean", "std", "max", "min", "delta"):
        x = eval_expr(e[1], fields)
        w = int(e[2][1])
        if op == "ref":
            return x.shift(w)
        if op == "mean":
            return x.rolling(w, min_periods=w).mean()
        if op == "std":
            return x.rolling(w, min_periods=max(2, int(w * 0.6))).std(ddof=1)
        if op == "max":
            return x.rolling(w, min_periods=w).max()
        if op == "min":
            return x.rolling(w, min_periods=w).min()
        return x - x.shift(w)  # delta
    x = eval_expr(e[1], fields)
    if op == "csrank":
        return x.rank(axis=1, pct=True)
    if op == "neg":
        return -x
    if op == "abs":
        return x.abs()
    if op == "log":
        return np.sign(x) * np.log1p(x.abs())
    y = eval_expr(e[2], fields)
    yv = y if isinstance(y, pd.DataFrame) else float(e[2][1])
    if op == "add":
        return x + yv
    if op == "sub":
        return x - yv
    if op == "mul":
        return x * yv
    if op == "div":
        return _safe_div(x, yv)
    raise KeyError(f"未知算子: {op}")


# ---------------- 引擎 ----------------
@dataclass
class GpParams:
    population_size: int = 60
    generations: int = 15
    tournament_k: int = 4
    p_crossover: float = 0.7
    p_mutate: float = 0.25
    elitism: int = 2
    horizon: int = 5
    seed: int = 7
    top_candidates: int = 12
    train_ratio: float = 0.6
    valid_ratio: float = 0.2
    split_bounds: dict | None = None     # 显式三段边界（给了就不用比例）

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class GpResult:
    candidates: list[dict] = dc_field(default_factory=list)
    curve: list[dict] = dc_field(default_factory=list)   # {generation, best_ic, mean_ic, n_eval}
    log: list[str] = dc_field(default_factory=list)


def segment_split(dates: pd.DatetimeIndex, purge: int = 0,
                  train_ratio: float = 0.6, valid_ratio: float = 0.2,
                  bounds: dict | None = None) -> dict[str, pd.DatetimeIndex]:
    """训练/验证/测试三段切分。两种模式：

    ① 比例模式（默认）：train_ratio / valid_ratio 定边界，测试段取剩余。
    ② **显式边界模式**（给了 bounds 里任意一项就走这条）：用户自己定三个区间。
       bounds 可含 train_start/train_end/valid_start/valid_end/test_start/test_end，
       缺省项按相邻边界或区间首末补齐（所以只填 valid_start 和 test_start 也成立）。

    防泄漏：h 期标签 y(t)=C(t+h)/C(t)-1 会用到 t+h 的价格，故每段末尾 h 天的标签
    落在下一段内。purge=h 把这些日期从 train/valid 末尾剔除 —— 两段之间因此留出
    h 天 embargo 空档（该空档不属于任何一段）。用户声明的边界照常 purge，
    不会被静默放宽；purge 后不足 1 天的段直接报错。
    """
    n = len(dates)
    if not n:
        raise ValueError("切分区间为空")

    def _pos(d, default: int) -> int:
        """日期 -> 该日期在日历中的下标（不在日历上则取其后第一个交易日）。"""
        if not d:
            return default
        ts = pd.Timestamp(d)
        i = int(dates.searchsorted(ts, side="left"))
        return min(max(i, 0), n)

    if bounds and any(bounds.get(k) for k in
                      ("train_start", "train_end", "valid_start", "valid_end",
                       "test_start", "test_end")):
        b = bounds
        # 比例模式的边界作为缺省兜底
        tr_r = min(max(float(train_ratio), 0.1), 0.8)
        va_r = min(max(float(valid_ratio), 0.05), 0.9 - tr_r)
        d_vs = max(1, int(round(n * tr_r)))
        d_ts = min(max(d_vs + 1, int(round(n * (tr_r + va_r)))), n - 1)

        def _idx(key):
            v = b.get(key)
            return _pos(v, -1) if v else None

        # 两个切分点（valid 起点 / test 起点）先定；缺的用相邻边界或比例推。
        # 这样用户只填 valid_start + test_start 也成立，只填 train_end 也成立。
        i_vs = _idx("valid_start")
        i_te = _idx("train_end")
        if i_vs is None:
            i_vs = (i_te + 1) if i_te is not None else d_vs
        if i_te is None:
            i_te = i_vs                       # train 到 valid 起点为止（闭区间 [.., i_vs-1]）
        else:
            i_te = i_te + 1                   # 闭区间 -> 开区间边界

        i_ts2 = _idx("test_start")
        i_ve = _idx("valid_end")
        if i_ts2 is None:
            i_ts2 = (i_ve + 1) if i_ve is not None else d_ts
        i_ve = i_ts2 if i_ve is None else i_ve + 1

        i_0 = _pos(b.get("train_start"), 0)
        i_n = _idx("test_end")
        i_n = (n - 1) if i_n is None else i_n          # 闭区间
        if not (0 <= i_0 <= i_te <= i_vs <= i_ve <= i_ts2 <= i_n + 1 <= n):
            raise ValueError(
                "三段边界必须满足 train_start ≤ train_end < valid_start ≤ valid_end "
                f"< test_start ≤ test_end（都在挖掘区间内）。解析出的下标："
                f"train=[{i_0},{i_te - 1}] valid=[{i_vs},{i_ve - 1}] "
                f"test=[{i_ts2},{i_n}]，区间共 {n} 个交易日")
        tr, va, te = dates[i_0:i_te], dates[i_vs:i_ve], dates[i_ts2:i_n + 1]
    else:
        tr_r = min(max(float(train_ratio), 0.1), 0.8)
        va_r = min(max(float(valid_ratio), 0.05), 0.9 - tr_r)
        i1 = max(1, int(round(n * tr_r)))
        i2 = min(max(i1 + 1, int(round(n * (tr_r + va_r)))), n - 1)   # 测试段至少留 1 天
        tr, va, te = dates[:i1], dates[i1:i2], dates[i2:]

    if purge > 0:
        tr = tr[: max(0, len(tr) - purge)]
        va = va[: max(0, len(va) - purge)]
    if not len(tr) or not len(va) or not len(te):
        raise ValueError(
            f"purge({purge} 天) 之后有段为空：train={len(tr)} valid={len(va)} test={len(te)}。"
            f"请把区间拉长或把边界拉开（每段在 purge 后至少要留 1 个交易日）")
    return {"train": tr, "valid": va, "test": te}


def rankic_mean(values: pd.DataFrame, close_adj: pd.DataFrame, dates_idx, h: int, min_n: int = 10) -> float:
    y = (close_adj.shift(-h) / close_adj - 1).loc[dates_idx]
    f = values.loc[dates_idx]
    ics = []
    farr, yarr = f.to_numpy(), y.to_numpy()
    for i in range(farr.shape[0]):
        m = np.isfinite(farr[i]) & np.isfinite(yarr[i])
        if m.sum() < min_n:
            continue
        xv = pd.Series(farr[i][m]).rank(method="average").to_numpy()
        yv = pd.Series(yarr[i][m]).rank(method="average").to_numpy()
        if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
            continue
        ics.append(np.corrcoef(xv, yv)[0, 1])
    return float(np.mean(ics)) if ics else -1.0


class GpEngine:
    """实现 MiningEngine 协议的 GP 适配器（课程版内置，进程内运行）。"""

    engine_id = "gp_daily"
    display_name = "GP 遗传编程（日频）"
    supported_objectives = ["factor_mining"]
    resource_class = "cpu"

    def __init__(self, market, params: GpParams):
        self.market = market
        self.p = params
        # purge = 持有期 h：段末 h 天的标签会跨入下一段，须剔除后再评估
        self.segments = segment_split(market.dates, purge=params.horizon,
                                      train_ratio=params.train_ratio,
                                      valid_ratio=params.valid_ratio,
                                      bounds=getattr(params, "split_bounds", None))
        self.y = market.close_adj.shift(-params.horizon) / market.close_adj - 1
        self.cache: dict[str, float] = {}
        self.rng = random.Random(params.seed)
        self.result = GpResult()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _segment_report(self) -> dict:
        """三段区间的可展示形状（起止 + 交易日数 + purge 说明）。"""
        out = {}
        for k, idx in self.segments.items():
            out[k] = {"start": str(idx.min().date()) if len(idx) else None,
                      "end": str(idx.max().date()) if len(idx) else None,
                      "n_bars": int(len(idx))}
        out["purge_days"] = int(self.p.horizon)
        out["note"] = (f"段末按持有期 {int(self.p.horizon)} 天 purge，"
                       f"两段之间留出同长度的 embargo 空档（该空档不计入任何段）")
        return out

    # ---- 适应度（带缓存） ----
    def fitness(self, expr: tuple) -> float:
        if expr_tokens(expr) > MAX_TOKENS:
            return -1.0
        k = expr_key(expr)
        if k in self.cache:
            return self.cache[k]
        try:
            with np.errstate(all="ignore"):
                vals = eval_expr(expr, self.market.fields)
            if not isinstance(vals, pd.DataFrame):
                vals = pd.DataFrame(vals, index=self.market.dates, columns=self.market.codes)
            vals = vals.replace([np.inf, -np.inf], np.nan)
            if np.isfinite(vals.to_numpy()).sum() < 100:
                ic = -1.0
            else:
                ic = rankic_mean(vals, self.market.close_adj, self.segments["train"],
                                 self.p.horizon)
        except Exception:
            ic = -1.0
        ic = -1.0 if (ic is None or not math.isfinite(ic)) else ic
        self.cache[k] = ic
        return ic

    def _tournament(self, pop: list[tuple], fits: list[float]) -> tuple:
        best = None
        for _ in range(self.p.tournament_k):
            i = self.rng.randrange(len(pop))
            if best is None or fits[i] > fits[best]:
                best = i
        return pop[best]

    def run(self, report: Callable[..., None] | None = None) -> GpResult:
        p = self.p
        pop = [rand_expr(self.rng) for _ in range(p.population_size)]
        hist: dict[str, dict] = {}
        t0 = time.time()
        for gen in range(1, p.generations + 1):
            if self._cancel:
                self.result.log.append("cancelled")
                break
            fits = [self.fitness(e) for e in pop]
            order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
            best_i = order[0]
            curve_row = {"generation": gen, "best_ic": fits[best_i],
                         "mean_ic": float(np.mean(fits)), "n_eval": len(self.cache)}
            self.result.curve.append(curve_row)
            line = f"[gen {gen:>3}/{p.generations}] best RankIC(train)={fits[best_i]:+.4f} " \
                   f"mean={np.mean(fits):+.4f} evals={len(self.cache)} elapsed={time.time()-t0:.0f}s"
            self.result.log.append(line)
            if report:
                report(progress=gen / p.generations, stage="training",
                       metrics={"best_train_rank_ic": fits[best_i]}, log_lines=[line])

            # 登记候选（含验证/测试段 IC，诚实展示）
            for i in order[:5]:
                k = expr_key(pop[i])
                if fits[i] <= -0.5 or k in hist:
                    continue
                with np.errstate(all="ignore"):
                    vals = eval_expr(pop[i], self.market.fields)
                    if not isinstance(vals, pd.DataFrame):
                        vals = pd.DataFrame(vals, index=self.market.dates, columns=self.market.codes)
                    vals = vals.replace([np.inf, -np.inf], np.nan)
                hist[k] = {
                    "expression": expr_infix(pop[i]),
                    "expr_json": pop[i],
                    "train_ic": round(fits[i], 6),
                    "valid_ic": round(rankic_mean(vals, self.market.close_adj,
                                                  self.segments["valid"], p.horizon), 6),
                    "test_ic": round(rankic_mean(vals, self.market.close_adj,
                                                 self.segments["test"], p.horizon), 6),
                    "size": expr_tokens(pop[i]),
                }

            # 进化
            new_pop = [pop[i] for i in order[: p.elitism]]
            while len(new_pop) < p.population_size:
                r = self.rng.random()
                if r < p.p_crossover:
                    child = crossover(self._tournament(pop, fits), self._tournament(pop, fits), self.rng)
                elif r < p.p_crossover + p.p_mutate:
                    child = mutate(self._tournament(pop, fits), self.rng)
                else:
                    child = rand_expr(self.rng)
                new_pop.append(child)
            pop = new_pop

        ranked = sorted(hist.values(), key=lambda c: c["train_ic"], reverse=True)
        self.result.candidates = ranked[: p.top_candidates]
        self.result.log.append(f"[done] 候选 {len(self.result.candidates)} 条，"
                               f"用时 {time.time()-t0:.1f}s， evaluations={len(self.cache)}")
        return self.result
