"""alphagen5m RPN 表达式解释器（平台侧独立实现，语义与引擎 expressions.py 逐一对齐）。

用途：算法实验室的 alphagen_5m 候选（RPN token 串）在平台真实 5 分钟面板上求值，
实现"候选 → 一键采纳 → 平台统一评估"的闭环（与 GP 候选走同一条链路）。

对齐要点（源自 alphagen5m/expressions.py）：
  - 滚动窗口要求窗内全 finite（否则 NaN）；Std/Var 为总体口径（ddof=0）
  - Med/Mad/Rank/WMA/EMA 用显式 [time, stock, window] 分块表示（64M 元素预算）
  - WMA 核 = 1..w 归一；EMA 核 = (1-α)^(w-1..0)，α=2/(w+1)，归一
  - Rank = ((#<last) + (#<=last)) / (2w)；Mad = mean(|x - 窗均值|)
  - Cov/Corr 为总体口径，且要求成对全 finite
  - Div 分母 |y|<=1e-12 → NaN；Greater/Less = 逐元素 max/min
  - Log = log(|x|.clamp_min(1e-12))；CSRank = 截面序数秩 / n
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

UNARY = ("Abs", "Sign", "Log", "CSRank")
BINARY = ("Add", "Sub", "Mul", "Div", "Greater", "Less")
ROLLING1 = ("Ref", "Mean", "Sum", "Std", "Var", "Max", "Min", "Med", "Mad", "Rank", "Delta", "WMA", "EMA")
ROLLING2 = ("Cov", "Corr")
_BUDGET = 64 * 1024 * 1024  # 与引擎一致的显式窗口临时预算（~256MiB float32）


class MinuteExprEvaluator:
    """在真实 5 分钟面板上求值 RPN token 串。字段面板按需加载并缓存。"""

    def __init__(self, minute_dir: Path):
        self.dir = Path(minute_dir)
        self._fields: dict[str, pd.DataFrame] = {}

    # ---- 字段面板 ----
    def field(self, name: str) -> pd.DataFrame:
        key = name.lower()
        if key in self._fields:
            return self._fields[key]
        from .minute import panel_frame
        if key == "vwap" or key == "turnover_rate":
            amt, vol = self.field("amount"), self.field("volume")
            df = amt / vol.where(vol.abs() > 1e-12, np.nan)  # 引擎 5m 口径 vwap=amount/volume
        else:
            df = panel_frame(self.dir, field=key)
        self._fields[key] = df
        return df

    # ---- 求值 ----
    def eval_token_string(self, token_string: str) -> pd.DataFrame:
        stack: list[pd.DataFrame | float | int] = []
        for tok in token_string.split():
            if tok.startswith("$"):
                stack.append(self.field(tok[1:]))
            elif re.fullmatch(r"-?\d+", tok):
                stack.append(int(tok))            # 窗口
            elif tok in UNARY or tok in BINARY or tok in ROLLING1 or tok in ROLLING2:
                stack.append(self._apply(tok, stack))
            else:
                stack.append(float(tok))          # 常数
        if len(stack) != 1 or not isinstance(stack[0], pd.DataFrame):
            raise ValueError(f"RPN 表达式未收敛为面板: {token_string!r}")
        out = stack[0]
        return out.replace([np.inf, -np.inf], np.nan)

    # ---- 算子 ----
    def _apply(self, name: str, stack: list) -> pd.DataFrame:
        if name in UNARY:
            return self._unary(name, stack.pop())
        if name in BINARY:
            b, a = stack.pop(), stack.pop()
            return self._binary(name, a, b)
        arity = 3 if name in ROLLING2 else 2
        args = [stack.pop() for _ in range(arity)][::-1]
        w = int(args[-1])
        return self._rolling(name, args[0], w) if arity == 2 else self._pair(name, args[0], args[1], w)

    def _unary(self, name: str, x) -> pd.DataFrame:
        arr = x.to_numpy(dtype="float64")
        if name == "Abs":
            out = np.abs(arr)
        elif name == "Sign":
            out = np.sign(arr)
        elif name == "Log":
            out = np.log(np.clip(np.abs(arr), 1e-12, None))
        else:  # CSRank：截面序数秩 / n（NaN → +inf 排末位后掩蔽）
            safe = np.where(np.isnan(arr), np.inf, arr)
            order = np.argsort(safe, axis=1, kind="stable")
            ranks = np.empty_like(order)
            rows = np.arange(order.shape[0])[:, None]
            ranks[rows, order] = np.arange(order.shape[1])[None, :]
            n = np.isfinite(arr).sum(axis=1, keepdims=True).clip(min=1)
            out = ranks / n
            out[np.isnan(arr)] = np.nan
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    def _binary(self, name: str, a, b) -> pd.DataFrame:
        av = a.to_numpy(dtype="float64") if isinstance(a, pd.DataFrame) else np.array(a, dtype="float64")
        bv = b.to_numpy(dtype="float64") if isinstance(b, pd.DataFrame) else np.array(b, dtype="float64")
        if name == "Add":
            out = av + bv
        elif name == "Sub":
            out = av - bv
        elif name == "Mul":
            out = av * bv
        elif name == "Div":
            with np.errstate(invalid="ignore", divide="ignore"):
                out = av / np.where(np.abs(bv) > 1e-12, bv, np.nan)
        elif name == "Greater":
            out = np.maximum(av, bv)   # NaN 传播，与 torch.maximum 一致
        else:
            out = np.minimum(av, bv)
        return pd.DataFrame(out, index=self._idx(a, b), columns=self._cols(a, b))

    @staticmethod
    def _idx(a, b):
        return (a.index if isinstance(a, pd.DataFrame) else b.index)

    @staticmethod
    def _cols(a, b):
        return (a.columns if isinstance(a, pd.DataFrame) else b.columns)

    @staticmethod
    def _as_df(x, ref: pd.DataFrame) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x
        return pd.DataFrame(np.full(ref.shape, float(x), dtype="float64"),
                            index=ref.index, columns=ref.columns)

    def _rolling(self, name: str, x, w: int) -> pd.DataFrame:
        df = self._as_df(x, self.field("close")) if not isinstance(x, pd.DataFrame) else x
        arr = df.to_numpy(dtype="float64")
        finite = np.isfinite(arr)
        # Ref/Delta 必须先于 w==1 快捷分支处理（引擎语义：真正的 shift；
        # 历史 bug：w==1 时曾走恒等分支，Ref(x,1) 错误地返回 x）
        if name == "Ref":
            return pd.DataFrame(df.shift(w).to_numpy(), index=df.index, columns=df.columns)
        if name == "Delta":
            return pd.DataFrame((df - df.shift(w)).to_numpy(), index=df.index, columns=df.columns)
        if w == 1:
            if name in ("Std", "Var", "Mad"):
                out = np.where(finite, 0.0, np.nan)
            elif name == "Rank":
                out = np.where(finite, 0.5, np.nan)
            else:
                out = arr
            return pd.DataFrame(out, index=df.index, columns=df.columns)

        r = df.rolling(w, min_periods=w)  # min_periods=w ⇒ 窗内全 finite 才有效
        if name == "Mean":
            out = r.mean().to_numpy()
        elif name == "Sum":
            out = r.sum().to_numpy()
        elif name == "Std":
            out = r.std(ddof=0).to_numpy()
        elif name == "Var":
            out = r.var(ddof=0).to_numpy()
        elif name == "Max":
            out = r.max().to_numpy()
        elif name == "Min":
            out = r.min().to_numpy()
        elif name in ("Med", "Mad", "Rank", "WMA", "EMA"):
            out = self._window_ops(name, arr.astype("float32"), w)
        else:
            raise KeyError(name)
        return pd.DataFrame(out, index=df.index, columns=df.columns)

    @staticmethod
    def _window_ops(name: str, arr: np.ndarray, w: int) -> np.ndarray:
        out = np.full(arr.shape, np.nan, dtype="float64")
        if name in ("WMA", "EMA"):
            if name == "WMA":
                weights = np.arange(1, w + 1, dtype="float64")
            else:
                alpha = 2 / (w + 1)
                weights = (1 - alpha) ** np.arange(w - 1, -1, -1, dtype="float64")
            kernel = weights / weights.sum()
        cols = arr.shape[1]
        time_chunk = max(1, _BUDGET // max(1, cols * w))  # 64M 元素预算（镜像引擎）
        for start in range(0, len(arr), time_chunk):
            seg = arr[start:start + time_chunk]
            if len(seg) < w:  # 尾块不足一个窗口：整块对应行本就因窗口不全而无效
                break
            win = np.lib.stride_tricks.sliding_window_view(seg, w, axis=0)
            valid = (~np.isnan(win)).all(axis=-1)
            safe = np.nan_to_num(win).astype("float64")
            if name == "Med":
                o = np.median(safe, axis=-1)
            elif name == "Mad":
                o = np.abs(safe - safe.mean(axis=-1, keepdims=True)).mean(axis=-1)
            elif name == "Rank":
                last = safe[..., -1]
                o = ((safe < last[..., None]).sum(axis=-1)
                     + (safe <= last[..., None]).sum(axis=-1)) / (2 * w)
            else:  # WMA / EMA
                o = np.einsum("...t,t->...", safe, kernel)
            o = np.where(valid, o, np.nan)
            i0 = start + w - 1  # 滑窗第 i 个视图对应全局行 start+i+w-1
            out[i0:i0 + o.shape[0]] = o
        return out

    def _pair(self, name: str, x, y, w: int) -> pd.DataFrame:
        a = self._as_df(x, self.field("close"))
        b = self._as_df(y, self.field("close"))
        av, bv = a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64")
        both = np.isfinite(av) & np.isfinite(bv)
        if w == 1:
            out = np.where(both, 0.0, np.nan)
            return pd.DataFrame(out, index=a.index, columns=a.columns)
        cnt = pd.DataFrame(both.astype("float64"), index=a.index).rolling(w, min_periods=w).sum()
        az = pd.DataFrame(np.where(both, av, 0.0), index=a.index)
        bz = pd.DataFrame(np.where(both, bv, 0.0), index=a.index)
        sx, sy = az.rolling(w, min_periods=w).sum(), bz.rolling(w, min_periods=w).sum()
        sxx = (az * az).rolling(w, min_periods=w).sum()
        syy = (bz * bz).rolling(w, min_periods=w).sum()
        sxy = (az * bz).rolling(w, min_periods=w).sum()
        mx, my = sx / w, sy / w
        cov = sxy / w - mx * my
        if name == "Cov":
            out = cov
        else:
            vx = (sxx / w - mx * mx).clip(lower=0)
            vy = (syy / w - my * my).clip(lower=0)
            out = cov / np.sqrt(vx * vy).clip(lower=1e-12)
        out = out.where(cnt == w)
        return pd.DataFrame(out.to_numpy(), index=a.index, columns=a.columns)


def eval_5m_token_string(token_string: str, minute_dir: Path,
                         evaluator: MinuteExprEvaluator | None = None) -> pd.DataFrame:
    ev = evaluator or MinuteExprEvaluator(minute_dir)
    return ev.eval_token_string(token_string)
