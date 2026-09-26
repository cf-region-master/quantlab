from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor


MAX_LOOKBACK = 1200


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    TURNOVER_RATE = 5
    VWAP = 5  # Backward-compatible alias for the old 5-minute cache.


@dataclass(frozen=True)
class Expr:
    name: str
    args: tuple["Expr", ...] = ()
    value: float | int | None = None

    @property
    def featured(self) -> bool:
        return self.name == "feature" or any(arg.featured for arg in self.args)

    @property
    def depth(self) -> int:
        return 1 if not self.args else 1 + max(arg.depth for arg in self.args)

    @property
    def length(self) -> int:
        return 1 + sum(arg.length for arg in self.args)

    @property
    def lookback(self) -> int:
        own = 0
        if self.name in {"Ref", "Delta"}: own = int(self.value or 0)
        elif self.name in ROLLING_NAMES: own = max(0, int(self.value or 0) - 1)
        return own + max((arg.lookback for arg in self.args), default=0)

    def __str__(self) -> str:
        if self.name == "feature":
            return f"${FeatureType(int(self.value)).name.lower()}"
        if self.name == "constant":
            return str(self.value)
        if not self.args:
            return self.name
        suffix = f",{int(self.value)}" if self.name in ROLLING_NAMES else ""
        return f"{self.name}({','.join(map(str, self.args))}{suffix})"

    def evaluate(self, data: Tensor, active_mask: Tensor | None = None) -> Tensor:
        if self.name == "feature":
            return data[:, int(self.value), :]
        if self.name == "constant":
            return torch.full_like(data[:, 0, :], float(self.value))
        xs = [arg.evaluate(data, active_mask) for arg in self.args]
        if self.name == "Abs": return xs[0].abs()
        if self.name == "Sign": return xs[0].sign()
        if self.name == "Log": return torch.log(xs[0].abs().clamp_min(1e-12))
        if self.name == "CSRank":
            return cross_section_rank(xs[0] if active_mask is None else xs[0].masked_fill(~active_mask, torch.nan))
        if self.name == "Add": return xs[0] + xs[1]
        if self.name == "Sub": return xs[0] - xs[1]
        if self.name == "Mul": return xs[0] * xs[1]
        if self.name == "Div": return xs[0] / xs[1].where(xs[1].abs() > 1e-12, torch.nan)
        if self.name == "Greater": return torch.maximum(xs[0], xs[1])
        if self.name == "Less": return torch.minimum(xs[0], xs[1])
        w = int(self.value)
        if self.name == "Ref": return shift(xs[0], w)
        if self.name == "Delta": return xs[0] - shift(xs[0], w)
        if self.name in PAIR_ROLLING_NAMES:
            return pair_rolling(self.name, xs[0], xs[1], w)
        return rolling(self.name, xs[0], w)


def shift(x: Tensor, n: int) -> Tensor:
    out = torch.full_like(x, torch.nan)
    if n < len(x):
        out[n:] = x[:-n]
    return out


def _windows(x: Tensor, w: int) -> Tensor:
    padded = torch.full((w - 1, x.shape[1]), torch.nan, device=x.device, dtype=x.dtype)
    return torch.cat([padded, x]).unfold(0, w, 1)


def _window_sum(x: Tensor, w: int) -> Tensor:
    prefix = torch.cat([torch.zeros_like(x[:1]), x.cumsum(0)], dim=0)
    return prefix[w:] - prefix[:-w]


def _window_moments(x: Tensor, w: int) -> tuple[Tensor, Tensor, Tensor]:
    finite = torch.isfinite(x)
    safe = x.nan_to_num()
    padded = F.pad(safe, (0, 0, w - 1, 0))
    count = _window_sum(F.pad(finite.to(x.dtype), (0, 0, w - 1, 0)), w)
    total = _window_sum(padded, w)
    total2 = _window_sum(padded.square(), w)
    return total, total2, count == w


def rolling(name: str, x: Tensor, w: int) -> Tensor:
    if w == 1:
        if name in {"Std", "Var", "Mad"}:
            return torch.zeros_like(x).where(torch.isfinite(x), torch.nan)
        if name == "Rank":
            return torch.full_like(x, 0.5).where(torch.isfinite(x), torch.nan)
        return x
    if name in {"Mean", "Sum", "Std", "Var"}:
        total, total2, valid = _window_moments(x, w)
        mean = total / w
        var = (total2 / w - mean.square()).clamp_min(0)
        out = {"Mean": mean, "Sum": total, "Std": var.sqrt(), "Var": var}[name]
        return out.where(valid, torch.nan)
    if name in {"Max", "Min"}:
        finite = torch.isfinite(x)
        count = _window_sum(F.pad(finite.to(x.dtype), (0, 0, w - 1, 0)), w)
        fill = -torch.inf if name == "Max" else torch.inf
        padded = F.pad(x.masked_fill(~finite, fill), (0, 0, w - 1, 0), value=fill)
        values = padded.T.unsqueeze(1)
        if name == "Max": out = F.max_pool1d(values, w, stride=1)
        else: out = -F.max_pool1d(-values, w, stride=1)
        return out.squeeze(1).T.where(count == w, torch.nan)
    if name in {"WMA", "EMA"}:
        finite = torch.isfinite(x)
        count = _window_sum(F.pad(finite.to(x.dtype), (0, 0, w - 1, 0)), w)
        if name == "WMA": weights = torch.arange(1, w + 1, device=x.device, dtype=x.dtype)
        else:
            alpha = 2 / (w + 1)
            weights = (1 - alpha) ** torch.arange(w - 1, -1, -1, device=x.device, dtype=x.dtype)
        kernel = (weights / weights.sum()).view(1, 1, w)
        padded = F.pad(x.nan_to_num().T.unsqueeze(1), (w - 1, 0))
        out = F.conv1d(padded, kernel).squeeze(1).T
        return out.where(count == w, torch.nan)

    if name not in {"Med", "Mad", "Rank"}:
        raise KeyError(name)
    # These operators need an explicit [time, stock, window] representation.
    # Slice the time dimension so a long 1-minute block and w=300 cannot create
    # a many-GB temporary tensor. The unfold itself is a view.
    z = _windows(x, w)
    max_window_elements = 64 * 1024 * 1024  # ~256 MiB in float32
    time_chunk = max(1, max_window_elements // max(1, x.shape[1] * w))
    outputs = []
    for start in range(0, len(z), time_chunk):
        window = z[start:start + time_chunk]
        valid = (~window.isnan()).all(dim=-1)
        safe = window.nan_to_num()
        if name == "Med":
            out = safe.median(-1).values
        elif name == "Mad":
            out = (safe - safe.mean(-1, keepdim=True)).abs().mean(-1)
        else:
            last = safe[..., -1, None]
            out = ((safe < last).sum(-1) + (safe <= last).sum(-1)).float() / (2 * w)
        outputs.append(out.where(valid, torch.nan))
    return torch.cat(outputs, dim=0)


def pair_rolling(name: str, x: Tensor, y: Tensor, w: int) -> Tensor:
    finite = torch.isfinite(x) & torch.isfinite(y)
    if w == 1:
        return torch.zeros_like(x).where(finite, torch.nan)
    a = x.where(finite, 0)
    b = y.where(finite, 0)
    pad = (0, 0, w - 1, 0)
    count = _window_sum(F.pad(finite.to(x.dtype), pad), w)
    sx = _window_sum(F.pad(a, pad), w); sy = _window_sum(F.pad(b, pad), w)
    sxx = _window_sum(F.pad(a.square(), pad), w); syy = _window_sum(F.pad(b.square(), pad), w)
    sxy = _window_sum(F.pad(a * b, pad), w)
    mx, my = sx / w, sy / w
    cov = sxy / w - mx * my
    if name == "Cov": out = cov
    else:
        vx = (sxx / w - mx.square()).clamp_min(0)
        vy = (syy / w - my.square()).clamp_min(0)
        out = cov / (vx * vy).sqrt().clamp_min(1e-12)
    return out.where(count == w, torch.nan)


def cross_section_rank(x: Tensor) -> Tensor:
    mask = x.isnan()
    safe = x.masked_fill(mask, torch.inf)
    ranks = safe.argsort(dim=1).argsort(dim=1).float()
    n = (~mask).sum(dim=1, keepdim=True).clamp_min(1)
    return (ranks / n).masked_fill(mask, torch.nan)


UNARY_NAMES = ("Abs", "Sign", "Log", "CSRank")
BINARY_NAMES = ("Add", "Sub", "Mul", "Div", "Greater", "Less")
ROLLING_NAMES = ("Ref", "Mean", "Sum", "Std", "Var", "Max", "Min", "Med", "Mad", "Rank", "Delta", "WMA", "EMA", "Cov", "Corr")
PAIR_ROLLING_NAMES = {"Cov", "Corr"}
OPERATORS = UNARY_NAMES + BINARY_NAMES + ROLLING_NAMES
WINDOWS = (1, 3, 6, 12, 24, 30, 48, 60, 120, 240, 300, 600, 1200)
CONSTANTS = (-10.0, -2.0, -1.0, -0.5, -0.01, 0.01, 0.5, 1.0, 2.0, 10.0)
# Long windows are reserved for operators whose implementation is based on
# shifts or prefix sufficient statistics.  WMA/EMA are mathematically linear
# but currently use a length-w convolution kernel, so they remain capped until
# they are rewritten as prefix/scan operators.  Max/Min use pooling rather
# than prefix sums and are conservatively capped as well.
LONG_WINDOW_OPERATORS = frozenset({"Ref", "Delta", "Mean", "Sum", "Std", "Var", "Cov", "Corr"})
NONLINEAR_WINDOW_MAX = 300


@dataclass(frozen=True)
class Token:
    kind: str
    value: str | float | int

    def __str__(self) -> str:
        if self.kind == "feature": return f"${str(self.value).lower()}"
        return str(self.value)


class RPNBuilder:
    def __init__(self, max_lookback: int = MAX_LOOKBACK) -> None:
        self.stack: list[Expr | int] = []
        self.tokens: list[Token] = []
        self.max_lookback = max_lookback

    def valid(self, token: Token) -> bool:
        if token.kind == "feature": return not (self.stack and isinstance(self.stack[-1], int))
        if token.kind == "constant": return not (self.stack and isinstance(self.stack[-1], int))
        if token.kind == "window":
            return (
                bool(self.stack) and isinstance(self.stack[-1], Expr)
                and self.stack[-1].featured
                and self.stack[-1].lookback + max(0, int(token.value) - 1) <= self.max_lookback
            )
        if token.kind == "stop": return len(self.stack) == 1 and isinstance(self.stack[0], Expr) and self.stack[0].featured
        if token.kind != "operator": return False
        name = str(token.value)
        if name in UNARY_NAMES:
            return bool(self.stack) and isinstance(self.stack[-1], Expr) and self.stack[-1].featured
        if name in BINARY_NAMES:
            return len(self.stack) >= 2 and all(isinstance(x, Expr) for x in self.stack[-2:]) and any(x.featured for x in self.stack[-2:])
        arity = 3 if name in PAIR_ROLLING_NAMES else 2
        structurally_valid = len(self.stack) >= arity and isinstance(self.stack[-1], int) and all(
            isinstance(x, Expr) and x.featured for x in self.stack[-arity:-1]
        )
        if not structurally_valid:
            return False
        window = int(self.stack[-1])
        if window > NONLINEAR_WINDOW_MAX and name not in LONG_WINDOW_OPERATORS:
            return False
        args = self.stack[-arity:-1]
        own = window if name in {"Ref", "Delta"} else max(0, window - 1)
        return own + max(x.lookback for x in args if isinstance(x, Expr)) <= self.max_lookback

    def add(self, token: Token) -> None:
        if not self.valid(token): raise ValueError(f"非法 token: {token}")
        self.tokens.append(token)
        if token.kind == "feature": self.stack.append(Expr("feature", value=int(FeatureType[str(token.value).upper()])))
        elif token.kind == "constant": self.stack.append(Expr("constant", value=float(token.value)))
        elif token.kind == "window": self.stack.append(int(token.value))
        elif token.kind == "operator":
            name = str(token.value)
            arity = 1 if name in UNARY_NAMES else 2 if name in BINARY_NAMES else 3 if name in PAIR_ROLLING_NAMES else 2
            args = self.stack[-arity:]; del self.stack[-arity:]
            if name in ROLLING_NAMES:
                self.stack.append(Expr(name, tuple(args[:-1]), int(args[-1])))  # type: ignore[arg-type]
            else: self.stack.append(Expr(name, tuple(args)))  # type: ignore[arg-type]

    def expression(self) -> Expr:
        if len(self.stack) != 1 or not isinstance(self.stack[0], Expr): raise ValueError("表达式未完成")
        return self.stack[0]

    @property
    def token_string(self) -> str:
        return " ".join(map(str, self.tokens))
