"""因子算子与因子卡注册表。

统一约定：
  - 输入输出均为宽表 DataFrame（index=交易日, columns=资产代码），NaN 表示缺失
  - 统一方向：分数越高，预期收益越高（rev5/lowvol20 在公式内取负号）
  - 通过 configs/factors.yaml 配置切换因子与参数，公式与回测核心分离
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


# ---------------- 算子 ----------------
def op_momentum(fields: dict[str, pd.DataFrame], window: int = 20, **_) -> pd.DataFrame:
    c = fields["close"]
    return c / c.shift(window) - 1


def op_reversal(fields: dict[str, pd.DataFrame], window: int = 5, **_) -> pd.DataFrame:
    c = fields["close"]
    return -(c / c.shift(window) - 1)


def op_lowvol(fields: dict[str, pd.DataFrame], window: int = 20, **_) -> pd.DataFrame:
    r = fields["close"] / fields["close"].shift(1) - 1
    min_p = max(2, int(window * 0.6))
    return -r.rolling(window, min_periods=min_p).std(ddof=1)


OPERATORS: dict[str, Callable[..., pd.DataFrame]] = {
    "momentum": op_momentum, "reversal": op_reversal, "lowvol": op_lowvol,
}


# ---------------- 因子卡 ----------------
@dataclass
class FactorCard:
    key: str
    name: str
    hypothesis: str
    formula: str
    fields: list[str]
    window: int
    direction: str
    missing_policy: str
    failure_modes: list[str]
    operator: str
    params: dict[str, Any]
    preprocess: list[str]

    def card_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "hypothesis": self.hypothesis,
            "formula": self.formula, "fields": self.fields, "window": self.window,
            "direction": self.direction, "missing_policy": self.missing_policy,
            "failure_modes": self.failure_modes, "operator": self.operator,
            "params": self.params, "preprocess": self.preprocess,
        }


def load_cards(factors_cfg: dict[str, Any]) -> dict[str, FactorCard]:
    cards = {}
    for key, spec in factors_cfg["factors"].items():
        cards[key] = FactorCard(
            key=key, name=spec["name"], hypothesis=spec["hypothesis"], formula=spec["formula"],
            fields=list(spec["fields"]), window=int(spec["window"]), direction=spec["direction"],
            missing_policy=spec["missing_policy"], failure_modes=list(spec["failure_modes"]),
            operator=spec["operator"], params=dict(spec.get("params", {})),
            preprocess=list(spec.get("preprocess", [])),
        )
    return cards


def compute_factor(card: FactorCard, fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if card.operator not in OPERATORS:
        raise KeyError(f"未注册的因子算子: {card.operator}")
    return OPERATORS[card.operator](fields, **card.params)


def values_digest(values: pd.DataFrame) -> str:
    """因子取值摘要：纳入列名、索引，并把 NaN 与 0 区分开。

    原实现仅对 `fillna(0.0)` 后的数值字节取哈希，导致：
      - 日期索引整体平移后哈希不变；
      - NaN 被当成 0，缺失与真实 0 值不可区分。
    两者都会让 store 的幂等去重返回错误的既有因子。
    """
    h = hashlib.sha256()
    h.update(("[%s]" % ",".join(str(c) for c in values.columns)).encode("utf-8"))
    h.update(("[%s]" % ",".join(str(i) for i in values.index)).encode("utf-8"))
    arr = values.to_numpy(dtype="float64")
    h.update(np.ascontiguousarray(np.isnan(arr)).tobytes())            # NaN 掩码
    h.update(np.ascontiguousarray(np.nan_to_num(arr, nan=0.0)).tobytes())
    return h.hexdigest()


def factor_hash(card: FactorCard, values: pd.DataFrame) -> str:
    """因子内容哈希（定义+取值），用于因子库幂等去重。"""
    payload = json.dumps(card.card_dict(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256((payload + values_digest(values)).encode("utf-8")).hexdigest()[:32]
