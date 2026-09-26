"""动态股票池：把 PIT 指数成分区间表展开为逐日布尔掩码。

数据来源：`data/reference/universe_intervals.csv`（见该目录 README 及源数据验收记录）
语义：某股票在 [opt-in, opt-out) 区间内属于该 index；同一 (index, symbol) 可有多行
      （调出后又被调回 = 两段独立区间），必须逐段标记，不能压扁成一行。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT

REF_DIR = ROOT / "data" / "reference"
UNIVERSE_FILE = REF_DIR / "universe_intervals.csv"
TRADE_DAYS_FILE = REF_DIR / "trade_days.csv"

INDEX_LABELS = {
    "csi300": "沪深300",
    "csi500": "中证500",
    "csi1000": "中证1000",
}

_OPEN = pd.Timestamp("1900-01-01")
_CLOSE = pd.Timestamp("2100-01-01")


def to_plain_code(symbol: str) -> str:
    """SH600000 -> 600000（与本项目 market.codes 的六位代码对齐）"""
    s = str(symbol).strip().upper()
    if s[:2] in ("SH", "SZ", "BJ"):
        s = s[2:]
    return s.zfill(6)


@lru_cache(maxsize=1)
def load_universe_intervals() -> pd.DataFrame:
    """读取并规范化区间表；列：index, symbol(六位), name, start, end。"""
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"缺少 PIT 成分区间表 {UNIVERSE_FILE}；请把 data/reference 一并提供")
    iv = pd.read_csv(UNIVERSE_FILE, parse_dates=["opt-in", "opt-out"])
    iv = iv.rename(columns={"opt-in": "start", "opt-out": "end"})
    iv["start"] = iv["start"].fillna(_OPEN)
    iv["end"] = iv["end"].fillna(_CLOSE)
    iv["symbol"] = iv["symbol"].map(to_plain_code)
    return iv[["index", "symbol", "name", "start", "end"]].copy()


@lru_cache(maxsize=1)
def load_trade_days() -> pd.DatetimeIndex:
    if not TRADE_DAYS_FILE.exists():
        raise FileNotFoundError(f"缺少交易日历 {TRADE_DAYS_FILE}")
    d = pd.read_csv(TRADE_DAYS_FILE, parse_dates=["datetime"])["datetime"]
    return pd.DatetimeIndex(sorted(d.unique()))


def available_indices() -> list[str]:
    try:
        return sorted(load_universe_intervals()["index"].unique().tolist())
    except FileNotFoundError:
        return []


def index_member_mask(indices: list[str],
                      dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """展开为逐日布尔掩码（index=日期, columns=六位代码）。

    dates 为 None 时用参考数据的完整交易日历；否则按给定日历 reindex
    （给定日历之外的日期不会出现，缺失一律为 False，不静默视为在池内）。
    """
    iv = load_universe_intervals()
    want = [i for i in indices if i]
    if not want:
        raise ValueError("至少需要一个指数代码，如 ['csi300']")
    unknown = sorted(set(want) - set(iv["index"].unique()))
    if unknown:
        raise ValueError(f"未知指数代码: {unknown}；可用: {available_indices()}")

    sub = iv[iv["index"].isin(want)]
    all_days = load_trade_days()
    if dates is None:
        days = all_days
    else:
        days = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    symbols = sorted(sub["symbol"].unique())

    mask = pd.DataFrame(False, index=days, columns=symbols)
    # 只对落在目标日历内的区间片段打标；逐区间切片，4033 段规模可接受
    for sym, grp in sub.groupby("symbol", sort=False):
        col = mask[sym].to_numpy()
        pos = days
        for lo, hi in zip(grp["start"].to_numpy(), grp["end"].to_numpy()):
            lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
            if hi <= pos[0] or lo > pos[-1]:
                continue
            sel = (pos >= lo) & (pos < hi)
            col |= sel
        mask[sym] = col
    mask.columns.name = "code"
    return mask


def resolve_pool(market, spec: dict | None) -> tuple[pd.DataFrame | None, str]:
    """把股票池定义解析为与 market 对齐的逐日掩码。

    spec:
      {"kind": "all"}                          全部可用资产（返回 None 表示不过滤）
      {"kind": "index", "indices": ["csi300"]} PIT 指数成分（动态）
      {"kind": "static", "codes": ["600519"]}   静态代码集合
    返回 (掩码 或 None, 说明文字)
    """
    if not spec or spec.get("kind") in (None, "all"):
        return None, "全部可用资产（不按指数过滤）"

    kind = spec["kind"]
    if kind == "index":
        idxs = list(spec.get("indices") or [])
        if not idxs:
            return None, "全部可用资产（未指定指数）"
        full = index_member_mask(idxs, dates=market.dates)
        # 与本项目实际拥有的代码取交集；池里有而我们没抓到的股票不参与
        have = [c for c in full.columns if c in set(market.codes)]
        missing = [c for c in full.columns if c not in set(market.codes)]
        out = full.reindex(index=market.dates, columns=have).fillna(False).astype(bool)
        names = "、".join(INDEX_LABELS.get(i, i) for i in idxs)
        note = f"动态股票池：{names}（PIT 逐日成分），池内 {len(have)} 只有本地行情"
        if missing:
            note += f"；池中另有 {len(missing)} 只缺行情，已排除"
        return out, note

    if kind == "static":
        codes = [str(c).zfill(6) for c in (spec.get("codes") or [])]
        have = [c for c in codes if c in set(market.codes)]
        out = pd.DataFrame(False, index=market.dates, columns=have or market.codes[:0])
        if have:
            out.loc[:, have] = True
        return out, f"静态股票池：指定 {len(codes)} 只，其中 {len(have)} 只有本地行情"

    raise ValueError(f"未知股票池类型: {kind}")


def pool_daily_counts(mask: pd.DataFrame) -> pd.Series:
    """逐日池内资产数（用于页面展示与健全性检查）。"""
    return mask.sum(axis=1).astype(int)
