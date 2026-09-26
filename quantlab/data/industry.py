"""PIT 行业分类：把申万行业区间表展开为逐日标签 / 分组编码。

数据来源：`data/reference/industry_intervals.csv`（聚宽采集，申万 2021 版命名）

⚠️ 口径边界（源数据已声明的局限，本模块不掩盖）：
   2021-12 之前的历史期间反映【申万 2014 版结构】，命名归一化只保证"当前"与 pipeline 一致。
   跨 2021-12 用行业做哑变量中性化会出现类别断裂；如需全期统一口径，必须显式映射并在报告中说明。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import ROOT

REF_DIR = ROOT / "data" / "reference"
INDUSTRY_FILE = REF_DIR / "industry_intervals.csv"

LEVELS = ("sw1", "sw2", "sw3")
_OPEN = pd.Timestamp("1900-01-01")
_CLOSE = pd.Timestamp("2100-01-01")


def to_plain_code(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if s[:2] in ("SH", "SZ", "BJ"):
        s = s[2:]
    return s.zfill(6)


@lru_cache(maxsize=1)
def load_industry_intervals() -> pd.DataFrame:
    if not INDUSTRY_FILE.exists():
        raise FileNotFoundError(
            f"缺少 PIT 行业区间表 {INDUSTRY_FILE}；请把 data/reference 一并提供")
    iv = pd.read_csv(INDUSTRY_FILE, parse_dates=["start", "end"])
    iv["start"] = iv["start"].fillna(_OPEN)
    iv["end"] = iv["end"].fillna(_CLOSE)
    iv["instrument"] = iv["instrument"].map(to_plain_code)
    return iv[["instrument", *LEVELS, "start", "end"]].copy()


def industry_labels(dates: pd.DatetimeIndex, codes: list[str],
                    level: str = "sw1") -> pd.DataFrame:
    """逐日行业标签（index=日期, columns=代码）。

    取"该日生效的区间"的值；同一 (股票, 区间) 重叠时 start 较晚者覆盖较早者
    （= 更近的一次行业变更，与原实现一致）。
    无覆盖一律为 NaN —— 不推断、不前向填充，中性化时会被显式排除并计数。

    实现说明：直接在 numpy 数组上按【区间】切片赋值（几千次切片），
    而不是逐列取出/写回 DataFrame（2431 次全表列写，实测 17s）。
    """
    if level not in LEVELS:
        raise ValueError(f"level 必须是 {LEVELS} 之一")
    iv = load_industry_intervals()
    days = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    codes = [str(c) for c in codes]
    if not len(days) or not codes:
        return pd.DataFrame(index=days, columns=codes, dtype=object)

    sub = iv[iv["instrument"].isin(set(codes))].sort_values("start")
    arr = np.full((len(days), len(codes)), np.nan, dtype=object)
    col_of = {c: i for i, c in enumerate(codes)}
    dvals = days.to_numpy()

    for instrument, lab, lo, hi in zip(sub["instrument"].to_numpy(),
                                       sub[level].to_numpy(),
                                       sub["start"].to_numpy(),
                                       sub["end"].to_numpy()):
        j = col_of.get(instrument)
        if j is None or lab is None or (isinstance(lab, float) and np.isnan(lab)):
            continue
        # [lo, hi) 区间 -> 日历上的位置区间
        i0 = int(np.searchsorted(dvals, np.datetime64(pd.Timestamp(lo)), side="left"))
        i1 = int(np.searchsorted(dvals, np.datetime64(pd.Timestamp(hi)), side="left"))
        if i1 > i0:
            arr[i0:i1, j] = lab

    out = pd.DataFrame(arr, index=days, columns=codes)
    out.columns.name = "code"
    return out


def industry_group_codes(dates: pd.DatetimeIndex, codes: list[str],
                         level: str = "sw1") -> tuple[pd.DataFrame, dict[int, str]]:
    """把行业标签因子化为整数编码，供 `preprocess.neutralize_cs` 使用。

    返回 (编码宽表, {编码: 行业名})；未知行业为 NaN（float 型），中性化时自动排除。
    """
    lab = industry_labels(dates, codes, level)
    all_lab = pd.unique(lab.to_numpy().ravel())
    names = sorted([x for x in all_lab if isinstance(x, str)])
    mapping = {i: n for i, n in enumerate(names)}
    inv = {n: i for i, n in mapping.items()}
    arr = lab.to_numpy()
    out = np.full(arr.shape, np.nan, dtype="float64")
    for i, n in mapping.items():
        out[arr == n] = float(i)
    return pd.DataFrame(out, index=lab.index, columns=lab.columns), mapping


def coverage_summary(dates: pd.DatetimeIndex, codes: list[str], level: str = "sw1") -> dict:
    """行业覆盖情况（用于页面展示，明确告诉用户有多少格子无行业）。"""
    lab = industry_labels(dates, codes, level)
    arr = lab.to_numpy()
    total = arr.size
    known = int(pd.notna(arr).sum())
    return {
        "level": level,
        "n_cells": int(total),
        "n_known": known,
        "coverage": round(known / total, 6) if total else 0.0,
        "n_industries": int(len({x for x in pd.unique(arr.ravel()) if isinstance(x, str)})),
        "n_codes_without_industry": int(sum(1 for c in lab.columns if lab[c].isna().all())),
    }


# ---------------------------------------------------------------------------
# 缓存层：全市场（969×2431）的标签表构造约数秒，页面与批量重算会反复用到，
# 按 (level, 日期序列, 代码序列) 缓存；返回副本，调用方可以随意 reindex/切片。
# ---------------------------------------------------------------------------
@lru_cache(maxsize=6)
def _labels_cached(level: str, dates_key: tuple, codes_key: tuple) -> pd.DataFrame:
    return industry_labels(pd.DatetimeIndex(list(dates_key)), list(codes_key), level)


@lru_cache(maxsize=6)
def _group_codes_cached(level: str, dates_key: tuple, codes_key: tuple
                        ) -> tuple[pd.DataFrame, tuple]:
    codes, mapping = industry_group_codes(
        pd.DatetimeIndex(list(dates_key)), list(codes_key), level)
    return codes, tuple(sorted(mapping.items()))


def industry_labels_fast(dates: pd.DatetimeIndex, codes: list[str],
                         level: str = "sw1") -> pd.DataFrame:
    """带缓存的 `industry_labels`（返回副本，调用方可安全改写）。"""
    d, c = pd.DatetimeIndex(dates), [str(x) for x in codes]
    return _labels_cached(level, tuple(d), tuple(c)).copy()


def industry_group_codes_fast(dates: pd.DatetimeIndex, codes: list[str],
                              level: str = "sw1") -> tuple[pd.DataFrame, dict[int, str]]:
    """带缓存的 `industry_group_codes`（返回副本，调用方可安全改写）。"""
    d, c = pd.DatetimeIndex(dates), [str(x) for x in codes]
    tbl, mapping = _group_codes_cached(level, tuple(d), tuple(c))
    return tbl.copy(), dict(mapping)


def coverage_summary_fast(dates: pd.DatetimeIndex, codes: list[str],
                          level: str = "sw1") -> dict:
    """带缓存的覆盖率统计。"""
    lab = industry_labels_fast(dates, codes, level)
    arr = lab.to_numpy()
    total = arr.size
    known = int(pd.notna(arr).sum())
    iv = load_industry_intervals()
    return {
        "level": level,
        "n_cells": int(total),
        "n_known": known,
        "coverage": round(known / total, 6) if total else 0.0,
        "n_industries": int(len({x for x in pd.unique(arr.ravel()) if isinstance(x, str)})),
        "n_codes_without_industry": int(sum(1 for c in lab.columns if lab[c].isna().all())),
        # 区间表的实际覆盖范围：早于 min_start 的日期没有任何行业记录，
        # 使用行业中性化的因子在这些日子会被整体排除 —— 页面必须能看见这条边界。
        "span": [str(iv["start"].min().date()), str(iv["end"].max().date())],
        "requested_span": [str(dates.min().date()), str(dates.max().date())],
        "n_days_before_data": int((dates < iv["start"].min()).sum()),
    }
