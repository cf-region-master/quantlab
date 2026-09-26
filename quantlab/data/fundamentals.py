"""估值/市值数据（Tushare daily_basic）：为市值中性化与股票池行情页提供字段。

数据由 `scripts/fetch_daily_basic.py` 按交易日整市场拉取，落在
`data/raw/daily_basic.parquet`（长表）。本模块把它整理成与 market 对齐的宽表面板。

口径（不做静默换算以外的加工）：
  - `total_mv` / `circ_mv`  Tushare 单位为万元，本模块 ×1e4 转成【元】后对外提供
  - `total_share` / `float_share` / `free_share`  万股 -> ×1e4 转成【股】
  - `turnover_rate` / `turnover_rate_f`  Tushare 为百分数（4.21 表示 4.21%），÷100 转比例
  - `pe/pb/ps/dv_ratio` 等为倍/百分数，原样提供，不做截断
  - 缺失保持 NaN，不填 0、不 ffill（与全项目「缺失不填补」口径一致）

若文件不存在，`available()` 返回 False，页面与 spec 会明确说明「无市值数据」，
而不是退化成用别的字段冒充市值。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT

RAW = ROOT / "data" / "raw"
DAILY_BASIC = RAW / "daily_basic.parquet"
MANIFEST = RAW / "daily_basic_manifest.json"

# 需要单位换算的列：列名 -> 乘数
SCALE = {"total_mv": 1e4, "circ_mv": 1e4,
         "total_share": 1e4, "float_share": 1e4, "free_share": 1e4,
         "turnover_rate": 0.01, "turnover_rate_f": 0.01}

# 估值/市值类字段的中文标签（页面表头用）
FIELD_LABELS = {
    "close": "收盘(未复权)", "pct_chg": "涨跌幅", "turnover_rate": "换手率",
    "turnover_rate_f": "换手率(自由流通)", "volume_ratio": "量比",
    "pe_ttm": "PE(TTM)", "pe": "PE", "pb": "PB", "ps_ttm": "PS(TTM)",
    "dv_ratio": "股息率", "total_mv": "总市值", "circ_mv": "流通市值",
    "amount": "成交额", "volume": "成交量", "close_adj": "收盘(复权)",
}


def available() -> bool:
    return DAILY_BASIC.exists()


def manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


@lru_cache(maxsize=1)
def _long() -> pd.DataFrame:
    if not DAILY_BASIC.exists():
        raise FileNotFoundError(
            f"缺少市值/估值数据 {DAILY_BASIC}；请先运行 "
            "python scripts/fetch_daily_basic.py --start 20220101 --end 20251231")
    df = pd.read_parquet(DAILY_BASIC)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    for c, k in SCALE.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") * k
    return df


def _wide(field: str, dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    df = _long()
    if field not in df.columns:
        raise FileNotFoundError(f"daily_basic 中缺少字段 {field}")
    w = df.pivot_table(index="date", columns="code", values=field, aggfunc="last")
    w.columns.name = "code"
    return w.reindex(index=dates, columns=codes)


def clear_cache() -> None:
    _long.cache_clear()


def total_mv_panel(dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    """总市值面板（元），与给定日期/代码对齐；无市值处为 NaN。"""
    return _wide("total_mv", dates, codes)


def circ_mv_panel(dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    """流通市值面板（元）。"""
    return _wide("circ_mv", dates, codes)


def field_panel(field: str, dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    """任意 daily_basic 字段的宽表面板（已按 SCALE 换算）。"""
    return _wide(field, dates, codes)


def snapshot(date: pd.Timestamp, codes: list[str]) -> dict[str, dict]:
    """某个交易日的估值快照 {code: {字段: 值}}（用于股票池成员表）。"""
    df = _long()
    d = pd.Timestamp(date)
    avail = df["date"].unique()
    # 取不晚于 date 的最近一个有数据的交易日（当日停牌/未上市时不至于整表空）
    le = sorted([x for x in avail if pd.Timestamp(x) <= d])
    if not le:
        return {}
    use = le[-1]
    sub = df[df["date"] == use]
    sub = sub[sub["code"].isin(set(codes))]
    out: dict[str, dict] = {}
    for r in sub.to_dict("records"):
        out[str(r["code"])] = {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v))
                                  else float(v))
                               for k, v in r.items() if k not in ("date", "code", "ts_code")}
    return out


@lru_cache(maxsize=1)
def names() -> dict[str, str]:
    """代码 -> 证券简称（来自 data/reference/instruments.csv）。"""
    p = ROOT / "data" / "reference" / "instruments.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype={"instrument": str})
    out = {}
    for it, nm in zip(df["instrument"], df["name"]):
        s = str(it).strip().upper()
        if s[:2] in ("SH", "SZ", "BJ"):
            s = s[2:]
        out[s.zfill(6)] = str(nm)
    return out


def name_of(code: str) -> str:
    return names().get(str(code).zfill(6), "")


def coverage_note() -> str:
    """给页面用的一句话数据说明（不虚构：没数据就说没数据）。"""
    if not available():
        return "未接入市值/估值数据（缺 data/raw/daily_basic.parquet）"
    mf = manifest()
    rng = mf.get("date_range_in_file") or ["?", "?"]
    return (f"Tushare daily_basic，按交易日整市场拉取，{rng[0]} ~ {rng[1]}，"
            f"{mf.get('n_rows', 0):,} 行 / {mf.get('n_codes', 0)} 只")
