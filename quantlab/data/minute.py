"""分钟级数据层：从真实 5 分钟宽表（hft_5m_csi300.csv，~918MB）构建多字段面板。

数据格式（与 alphagen_5m 引擎一致）：每行 = (datetime 交易日, instrument 资产)，
{field}1..{field}48 为当日 48 根 5 分钟 bar（9:30–11:25 共24根，13:00–14:55 共24根，开始时刻标注）。

低内存设计（目标环境可用内存可能 <2GB）：
  Pass A 只读 datetime+instrument 建立日期/资产索引；
  Pass B 分块(2万行)读字段列，向量写入预分配 float32 面板（n_dates*48 × n_inst）。
缺失（停牌/未覆盖）保留 NaN，不填零。
"""
from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BARS_PER_DAY = 48
# 每根 bar 的钟点偏移（分钟，自 00:00 起，按 bar 开始时刻标注）：
# 上午 24 根：9:30, 9:35, ..., 11:25；下午 24 根：13:00, 13:05, ..., 14:55
BAR_OFFSET_MIN = ([570 + 5 * i for i in range(24)]
                  + [780 + 5 * i for i in range(24)])


def build_5m_close(csv_path: Path, out_dir: Path, force: bool = False) -> Path:
    return build_5m_fields(csv_path, out_dir, fields=("close",), force=force)


def build_5m_fields(csv_path: Path, out_dir: Path,
                    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount"),
                    force: bool = False) -> Path:
    """构建分钟面板缓存：<out_dir>/<field>_5m.npy（meta/dates 共享）。幂等，只补缺失字段。"""
    csv_path, out_dir = Path(csv_path), Path(out_dir)
    meta_path = out_dir / "close_5m_meta.json"
    todo = tuple(f for f in fields if force or not (out_dir / f"{f}_5m.npy").exists())
    if not todo and meta_path.exists():
        return out_dir

    # ---- Pass A：日期与资产索引（轻量列） ----
    dates_seen: dict[str, int] = {}
    insts_seen: dict[str, int] = {}
    for chunk in pd.read_csv(csv_path, usecols=["datetime", "instrument"], chunksize=200_000):
        for d in chunk["datetime"].astype(str):
            if d not in dates_seen:
                dates_seen[d] = len(dates_seen)
        for c in chunk["instrument"].astype(str):
            if c not in insts_seen:
                insts_seen[c] = len(insts_seen)
    dates = sorted(dates_seen)
    insts = sorted(insts_seen)
    d_idx = {d: i for i, d in enumerate(dates)}
    c_idx = {c: i for i, c in enumerate(insts)}
    n_ts = len(dates) * BARS_PER_DAY

    panels = {f: np.full((n_ts, len(insts)), np.nan, dtype="float32") for f in todo}
    cols = [f"{f}{i}" for f in todo for i in range(1, BARS_PER_DAY + 1)]
    d_buf, c_buf, v_bufs = [], [], {f: [] for f in todo}
    for chunk in pd.read_csv(csv_path, usecols=["datetime", "instrument", *cols], chunksize=20_000):
        d_buf.append(chunk["datetime"].astype(str).map(d_idx).to_numpy(dtype="int64"))
        c_buf.append(chunk["instrument"].astype(str).map(c_idx).to_numpy(dtype="int64"))
        for f in todo:
            v_bufs[f].append(chunk[[f"{f}{i}" for i in range(1, BARS_PER_DAY + 1)]]
                             .to_numpy(dtype="float32"))
        if len(d_buf) >= 10:  # 每 ~20 万行落一次盘，控制驻留内存
            _flush_all(panels, d_buf, c_buf, v_bufs)
            gc.collect()
    _flush_all(panels, d_buf, c_buf, v_bufs)

    out_dir.mkdir(parents=True, exist_ok=True)
    built = []
    if meta_path.exists():
        built = json.loads(meta_path.read_text(encoding="utf-8")).get("fields", [])
    for f, panel in panels.items():
        np.save(out_dir / f"{f}_5m.npy", panel)
        built = sorted(set(built) | {f})
    (out_dir / "close_5m_dates.json").write_text(json.dumps(dates), encoding="utf-8")
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "source_csv_bytes": csv_path.stat().st_size,
        "bars_per_day": BARS_PER_DAY,
        "n_dates": len(dates),
        "n_instruments": len(insts),
        "n_timestamps": int(n_ts),
        "fields": built,
        "dtype": "float32",
        "bar_offsets_min": BAR_OFFSET_MIN,
        "bar_time_convention": "开始时刻标注：上午9:30起、下午13:00起，每5分钟一根",
        "missing_policy": "停牌/缺失保留 NaN，不填零",
        "note": "因子与标签均在此 5 分钟面板上计算；频率口径=5分钟",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def _flush_all(panels: dict[str, np.ndarray], d_buf: list, c_buf: list, v_bufs: dict) -> None:
    if not d_buf:
        return
    d = np.concatenate(d_buf)
    c = np.concatenate(c_buf)
    rows = d * BARS_PER_DAY
    bar = np.tile(np.arange(BARS_PER_DAY, dtype="int64"), (len(d), 1))
    for f, buf in v_bufs.items():
        v = np.concatenate(buf)
        panels[f][rows[:, None] + bar, c[:, None]] = v
        buf.clear()
    d_buf.clear(); c_buf.clear()


def panel_frame(out_dir: Path, field: str = "close") -> pd.DataFrame:
    """按缓存重建时间索引的分钟面板 DataFrame（float32，index=bar 时间戳）。"""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "close_5m_meta.json").read_text(encoding="utf-8"))
    panel = np.load(out_dir / f"{field}_5m.npy")
    dates = json.loads((out_dir / "close_5m_dates.json").read_text())
    bar_min = np.array(meta.get("bar_offsets_min", BAR_OFFSET_MIN), dtype="int64")
    ts = (np.repeat(pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]"), BARS_PER_DAY)
          + np.tile(bar_min.astype("timedelta64[m]"), len(dates)))
    return pd.DataFrame(panel, index=pd.DatetimeIndex(ts))
