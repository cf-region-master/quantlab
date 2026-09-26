from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from torch import Tensor

from .expressions import FeatureType


FEATURE_NAMES = ("open", "close", "high", "low", "volume", "vwap")
RAW_NAMES = ("open", "close", "high", "low", "volume", "amount")
BARS_PER_DAY = 48


def build_cache(csv_path: Path, cache_dir: Path, max_dates: int | None = None, force: bool = False) -> Path:
    """把日频宽表流式转换为可内存映射的 [bar, feature, stock] float32 数组。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = cache_dir / "market.npy", cache_dir / "meta.json"
    if data_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source_size") == csv_path.stat().st_size and meta.get("max_dates") == max_dates:
            return cache_dir

    dates: set[str] = set()
    stocks: set[str] = set()
    for chunk in pd.read_csv(csv_path, usecols=["datetime", "instrument"], chunksize=200_000):
        dates.update(chunk["datetime"].astype(str).unique())
        stocks.update(chunk["instrument"].astype(str).unique())
    date_list = sorted(dates)
    if max_dates is not None:
        date_list = date_list[:max_dates]
    stock_list = sorted(stocks)
    date_to_idx = {x: i for i, x in enumerate(date_list)}
    stock_to_idx = {x: i for i, x in enumerate(stock_list)}

    arr = open_memmap(data_path, mode="w+", dtype="float32",
                      shape=(len(date_list) * BARS_PER_DAY, len(FEATURE_NAMES), len(stock_list)))
    arr[:] = np.nan
    raw_cols = [f"{name}{bar}" for name in RAW_NAMES for bar in range(1, BARS_PER_DAY + 1)]
    usecols = ["datetime", "instrument", *raw_cols]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=20_000):
        keep = chunk["datetime"].astype(str).isin(date_to_idx)
        chunk = chunk.loc[keep]
        if chunk.empty: continue
        d = chunk["datetime"].astype(str).map(date_to_idx).to_numpy()
        s = chunk["instrument"].astype(str).map(stock_to_idx).to_numpy()
        t = d[:, None] * BARS_PER_DAY + np.arange(BARS_PER_DAY)[None, :]
        si = np.broadcast_to(s[:, None], t.shape)
        for fi, raw in enumerate(RAW_NAMES):
            values = chunk[[f"{raw}{bar}" for bar in range(1, BARS_PER_DAY + 1)]].to_numpy(dtype=np.float32)
            arr[t, fi, si] = values
        volume = arr[t, int(FeatureType.VOLUME), si]
        amount = arr[t, int(FeatureType.VWAP), si]
        arr[t, int(FeatureType.VWAP), si] = np.divide(
            amount, volume, out=np.full_like(amount, np.nan), where=volume > 0
        )
    arr.flush()
    meta_path.write_text(json.dumps({
        "dates": date_list, "stocks": stock_list, "bars_per_day": BARS_PER_DAY,
        "features": FEATURE_NAMES, "source": str(csv_path.resolve()),
        "source_size": csv_path.stat().st_size, "max_dates": max_dates,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_dir


@dataclass
class MarketSplit:
    name: str
    values: Tensor
    target: Tensor
    bar_in_day: Tensor


@dataclass
class MarketData:
    splits: dict[str, MarketSplit]
    dates: list[str]
    stocks: list[str]
    label_horizon: int
    same_day_label: bool

    @classmethod
    def load(
        cls, cache_dir: Path, device: torch.device, label_horizon: int = 6,
        same_day_label: bool = False,
        train_start: str = "2019-01-01", train_end: str = "2022-12-31",
        valid_start: str = "2023-01-01", valid_end: str = "2023-12-31",
        test_start: str = "2024-01-01", test_end: str = "2025-12-31",
    ) -> "MarketData":
        meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        mmap = np.load(cache_dir / "market.npy", mmap_mode="r")
        bars = int(meta["bars_per_day"])
        ranges = {
            "train": (train_start, train_end),
            "validation": (valid_start, valid_end),
            "test": (test_start, test_end),
        }
        date_array = np.asarray(meta["dates"], dtype="U10")
        splits: dict[str, MarketSplit] = {}
        for name, (start_date, end_date) in ranges.items():
            day_mask_np = (date_array >= start_date) & (date_array <= end_date)
            selected = np.flatnonzero(day_mask_np)
            if not len(selected):
                raise ValueError(f"{name} 日期范围 {start_date}..{end_date} 没有数据")
            first, last = int(selected[0]) * bars, (int(selected[-1]) + 1) * bars
            values = torch.from_numpy(np.array(mmap[first:last], copy=True)).to(device)
            close = values[:, int(FeatureType.CLOSE), :]
            h = label_horizon
            if h <= 0 or h >= len(close):
                raise ValueError(f"label_horizon={h} must be in [1, {len(close) - 1}] for {name}")
            target = close[h:] / close[:-h] - 1
            bar_in_day = torch.arange(len(target), device=device) % bars
            if same_day_label:
                target[(bar_in_day + h) >= bars] = torch.nan
            splits[name] = MarketSplit(name, values, target, bar_in_day)
        return cls(splits, list(meta["dates"]), list(meta["stocks"]), label_horizon, same_day_label)

    def summary(self) -> dict:
        return {
            "date_start": self.dates[0], "date_end": self.dates[-1],
            "n_dates": len(self.dates), "n_stocks": len(self.stocks),
            "n_bars": sum(len(split.target) for split in self.splits.values()),
            "label_horizon": self.label_horizon,
            "same_day_label": self.same_day_label,
            "split_bars": {name: len(split.target) for name, split in self.splits.items()},
        }
