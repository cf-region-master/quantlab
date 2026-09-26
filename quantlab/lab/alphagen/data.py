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
