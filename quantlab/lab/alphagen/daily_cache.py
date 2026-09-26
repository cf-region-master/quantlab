"""日频缓存构建：把本项目的 clean 面板写成 AlphaGen 引擎要的 [bar, feature, stock] 布局。

与原版（`data.py`，5 分钟 48 bar/日）的唯一区别是 **bars_per_day = 1**。
这样 `MarketData.load` / `Calculator` / `AlphaPool` / `BatchedAlphaVecEnv` 全部可原样复用：
  - `bar_in_day = arange(T) % 1 = 0`（恒为 0）
  - `pool.py` 的"日内预热屏蔽"由 `same_day_label` 控制，日频传 False 即自动跳过；
    滚动窗口的预热期由算子自身的 NaN 传播处理。

特征与顺序必须与 `alphagen.data.FEATURE_NAMES` 一致：
    (open, close, high, low, volume, vwap)
其中 vwap = amount / volume（本项目 amount 单位元、volume 单位股 → 元/股）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ...data.clean import MarketData as QuantlabMarket
from .data import FEATURE_NAMES

BARS_PER_DAY_DAILY = 1


def build_daily_cache(market: QuantlabMarket, cache_dir: Path,
                      force: bool = False) -> Path:
    """把 clean 宽表写成 market.npy + meta.json（float32）。

    股票池与日期区间由调用方通过 `MarketData.scoped_view()` 施加（池外整格 NaN），
    这里只负责布局转换 —— 掩码只在一处实现，GP / AlphaGen 两个引擎口径自然一致。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = cache_dir / "market.npy", cache_dir / "meta.json"
    dates = [str(d.date()) for d in market.dates]
    stocks = [str(c) for c in market.codes]

    if data_path.exists() and meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (meta.get("dates") == dates and meta.get("stocks") == stocks
                    and meta.get("bars_per_day") == BARS_PER_DAY_DAILY):
                return cache_dir
        except Exception:  # noqa: BLE001 —— 缓存损坏则重建
            pass

    T, N = len(dates), len(stocks)
    arr = np.full((T, len(FEATURE_NAMES), N), np.nan, dtype="float32")

    def put(name: str, df) -> None:
        i = FEATURE_NAMES.index(name)
        block = df.reindex(index=market.dates, columns=market.codes).to_numpy(dtype="float64")
        arr[:, i, :] = block.astype("float32")

    put("open", market.open_adj)
    put("close", market.close_adj)
    put("high", market.high_adj)
    put("low", market.low_adj)
    put("volume", market.volume)
    # vwap = amount / volume，volume<=0 处保持 NaN（不填 0）
    vol = market.volume.reindex(index=market.dates, columns=market.codes).to_numpy(dtype="float64")
    amt = market.amount.reindex(index=market.dates, columns=market.codes).to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(vol > 0, amt / vol, np.nan)
    arr[:, FEATURE_NAMES.index("vwap"), :] = vwap.astype("float32")

    np.save(data_path, arr)
    meta_path.write_text(json.dumps({
        "dates": dates, "stocks": stocks, "bars_per_day": BARS_PER_DAY_DAILY,
        "features": list(FEATURE_NAMES), "source": "quantlab:data/clean",
        "n_dates": T, "n_stocks": N,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_dir


def cache_summary(cache_dir: Path) -> dict:
    meta = json.loads((Path(cache_dir) / "meta.json").read_text(encoding="utf-8"))
    arr = np.load(Path(cache_dir) / "market.npy", mmap_mode="r")
    fin = 0
    for i in range(0, arr.shape[0], max(1, arr.shape[0] // 40)):   # 抽样，避免全量扫描
        fin += int(np.isfinite(arr[i]).sum())
    return {
        "shape": list(arr.shape), "bars_per_day": meta["bars_per_day"],
        "n_dates": meta["n_dates"], "n_stocks": meta["n_stocks"],
        "features": meta["features"],
        "date_range": [meta["dates"][0], meta["dates"][-1]],
        "finite_sample": fin,
    }
