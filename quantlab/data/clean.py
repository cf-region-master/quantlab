"""数据清洗与质量报告（Project1 数据模块）。

输入 data/raw 快照（fetch_data.py 产出），输出 data/clean：
  {open,high,low,close}_raw.parquet  原始未复权价宽表（date×code）
  adj_factor.parquet           乘法累计复权因子 a(t)
  volume/amount/suspended.parquet
  benchmark.csv                基准指数
  quality_report.json          质量报告（重复键/日期顺序/缺失/非法值/覆盖率/清洗前后数量）
  clean_manifest.json          清洗时间、输入快照校验值、参数、输出校验值

  **不落调整价**：调整价是纯派生量 P_adj(t) = P_raw(t)·a(t)/a(τ)，τ 是口径参数。
  存派生结果等于把某个 τ 焊死在数据里（换口径要重跑清洗）；存 raw + factor 后
  任意口径都能在加载时精确还原，见 `MarketData.price(field, basis)`。

**「清洗层」做的是什么（以及不是什么）**
  它是 raw → 宽表面的 ETL 与质检：把 2431 个逐股 CSV 拼成 (日期×股票) 面板、去重、
  非法值置 NaN 并计数、区分未上市/停牌、产出质量报告与内容哈希。
  它**不做复权** —— 复权是加载时按所选口径算的。

清洗策略（与质量报告一致，缺失值一律不填零）：
  - 重复(日期,资产)键：保留首条并计数
  - 非法价格/成交量（<=0、high<low、high<max(open,close)等）：记数后置 NaN，整行不删
  - 缺失：保留 NaN；截面/时序使用处显式跳过并计数
  - 停牌：volume<=0 或收盘缺失 → suspended 标记（不可成交，估值用最后有效价）
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PRICE_FIELDS = ("open", "high", "low", "close")

# 清洗产物结构版本：改了落盘内容就 +1，让旧缓存自动失效（否则会静默沿用旧结构）
#   1 = 落 {f}_adj 调整价（旧）
#   2 = 只落 {f}_raw 原始价 + adj_factor，调整价改为加载时按口径派生
CLEAN_VERSION = 2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv().encode("utf-8")).hexdigest()


def _check_illegal(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """非法值检查：命中处置 NaN 并逐条计数（不删行）。"""
    counts = {k: 0 for k in ("nonpositive_price", "nonpositive_volume", "high_lt_low", "inconsistent_ohlc")}
    o, h, l, c, v = (df.get(f) for f in ("open", "high", "low", "close", "volume"))
    price_min = df[["open", "high", "low", "close"]].min(axis=1)

    bad = (price_min <= 0) & price_min.notna()
    counts["nonpositive_price"] = int(bad.sum())
    for f in PRICE_FIELDS:
        df.loc[bad, f] = np.nan

    if v is not None:
        bad_v = v < 0
        counts["nonpositive_volume"] = int(bad_v.sum())
        df.loc[bad_v, "volume"] = np.nan

    hl_bad = h.notna() & l.notna() & (h < l)
    counts["high_lt_low"] = int(hl_bad.sum())
    df.loc[hl_bad, ["high", "low"]] = np.nan

    leg = df[["open", "high", "low", "close"]]
    inc = (leg.max(axis=1) > h) | (leg.min(axis=1) < l)
    inc &= leg.notna().all(axis=1) & h.notna() & l.notna()
    counts["inconsistent_ohlc"] = int(inc.sum())
    df.loc[inc, ["open", "high", "low", "close"]] = np.nan
    return df, counts


def build_clean(raw_dir: Path, clean_dir: Path, period: dict[str, str]) -> dict[str, Any]:
    raw_dir, clean_dir = Path(raw_dir), Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    codes = sorted(
        p.stem for p in (raw_dir / "stocks").glob("*.csv")
        if raw_manifest.get("files", {}).get(f"stocks/{p.stem}.csv")
    )

    bench = pd.read_csv(raw_dir / "index" / "000300.csv", parse_dates=["date"]).set_index("date").sort_index()
    start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
    bench = bench.loc[(bench.index >= start) & (bench.index <= end)]
    calendar = bench.index

    panels: dict[str, dict[str, pd.DataFrame]] = {f: {} for f in ("close_raw", "volume", "amount")}
    factor_panels: dict[str, pd.Series] = {}
    per_stock_quality: dict[str, dict] = {}
    warnings: list[str] = []

    for code in codes:
        raw = pd.read_csv(raw_dir / "stocks" / f"{code}.csv", parse_dates=["date"])
        rows_raw = len(raw)
        raw = raw.drop_duplicates(subset="date", keep="first").sort_values("date").set_index("date")
        dup = rows_raw - len(raw)
        monotonic = raw.index.is_monotonic_increasing
        raw, illegal = _check_illegal(raw)

        fac = pd.read_csv(raw_dir / "adj_factor" / f"{code}.csv", parse_dates=["date"])
        fac = fac.drop_duplicates(subset="date", keep="last").sort_values("date").set_index("date")["adj_factor"]
        fac = fac.groupby(level=0).last()
        # 复权因子是阶梯函数：公告日之间沿用最近一次公告值（沿用已发布值，非构造价格）
        fac = fac[~fac.index.duplicated(keep="last")].asfreq("D").ffill()
        fac = fac.reindex(fac.index.intersection(raw.index))

        raw_in_window = raw.loc[(raw.index >= start) & (raw.index <= end)]
        fac_in = fac.reindex(raw_in_window.index)
        if fac_in.dropna().empty:
            warnings.append(f"{code}: 样本期内无可用的复权因子，已剔除")
            continue
        a_tau = fac_in.dropna().iloc[0]
        # 仅用于 per_stock 的「因子公告覆盖缺口」统计（missing_close_adj）；
        # 正式的加载派生见 MarketData.price（用 ffill 后的因子面板）
        adj = raw_in_window[["open", "high", "low", "close"]].mul(fac_in / a_tau, axis=0)
        adj.columns = [f"{c}_adj" for c in adj.columns]

        panels["volume"][code] = raw_in_window["volume"]
        panels["amount"][code] = raw_in_window["amount"]
        for c in PRICE_FIELDS:
            # 原始价：唯一落盘的价格来源（调整价在加载时按口径派生）
            panels.setdefault(f"{c}_raw", {})[code] = raw_in_window[c]
        factor_panels[code] = fac_in

        span = raw_in_window.dropna(subset=["close"]).index
        miss_raw = int(raw_in_window["close"].isna().sum())
        suspended_days = int(((raw_in_window["volume"] <= 0) | raw_in_window["close"].isna()).sum())
        per_stock_quality[code] = {
            "rows_raw": rows_raw, "rows_in_window": len(raw_in_window),
            "duplicate_dates_dropped": dup, "raw_monotonic": bool(monotonic),
            "illegal_counts": illegal, "missing_close_raw": miss_raw,
            "missing_close_adj": int(adj["close_adj"].isna().sum()),
            "suspended_days": suspended_days,
            "coverage": round(float(raw_in_window["close"].notna().mean()), 6),
            "first_date": str(span.min().date()) if len(span) else None,
            "last_date": str(span.max().date()) if len(span) else None,
            "adj_factor_ref": float(a_tau),
        }
        if per_stock_quality[code]["coverage"] < 0.9:
            warnings.append(f"{code}: 覆盖率仅 {per_stock_quality[code]['coverage']:.1%}，存在较多缺失")

    wide: dict[str, pd.DataFrame] = {}
    for name, d in panels.items():
        m = pd.DataFrame(d).reindex(calendar).sort_index()
        m.columns.name = "code"
        wide[name] = m
    adj_factor_wide = pd.DataFrame(factor_panels).reindex(calendar).ffill()
    # 区分"未上市"（首有效收盘之前）与"停牌"（上市后 volume<=0 或原始价缺失）
    # 注意：必须用【原始】收盘价判定停牌。原实现误用复权价 close_adj，
    # 使得复权因子覆盖不到的日期（原始价其实存在）被误判为停牌。
    first_valid = {}
    for code in wide["close_raw"].columns:
        s = wide["close_raw"][code].dropna()
        first_valid[code] = s.index.min() if len(s) else pd.NaT
    pre_listed = pd.DataFrame({c: wide["close_raw"].index < fv for c, fv in first_valid.items()},
                              index=wide["close_raw"].index)
    listed = ~pre_listed
    suspended = listed & (wide["volume"].isna() | (wide["volume"] <= 0)
                          | wide["close_raw"].isna())

    agg = {
        "n_assets_requested": len(codes),
        "n_assets_cleaned": wide["close_raw"].shape[1],
        "n_trading_days": int(wide["close_raw"].shape[0]),
        "date_range": [str(wide["close_raw"].index.min().date()), str(wide["close_raw"].index.max().date())],
        "rows_raw_total": int(sum(q["rows_raw"] for q in per_stock_quality.values())),
        "rows_in_window_total": int(sum(q["rows_in_window"] for q in per_stock_quality.values())),
        "duplicate_dates_dropped_total": int(sum(q["duplicate_dates_dropped"] for q in per_stock_quality.values())),
        "illegal_total": {k: int(sum(q["illegal_counts"][k] for q in per_stock_quality.values()))
                          for k in next(iter(per_stock_quality.values()))["illegal_counts"]},
        # 口径=加载派生语义：raw 缺失 或 【ffill 后的】复权因子缺失。
        # 历史 bug：这里曾用"未 ffill 的段内因子"在内存里拼调整价统计（1883），
        # 而用户实际加载到的派生面板只有 601 格缺失 —— 报告与数据不符。
        "missing_close_adj_cells": int(
            (wide["close_raw"].isna() | adj_factor_wide.isna()).sum().sum()),
        "pre_listed_cells": int(pre_listed.sum().sum()),
        "suspended_cells_listed": int(suspended.sum().sum()),
        "suspended_cell_ratio_listed": round(float(suspended[listed.columns].stack().mean()), 6),
        "coverage_mean": round(float(np.mean([q["coverage"] for q in per_stock_quality.values()])), 6),
        "coverage_min": round(float(np.min([q["coverage"] for q in per_stock_quality.values()])), 6),
    }
    quality = {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "duplicate_key": "按(日期,资产)去重保留首条",
            "illegal": "置NaN并计数，不删行",
            "missing": "保留NaN，下游显式跳过并计数；缺失值一律不统一填零",
            "adjustment": "复权因子为阶梯函数，公告日间沿用最近公告值（非构造价格）",
        },
        "aggregate": agg,
        "per_stock": per_stock_quality,
        "warnings": warnings,
    }

    outputs = {
        "volume.parquet": wide["volume"],
        "amount.parquet": wide["amount"], "adj_factor.parquet": adj_factor_wide,
        "suspended.parquet": suspended.astype(bool),
    }
    # 只落【原始价】+【复权因子】，不落调整价：
    #   调整价是纯派生量 P_adj(t) = P_raw(t)·a(t)/a(τ)，τ 是口径参数。
    #   把派生结果当存储结果，等于把某一个 τ 焊死在数据里 —— 换口径就得重跑清洗。
    #   存 raw + factor 后，任意 τ（前后复权、任意基准日）都能在加载时精确算出，
    #   且不占额外存储（4 张原始价，取代原来的 4 张调整价）。
    for f in PRICE_FIELDS:
        outputs[f"{f}_raw.parquet"] = wide[f"{f}_raw"]

    out_hashes = {}
    # 先删掉本版本不再产出的旧文件：否则上一版留下的 {f}_adj.parquet 会留在目录里，
    # 让"当前口径是什么"变成靠文件名猜 —— 旧文件必须消失，避免被误读为有效产物。
    for old in clean_dir.glob("*.parquet"):
        if old.name not in outputs:
            old.unlink()
            warnings.append(f"清理上一版本残留产物: {old.name}")
    for fname, df in outputs.items():
        df.to_parquet(clean_dir / fname)
        out_hashes[fname] = _sha256_df(df)
    bench.reset_index().rename(columns={"index": "date"}).to_csv(clean_dir / "benchmark.csv", index=False)
    out_hashes["benchmark.csv"] = sha256_file(clean_dir / "benchmark.csv")
    (clean_dir / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    out_hashes["quality_report.json"] = sha256_file(clean_dir / "quality_report.json")

    manifest = {
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "clean_version": CLEAN_VERSION,
        "source_snapshot": {"raw_manifest_sha256": sha256_file(raw_dir / "manifest.json"),
                            "fetched_at": raw_manifest.get("fetched_at"),
                            "source": raw_manifest.get("source")},
        "params": {"period": period,
                   "prices": "只落原始未复权价 {f}_raw + adj_factor；调整价在加载时按口径算",
                   "adjustment": "P_adj(t) = P_raw(t)·a(t)/a(τ)，τ 由复权口径决定（hfq=τ首 / qfq=τ末）",
                   "missing_policy": "不填零"},
        "outputs": out_hashes,
        "aggregate": agg,
    }
    (clean_dir / "clean_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"quality": quality, "manifest": manifest}


class MarketData:
    """清洗后的市场数据面板（宽表 date×code）。复权在**加载时**按口径算，不在清洗时落盘。

    落盘的是原始未复权价 `{f}_raw` 与乘法累计复权因子 `a(t)`：

        P_adj(t) = P_raw(t) · a(t) / a(τ)

    τ 由口径决定：`hfq` → 该股在样本期内首个有效 a(t)（后复权）；
                  `qfq` → 末个有效 a(t)（前复权）。
    两个口径只差一个【每股常数】k = a(τ首)/a(τ末)：P_qfq = P_hfq·k，与 t 无关 ——
    所以任意 τ（含任意中间基准日）都能零成本还原，不需要重跑清洗或多存价格表。

    口径只影响**价格水平**，不影响收益率：hfq/qfq 的日收益率逐值相同（实测差 2.2e-16）。
    """

    BASES = ("hfq", "qfq")
    BASIS_LABELS = {"hfq": "后复权（τ = 样本首日）", "qfq": "前复权（τ = 样本末日）"}

    def __init__(self, clean_dir: Path):
        clean_dir = Path(clean_dir)
        try:
            self._raw = {f: pd.read_parquet(clean_dir / f"{f}_raw.parquet")
                         for f in PRICE_FIELDS}
        except FileNotFoundError as e:
            # 明确报错，不回退到任何"看起来像价格"的旧文件：
            # 静默读旧口径的面板，会让"复权方式"这个参数变成谎话。
            raise FileNotFoundError(
                f"{clean_dir} 缺少原始价面板（{e.filename}）。当前清洗层只落"
                f" {{open,high,low,close}}_raw.parquet + adj_factor.parquet，"
                f"调整价在加载时按口径派生。请重新清洗："
                f"python scripts/run_pipeline.py --force-clean") from e
        self.adj_factor = pd.read_parquet(clean_dir / "adj_factor.parquet")
        self.volume = pd.read_parquet(clean_dir / "volume.parquet")
        self.amount = pd.read_parquet(clean_dir / "amount.parquet")
        self.suspended = pd.read_parquet(clean_dir / "suspended.parquet")
        bench = pd.read_csv(clean_dir / "benchmark.csv", parse_dates=["date"]).set_index("date")
        self.benchmark_close = bench["close"]
        # 未复权 vwap（成交额/成交量；成交量<=0 处保持 NaN，不填 0）
        with np.errstate(invalid="ignore", divide="ignore"):
            self.vwap_raw = self.amount / self.volume.where(self.volume > 0, np.nan)
        self._tau_cache: dict[str, pd.Series] = {}
        self._price_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._view_cache: dict[str, Any] = {}
        qr = json.loads((clean_dir / "quality_report.json").read_text(encoding="utf-8"))
        self.quality = qr

    # ---------------- 基础属性 ----------------
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._raw["close"].index

    @property
    def codes(self) -> list[str]:
        return list(self._raw["close"].columns)

    @property
    def close_raw(self) -> pd.DataFrame:
        return self._raw["close"]

    def daily_returns(self, basis: str = "hfq") -> pd.DataFrame:
        px = self.price("close", basis)
        return px / px.shift(1) - 1

    # ---------------- 复权口径 ----------------
    def tau(self, basis: str = "hfq") -> pd.Series:
        """各股的复权基准因子 a(τ)：hfq 取首个有效值，qfq 取末个有效值。"""
        basis = self._check_basis(basis)
        if basis in self._tau_cache:
            return self._tau_cache[basis]
        f = self.adj_factor
        notna = f.notna().to_numpy()
        any_ok = notna.any(axis=0)
        vals = f.to_numpy(dtype="float64")
        n, m = vals.shape
        cols = np.arange(m)
        if basis == "hfq":
            pick = np.argmax(notna, axis=0)
        else:
            pick = n - 1 - np.argmax(notna[::-1], axis=0)
        s = pd.Series(np.where(any_ok, vals[pick, cols], np.nan),
                      index=f.columns, name=f"tau_{basis}")
        self._tau_cache[basis] = s
        return s

    def basis_scale(self, basis: str = "hfq") -> pd.Series | None:
        """把后复权面板换算成指定口径所需的【每股常数】；hfq 无需缩放，返回 None。

        k = a(τ首)/a(τ末)：P_adj_qfq(t) = P_adj_hfq(t)·k，与 t 无关，从未除权的股票 k=1。
        """
        basis = self._check_basis(basis)
        if basis == "hfq":
            return None
        key = "scale_qfq"
        if key not in self._tau_cache:
            with np.errstate(invalid="ignore", divide="ignore"):
                self._tau_cache[key] = self.tau("hfq") / self.tau("qfq")
        return self._tau_cache[key]

    def _check_basis(self, basis: str) -> str:
        b = basis or "hfq"
        if b not in self.BASES:
            raise ValueError(f"未知复权口径 {b!r}；可选 {list(self.BASES)}")
        return b

    def price(self, field: str = "close", basis: str = "hfq") -> pd.DataFrame:
        """指定复权口径的价格面板（open/high/low/close/vwap）。"""
        if field == "vwap":
            raw = self.vwap_raw
        elif field in PRICE_FIELDS:
            raw = self._raw[field]
        else:
            raise KeyError(f"复权口径只对价格类字段有意义，收到 {field!r}")
        basis = self._check_basis(basis)
        key = (field, basis)
        if key not in self._price_cache:
            with np.errstate(invalid="ignore", divide="ignore"):
                self._price_cache[key] = raw * (self.adj_factor / self.tau(basis))
        return self._price_cache[key]

    # 兼容旧属性名：默认后复权
    @property
    def open_adj(self) -> pd.DataFrame:
        return self.price("open", "hfq")

    @property
    def high_adj(self) -> pd.DataFrame:
        return self.price("high", "hfq")

    @property
    def low_adj(self) -> pd.DataFrame:
        return self.price("low", "hfq")

    @property
    def close_adj(self) -> pd.DataFrame:
        return self.price("close", "hfq")

    @property
    def vwap(self) -> pd.DataFrame:
        return self.price("vwap", "hfq")

    @property
    def fields(self) -> dict[str, pd.DataFrame]:
        return self.fields_for("hfq")

    def prices(self, basis: str = "hfq") -> dict[str, pd.DataFrame]:
        return {f: self.price(f, basis) for f in ("open", "high", "low", "close", "vwap")}

    def fields_for(self, basis: str = "hfq") -> dict[str, pd.DataFrame]:
        """按复权口径取字段字典（volume/amount 是量，不随口径变）。"""
        basis = self._check_basis(basis)
        out = {"volume": self.volume, "amount": self.amount}
        for f in ("open", "high", "low", "close"):
            out[f] = self.price(f, basis)
        out["vwap"] = self.price("vwap", basis)
        return out

    def basis_view(self, basis: str = "hfq"):
        """轻量视图：把 open_adj/close_adj 换成指定口径，其余属性透传。

        回测引擎只有 `market.open_adj/close_adj/volume/suspended/dates` 这几处依赖，
        用视图注入口径即可，不必给引擎加参数、也不必复制整份 MarketData。
        """
        b = self._check_basis(basis)
        if b == "hfq":
            return self
        if b not in self._view_cache:
            self._view_cache[b] = _BasisView(self, b)
        return self._view_cache[b]

    def scoped_view(self, basis: str = "hfq", start=None, end=None, pool_mask=None):
        """限定【复权口径 + 日期区间 + 股票池】的视图。

        日期区间是**闭区间**，按本地交易日切；区间内不足 2 天会明确报错。

        股票池以**逐日掩码置 NaN** 的方式施加到所有价格/量面板上，不做静态裁剪：
        PIT 指数成分逐日变化，静态裁成"曾经属于过"的并集等于放松池子；
        而 NaN 会一路传播到因子求值与标签（`close_adj` 也是 NaN），
        因此 IC、GP 适应度、AlphaGen 的特征与奖励**都只在池内计算**，
        两个挖矿引擎自动得到同一套口径，不必各自实现一遍。
        """
        b = self._check_basis(basis)
        lo = pd.Timestamp(start) if start else self.dates.min()
        hi = pd.Timestamp(end) if end else self.dates.max()
        idx = self.dates[(self.dates >= lo) & (self.dates <= hi)]
        if len(idx) < 2:
            raise ValueError(
                f"挖掘区间 [{lo.date()}, {hi.date()}] 内只有 {len(idx)} 个交易日，"
                f"至少需要 2 天（本地行情 {self.dates.min().date()} ~ {self.dates.max().date()}）")
        m = None
        if pool_mask is not None:
            m = pool_mask.reindex(index=idx, columns=self.codes).fillna(False)
            m = m.to_numpy(dtype=bool)
        key = f"{b}|{idx.min().date()}|{idx.max().date()}|p{0 if m is None else int(m.sum())}"
        if key not in self._view_cache:
            self._view_cache[key] = _ScopedView(self, b, idx, m)
        return self._view_cache[key]


class _BasisView:
    """MarketData 的复权口径视图（其余属性透传给底层对象）。"""

    def __init__(self, md: MarketData, basis: str):
        object.__setattr__(self, "_md", md)
        object.__setattr__(self, "_basis", basis)

    @property
    def basis(self) -> str:
        return self._basis

    @property
    def open_adj(self) -> pd.DataFrame:
        return self._md.price("open", self._basis)

    @property
    def close_adj(self) -> pd.DataFrame:
        return self._md.price("close", self._basis)

    @property
    def fields(self) -> dict[str, pd.DataFrame]:
        return self._md.fields_for(self._basis)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_md"), name)


class _ScopedView(_BasisView):
    """在口径视图之上再限定日期区间与股票池（用于挖掘任务的「计算范围」）。"""

    def __init__(self, md: MarketData, basis: str, idx: pd.DatetimeIndex, mask=None):
        super().__init__(md, basis)
        object.__setattr__(self, "_idx", idx)
        object.__setattr__(self, "_mask", mask)      # (T, N) bool 或 None

    @property
    def dates(self) -> pd.DatetimeIndex:
        return object.__getattribute__(self, "_idx")

    def _slice(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.reindex(index=self.dates)
        m = object.__getattribute__(self, "_mask")
        if m is not None:
            out = out.where(m, np.nan)               # 池外整格 NaN（逐日生效）
        return out

    @property
    def close_adj(self) -> pd.DataFrame:
        return self._slice(self._md.price("close", self._basis))

    @property
    def open_adj(self) -> pd.DataFrame:
        return self._slice(self._md.price("open", self._basis))

    @property
    def high_adj(self) -> pd.DataFrame:
        return self._slice(self._md.price("high", self._basis))

    @property
    def low_adj(self) -> pd.DataFrame:
        return self._slice(self._md.price("low", self._basis))

    @property
    def volume(self) -> pd.DataFrame:
        return self._slice(self._md.volume)

    @property
    def amount(self) -> pd.DataFrame:
        return self._slice(self._md.amount)

    @property
    def suspended(self) -> pd.DataFrame:
        return self._slice(self._md.suspended)

    @property
    def fields(self) -> dict[str, pd.DataFrame]:
        out = self._md.fields_for(self._basis)
        return {k: self._slice(v) for k, v in out.items()}

    def cache_key(self) -> str:
        """缓存签名：口径 + 区间 + 池（池不同则数据不同，必须分目录）。"""
        m = object.__getattribute__(self, "_mask")
        cells = 0 if m is None else int(m.sum())
        return f"{self._basis}_{self.dates.min().date()}_{self.dates.max().date()}_p{cells}"
