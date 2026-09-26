"""补拉 Tushare daily_basic（按交易日整市场一次），为估值/市值类字段与市值中性化提供数据。

与 fetch_data_tushare.py 的区别（以及为什么不让它顺手拉）：
  逐股方式需要 2431 次 daily_basic 调用；**按交易日**只需 969 次，每次返回当日全市场，
  量级差 2.5 倍且更快。daily_basic 是「按日快照」的接口，本就应该按日拉。

产出（data/raw/daily_basic.parquet，长表）：
  date, code, close, turnover_rate, turnover_rate_f, volume_ratio,
  pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
  total_share, float_share, free_share, total_mv, circ_mv

单位：total_mv / circ_mv 单位为【万元】（Tushare 原样，不做换算，字段名保持 dark 语义清晰，
加载层 fundamentals.py 会再乘 1e4 转成元并记录口径）。

用法：
  python scripts/fetch_daily_basic.py                    # 用 data/reference/trade_days.csv 的日历
  python scripts/fetch_daily_basic.py --start 20220101 --end 20251231
  python scripts/fetch_daily_basic.py --force            # 全量重拉
  python scripts/fetch_daily_basic.py --resume           # 断点续拉（默认行为，已有日期跳过）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "daily_basic_parts"
OUT = RAW / "daily_basic.parquet"

sys.path.insert(0, str(ROOT))
from scripts.fetch_data_tushare import TushareError, api, load_token  # noqa: E402


def to_plain_code(ts_code: str) -> str:
    """600000.SH -> 600000（与本项目 market.codes 的六位代码对齐）"""
    return str(ts_code).split(".")[0].zfill(6)

FIELDS = ("ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
          "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,"
          "total_share,float_share,free_share,total_mv,circ_mv")

NUM_COLS = ["close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm",
            "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm",
            "total_share", "float_share", "free_share", "total_mv", "circ_mv"]


def trade_days(start: str, end: str) -> list[str]:
    """交易日历：优先 data/reference/trade_days.csv，落到区间内。"""
    ref = ROOT / "data" / "reference" / "trade_days.csv"
    if ref.exists():
        d = pd.read_csv(ref, parse_dates=["datetime"])["datetime"]
        days = pd.DatetimeIndex(sorted(d.unique()))
    else:
        # 退化：用基准指数文件（若存在）
        idx = RAW / "index" / "000300.csv"
        if not idx.exists():
            raise TushareError("既无 data/reference/trade_days.csv 也无 data/raw/index/000300.csv")
        days = pd.DatetimeIndex(pd.read_csv(idx, parse_dates=["date"])["date"])
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    sel = days[(days >= lo) & (days <= hi)]
    return [d.strftime("%Y%m%d") for d in sel]


def fetch_day(token: str, day: str) -> pd.DataFrame:
    df = api("daily_basic", token, {"trade_date": day}, FIELDS)
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["code"] = df["ts_code"].map(to_plain_code)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["date", "code"] + [c for c in NUM_COLS if c in df.columns]
    return df[keep]


def main() -> int:
    ap = argparse.ArgumentParser(description="按交易日补拉 Tushare daily_basic")
    ap.add_argument("--start", default="20220101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--sleep", type=float, default=0.16, help="每次请求间隔秒数（限频保护）")
    ap.add_argument("--force", action="store_true", help="忽略分片缓存全量重拉")
    args = ap.parse_args()

    token = load_token()
    CACHE.mkdir(parents=True, exist_ok=True)

    days = trade_days(args.start, args.end)
    print(f"[daily_basic] 交易日 {len(days)} 天：{days[0]} ~ {days[-1]}", flush=True)

    ok, skipped, empty, failed = 0, 0, 0, []
    for i, day in enumerate(days, 1):
        part = CACHE / f"{day}.csv"
        if part.exists() and not args.force:
            skipped += 1
            if i % 100 == 0 or i == len(days):
                print(f"[{i:>4}/{len(days)}] {day} 已缓存（跳过 {skipped}）", flush=True)
            continue
        try:
            df = fetch_day(token, day)
            if df.empty:
                empty += 1
                part.write_text("empty\n", encoding="utf-8")
                print(f"[{i:>4}/{len(days)}] {day} 空（非交易日/无数据）", flush=True)
            else:
                df.to_csv(part, index=False, encoding="utf-8")
                ok += 1
                if i % 25 == 0 or i == len(days):
                    print(f"[{i:>4}/{len(days)}] {day} {len(df)} 只", flush=True)
        except Exception as e:  # noqa: BLE001 —— 单日失败不中断
            failed.append((day, str(e)))
            print(f"[{i:>4}/{len(days)}] {day} 失败: {e}", flush=True)
        time.sleep(args.sleep)

    # 合并分片
    parts = []
    for day in days:
        part = CACHE / f"{day}.csv"
        if not part.exists():
            continue
        try:
            d = pd.read_csv(part)
            if "code" in d.columns and len(d):
                d["date"] = pd.to_datetime(d["date"])
                parts.append(d)
        except Exception:  # noqa: BLE001
            continue
    if not parts:
        print("[daily_basic] 没有任何分片可合并", file=sys.stderr)
        return 1
    all_df = pd.concat(parts, ignore_index=True).sort_values(["date", "code"])
    all_df.to_parquet(OUT, index=False)

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": {"library": "tushare", "endpoint": "daily_basic", "mode": "by trade_date",
                   "access": "HTTP API (http://api.tushare.pro)"},
        "period": {"start": days[0], "end": days[-1]},
        "n_trade_days_requested": len(days),
        "n_days_fetched_now": ok, "n_days_cached_skipped": skipped,
        "n_days_empty": empty, "n_days_failed": len(failed),
        "n_rows": int(len(all_df)),
        "n_codes": int(all_df["code"].nunique()),
        "units": {
            "total_mv": "万元（Tushare 原样；加载层 ×1e4 转元）",
            "circ_mv": "万元（同上）",
            "total_share": "万股", "float_share": "万股", "free_share": "万股",
            "turnover_rate": "%（未除以100，加载层换算）",
            "pe/pe_ttm/pb/ps/ps_ttm/dv_ratio/dv_ttm": "倍 / 百分数（Tushare 原样）",
        },
        "date_range_in_file": [str(all_df["date"].min().date()), str(all_df["date"].max().date())],
        "failed": failed[:50],
    }
    (RAW / "daily_basic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] {OUT}  {len(all_df):,} 行  {all_df['code'].nunique()} 只  "
          f"{manifest['date_range_in_file'][0]} ~ {manifest['date_range_in_file'][1]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
