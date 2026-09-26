"""Project1 数据接入：拉取沪深300成分子集的日频行情快照。

产出（data/raw/）：
  constituents.csv      成分股名单及抽样规则说明
  stocks/{code}.csv     单只股票日频原始行情（未复权，eastmoney，成交量单位=手）
  adj_factor/{code}.csv 乘法累计复权因子（后复权口径，sina，最早交易日锚定=1）
  index/000300.csv      沪深300指数日行情（基准）
  manifest.json         下载时间、数据源与版本、抽样规则、逐文件 sha256 与行数（校验值）

幂等：已存在且 manifest 中登记的文件跳过；--force 强制重拉。
用法：python scripts/fetch_data.py [--universe-size 50] [--start 20220101] [--end 20251231] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sina_symbol(code: str) -> str:
    if code.startswith(("6", "9", "5")):
        return f"sh{code}"
    if code.startswith(("8", "4")):
        return f"bj{code}"
    return f"sz{code}"


def constituent_codes(cons_df) -> list[str]:
    col = next((c for c in cons_df.columns if "成分券代码" in c), None)
    if col is None:  # 兜底：排除指数代码列
        col = next(c for c in cons_df.columns if "代码" in c and "指数" not in c)
    return sorted(str(x).zfill(6) for x in cons_df[col].astype(str))


def pick_universe(cons_df, size: int) -> list[str]:
    """确定性抽样：按代码升序等步长抽取 size 只，保证可复现。"""
    codes = constituent_codes(cons_df)
    step = max(1, len(codes) // size)
    return sorted(codes[::step][:size])


def _with_retry(fn, *args, attempts: int = 4, wait: float = 1.5, **kwargs):
    """网络抖动/代理不稳时的重试退避。"""
    last = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(wait * (i + 1))
    raise last


def fetch_one(ak, code: str, start: str, end: str) -> tuple[int, int, str]:
    """返回 (行情行数, 因子行数, 数据源)。sina 为主源（原始未复权日线+hfq因子），eastmoney 兜底。"""
    sym = sina_symbol(code)
    try:
        hist = _with_retry(ak.stock_zh_a_daily, symbol=sym, start_date=start, end_date=end)
        hist = hist[["date", "open", "high", "low", "close", "volume", "amount", "turnover"]]
        hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
        source = "sina(stock_zh_a_daily)"
    except Exception:
        hist = _with_retry(ak.stock_zh_a_hist, symbol=code, period="daily",
                           start_date=start, end_date=end, adjust="")
        hist = hist.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount", "换手率": "turnover_rate",
        })[["date", "open", "high", "low", "close", "volume", "amount", "turnover_rate"]]
        source = "eastmoney(stock_zh_a_hist)"
    out = RAW / "stocks" / f"{code}.csv"
    hist.to_csv(out, index=False, encoding="utf-8")

    factor = _with_retry(ak.stock_zh_a_daily, symbol=sym, adjust="hfq-factor")
    factor = factor.rename(columns={"date": "date", "hfq_factor": "adj_factor"})[["date", "adj_factor"]]
    factor["date"] = pd.to_datetime(factor["date"]).dt.strftime("%Y-%m-%d")
    fout = RAW / "adj_factor" / f"{code}.csv"
    factor.to_csv(fout, index=False, encoding="utf-8")
    return len(hist), len(factor), source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-size", type=int, default=50)
    ap.add_argument("--start", default="20220101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import akshare as ak

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "stocks").mkdir(exist_ok=True)
    (RAW / "adj_factor").mkdir(exist_ok=True)
    (RAW / "index").mkdir(exist_ok=True)

    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() and not args.force else {
        "files": {}
    }

    print(f"[universe] 获取沪深300成分名单 ...")
    cons = ak.index_stock_cons_csindex(symbol="000300")
    codes = pick_universe(cons, args.universe_size)
    cons_out = RAW / "constituents.csv"
    cons_out.write_text(
        "code\n" + "\n".join(constituent_codes(cons)) + "\n", encoding="utf-8")
    print(f"[universe] 共{len(cons)}只，按代码升序步长抽样 {len(codes)} 只: {codes[:5]} ...")

    files: dict[str, dict] = {}

    def register(rel: str, rows: int, note: str) -> None:
        p = RAW / rel
        files[rel] = {"sha256": sha256_file(p), "rows": int(rows), "note": note}

    index_path = RAW / "index" / "000300.csv"
    if args.force or not index_path.exists() or manifest.get("files", {}).get("index/000300.csv") is None:
        # eastmoney 80.push2 在当前网络不可达，基准指数改用 sina 全历史再按样本期切片
        idx = ak.stock_zh_index_daily(symbol="sh000300")
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx[(idx["date"] >= args.start) & (idx["date"] <= args.end)]
        idx = idx[["date", "open", "high", "low", "close", "volume"]]
        idx.to_csv(index_path, index=False, encoding="utf-8")
        print(f"[index] 沪深300 {len(idx)} 行（sina）")
    register("index/000300.csv", sum(1 for _ in open(index_path, encoding="utf-8")) - 1, "benchmark 沪深300 日行情")
    files["constituents.csv"] = {"sha256": sha256_file(cons_out), "rows": len(cons),
                                 "note": "沪深300成分名单（当期，非PIT，见局限）"}

    failed = []
    for i, code in enumerate(codes, 1):
        s_path, f_path = RAW / "stocks" / f"{code}.csv", RAW / "adj_factor" / f"{code}.csv"
        if (not args.force and s_path.exists() and f_path.exists()
                and manifest.get("files", {}).get(f"stocks/{code}.csv")):
            files[f"stocks/{code}.csv"] = manifest["files"][f"stocks/{code}.csv"]
            files[f"adj_factor/{code}.csv"] = manifest["files"][f"adj_factor/{code}.csv"]
            continue
        try:
            n_hist, n_fac, src = fetch_one(ak, code, args.start, args.end)
            register(f"stocks/{code}.csv", n_hist, f"日频原始行情（未复权；volume单位=股）[{src}]")
            register(f"adj_factor/{code}.csv", n_fac, "hfq乘法累计复权因子（锚定最早交易日=1）")
            print(f"[{i:>3}/{len(codes)}] {code} 行情{n_hist} 因子{n_fac} ({src.split('(')[0]})")
        except Exception as e:  # noqa: BLE001 —— 单票失败不中断整体，记录后人工核查
            failed.append((code, str(e)))
            print(f"[{i:>3}/{len(codes)}] {code} 失败: {e}")
        time.sleep(0.3)

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": {"library": "akshare", "version": ak.__version__,
                   "endpoints": ["stock_zh_a_hist(eastmoney)", "stock_zh_a_daily(sina hfq-factor)",
                                 "stock_zh_index_daily(sina)", "index_stock_cons_csindex"]},
        "universe_rule": f"沪深300当期成分按代码升序步长抽样{len(codes)}只（确定性）",
        "period": {"start": args.start, "end": args.end},
        "units": {"price": "元（未复权原始价）", "volume": "股（sina口径）", "amount": "元",
                  "turnover_rate": "比例（sina口径，0-1）"},
        "adjustment": "乘法累计因子；调整价 P_adj(t)=P_raw(t)*a(t)/a(τ)，τ=样本首日（见 configs/data.yaml）",
        "failed": failed,
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [c for c, _ in [(k.split("/")[1][:-4], 0) for k in files if k.startswith("stocks/")]]
    print(f"[done] 成功 {len(ok)} 只，失败 {len(failed)} 只；manifest: {manifest_path}")
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
