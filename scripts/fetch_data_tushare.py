"""Project1 数据接入（Tushare 版）：拉取沪深300成分子集的日频行情快照。

与 akshare 版（scripts/fetch_data.py）产出**完全相同的目录结构与字段口径**，
因此下游 quantlab/data/clean.py 无需任何改动：

产出（data/raw/）：
  constituents.csv      成分股名单（沪深300当期成分，非 PIT）
  stocks/{code}.csv     date,open,high,low,close,volume,amount,turnover
  adj_factor/{code}.csv date,adj_factor
  index/000300.csv      date,open,high,low,close,volume
  manifest.json         下载时间、数据源、抽样规则、逐文件 sha256 与行数

单位换算（Tushare 口径 -> 本仓库口径，已用 akshare 快照逐值核对）：
  vol     手    -> 股     × 100
  amount  千元  -> 元     × 1000
  turnover_rate %  -> 比例(0-1)  ÷ 100
  OHLC    原样（与 sina 未复权价逐值一致）

凭证：从环境变量 TUSHARE_TOKEN 或 quantlab/.env 读取，不写死在代码里。

用法：
  python scripts/fetch_data_tushare.py                    # 默认 50 只，2022-01-04~2025-12-31
  python scripts/fetch_data_tushare.py --universe-size 50 --start 20220101 --end 20251231
  python scripts/fetch_data_tushare.py --force             # 忽略已有文件强制重拉
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
API = "http://api.tushare.pro"

# 沪深300：指数行情用 000300.SH，成分权重用 399300.SZ（同指数两所代码）
INDEX_CODE_DAILY = "000300.SH"
INDEX_CODE_WEIGHT = "399300.SZ"
BENCH_FILE = "000300.csv"


class TushareError(RuntimeError):
    pass


# ---------------- 凭证 ----------------
def load_token() -> str:
    """优先环境变量，其次 quantlab/.env（已在 .gitignore 排除）。"""
    tok = os.environ.get("TUSHARE_TOKEN", "").strip()
    if tok:
        return tok
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TUSHARE_TOKEN="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    return v
    raise TushareError(
        "未找到 TUSHARE_TOKEN。请设置环境变量，或在 "
        f"{env} 中写入 TUSHARE_TOKEN=<你的token>"
    )


# ---------------- HTTP 调用 ----------------
def _post(api_name: str, token: str, params: dict, fields: str, timeout: int = 60) -> dict:
    import urllib.request

    payload = json.dumps({"api_name": api_name, "token": token,
                          "params": params, "fields": fields}).encode()
    req = urllib.request.Request(API, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def api(api_name: str, token: str, params: dict | None = None, fields: str = "",
        attempts: int = 5, wait: float = 3.0) -> pd.DataFrame:
    """调用 Tushare 并返回 DataFrame；对频次限制/网络抖动做指数退避重试。"""
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = _post(api_name, token, params or {}, fields)
        except Exception as e:  # noqa: BLE001 —— 网络层重试
            last = e
            time.sleep(wait * (i + 1))
            continue
        code = r.get("code")
        if code == 0:
            data = r.get("data") or {}
            cols = data.get("fields") or []
            items = data.get("items") or []
            return pd.DataFrame(items, columns=cols)
        msg = (r.get("msg") or "").strip()
        # 频次超限 / 积分不足的临时性错误 -> 退避重试；权限类错误直接抛出
        if any(k in msg for k in ("每分钟", "频率", "频次", "超过", "timeout", "busy")):
            last = TushareError(f"{api_name}: {msg}")
            time.sleep(wait * (i + 2))
            continue
        raise TushareError(f"{api_name} 失败 code={code} msg={msg}")
    raise TushareError(f"{api_name} 重试 {attempts} 次仍失败: {last}")


# ---------------- 代码格式 ----------------
def to_ts_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "9", "5")):
        return f"{code}.SH"
    if code.startswith(("00", "30", "20")):
        return f"{code}.SZ"
    if code.startswith(("4", "8", "92")):
        return f"{code}.BJ"
    raise ValueError(f"无法判断交易所: {code}")


def strip_code(ts_code: str) -> str:
    return str(ts_code).split(".")[0].zfill(6)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- 成分名单 ----------------
def latest_index_weight(token: str, lookback_months: int = 8) -> tuple[pd.DataFrame, str]:
    """取最近一个可用月份的沪深300成分快照（当期成分，非 PIT，与原实现口径一致）。"""
    today = date.today()
    y, m = today.year, today.month
    for _ in range(lookback_months):
        first = date(y, m, 1)
        nxt = date(y + (m == 12), 1 if m == 12 else m + 1, 1)
        last = date.fromordinal(nxt.toordinal() - 1)
        df = api("index_weight", token,
                 {"index_code": INDEX_CODE_WEIGHT,
                  "start_date": first.strftime("%Y%m%d"),
                  "end_date": last.strftime("%Y%m%d")},
                 "index_code,con_code,trade_date,weight")
        if not df.empty:
            return df, str(df["trade_date"].max())
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    raise TushareError("近 8 个月均未取到沪深300成分权重")


def pick_universe(codes: list[str], size: int) -> list[str]:
    """确定性抽样：按代码升序等步长抽取 size 只（与 akshare 版同一规则）。"""
    codes = sorted(str(c).zfill(6) for c in codes)
    step = max(1, len(codes) // size)
    return sorted(codes[::step][:size])


def strip_prefix_symbol(sym: str) -> str:
    """SH600000 / SZ000001 / BJ430047 -> 600000 / 000001 / 430047"""
    s = str(sym).strip().upper()
    if s[:2] in ("SH", "SZ", "BJ"):
        s = s[2:]
    return s.zfill(6)


def pit_union_codes(start: str, end: str) -> tuple[list[str], str]:
    """HS300/ZZ500/ZZ1000 在 [start,end] 内出现过的成分【并集】。

    这是动态股票池的抓取范围：只抓并集，逐日归属由 data/reference 的区间表在下游决定。
    返回 (代码列表, 规则说明)。
    """
    ref = ROOT / "data" / "reference" / "universe_intervals.csv"
    if not ref.exists():
        raise TushareError(
            f"缺少 PIT 成分区间表 {ref}；请先导入 data/reference，或改用 --universe sample")
    iv = pd.read_csv(ref, parse_dates=["opt-in", "opt-out"])
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    iv["opt-in"] = iv["opt-in"].fillna(pd.Timestamp("1900-01-01"))
    iv["opt-out"] = iv["opt-out"].fillna(pd.Timestamp("2100-01-01"))
    w = iv[(iv["opt-in"] <= hi) & (iv["opt-out"] > lo)]
    if w.empty:
        raise TushareError(f"区间表在 {start}~{end} 内没有任何成分，请检查日期")
    codes = sorted({strip_prefix_symbol(s) for s in w["symbol"]})
    per_idx = {i: int(w[w["index"] == i]["symbol"].nunique()) for i in sorted(w["index"].unique())}
    detail = "、".join(f"{k}={v}" for k, v in per_idx.items())
    rule = (f"HS300/ZZ500/ZZ1000 逐日 PIT 成分并集（data/reference/universe_intervals.csv；"
            f"{detail}）")
    return codes, rule


# ---------------- 单只股票 ----------------
def fetch_one(token: str, code: str, start: str, end: str,
              with_basic: bool = False) -> tuple[int, int]:
    ts_code = to_ts_code(code)

    day = api("daily", token, {"ts_code": ts_code, "start_date": start, "end_date": end},
              "ts_code,trade_date,open,high,low,close,vol,amount")
    if day.empty:
        raise TushareError(f"{code} 区间内无行情数据")

    fac = api("adj_factor", token, {"ts_code": ts_code, "start_date": start, "end_date": end},
              "ts_code,trade_date,adj_factor")
    if fac.empty:
        raise TushareError(f"{code} 区间内无复权因子")

    # 换手率：默认**不逐股拉**（省 1/3 次调用）—— daily_basic 由
    # scripts/fetch_daily_basic.py 按交易日整市场拉取，落在 data/raw/daily_basic.parquet，
    # 字段更全（市值/PE/PB 等）且调用次数少一个数量级。clean.py 也不用这个列。
    turnover = None
    if with_basic:
        try:
            bas = api("daily_basic", token,
                      {"ts_code": ts_code, "start_date": start, "end_date": end},
                      "ts_code,trade_date,turnover_rate")
            if not bas.empty:
                bas = bas.copy()
                bas["date"] = pd.to_datetime(bas["trade_date"], format="%Y%m%d")
                bas["turnover"] = pd.to_numeric(bas["turnover_rate"], errors="coerce") / 100.0
                turnover = bas[["date", "turnover"]]
        except TushareError:
            turnover = None

    day = day.copy()
    day["date"] = pd.to_datetime(day["trade_date"], format="%Y%m%d")
    for c in ("open", "high", "low", "close"):
        day[c] = pd.to_numeric(day[c], errors="coerce")
    # 单位换算：手 -> 股，千元 -> 元
    day["volume"] = pd.to_numeric(day["vol"], errors="coerce") * 100.0
    day["amount"] = pd.to_numeric(day["amount"], errors="coerce") * 1000.0

    if turnover is not None:
        day = day.merge(turnover, on="date", how="left")
    else:
        day["turnover"] = pd.NA

    day = day.sort_values("date")
    hist = day[["date", "open", "high", "low", "close", "volume", "amount", "turnover"]].copy()
    hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    out = RAW / "stocks" / f"{code}.csv"
    hist.to_csv(out, index=False, encoding="utf-8")

    fac = fac.copy()
    fac["date"] = pd.to_datetime(fac["trade_date"], format="%Y%m%d")
    fac["adj_factor"] = pd.to_numeric(fac["adj_factor"], errors="coerce")
    fac = fac.sort_values("date")[["date", "adj_factor"]]
    fac["date"] = fac["date"].dt.strftime("%Y-%m-%d")
    fout = RAW / "adj_factor" / f"{code}.csv"
    fac.to_csv(fout, index=False, encoding="utf-8")

    return len(hist), len(fac)


def _recorded_range(old: dict, rel: str) -> tuple[str, str] | None:
    """某个文件上次是按什么区间拉的。

    优先取逐文件记录；老 manifest 没有该字段时回落到顶层 period
    （顶层 period 就是那次运行所有文件共同的请求区间）。
    取不到返回 None —— 视为**未覆盖**，宁可重拉也不静默沿用。
    """
    rec = old.get("files", {}).get(rel) or {}
    rng = rec.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return str(rng[0]), str(rng[1])
    p = old.get("period") or {}
    if p.get("start") and p.get("end"):
        return str(p["start"]), str(p["end"])
    return None


def _covers(rec: tuple[str, str] | None, start: str, end: str) -> bool:
    """上次拉的区间是否已覆盖本次请求区间（字符串 YYYYMMDD 可直接比较）。"""
    if rec is None:
        return False
    return str(rec[0]) <= str(start) and str(rec[1]) >= str(end)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tushare 数据接入（沪深300子集日频快照）")
    ap.add_argument("--universe", choices=["pit", "sample"], default="pit",
                    help="pit=HS300/ZZ500/ZZ1000 逐日 PIT 成分并集（动态股票池，默认）；"
                         "sample=旧的沪深300当期成分步长抽样")
    ap.add_argument("--universe-size", type=int, default=50,
                    help="仅 --universe sample 时生效")
    ap.add_argument("--start", default="20220101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.25, help="每次请求间隔秒数（限频保护）")
    ap.add_argument("--with-basic", action="store_true",
                    help="额外逐股拉 daily_basic（仅换手率）。默认不拉："
                         "fetch_daily_basic.py 已按交易日整市场拉取，字段更全、调用数少一个数量级")
    args = ap.parse_args()

    token = load_token()
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "stocks").mkdir(exist_ok=True)
    (RAW / "adj_factor").mkdir(exist_ok=True)
    (RAW / "index").mkdir(exist_ok=True)

    manifest_path = RAW / "manifest.json"
    if args.force or not manifest_path.exists():
        old = {"files": {}}
    else:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.universe == "pit":
        print("[universe] 展开 HS300/ZZ500/ZZ1000 逐日 PIT 成分并集 ...")
        codes, rule = pit_union_codes(args.start, args.end)
        cons_codes, snap_date = codes, None
        print(f"[universe] 并集 {len(codes)} 只: {codes[:5]} ...")
    else:
        print("[universe] 获取沪深300成分权重 ...")
        cons, snap_date = latest_index_weight(token)
        cons_codes = sorted(strip_code(c) for c in cons["con_code"].unique())
        codes = pick_universe(cons_codes, args.universe_size)
        rule = f"沪深300当期成分按代码升序步长抽样{len(codes)}只（确定性）"
        print(f"[universe] 快照日={snap_date} 共{len(cons_codes)}只，"
              f"步长抽样 {len(codes)} 只: {codes[:5]} ...")
    cons_out = RAW / "constituents.csv"
    cons_out.write_text("code\n" + "\n".join(cons_codes) + "\n", encoding="utf-8")

    files: dict[str, dict] = {}

    def register(rel: str, rows: int, note: str) -> None:
        p = RAW / rel
        files[rel] = {"sha256": sha256_file(p), "rows": int(rows), "note": note}

    # 基准指数：同样必须按区间判断是否重拉。
    # ⚠️ clean.py 用 benchmark.csv 的日期当【交易日历】（calendar = bench.index），
    # 指数文件一旦被跳过，扩区间就会变成"股票有 2020~2026、日历只有 2022~2025"，
    # 整个清洗结果被静默截断回旧区间。
    index_path = RAW / "index" / BENCH_FILE
    idx_rel = f"index/{BENCH_FILE}"
    if (args.force or not index_path.exists()
            or not _covers(_recorded_range(old, idx_rel), args.start, args.end)):
        idx = api("index_daily", token,
                  {"ts_code": INDEX_CODE_DAILY, "start_date": args.start, "end_date": args.end},
                  "trade_date,open,high,low,close,vol")
        if idx.empty:
            raise TushareError("未取到沪深300指数行情")
        idx = idx.copy()
        idx["date"] = pd.to_datetime(idx["trade_date"], format="%Y%m%d")
        for c in ("open", "high", "low", "close"):
            idx[c] = pd.to_numeric(idx[c], errors="coerce")
        idx["volume"] = pd.to_numeric(idx["vol"], errors="coerce") * 100.0  # 手 -> 股
        idx = idx.sort_values("date")[["date", "open", "high", "low", "close", "volume"]]
        idx["date"] = idx["date"].dt.strftime("%Y-%m-%d")
        idx.to_csv(index_path, index=False, encoding="utf-8")
        print(f"[index] 沪深300 {len(idx)} 行（tushare index_daily）"
              f" {idx['date'].iloc[0]} ~ {idx['date'].iloc[-1]}")
    else:
        print("[index] 区间已覆盖，跳过")
    register(idx_rel, sum(1 for _ in open(index_path, encoding="utf-8")) - 1,
             "benchmark 沪深300 日行情")
    files[idx_rel]["range"] = [args.start, args.end]
    files["constituents.csv"] = {
        "sha256": sha256_file(cons_out), "rows": len(cons_codes),
        "note": f"沪深300成分名单（Tushare index_weight 快照日 {snap_date}，非PIT，见局限）"}

    failed: list[tuple[str, str]] = []
    skipped = 0
    for i, code in enumerate(codes, 1):
        s_path = RAW / "stocks" / f"{code}.csv"
        f_path = RAW / "adj_factor" / f"{code}.csv"
        s_rel, f_rel = f"stocks/{code}.csv", f"adj_factor/{code}.csv"
        # 只有【上次拉的区间已覆盖本次请求区间】才跳过。
        # 原实现只看文件是否存在 —— 扩区间时会把 2022~2025 的旧文件当成已覆盖而静默跳过，
        # 于是"扩了区间"实际上没扩（踩过一次）。
        if (not args.force and s_path.exists() and f_path.exists()
                and _covers(_recorded_range(old, s_rel), args.start, args.end)
                and _covers(_recorded_range(old, f_rel), args.start, args.end)):
            files[s_rel] = old["files"][s_rel]
            files[f_rel] = old["files"][f_rel]
            skipped += 1
            print(f"[{i:>4}/{len(codes)}] {code} 区间已覆盖，跳过")
            continue
        try:
            n_hist, n_fac = fetch_one(token, code, args.start, args.end,
                                      with_basic=args.with_basic)
            register(s_rel, n_hist,
                     "日频原始行情（未复权；volume单位=股）[tushare(daily)]")
            register(f_rel, n_fac, "乘法累计复权因子[tushare(adj_factor)]")
            files[s_rel]["range"] = [args.start, args.end]
            files[f_rel]["range"] = [args.start, args.end]
            print(f"[{i:>4}/{len(codes)}] {code} 行情{n_hist} 因子{n_fac} (tushare)")
        except Exception as e:  # noqa: BLE001 —— 单票失败不中断整体
            failed.append((code, str(e)))
            print(f"[{i:>4}/{len(codes)}] {code} 失败: {e}")
        time.sleep(args.sleep)
    if skipped:
        print(f"[skip] {skipped} 只区间已覆盖，未重复请求")

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "library": "tushare",
            "access": "HTTP API (http://api.tushare.pro)",
            "endpoints": ["index_weight(csi300)",
                          "daily(未复权行情)", "adj_factor(复权因子)",
                          "daily_basic(换手率)", "index_daily(沪深300基准)"],
        },
        "universe_rule": rule,
        "universe_mode": args.universe,
        "universe_snapshot_date": snap_date,
        "n_symbols": len(codes),
        "period": {"start": args.start, "end": args.end},
        "units": {"price": "元（未复权原始价）", "volume": "股（由 tushare 手×100 换算）",
                  "amount": "元（由 tushare 千元×1000 换算）",
                  "turnover_rate": "比例（由 tushare %÷100 换算）"},
        "adjustment": "乘法累计因子；调整价 P_adj(t)=P_raw(t)*a(t)/a(τ)，τ=样本首日（见 configs/data.yaml）",
        "notes": [
            "tushare adj_factor 精度为 2-4 位小数，与 akshare/sina hfq_factor(6位) 存在 ~1e-4 量级差异",
            "OHLC 未复权价与 akshare 快照逐值一致；vol/amount 单位已换算并核对",
        ],
        "failed": failed,
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ok = sum(1 for k in files if k.startswith("stocks/"))
    print(f"[done] 成功 {n_ok} 只，失败 {len(failed)} 只；manifest: {manifest_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
