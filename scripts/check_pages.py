"""全页面稳定性巡检：验证所有页面/端点返回 200。

用法：
  python scripts/check_pages.py [--base http://127.0.0.1:8001]

对部署后的持续监控有用——任何页面异常立即非零退出。
"""
from __future__ import annotations

import argparse
import sys
import urllib.request

PAGES = [
    "/", "/factors", "/factors/new", "/factors/1",
    "/signals", "/signals/new", "/signals/1",
    "/backtests", "/backtests/new", "/backtests/compare?ids=1,2,3",
    "/pools", "/lab", "/experiments", "/runs",
    "/api/factors", "/api/runs/latest",
    "/api/signals/similarity-matrix",
    "/api/factors/correlation?ids=1,2,3",
    "/api/factors/incremental-ic?ids=1,2,3",
    "/api/signals/1/rebalance-hint",
    "/api/signals/1/best-horizon",
    "/api/signals/1/similarity",
    "/api/signals/1/combination-report.csv",
    "/api/factors/decay-compare?ids=1,2,3",
    "/api/signals/1/rolling-ic",
    "/api/experiments/summary.csv",
    "/api/factors/16/decay",
    "/backtests/compare?ids=3&ids=4",
    "/api/signals/9/combination-report.pdf",
    "/api/factors/1/ic.csv?horizon=5",
    "/api/signals/9/ic.csv",
    "/signals/15",
    "/signals/16",
    "/factors/16",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="QuantLab 全页面稳定性巡检")
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    args = ap.parse_args()

    failures = []
    print(f"巡检目标: {args.base}")
    print("-" * 50)
    for pg in PAGES:
        try:
            req = urllib.request.Request(args.base + pg)
            with urllib.request.urlopen(req, timeout=30) as r:
                status = r.status
        except Exception as e:
            status = f"ERR: {e}"
        ok = status == 200
        mark = "✓" if ok else "✗"
        print(f"  {mark} {pg} -> {status}")
        if not ok:
            failures.append(pg)

    print("-" * 50)
    if failures:
        print(f"巡检失败 {len(failures)}/{len(PAGES)} 个页面: {failures}")
        return 1
    print(f"全部 {len(PAGES)} 个页面/端点 200 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
