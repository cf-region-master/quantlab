"""组合方式对照实验（一键）：同因子集，逐组合模型对照 RankIC。

用法：
  python scripts/compare_combinations.py --factors 1,2,3 --start 2023-01-01 --end 2025-06-30
  python scripts/compare_combinations.py --factors 1,2,3 --horizon 5 --models equal_weight,ic_meanvar

输出对照表并落盘 reports/combination_experiments/<stamp>.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from quantlab.lab.comparison import compare_combinations  # noqa: E402
from quantlab.config import ROOT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="组合方式对照实验")
    ap.add_argument("--factors", required=True, help="因子 id，逗号分隔，如 1,2,3")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--horizon", type=int, default=5, help="标签持有期（交易日）")
    ap.add_argument("--models", default=None,
                    help="逗号分隔；默认全部轻量组合模型")
    ap.add_argument("--backtest", action="store_true",
                    help="每个模型追加一次真实模拟成交（周度 top10 等权）")
    args = ap.parse_args()

    factor_ids = [int(x) for x in args.factors.split(",") if x.strip()]
    models = [x for x in args.models.split(",") if x.strip()] if args.models else None
    res = compare_combinations(factor_ids, args.start, args.end,
                               horizon=args.horizon, models=models,
                               include_backtest=args.backtest)

    print(f"\n组合方式对照实验（因子 {res['factor_ids']} · h={args.horizon} · "
          f"{args.start} ~ {args.end}）")
    print("-" * 72)
    print(f"{'模型':<28} {'RankIC均值':>10} {'t(普通)':>8} {'t(NW)':>8} {'N':>6}")
    for m, v in res["models"].items():
        if "error" in v:
            print(f"{m:<28} ERROR: {v['error'][:44]}")
        else:
            tnw = f"{v['t_nw']:.2f}" if v["t_nw"] is not None else "—"
            print(f"{m:<28} {v['rank_ic_mean']:>10.4f} {v['t_naive']:>8.2f} "
                  f"{tnw:>8} {v['n_obs']:>6}")
    print("-" * 72)
    print(res["note"])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / "combination_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cmp_{stamp}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"[saved] {out}")
    return 0




if __name__ == "__main__":
    sys.exit(main())
