"""从指定 run 生成研究报告（Markdown + PDF）。

用法：python scripts/build_report.py --run v1
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from quantlab.report import build_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_id（reports/runs 下的目录名）")
    args = ap.parse_args()
    md, pdf = build_report(args.run)
    print(f"[report] {md}")
    if pdf:
        print(f"[report] {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
