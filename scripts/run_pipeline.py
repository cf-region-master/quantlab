"""一键运行完整研究流水线：数据清洗 → 因子诊断 → 回测 → 对照实验 → 复现 manifest。

用法：
  python scripts/run_pipeline.py              # 常规运行（自动生成 run_id）
  python scripts/run_pipeline.py --run-id v1  # 指定运行 ID
  python scripts/run_pipeline.py --force-clean # 强制重清洗
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from quantlab.pipeline import run_full_pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="QuantLab 端到端研究流水线")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--force-clean", action="store_true")
    args = ap.parse_args()
    out = run_full_pipeline(run_id=args.run_id, force_clean=args.force_clean)
    print(f"[entry] 研究报告: python scripts/build_report.py --run {out.name}")
    print(f"[entry] Web 平台: python -m uvicorn quantlab.api.main:app --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
