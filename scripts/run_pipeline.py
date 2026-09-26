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

    # 数据/清洗产物已更新：通知本机 Web 进程重载缓存（若在运行）。
    # 历史问题：Web 进程的 market 单例与池缓存永不失效，重跑流水线后页面继续用旧面板。
    import os
    import urllib.request

    for base in (os.environ.get("QUANTLAB_WEB_URL", "http://127.0.0.1:8000"),
                 "http://127.0.0.1:8001"):
        try:
            req = urllib.request.Request(base + "/api/admin/cache/reset", method="POST")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    print(f"[hook] 已通知 Web 进程重载缓存 ({base})")
                    break
        except Exception:  # noqa: BLE001 —— Web 未运行则静默
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
