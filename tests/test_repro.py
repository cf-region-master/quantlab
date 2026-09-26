"""复现与数据管线测试（小样本端到端）。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_raw_snapshot_manifest_integrity():
    """原始快照完整性：manifest 中每个文件的 sha256 与实际文件一致。"""
    manifest = json.loads((ROOT / "data" / "raw" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"], "manifest 不应为空"
    import hashlib
    bad = []
    for rel, meta in manifest["files"].items():
        p = ROOT / "data" / "raw" / rel
        if not p.exists():
            bad.append((rel, "missing"))
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != meta["sha256"]:
            bad.append((rel, "hash mismatch"))
    assert not bad, f"快照校验失败: {bad[:5]}"


def test_clean_quality_report_fields():
    """质量报告包含 Project1 要求的关键项，且缺失值不填零。"""
    qr = json.loads((ROOT / "data" / "clean" / "quality_report.json").read_text(encoding="utf-8"))
    agg = qr["aggregate"]
    for key in ("duplicate_dates_dropped_total", "illegal_total", "missing_close_adj_cells",
                "coverage_mean", "rows_raw_total", "rows_in_window_total",
                "suspended_cells_listed", "pre_listed_cells"):
        assert key in agg, f"质量报告缺少 {key}"
    assert qr["policy"]["missing"] and "不统一填零" in qr["policy"]["missing"]
    # 缺失在面板中应保持 NaN 而非 0。
    # 注意：清洗层现在只落原始价 {f}_raw + adj_factor，调整价在加载时按口径派生；
    # 「missing_close_adj_cells」统计的是"原始价缺失 或 复权因子缺失"的格子，
    # 与 τ 无关（缩放不会制造或消除缺失），因此用派生面板核对同样成立。
    from quantlab.data.clean import MarketData

    md = MarketData(ROOT / "data" / "clean")
    close = md.price("close", "hfq")
    assert int(close.isna().sum().sum()) == agg["missing_close_adj_cells"]
    # 落盘的原始价必须存在且与派生面板口径无关
    assert "close" in md._raw and md.close_raw.isna().sum().sum() <= close.isna().sum().sum()


def test_pipeline_rerun_deterministic(tmp_path):
    """同数据+同配置重跑：主要数值结果哈希一致（声明容差=0）。"""
    from quantlab.config import load_config
    from quantlab.pipeline import ensure_clean, build_factor_suite, build_signal_panel, run_single_backtest
    cfg = load_config()
    market = ensure_clean(cfg)
    suite = build_factor_suite(market, cfg)
    sig = build_signal_panel(cfg, suite, market)
    a = run_single_backtest(market, sig, cfg, name="repro")
    b = run_single_backtest(market, sig, cfg, name="repro")
    ha = hashlib.sha256(json.dumps(a["nav"], sort_keys=True).encode()).hexdigest()
    hb = hashlib.sha256(json.dumps(b["nav"], sort_keys=True).encode()).hexdigest()
    assert ha == hb, "重跑净值不一致，违反可复现要求"
