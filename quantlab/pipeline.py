"""端到端研究流水线：数据 → 因子 → 诊断 → 回测 → 对照实验 → 运行产物与复现 manifest。

一次 run 产出 reports/runs/<run_id>/：
  manifest.json          复现信息（版本/配置/数据校验值/耗时/检查输出/产物校验值）
  quality_summary.json   数据质量报告摘要
  factors/<key>/         因子卡 + 诊断结果（含逐日 IC 序列与分层汇总）
  backtests/<name>.json  回测结果（净值/交易/换手/成本/指标/检查）
  experiments/<exp>.json 对照实验结果与对照表
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ROOT, code_version, load_config, Config
from .data.clean import MarketData, build_clean, sha256_file
from .factors.diagnostics import diagnostics_json, run_diagnostics
from .factors.operators import compute_factor, factor_hash, load_cards
from .factors.preprocess import apply_preprocess

RUNS = ROOT / "reports" / "runs"
STORE = ROOT / "data" / "store"


def _jdump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def clean_cache_stale(cfg: Config) -> tuple[bool, str]:
    """清洗缓存是否失效：比对 raw 快照校验值、样本区间与产物结构版本。

    修复点（原实现只检查 clean_manifest.json 是否存在，换数据源/换区间后会静默沿用旧结果）。
    返回 (是否过期, 原因)。
    """
    from .data.clean import CLEAN_VERSION

    raw_dir = ROOT / "data" / "raw"
    marker = ROOT / "data" / "clean" / "clean_manifest.json"
    if not marker.exists():
        return True, "缺少 clean_manifest.json"
    try:
        cm = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 —— 清单损坏视为过期
        return True, f"clean_manifest.json 不可解析: {e}"
    if int(cm.get("clean_version") or 0) != CLEAN_VERSION:
        return True, (f"清洗产物结构版本变化"
                      f"（清单={cm.get('clean_version')} 当前={CLEAN_VERSION}）")
    rec = (cm.get("source_snapshot") or {}).get("raw_manifest_sha256")
    cur = sha256_file(raw_dir / "manifest.json") if (raw_dir / "manifest.json").exists() else None
    if rec != cur:
        return True, f"raw 快照校验值变化（清单={str(rec)[:12]} 当前={str(cur)[:12]}）"
    rec_period = (cm.get("params") or {}).get("period")
    if rec_period != cfg.data["period"]:
        return True, f"样本区间变化（清单={rec_period} 配置={cfg.data['period']}）"
    return False, "命中缓存"


def ensure_clean(cfg: Config, force: bool = False) -> MarketData:
    clean_dir = ROOT / "data" / "clean"
    raw_dir = ROOT / "data" / "raw"
    stale, why = clean_cache_stale(cfg)
    if force or stale:
        if stale and not force:
            print(f"[clean] 缓存失效（{why}），重新清洗 ...")
        res = build_clean(raw_dir, clean_dir, cfg.data["period"])
        print(f"[clean] 完成：{res['manifest']['aggregate']['n_assets_cleaned']} 资产 × "
              f"{res['manifest']['aggregate']['n_trading_days']} 交易日；"
              f"覆盖率均值 {res['manifest']['aggregate']['coverage_mean']:.2%}")
    else:
        print(f"[clean] {why}，跳过重新清洗")
    return MarketData(clean_dir)


def build_factor_suite(market: MarketData, cfg: Config) -> dict[str, Any]:
    """计算全部注册因子的原始值与信号值，并执行诊断。"""
    cards = load_cards(cfg.factors)
    diag_cfg = cfg.factors["diagnosis"]
    suite: dict[str, Any] = {}
    for key, card in cards.items():
        raw = compute_factor(card, market.fields)
        sig = apply_preprocess(raw, card.preprocess, diag_cfg)
        h = factor_hash(card, raw)
        diag = run_diagnostics(raw, market.close_adj, diag_cfg)
        suite[key] = {"card": card, "raw": raw, "signal": sig, "hash": h, "diagnostics": diag}
        ic5 = diag["5"]["ic_summary"]["rank_ic"]["mean"] if "5" in diag else np.nan
        ic20 = diag["20"]["ic_summary"]["rank_ic"]["mean"] if "20" in diag else np.nan
        print(f"[factor] {key:<10} hash={h[:8]} RankIC(5d)={ic5:+.4f} RankIC(20d)={ic20:+.4f}")
    return suite


def build_signal_panel(cfg: Config, suite: dict[str, Any], market: MarketData) -> pd.DataFrame:
    """按 backtest.yaml 生成信号面板：单因子预处理后取值，或 composite 加权合成。"""
    sig_cfg = cfg.backtest["signal"]
    bt_start = pd.Timestamp(cfg.backtest["sample"]["start"])
    bt_end = pd.Timestamp(cfg.backtest["sample"]["end"])
    if sig_cfg.get("type", "factor") == "factor":
        key = sig_cfg["factor"]
        sig = apply_preprocess(suite[key]["raw"], sig_cfg.get("preprocess",
                               suite[key]["card"].preprocess), cfg.factors["diagnosis"])
    else:  # composite：因子 zscore 加权（策略库的 equal_weight 模型）
        weights = sig_cfg["weights"]
        zs = [apply_preprocess(suite[k]["raw"], ["winsorize", "zscore"], cfg.factors["diagnosis"])
              for k in weights]
        sig = sum(z * w for z, w in zip(zs, weights.values()))
        sig.columns.name = "code"
    return sig.loc[(sig.index >= bt_start) & (sig.index <= bt_end)]


def resolve_bt_pool(market: MarketData, bt_cfg: dict) -> tuple:
    """解析回测配置里的股票池（`pool:` 段）为与 market 对齐的逐日掩码。

    不配置 `pool` 时返回 (None, "全部可用资产") —— 但这会让回测从全部已抓取资产里挑，
    包含当日并非任何指数成分的标的，需在报告中显式说明。
    """
    spec = bt_cfg.get("pool")
    if not spec:
        return None, "全部可用资产（未配置股票池）"
    from .data.universe import resolve_pool
    return resolve_pool(market, spec)


def run_single_backtest(market: MarketData, signal: pd.DataFrame, cfg: Config,
                        name: str, pool_mask: pd.DataFrame | None = None) -> dict[str, Any]:
    from .backtest.checks import finalize_checks, run_checks
    from .backtest.engine import run_backtest
    from .backtest.metrics import compute_metrics

    bt_cfg = dict(cfg.backtest)
    bt_cfg.setdefault("portfolio", {})
    bt_cfg["portfolio"] = {**bt_cfg["portfolio"]}
    if pool_mask is None:
        pool_mask, pool_note = resolve_bt_pool(market, bt_cfg)
    else:
        pool_note = "由调用方指定"
    bt_cfg["pool_note"] = pool_note
    net = run_backtest(signal, market, bt_cfg, name=name, pool_mask=pool_mask)
    gross = run_backtest(signal, market, bt_cfg, name=name, disable_cost=True,
                         pool_mask=pool_mask)
    net.nav_gross = gross.nav  # noqa: SLF001 —— 毛净值挂到结果上
    rf = float(cfg.data["risk_free_annual"])
    A = int(cfg.data["trading_days_per_year"])
    net.metrics = compute_metrics(net, rf, A)
    checks = run_checks(net, market, nav_gross=gross.nav, signal=signal, pool_mask=pool_mask)

    # 毛净归因：如实量化两条路径的差异，并按实测结果声明采用的归因口径。
    # 原实现把这一项【加在 run_checks 之后】，而 all_pass 已在 run_checks 内部算完，
    # 导致该检查即使判 False 也不影响 all_pass（已交付产物中即为此状态）。
    net_dates = [t["date"] for t in net.trades]
    gross_dates = [t["date"] for t in gross.trades]
    pairs = list(zip(net.trades, gross.trades))
    n_diff = sum(1 for a, g in pairs if abs(float(a["amount"]) - float(g["amount"])) > 1e-12)
    max_diff = max((abs(float(a["amount"]) - float(g["amount"])) for a, g in pairs), default=0.0)
    same_path = bool(net_dates == gross_dates and n_diff == 0)
    checks["gross_net_attribution"] = {
        "convention": "same_path" if same_path else "separate_reruns",
        "same_trade_dates": bool(net_dates == gross_dates),
        "n_trades_net": len(net.trades), "n_trades_gross": len(gross.trades),
        "n_trades_differing_notional": int(n_diff),
        "max_notional_diff": float(max_diff),
        "cost_total": float(net.metrics["cost"]["total_cost_currency"]),
        "gross_minus_net_final_nav": float(gross.nav.iloc[-1] - net.nav.iloc[-1]),
        "note": ("费用会通过可用现金影响后续下单金额，故毛/净两条序列可能不同。"
                 "convention=separate_reruns 时，毛净差异 = 费用 + 路径差异，"
                 "不作纯费用归因；差异幅度见 n_trades_differing_notional / max_notional_diff。"),
    }
    checks = finalize_checks(checks)   # 所有检查写入后统一计算 all_pass
    net.checks = checks
    print(f"[bt] {name:<28} 年化={net.metrics['net']['annualized_return']:+.2%} "
          f"Sharpe={net.metrics['net']['annualized_sharpe']:+.2f} "
          f"MDD={net.metrics['net']['max_drawdown']:.2%} "
          f"换手合计={net.metrics['turnover']['sum']:.1f} "
          f"成本={net.metrics['cost']['total_cost_currency']:.4f} "
          f"检查={'PASS' if checks['all_pass'] else 'FAIL'}")
    return net.to_json()


def run_experiments(market: MarketData, suite: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """对照实验：每个实验只改一个主要因素，其余口径保持一致。"""
    from .config import load_config
    out: dict[str, Any] = {}
    for exp in cfg.experiment.get("experiments", []):
        variants_out = {}
        for v in exp["variants"]:
            vcfg = load_config(overrides={**v["override"], "_name": f"{exp['name']}:{v['name']}"},
                               backtest_file="backtest.yaml")
            signal = build_signal_panel(vcfg, suite, market)
            variants_out[v["name"]] = run_single_backtest(market, signal, vcfg,
                                                          name=f"{exp['name']}:{v['name']}")
        # 对照表
        rows = []
        for vname, bt in variants_out.items():
            m = bt["metrics"]
            rows.append({
                "variant": vname,
                "cum_return": m["net"]["cumulative_return"],
                "annualized_return": m["net"]["annualized_return"],
                "annualized_vol": m["net"]["annualized_vol"],
                "sharpe": m["net"]["annualized_sharpe"],
                "max_drawdown": m["net"]["max_drawdown"],
                "turnover_sum": m["turnover"]["sum"],
                "total_cost": m["cost"]["total_cost_currency"],
                "n_trades": m["portfolio"]["n_trades"],
                "checks_pass": bt["checks"]["all_pass"],
            })
        out[exp["name"]] = {
            # 优先用实验自己的问题/预期；缺省时回落到顶层总纲（原实现对每个实验都用顶层，
            # 导致 E1/E2 显示同一句"问题/预期"）
            "question": exp.get("question") or cfg.experiment.get("question"),
            "expectation": exp.get("expectation") or cfg.experiment.get("expectation"),
            "keep_fixed": cfg.experiment.get("keep_fixed"),
            "description": exp["description"],
            "variants": variants_out,
            "comparison": rows,
        }
        print(f"[exp] {exp['name']}: " + " | ".join(
            f"{r['variant']} 年化{r['annualized_return']:+.2%}/Sharpe{r['sharpe']:+.2f}"
            for r in rows))
    return out


def run_full_pipeline(run_id: str | None = None, force_clean: bool = False) -> Path:
    t0 = time.time()
    cfg = load_config()
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RUNS / run_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    market = ensure_clean(cfg, force=force_clean)
    qr = market.quality
    _jdump({"aggregate": qr["aggregate"], "policy": qr["policy"],
            "warnings": qr["warnings"]}, out / "quality_summary.json")

    suite = build_factor_suite(market, cfg)

    factors_dir = out / "factors"
    for key, item in suite.items():
        _jdump({"card": item["card"].card_dict(), "hash": item["hash"],
                "diagnostics": diagnostics_json(item["diagnostics"])},
               factors_dir / f"{key}.json")

    # 基准回测（backtest.yaml 原样）
    signal = build_signal_panel(cfg, suite, market)
    base = run_single_backtest(market, signal, cfg, name="base")
    _jdump(base, out / "backtests" / "base.json")

    experiments = run_experiments(market, suite, cfg)
    for ename, e in experiments.items():
        _jdump(e, out / "experiments" / f"{ename}.json")

    # 复现 manifest
    raw_dir = ROOT / "data" / "raw"
    clean_dir = ROOT / "data" / "clean"
    output_files = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {
        "run_id": run_id,
        "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - t0, 3),
        "environment": {"python": platform.python_version(),
                        "pandas": pd.__version__, "numpy": np.__version__,
                        "os": platform.platform()},
        "code_version": code_version(),
        "data": {"raw_manifest_sha256": sha256_file(raw_dir / "manifest.json"),
                 "clean_manifest_sha256": sha256_file(clean_dir / "clean_manifest.json"),
                 "universe_size": len(market.codes),
                 "n_trading_days": len(market.dates)},
        "config": cfg.snapshot(),
        "checks": {"base_backtest_all_pass": base["checks"]["all_pass"]},
        "outputs": output_files,
        "manifest_note": "同数据+同代码+同配置重跑，输出哈希应一致（容差检查见 tests/test_repro.py）",
    }
    _jdump(manifest, out / "manifest.json")
    print(f"[done] run={run_id} 用时 {time.time()-t0:.1f}s → {out}")
    return out
