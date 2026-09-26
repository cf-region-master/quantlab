"""服务层：因子库/策略库/算法实验室的业务编排（API 层只做校验与调用本层）。"""
from __future__ import annotations

import json
import threading
import uuid
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ROOT, code_version
from ..data.clean import MarketData
from ..factors.diagnostics import diagnostics_json, run_diagnostics
from ..factors.operators import FactorCard, compute_factor, factor_hash, load_cards
from ..factors.preprocess import (DEFAULT_SPEC, PREPROCESS_SCHEMA, apply_preprocess,
                                  apply_preprocess_spec, normalize_spec, parse_horizons,
                                  spec_steps)
from .db import (AlgoCandidate, AlgoTask, BacktestRun, BatchJob, Factor, FactorMetric,
                 SessionLocal, Signal, StockPool, get_session, init_db, now)

VALUES_DIR = ROOT / "data" / "store" / "factor_values"
# 回测结果与"策略"解耦：一个信号可对应多个回测 run，文件名 bt{run_id}.json
RESULTS_DIR = ROOT / "data" / "store" / "backtest_results"
# （已移除同进程引擎注册表：引擎现于独立子进程执行，取消经 task_runner 文件旗标）

_market: MarketData | None = None


def reset_caches() -> dict:
    """清空进程内数据缓存（market 单例/股票池掩码/基本面面板）。

    使用场景：服务运行期间重跑了 pipeline/清洗/抓取后，页面数据仍来自旧面板 ——
    调用本函数后下一次请求重新加载。不影响 DB 与因子值 parquet（那两处本来
    就是每次查询现读的）。
    """
    global _market, _close_5m
    _market = None
    _close_5m = None
    _pool_cache.clear()
    try:
        from ..data import fundamentals
        fundamentals.clear_cache()
    except Exception:  # noqa: BLE001 —— fundamentals 可用性不影响其余缓存清理
        pass
    return {"status": "cache_reset"}


def market() -> MarketData:
    global _market
    if _market is None:
        _market = MarketData(ROOT / "data" / "clean")
    return _market


def latest_run_dir() -> Path | None:
    """最新一次流水线运行目录：按【修改时间】取，而非目录名字典序。

    原实现用 sorted(glob('*'))[-1]，run_id 为 v1..v10 时会选到 v9。
    """
    runs = [p for p in (ROOT / "reports" / "runs").glob("*") if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: (p / "manifest.json").stat().st_mtime
               if (p / "manifest.json").exists() else p.stat().st_mtime)


def list_run_summaries() -> list[dict]:
    """列出全部流水线 run 的可比摘要（供 /runs 对比页使用）。

    每个 run 目录都尽力解析：缺失文件不抛错，字段置 None，保证页面不因单个
    损坏的 run 而 500。
    """
    base = ROOT / "reports" / "runs"
    if not base.exists():
        return []
    out: list[dict] = []
    # 注：glob 与后续 stat/read 之间文件可能被清理（流水线重跑），
    # 每个 run 的解析都吞掉 OSError/JSON 错误，保证页面不 500。

    def _json(p: Path, default):
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
        except Exception:  # noqa: BLE001
            return default

    for d in base.glob("*"):
        if not d.is_dir():
            continue
        mf = _json(d / "manifest.json", None)
        if mf is None:
            continue
        bt = _json(d / "backtests" / "base.json", {})
        q = _json(d / "quality_summary.json", {})
        metrics = (bt.get("metrics") or {})
        net = metrics.get("net") or {}
        agg = (q.get("aggregate") or {})
        factors = {}
        for jf in sorted((d / "factors").glob("*.json")):
            fd = _json(jf, {})
            key = ((fd.get("card") or {}).get("key"))
            if key:
                factors[key] = str(fd.get("hash", ""))[:8]
        gn = ((bt.get("checks") or {}).get("gross_net_attribution") or {})
        data = mf.get("data") or {}
        checks = mf.get("checks") or {}
        out.append({
            "run_id": mf.get("run_id") or d.name,
            "started_at": (mf.get("started_at") or "")[:19].replace("T", " "),
            "duration_seconds": mf.get("duration_seconds"),
            "environment": mf.get("environment") or {},
            "raw_manifest_sha256": data.get("raw_manifest_sha256"),
            "universe_size": data.get("universe_size"),
            "n_trading_days": data.get("n_trading_days"),
            "n_code_files": len(mf.get("code_version") or {}),
            "factors": factors,
            "all_pass": checks.get("base_backtest_all_pass"),
            "annualized_return": net.get("annualized_return"),
            "sharpe": net.get("annualized_sharpe"),
            "max_drawdown": net.get("max_drawdown"),
            "cumulative_return": net.get("cumulative_return"),
            "turnover_sum": (metrics.get("turnover") or {}).get("sum"),
            "total_cost": (metrics.get("cost") or {}).get("total_cost_currency"),
            "n_trades": (metrics.get("portfolio") or {}).get("n_trades"),
            "gross_annualized": (metrics.get("gross") or {}).get("annualized_return"),
            "convention": gn.get("convention"),
            "missing_close_adj_cells": agg.get("missing_close_adj_cells"),
            "suspended_cells_listed": agg.get("suspended_cells_listed"),
            "mtime": (d / "manifest.json").stat().st_mtime,
        })
    out.sort(key=lambda r: r["mtime"])
    return out


# ---------------- 因子库 ----------------
def _save_values(factor_id: int, values: pd.DataFrame) -> str:
    VALUES_DIR.mkdir(parents=True, exist_ok=True)
    p = VALUES_DIR / f"{factor_id}.parquet"
    values.to_parquet(p)
    return str(p)


def _write_values_tmp(factor_id: int, values: pd.DataFrame) -> Path:
    """写临时文件（不触碰正式路径），由 _promote_values_tmp 在 DB 提交成功后原子替换。

    修复历史 bug：refresh_factor 曾先覆盖正式 parquet 再算诊断/提交，
    中途异常会留下「新值文件 + 旧元数据」，且旧值已不可恢复。
    """
    VALUES_DIR.mkdir(parents=True, exist_ok=True)
    p = VALUES_DIR / f"{factor_id}.parquet.tmp"
    values.to_parquet(p)
    return p


def _promote_values_tmp(tmp: Path, final: str | Path) -> None:
    """DB 提交成功后调用：原子替换正式文件（同盘 rename）。失败时临时文件残留不影响旧数据。"""
    if tmp is None:
        return
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp.replace(final)


def _values_path_of(factor_id: int) -> str:
    return str(VALUES_DIR / f"{factor_id}.parquet")


def register_factor(*, name: str, hash_code: str, source: str, expression: str,
                    hypothesis: str, direction: str, missing_policy: str,
                    failure_modes: list[str], operator: str, params: dict,
                    preprocess: list[str], values: pd.DataFrame,
                    algo_task_id: str | None = None, notes: str = "",
                    pool_id: int | None = None, universe_note: str = "",
                    code: str = "", hidden: bool = False, session=None) -> tuple[Factor, bool]:
    """登记因子（幂等：hash_code 已存在则返回既有记录）。评估指标即时计算。"""
    own = session is None
    s = session or get_session()
    try:
        exists = s.query(Factor).filter_by(hash_code=hash_code).first()
        if exists:
            return exists, False
        f = Factor(name=name, hash_code=hash_code, source=source, expression=expression,
                   hypothesis=hypothesis, direction=direction, missing_policy=missing_policy,
                   failure_modes=";".join(failure_modes), operator=operator, params=params,
                   preprocess=preprocess, algo_task_id=algo_task_id, notes=notes,
                   pool_id=pool_id, universe_note=universe_note, code=code, hidden=hidden)
        # 因子值的覆盖区间（用于"策略所需历史是否够用"的校验）
        if len(values.index):
            f.value_start = pd.Timestamp(values.index.min())
            f.value_end = pd.Timestamp(values.index.max())
        s.add(f)
        # 元数据、因子值文件、诊断指标必须原子：原实现先 commit 元数据再写 parquet，
        # 写盘失败会留下 values_path 为空、读取即 FileNotFoundError 的"僵尸因子"。
        values_path = None
        try:
            s.flush()                              # 取自增 id，但尚未提交
            values_path = _save_values(f.id, values)
            f.values_path = values_path
            _compute_and_store_metrics(f, values, s)
            s.commit()
        except Exception:
            s.rollback()
            if values_path:
                Path(values_path).unlink(missing_ok=True)
            raise
        return f, True
    finally:
        if own:
            s.close()


def _apply_pool(values: pd.DataFrame, pool_id: int | None, mk) -> tuple[pd.DataFrame, str]:
    """把因子面板限制到某个股票池（池外置 NaN），使截面运算与诊断只在该池内进行。

    这一步很关键：winsorize / zscore / IC / 分层都是【截面】运算，
    换池 = 换截面 = 因子含义可能大幅变化。因此换池必须显式确认。
    """
    mask, note = pool_mask_for(pool_id, mk)
    if mask is None:
        return values, note
    aligned = mask.reindex(index=values.index, columns=values.columns).fillna(False)
    return values.where(aligned), note


def _processed_values(f: Factor, values: pd.DataFrame | None = None) -> pd.DataFrame:
    """因子值 -> 限定股票池 -> 按该因子自己的预处理参数处理。

    诊断、分层回测、信号合成都必须走这里，保证「页面写的口径」=「实际算的口径」。
    （历史 bug：`_compute_and_store_metrics` 直接对原始值算 IC/分层，
    而页面口径表却按 `preprocess` 声明了 winsorize+zscore —— 两者不一致。）
    """
    if values is None:
        values = pd.read_parquet(f.values_path)
    scoped, _ = _apply_pool(values, f.pool_id, market())
    return apply_preprocess_spec(scoped, f.preprocess_spec)


def _diag_cfg_for(f: Factor) -> dict:
    """把因子的 spec 翻成 run_diagnostics 需要的评估参数。

    评估口径（IC 方法 / 持有期 / 分组数）与预处理参数**存在同一份 spec 里**，
    所以"页面上写的口径"就是"实际算的口径"，不会再出现两处配置不一致。
    """
    from ..factors.preprocess import normalize_spec, parse_horizons

    sp = normalize_spec(f.preprocess_spec)
    return {"horizon_days": parse_horizons(sp["horizons"]),
            "min_cross_section_samples": int(sp["min_n"]),
            "quantile_groups": int(sp["quantile"]),
            "ic_method": str(sp["ic_method"])}


def _compute_and_store_metrics(f: Factor, values: pd.DataFrame, s) -> None:
    """算因子诊断 + 分层净值（附属统计量），一起写进 FactorMetric。

    分层净值是**因子计算的附属统计量**：随因子计算自动产出，没有单独的"跑分层"入口。
    每个持有期 h 对应一组 K 层净值曲线（rebalance 间隔 = h）。
    """
    from ..config import load_config
    from ..factors.layers import layered_nav

    cfg = load_config()
    mk = market()
    basis = factor_basis(f)
    processed = _processed_values(f, values)
    diag_cfg = _diag_cfg_for(f)
    # 诊断的标签收益用该因子自己复权口径下的收盘价（收益率与口径无关，但保持同一口径）
    px_close = mk.price("close", basis)
    diag = run_diagnostics(processed, px_close, diag_cfg)

    pool_mask, _ = pool_mask_for(f.pool_id, mk)
    cost = cfg.backtest.get("cost", {})
    A = int(cfg.data["trading_days_per_year"])
    rf = float(cfg.data["risk_free_annual"])

    s.query(FactorMetric).filter_by(factor_id=f.id).delete()
    for h_str, r in diag.items():
        summ, icm = r["ic_summary"], r["quantile_summary"]
        payload = diagnostics_json({h_str: r})
        # 分层净值：附属统计量，失败不阻断因子登记（页面会显示"未产出"）
        try:
            payload[h_str]["layer_nav"] = layered_nav(
                processed, mk.basis_view(basis),
                groups=int(diag_cfg["quantile_groups"]), h=int(h_str),
                pool_mask=pool_mask,
                commission_buy=float(cost.get("commission_buy", 0.0)),
                commission_sell=float(cost.get("commission_sell", 0.0)),
                trading_days=A, rf_annual=rf)
        except Exception as e:  # noqa: BLE001
            payload[h_str]["layer_nav"] = {"error": f"{type(e).__name__}: {e}"}
        s.add(FactorMetric(
            factor_id=f.id, horizon=int(h_str),
            ic_mean=summ["ic"]["mean"], ic_std=summ["ic"]["std"],
            rank_ic_mean=summ["rank_ic"]["mean"], rank_ic_std=summ["rank_ic"]["std"],
            icir=summ["rank_ic"]["icir"], t_stat=summ["rank_ic"]["t_stat"],
            coverage_mean=summ["coverage_mean"], n_valid_mean=summ["n_valid_mean"],
            spread_mean=icm["spread_mean"], monotonicity=icm["monotonicity_spearman"],
            summary_json=payload,
        ))
    # 不在此处 commit：由 register_factor 统一提交，保证与因子行同一事务


def factor_values(factor_id: int) -> pd.DataFrame:
    s = get_session()
    try:
        f = s.get(Factor, factor_id)
        return pd.read_parquet(f.values_path)
    finally:
        s.close()


def builtin_card_for(f: Factor) -> FactorCard:
    return FactorCard(key=f.name, name=f.name, hypothesis=f.hypothesis or "",
                      formula=f.expression, fields=["close"], window=int(f.params.get("window", 20)),
                      direction=f.direction or "", missing_policy=f.missing_policy or "",
                      failure_modes=f.failure_modes.split(";") if f.failure_modes else [],
                      operator=f.operator, params=dict(f.params), preprocess=list(f.preprocess or []))


def factor_basis(f: Factor) -> str:
    """该因子的复权口径（hfq / qfq）。"""
    return str(f.preprocess_spec.get("adjust") or "hfq")


def recompute_values(f: Factor) -> pd.DataFrame:
    """按该因子的复权口径重算因子原始值。

    复权口径决定价格面板 → 决定因子值本身（用到价格绝对水平的因子会不同），
    因此这里必须按 factor_basis(f) 取 fields，而不是固定用后复权。
    """
    mk = market()
    fields = mk.fields_for(factor_basis(f))
    if f.operator == "user":
        from ..factors.user_code import compute_user_factor
        return compute_user_factor(f.code or "", fields)
    if f.operator == "expression":
        from ..lab.gp_engine import eval_expr
        tree = f.params.get("expr_json")
        with np.errstate(all="ignore"):
            v = eval_expr(tuple(tree), fields)
            if not isinstance(v, pd.DataFrame):
                v = pd.DataFrame(v, index=mk.dates, columns=mk.codes)
            return v.replace([np.inf, -np.inf], np.nan)
    return compute_factor(builtin_card_for(f), fields)


def register_user_factor(*, name: str, code: str, description: str = "",
                         hypothesis: str = "", direction: str = "",
                         missing_policy: str = "", failure_modes: list[str] | None = None,
                         preprocess: list[str] | None = None,
                         adjust: str = "hfq") -> dict:
    """登记用户手写因子：在受限沙箱内执行代码、算值、诊断并入库（幂等）。

    `adjust` 决定用哪个复权口径的价格面板算因子值（hfq 后复权 / qfq 前复权）。
    """
    from ..factors.operators import FactorCard, factor_hash
    from ..factors.user_code import compute_user_factor

    mk = market()
    spec = {**DEFAULT_SPEC, **(normalize_spec(preprocess) if preprocess else {}),
            "adjust": adjust}
    values = compute_user_factor(code, mk.fields_for(adjust))   # 失败会抛 FactorCodeError
    fm = list(failure_modes or ["用户手写代码，未经外部验证；请对照 valid/test 复核"])
    card = FactorCard(
        key=name, name=name, hypothesis=hypothesis or "用户手写因子（未填写研究假设）",
        formula=code.strip(), fields=["close", "open", "high", "low", "volume"],
        window=0, direction=direction or "分数越高，预期收益越高",
        missing_policy=missing_policy or "缺失记 NaN，不填零；截面使用处显式跳过",
        failure_modes=fm, operator="user", params={},
        preprocess=normalize_spec(spec))
    h = factor_hash(card, values)
    f, created = register_factor(
        name=name, hash_code=h, source="user", expression=code.strip(),
        hypothesis=card.hypothesis, direction=card.direction,
        missing_policy=card.missing_policy, failure_modes=fm,
        operator="user", params={}, preprocess=normalize_spec(spec),
        values=values, code=code, notes=description or "用户手写因子",
        universe_note="用户提交：在当前全部可用资产上计算")
    return {"id": f.id, "created": bool(created), "shape": list(values.shape),
            "adjust": adjust}


def delete_factors(factor_ids: list[int], *, force: bool = False) -> dict:
    """删除因子（含诊断、因子值文件、候选引用）。

    引用保护：若某因子已被信号（策略）用作分量，默认**拒绝删除**并列出是哪些信号 ——
    静默删掉会让信号的分量指向不存在的因子，之后回测报一个莫名其妙的错。
    确需删除时传 force=True（会连带删除这些信号及其回测记录，并如实报告）。
    """
    s = get_session()
    deleted, blocked, cascaded = [], [], []
    stale_paths: list[str | None] = []
    try:
        for fid in factor_ids:
            fid = int(fid)
            f = s.get(Factor, fid)
            if f is None:
                blocked.append({"id": fid, "reason": "因子不存在"})
                continue
            users = [sig for sig in s.query(Signal).all()
                     if any(int(c.get("factor_id", -1)) == fid for c in (sig.components or []))]
            if users and not force:
                blocked.append({"id": fid, "name": f.name,
                                "reason": "被信号引用：" + "、".join(x.name for x in users)})
                continue
            cascaded_here = []
            for sig in users:                      # force：连带删除引用它的信号
                cascaded_here.append(sig.name)
                for b in s.query(BacktestRun).filter_by(signal_id=sig.id).all():
                    RESULTS_DIR.joinpath(f"bt{b.id}.json").unlink(missing_ok=True)
                s.query(BacktestRun).filter_by(signal_id=sig.id).delete()
                cascaded.append({"id": sig.id, "name": sig.name, "factor_id": fid})
                s.delete(sig)
            s.query(FactorMetric).filter_by(factor_id=fid).delete()
            for c in s.query(AlgoCandidate).filter_by(factor_id=fid).all():
                c.adopted, c.factor_id = False, None   # 候选回到"未采纳"，可再次采纳
            p, name = f.values_path, f.name
            stale_paths.append(p)
            s.delete(f)
            deleted.append({"id": fid, "name": name, "cascaded_signals": cascaded_here})
        s.commit()
        # 提交成功后才删值文件：失败回滚时 DB 行与文件保持一致
        for p in stale_paths:
            if p:
                Path(p).unlink(missing_ok=True)
    finally:
        s.close()
    return {"n_requested": len(factor_ids), "n_deleted": len(deleted),
            "n_blocked": len(blocked), "deleted": deleted, "blocked": blocked,
            "n_cascaded_signals": len(cascaded), "cascaded_signals": cascaded}


def factor_usage(factor_ids: list[int]) -> list[dict]:
    """预览：这些因子分别被哪些信号引用（删除前给用户看清楚）。"""
    s = get_session()
    try:
        out = []
        for fid in factor_ids:
            f = s.get(Factor, int(fid))
            if f is None:
                continue
            users = [sig.name for sig in s.query(Signal).all()
                     if any(int(c.get("factor_id", -1)) == int(fid)
                            for c in (sig.components or []))]
            out.append({"id": int(fid), "name": f.name, "signals": users})
        return out
    finally:
        s.close()


# ---------------- 轻量后台任务（信号/回测等秒~十秒级计算出请求路径） ----------------
def start_simple_job(kind: str, payload: dict, fn) -> str:
    """把一个同步计算挪到后台线程执行，立即返回 job_id。

    与 BatchJob 共用表与进度页；fn() 的返回值（实体 id 等）存 result_json["result"]，
    完成页据此跳转到结果实体页。适用秒级计算；分钟级以上的挖掘任务走
    task_runner 子进程（见 _spawn_runner）。
    """
    job_id = str(uuid.uuid4())[:8]
    s = get_session()
    try:
        s.add(BatchJob(id=job_id, kind=kind, status="PENDING", total=1,
                       params=payload, stage="排队中"))
        s.commit()
    finally:
        s.close()

    def _run():
        ss = get_session()
        try:
            j = ss.get(BatchJob, job_id)
            j.status, j.stage = "RUNNING", "执行中"
            ss.commit()
        finally:
            ss.close()
        try:
            r = fn()
            ss = get_session()
            try:
                j = ss.get(BatchJob, job_id)
                j.status, j.done = "SUCCESS", 1
                j.result_json = {"result": r}
                j.finished_at = now()
                ss.commit()
            finally:
                ss.close()
        except Exception as e:  # noqa: BLE001 —— 失败信息进任务行，页面展示
            ss = get_session()
            try:
                j = ss.get(BatchJob, job_id)
                j.status, j.message = "FAILED", f"{type(e).__name__}: {e}"
                j.finished_at = now()
                ss.commit()
            finally:
                ss.close()

    threading.Thread(target=_run, name=f"job-{job_id}", daemon=True).start()
    return job_id


# ---------------- 批量后台任务 ----------------
def start_batch_job(kind: str, ids: list[int], **params) -> str:
    """建批量任务并起后台线程，立即返回 job_id（HTTP 请求不等待）。"""
    job_id = str(uuid.uuid4())[:8]
    s = get_session()
    try:
        s.add(BatchJob(id=job_id, kind=kind, status="PENDING", total=len(ids),
                       params={**params, "ids": [int(i) for i in ids]},
                       stage="排队中"))
        s.commit()
    finally:
        s.close()
    threading.Thread(target=_run_batch_job, args=(job_id, kind, [int(i) for i in ids], params),
                     daemon=True).start()
    return job_id


def _run_batch_job(job_id: str, kind: str, ids: list[int], params: dict) -> None:
    def bump(done: int, stage: str):
        ss = get_session()
        try:
            j = ss.get(BatchJob, job_id)
            j.status, j.done, j.stage = "RUNNING", int(done), stage
            ss.commit()
        finally:
            ss.close()

    bump(0, "开始")
    try:
        if kind == "refresh":
            res = batch_refresh_factors(
                ids, pool_id=params.get("pool_id"),
                confirm_pool_change=bool(params.get("confirm_pool_change")),
                spec=None, start=params.get("start"), end=params.get("end"),
                progress=bump)
        elif kind == "delete":
            res = delete_factors(ids, force=bool(params.get("force")))
        else:
            raise ValueError(f"未知批量任务类型: {kind}")
        ss = get_session()
        try:
            j = ss.get(BatchJob, job_id)
            j.status, j.done, j.stage = "SUCCESS", j.total, "完成"
            j.result_json = res
            j.finished_at = now()
            ss.commit()
        finally:
            ss.close()
    except Exception as e:  # noqa: BLE001 —— 失败落库，页面如实显示
        ss = get_session()
        try:
            j = ss.get(BatchJob, job_id)
            j.status, j.stage = "FAILED", "失败"
            j.message = f"{type(e).__name__}: {e}"
            j.result_json = {"n_requested": len(ids), "n_ok": 0, "n_failed": len(ids),
                             "ok": [], "errors": [{"id": i, "error": str(e)} for i in ids],
                             "n_deleted": 0, "n_blocked": len(ids), "deleted": [],
                             "blocked": [{"id": i, "reason": str(e)} for i in ids]}
            j.finished_at = now()
            ss.commit()
        finally:
            ss.close()


def get_batch_job(job_id: str) -> dict | None:
    s = get_session()
    try:
        j = s.get(BatchJob, job_id)
        return j.to_dict() if j else None
    finally:
        s.close()


def backfill_signal_diagnostics() -> int:
    """给缺少「分层净值」附属统计量的历史信号补算诊断（幂等，供后台线程调用）。

    分层净值是 diagnose_signal 的产物；本次改动之前建的信号没有这一段。
    启动时后台补一次，避免它们的策略页显示"未产出"却没有任何补救入口。
    """
    s = get_session()
    try:
        todo = []
        for sig in s.query(Signal).all():
            dj = sig.diagnostics_json or {}
            if not dj or dj.get("error"):
                continue
            if any((dj.get(h) or {}).get("layer_nav") for h in dj):
                continue
            todo.append(sig.id)
    finally:
        s.close()
    n = 0
    for sid in todo:
        try:
            diag = diagnose_signal(sid)
            s2 = get_session()
            try:
                sg = s2.get(Signal, sid)
                sg.diagnostics_json = diag
                s2.commit()
            finally:
                s2.close()
            n += 1
        except Exception:  # noqa: BLE001 —— 单个信号补算失败不影响其余
            continue
    return n


def last_task_params() -> dict:
    """最近一次挖掘任务的参数，用于把新建任务表单**预填成上次的输入**（记忆化）。

    拆成两块，因为它们的"上次"含义不同：
      common    —— 因子计算参数 / 计算范围 / 三段切分：取**最近一次任务**（不分引擎），
                   这是用户每次都要重填的部分；
      by_engine —— 算法参数：按引擎各取该引擎最近一次（换引擎时沿用该引擎自己的设置）。
    取不到就返回空，页面回落到 schema 默认值，不会造出假值。
    """
    s = get_session()
    try:
        tasks = s.query(AlgoTask).order_by(AlgoTask.created_at.desc()).all()
    finally:
        s.close()
    if not tasks:
        return {"common": {}, "by_engine": {}}
    p0 = tasks[0].params or {}
    common_keys = ("spec", "pool_id", "start_date", "end_date", "split_mode")
    common = {k: p0.get(k) for k in common_keys if p0.get(k) is not None}
    # 三段边界入库时被收进 split_bounds（不是顶层键），这里摊平回去供表单预填
    bounds = p0.get("split_bounds") or {}
    for k in ("train_start", "train_end", "valid_start", "valid_end",
              "test_start", "test_end"):
        if bounds.get(k):
            common[k] = bounds[k]
    skip = set(common_keys) | {"split_bounds", "spec"}
    by_engine: dict[str, dict] = {}
    for t in tasks:                                    # 已按时间倒序，首次见到的即最新
        if t.engine in by_engine:
            continue
        p = t.params or {}
        by_engine[t.engine] = {k: v for k, v in p.items() if k not in skip}
    return {"common": common, "by_engine": by_engine, "from_task": tasks[0].id}


def last_backtest_config() -> dict:
    """最近一次回测的配置，用于预填新建回测表单（记忆化）。没有则返回空。"""
    s = get_session()
    try:
        r = s.query(BacktestRun).order_by(BacktestRun.id.desc()).first()
        if r is None:
            return {}
        d = r.to_dict()
        cost = d.get("cost_json") or {}
        return {
            "signal_id": d.get("signal_id"),
            "pool_id": d.get("pool_id"),
            "top_n": d.get("top_n"),
            "weighting": d.get("weighting"),
            "rebalance_freq": d.get("rebalance_freq"),
            "initial_cash": d.get("initial_cash"),
            "start_date": (d.get("start_date") or "")[:10] or None,
            "end_date": (d.get("end_date") or "")[:10] or None,
            "commission_buy": cost.get("commission_buy"),
            "commission_sell": cost.get("commission_sell"),
            "slippage": cost.get("slippage"),
            "from_run_id": d.get("id"),
        }
    finally:
        s.close()


def seed_from_latest_run() -> int:
    """启动时把最新 pipeline run 的因子导入因子库（幂等）。"""
    run = latest_run_dir()
    if run is None:
        return 0
    n = 0
    for jf in sorted((run / "factors").glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        card_d, h = data["card"], data["hash"]
        s = get_session()
        try:
            if s.query(Factor).filter_by(hash_code=h).first():
                continue
        finally:
            s.close()
        card = FactorCard(key=card_d["key"], name=card_d["name"], hypothesis=card_d["hypothesis"],
                          formula=card_d["formula"], fields=card_d["fields"], window=card_d["window"],
                          direction=card_d["direction"], missing_policy=card_d["missing_policy"],
                          failure_modes=card_d["failure_modes"], operator=card_d["operator"],
                          params=card_d["params"], preprocess=card_d["preprocess"])
        values = compute_factor(card, market().fields)
        _, created = register_factor(
            name=card_d["name"], hash_code=h, source="builtin", expression=card_d["formula"],
            hypothesis=card_d["hypothesis"], direction=card_d["direction"],
            missing_policy=card_d["missing_policy"], failure_modes=card_d["failure_modes"],
            operator=card_d["operator"], params=card_d["params"], preprocess=card_d["preprocess"],
            values=values, notes=f"seeded from run {run.name}",
            universe_note="全部可用资产（pipeline 在全样本上计算因子值）")
        n += int(created)
    return n


# ---------------- 股票池 ----------------
DEFAULT_POOLS = [
    ("全部可用资产", "不按指数过滤，使用本地行情覆盖的全部资产", "all", [], []),
    ("沪深300", "动态 PIT 成分（逐日）", "index", ["csi300"], []),
    ("中证500", "动态 PIT 成分（逐日）", "index", ["csi500"], []),
    ("中证1000", "动态 PIT 成分（逐日）", "index", ["csi1000"], []),
    ("300+500+1000", "三指数 PIT 成分并集（逐日）", "index",
     ["csi300", "csi500", "csi1000"], []),
]


def ensure_default_pools() -> int:
    """确保内置股票池存在（幂等）。返回新建数量。"""
    s = get_session()
    made = 0
    try:
        for name, desc, kind, idxs, codes in DEFAULT_POOLS:
            if s.query(StockPool).filter_by(name=name).first():
                continue
            s.add(StockPool(name=name, description=desc, kind=kind, indices=idxs, codes=codes))
            made += 1
        s.commit()
        return made
    finally:
        s.close()


def list_pools() -> list[dict]:
    s = get_session()
    try:
        return [p.to_dict() for p in s.query(StockPool).order_by(StockPool.id).all()]
    finally:
        s.close()


def get_pool(pool_id: int | None):
    if pool_id is None:
        return None
    s = get_session()
    try:
        p = s.get(StockPool, pool_id)
        return p.to_dict() if p else None
    finally:
        s.close()


def norm_pool_id(pool_id) -> int | None:
    """把股票池 id 归一：`kind='all'` 的池等价于「不过滤」，统一记成 None。

    不这么做的话，None（不过滤）与 id=1（「全部可用资产」池）是同一件事却有两个表示，
    页面显示"全部可用资产"、库里存 None，一保存就被判成"换池"并弹警告 —— 假变更。
    """
    if pool_id in (None, 0, "", "0"):
        return None
    try:
        pid = int(pool_id)
    except (TypeError, ValueError):
        # 兼容内置池的字符串键（如 "index_csi300"）：按 DEFAULT_POOLS 的 kind/indices 找回数字 id。
        # 历史行为：直接 int() 抛 invalid literal，任务 FAILED。
        key = str(pool_id)
        probe = key[6:] if key.startswith("index_") else key  # index_csi300 -> csi300
        s = get_session()
        try:
            for p_ in s.query(StockPool).all():
                d = p_.to_dict()
                indices = d.get("indices") or []
                if key == d.get("kind") or probe in indices or key in indices:
                    return int(p_.id)
        finally:
            s.close()
        raise ValueError(f"未知股票池: {pool_id}")
    p = get_pool(pid)
    if p is not None and p.get("kind") == "all":
        return None
    return int(pool_id)


def pool_mask_for(pool_id: int | None, market) -> tuple:
    """把 pool_id 解析成与 market 对齐的逐日掩码；pool_id=None 视为不过滤。"""
    from ..data.universe import resolve_pool
    p = get_pool(pool_id)
    if p is None or p.get("kind") == "all":
        return None, "全部可用资产（不按指数过滤）"
    return resolve_pool(market, p)


# ---------------- 股票池行情（券商风格页面用） ----------------
_pool_cache: dict[str, Any] = {}


def _pool_bundle(pool_id: int) -> dict:
    """股票池的逐日掩码 + 池内等权指数（缓存，页面反复访问不做重复计算）。"""
    key = str(pool_id)
    if key in _pool_cache:
        return _pool_cache[key]
    mk = market()
    mask, note = pool_mask_for(pool_id, mk)
    if mask is None:
        mask = pd.DataFrame(True, index=mk.dates, columns=mk.codes)
    mask = mask.reindex(index=mk.dates, columns=mk.codes).fillna(False).astype(bool)

    rets = mk.daily_returns()
    m = mask.to_numpy(dtype=bool)
    r = rets.to_numpy(dtype="float64")
    rr = np.where(m & np.isfinite(r), r, np.nan)
    n_members = np.isfinite(rr).sum(axis=1)
    # 全 NaN 的行会让 nanmean 发 RuntimeWarning（不是浮点错误，errstate 挡不住）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(rr, axis=1)
    daily = pd.Series(np.where(n_members > 0, means, np.nan), index=mk.dates)
    nav = (1.0 + daily.fillna(0.0)).cumprod()
    nav[daily.isna()] = np.nan
    bundle = {"mask": mask, "note": note, "counts": pd.Series(n_members, index=mk.dates),
              "daily_ret": daily, "nav": nav}
    _pool_cache[key] = bundle
    return bundle


def pool_overview(pool_id: int) -> dict | None:
    """股票池概览：成分数走势 + 池内等权指数，用于页面主图。"""
    p = get_pool(pool_id)
    if p is None:
        return None
    b = _pool_bundle(pool_id)
    mk = market()
    bench = mk.benchmark_close.reindex(mk.dates).ffill()
    bench_nav = bench / bench.dropna().iloc[0] if bench.notna().any() else bench
    nav = b["nav"]
    base = nav.dropna()
    nav_n = nav / base.iloc[0] if len(base) else nav

    def _ser(s: pd.Series, nd=6) -> dict:
        return {"index": [str(d.date()) for d in s.index],
                "values": [None if not np.isfinite(v) else round(float(v), nd) for v in s]}

    counts = b["counts"]
    return {
        "pool": p,
        "note": b["note"],
        "n_codes_ever": int((b["mask"].sum(axis=0) > 0).sum()),
        "n_codes_today": int(counts.iloc[-1]) if len(counts) else 0,
        "avg_members": float(counts[counts > 0].mean()) if (counts > 0).any() else 0.0,
        "nav": _ser(nav_n),
        "benchmark": _ser(bench_nav),
        "counts": {"index": [str(d.date()) for d in counts.index],
                   "values": [int(v) for v in counts]},
        "daily_ret": _ser(b["daily_ret"]),
        "period": [str(mk.dates.min().date()), str(mk.dates.max().date())],
        # 池内等权、逐日再平衡、未扣费 —— 明说口径，避免被当成可交易净值
        "convention": "池内等权、逐日再平衡、未扣交易成本；仅用于观察池子整体走势",
    }


@lru_cache(maxsize=1)
def _listing_dates() -> dict[str, str]:
    """代码 -> 上市日期（data/reference/instruments.csv）。"""
    p = ROOT / "data" / "reference" / "instruments.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype={"instrument": str})
    out = {}
    for it, d in zip(df["instrument"], df["list_date"]):
        s = str(it).strip().upper()
        if s[:2] in ("SH", "SZ", "BJ"):
            s = s[2:]
        out[s.zfill(6)] = str(d)[:10]
    return out


def _listing_date(code: str) -> str | None:
    return _listing_dates().get(str(code).zfill(6)) or None


def pool_members(pool_id: int, date=None, *, sort: str = "total_mv", limit: int = 0,
                 order: str = "desc") -> dict | None:
    """某个交易日的池内成分快照（代码/名称/涨跌/换手/估值/市值/行业）。

    日期取「不晚于 date 的最近一个有数据的交易日」，并在返回里写明实际用的日期。
    """
    from ..data import fundamentals as fund
    from ..data.industry import industry_labels_fast

    p = get_pool(pool_id)
    if p is None:
        return None
    mk = market()
    b = _pool_bundle(pool_id)
    mask = b["mask"]
    avail = mask.index[mask.sum(axis=1) > 0]
    if not len(avail):
        return {"pool": p, "as_of": None, "rows": [], "n_members": 0}
    if date:
        d = pd.Timestamp(date)
        usable = avail[avail <= d]
        use = usable[-1] if len(usable) else avail[0]
    else:
        use = avail[-1]

    members = mask.columns[mask.loc[use].to_numpy(dtype=bool)]
    codes = [str(c) for c in members]
    rets = mk.daily_returns().loc[use].reindex(codes)
    close_adj = mk.close_adj.loc[use].reindex(codes)
    close_raw = mk.close_raw.loc[use].reindex(codes)
    amount = mk.amount.loc[use].reindex(codes)
    vol = mk.volume.loc[use].reindex(codes)
    prev = mk.dates[mk.dates < use]
    rets_5 = rets_20 = None
    if len(prev):
        p5 = prev[-5] if len(prev) >= 5 else prev[0]
        p20 = prev[-20] if len(prev) >= 20 else prev[0]
        rets_5 = (mk.close_adj.loc[use] / mk.close_adj.loc[p5] - 1).reindex(codes)
        rets_20 = (mk.close_adj.loc[use] / mk.close_adj.loc[p20] - 1).reindex(codes)
    snap = fund.snapshot(use, codes) if fund.available() else {}
    names = fund.names()
    try:
        # 复用中性化那张缓存表（(level,日期,代码) 级别缓存），避免按日构造
        ind_all = industry_labels_fast(mk.dates, mk.codes, "sw1")
        ind = ind_all.loc[use] if use in ind_all.index else pd.Series(dtype=object)
    except Exception:  # noqa: BLE001
        ind = pd.Series(dtype=object)

    def _f(s, c):
        if s is None:
            return None
        v = s.get(c)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    rows = []
    for c in codes:
        dm = snap.get(c) or {}
        rows.append({
            "code": c, "name": names.get(c, ""),
            "industry": None if pd.isna(ind.get(c)) else str(ind.get(c)),
            "close_raw": _f(close_raw, c), "close_adj": _f(close_adj, c),
            "pct_chg": _f(rets, c), "ret_5d": _f(rets_5, c), "ret_20d": _f(rets_20, c),
            "amount": _f(amount, c), "volume": _f(vol, c),
            "turnover_rate": dm.get("turnover_rate"), "volume_ratio": dm.get("volume_ratio"),
            "pe_ttm": dm.get("pe_ttm"), "pb": dm.get("pb"), "ps_ttm": dm.get("ps_ttm"),
            "dv_ratio": dm.get("dv_ratio"),
            "total_mv": dm.get("total_mv"), "circ_mv": dm.get("circ_mv"),
            "total_share": dm.get("total_share"), "float_share": dm.get("float_share"),
            "free_share": dm.get("free_share"),
            "listing_date": _listing_date(c),
            "suspended": bool(mk.suspended.loc[use].reindex(codes).get(c, False)),
        })

    key = {"total_mv": "total_mv", "circ_mv": "circ_mv", "pct_chg": "pct_chg",
           "turnover_rate": "turnover_rate", "pe_ttm": "pe_ttm", "pb": "pb",
           "amount": "amount", "code": "code", "ret_5d": "ret_5d", "ret_20d": "ret_20d",
           "name": "name", "industry": "industry"}.get(sort, "total_mv")
    rev = (order != "asc")
    num_rows = [r for r in rows if isinstance(r.get(key), (int, float))]
    str_rows = [r for r in rows if isinstance(r.get(key), str)]
    none_rows = [r for r in rows if r.get(key) is None]
    num_rows.sort(key=lambda r: float(r[key]), reverse=rev)
    str_rows.sort(key=lambda r: r[key], reverse=rev)
    rows = num_rows + str_rows + none_rows
    total = len(rows)
    if limit:
        rows = rows[: int(limit)]

    # 当日涨跌家数 / 分布（页面做红绿分布）
    chg = [r["pct_chg"] for r in rows if r["pct_chg"] is not None]
    breadth = {
        "n_up": sum(1 for v in chg if v > 0), "n_down": sum(1 for v in chg if v < 0),
        "n_flat": sum(1 for v in chg if v == 0), "n_total": len(chg),
        "limit_up": sum(1 for v in chg if v >= 0.095),
        "limit_down": sum(1 for v in chg if v <= -0.095),
        "median": float(np.median(chg)) if chg else None,
    }
    return {"pool": p, "as_of": str(use.date()), "rows": rows, "n_members": total,
            "shown": len(rows), "breadth": breadth,
            "has_fundamentals": bool(snap),
            "counts": int(b["counts"].loc[use]) if use in b["counts"].index else len(codes)}


def pool_date_bounds(pool_id: int) -> tuple[str | None, str | None]:
    """池内出现过的日期范围（页面日期选择器用）。"""
    p = get_pool(pool_id)
    if p is None:
        return None, None
    mk = market()
    b = _pool_bundle(pool_id)
    avail = b["mask"].index[b["mask"].sum(axis=1) > 0]
    if not len(avail):
        return None, None
    return str(avail.min().date()), str(avail.max().date())


def list_pools_with_stats() -> list[dict]:
    """股票池列表 + 每个池的成分数（用于 /pools 列表页）。"""
    out = []
    for p in list_pools():
        try:
            b = _pool_bundle(int(p["id"]))
            counts = b["counts"]
            out.append({**p,
                        "n_codes_ever": int((b["mask"].sum(axis=0) > 0).sum()),
                        "n_codes_today": int(counts.iloc[-1]) if len(counts) else 0,
                        "avg_members": float(counts[counts > 0].mean()) if (counts > 0).any() else 0.0,
                        "note": b["note"]})
        except Exception as e:  # noqa: BLE001
            out.append({**p, "n_codes_ever": 0, "n_codes_today": 0, "avg_members": 0.0,
                        "note": f"解析失败: {type(e).__name__}: {e}"})
    return out


def factor_data_policy(f: Factor) -> dict:
    """因子的数据处理口径（复权 / 缺失 / 去重 / 非法值 / 去极值 / 标准化 / 中性化）。

    全部从实际生效的产物与配置读取，不写死文案 —— 口径变了页面跟着变。
    """
    from ..config import ROOT

    pol: dict = {}

    cm = ROOT / "data" / "clean" / "clean_manifest.json"
    if cm.exists():
        try:
            m = json.loads(cm.read_text(encoding="utf-8"))
            params = m.get("params") or {}
            pol["adjustment_formula"] = params.get("adjustment")
            pol["clean_period"] = params.get("period")
            pol["clean_source"] = (m.get("source_snapshot") or {}).get("source")
        except Exception:  # noqa: BLE001
            pass

    qr = ROOT / "data" / "clean" / "quality_report.json"
    if qr.exists():
        try:
            q = json.loads(qr.read_text(encoding="utf-8"))
            p = q.get("policy") or {}
            pol["duplicate_policy"] = p.get("duplicate_key")
            pol["illegal_policy"] = p.get("illegal")
            pol["missing_policy"] = p.get("missing")
            pol["adjustment_policy"] = p.get("adjustment")
            agg = q.get("aggregate") or {}
            pol["quality"] = {
                "missing_close_adj_cells": agg.get("missing_close_adj_cells"),
                "pre_listed_cells": agg.get("pre_listed_cells"),
                "suspended_cells_listed": agg.get("suspended_cells_listed"),
                "coverage_mean": agg.get("coverage_mean"),
            }
        except Exception:  # noqa: BLE001
            pass

    pre = list(f.preprocess or [])
    spec = f.preprocess_spec
    steps = spec_steps(spec)

    pol["preprocess_applied"] = pre
    pol["preprocess_spec"] = spec
    pol["preprocess_steps"] = steps
    pol["preprocess_schema"] = PREPROCESS_SCHEMA
    pol["neutralize_enabled"] = spec["neutralize_method"] != "none"
    pol["neutralize_method"] = spec["neutralize_method"]
    pol["industry_level"] = spec["neutralize_level"]
    pol["size_neutralize_needs"] = "total_mv"

    # 中性化要用到的数据是否真的在本地（有就说有，没有就明说）
    from ..data.industry import coverage_summary_fast as _ind_cov
    from ..data import fundamentals as _fund
    try:
        mk = market()
        ic = _ind_cov(mk.dates, mk.codes, spec["neutralize_level"])
        pol["industry_data"] = {
            "available": True, "level": spec["neutralize_level"],
            "coverage": ic["coverage"], "n_industries": ic["n_industries"],
            "n_codes_without_industry": ic["n_codes_without_industry"],
            "span": ic["span"], "n_days_before_data": ic["n_days_before_data"],
            "source": "data/reference/industry_intervals.csv（PIT 逐日）",
        }
    except Exception as e:  # noqa: BLE001
        pol["industry_data"] = {"available": False, "error": f"{type(e).__name__}: {e}"}
    pol["market_cap_data"] = {
        "available": _fund.available(),
        "note": _fund.coverage_note(),
        "source": "data/raw/daily_basic.parquet（Tushare daily_basic，逐日）",
    }

    pol["diagnosis"] = {
        "horizon_days": parse_horizons(spec["horizons"]),
        "min_cross_section_samples": spec["min_n"],
        "quantile_groups": spec["quantile"],
        "ic_method": spec["ic_method"],
        "adjust": spec["adjust"] or "后复权（清洗层唯一口径）",
        "alpha": spec["alpha"], "k": spec["k"],
        "note": "评估口径（IC 方法 / 持有期 h / 分组数 K）与预处理参数一起存在"
                "本因子自己的 spec 里，可在因子页直接修改并立即重算。",
    }
    pol["universe_note"] = f.universe_note or "未登记（早期导入的因子）"
    pol["value_range"] = (
        [str(f.value_start.date()), str(f.value_end.date())]
        if f.value_start and f.value_end else None)
    return pol


# ---------------- 信号（策略） ----------------
def _ic_weights(factor_ids: list[int]) -> dict[int, float]:
    """ic_weight 模型：按各因子 20 日 RankIC 均值（带符号）归一化加权。"""
    s = get_session()
    try:
        ms = {fid: s.query(FactorMetric).filter_by(factor_id=fid, horizon=20).first()
              for fid in factor_ids}
        raw = {fid: (m.rank_ic_mean or 0.0) for fid, m in ms.items()}
    finally:
        s.close()
    denom = sum(abs(v) for v in raw.values()) or 1.0
    return {fid: v / denom for fid, v in raw.items()}


def signal_component_weights(sig: Signal) -> dict[int, float]:
    """信号的分量权重。

    ic_weight 按 RankIC 反推；linear / tree 走各自的 walk-forward 拟合路径；
    其余（equal_weight / **weighted** / alphagen）按 components 里存的显式权重归一化 ——
    `weighted` 专门用于"挖掘任务的因子池 + 学习到的线性权重"这种组合，
    权重是算法产出的，不是等权，因此不能标成 equal_weight（那会让页面口径说谎）。
    """
    comps = list(sig.components or [])
    fids = [int(c["factor_id"]) for c in comps]
    if not fids:
        return {}
    if sig.model_type == "ic_weight":
        return _ic_weights(fids)
    w = {int(c["factor_id"]): float(c.get("weight", 1.0)) for c in comps}
    tot = sum(abs(v) for v in w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


# 走显式权重相加（而非 walk-forward 拟合）的模型类型
WEIGHTED_MODELS = ("equal_weight", "ic_weight", "weighted")


_decay_cache: dict[tuple, dict] = {}


def factor_decay(factor_id: int) -> dict:
    """IC 衰减曲线与半衰期（按需计算，进程内缓存；调仓频率的量化依据）。

    horizons 取 (1,2,3,5,10,15,20,40) bar/交易日；因子值与标签都按其存储频率口径。
    """
    hit = next((v for k, v in _decay_cache.items() if k[0] == factor_id), None)
    if hit is not None:
        return hit
    s = get_session()
    try:
        f = s.get(Factor, factor_id)
        if f is None:
            raise ValueError(f"因子 {factor_id} 不存在")
        freq = getattr(f, "frequency", None) or "1d"
    finally:
        s.close()
    values = factor_values(factor_id)
    close = close_5m() if freq == "5m" else market().close_adj
    from ..factors.decay import full_decay_report
    rep = full_decay_report(values, close, min_n=10)
    rep["frequency"] = freq
    _decay_cache[(factor_id, freq)] = rep
    return rep


def factor_correlation(factor_ids: list[int]) -> dict:
    """选中因子的池化相关矩阵 + 去冗余建议（组合前的"体检"）。

    相关性用原始因子值（zscore 等仿射变换不改变相关系数）；分数取每个因子
    最接近 20 日（没有则最大）的 RankIC 均值。
    """
    from ..factors.combine import pooled_corr, redundancy_prune

    ids = [int(x) for x in factor_ids]
    if len(ids) < 2:
        raise ValueError("至少选择 2 个因子")
    if len(ids) > 12:
        raise ValueError("一次最多比较 12 个因子")
    panels, meta = {}, []
    s = get_session()
    try:
        for fid in ids:
            f = s.get(Factor, fid)
            if f is None:
                continue
            ms = s.query(FactorMetric).filter_by(factor_id=fid).all()
            if ms:
                pick = min(ms, key=lambda m: (abs(m.horizon - 20), -m.horizon))
                ric = pick.rank_ic_mean
            else:
                ric = None
            meta.append({"id": fid, "name": f.name, "rank_ic": ric,
                         "horizon": (pick.horizon if ms else None)})
            panels[str(fid)] = factor_values(fid)
    finally:
        s.close()
    if len(panels) < 2:
        raise ValueError("有效因子不足 2 个（缺因子值）")
    corr = pooled_corr(panels).fillna(0.0)
    scores = {str(m["id"]): (m["rank_ic"] or 0.0) for m in meta}
    red = redundancy_prune(corr, scores, max_abs_corr=0.8)
    name_of = {str(m["id"]): m["name"] for m in meta}
    red = {"keep": [name_of.get(k, k) for k in red["keep"]],
           "drop": [{**d, "key": name_of.get(d["key"], d["key"]),
                     "vs": name_of.get(d["vs"], d["vs"])} for d in red["drop"]],
           "threshold": red["threshold"]}
    return {"factors": meta,
            "keys": list(corr.columns),
            "matrix": [[round(float(x), 4) if np.isfinite(x) else None
                        for x in row] for row in corr.to_numpy()],
            "redundancy": red}


def combination_report(signal_id: int, horizon: int | None = None) -> dict:
    """组合增益报告：组合信号 vs 各分量单因子，同口径 RankIC 对照 + 相关性摘要。"""
    from ..factors.combine import pooled_corr

    s = get_session()
    try:
        sig = s.get(Signal, signal_id)
        if sig is None:
            raise ValueError(f"信号 {signal_id} 不存在")
        h_default = int((sig.model_params or {}).get("label_horizon", 20))
        comps = [int(c["factor_id"]) for c in (sig.components or [])]
    finally:
        s.close()
    h = int(horizon or h_default)

    diag = (get_signal(signal_id) or {}).get("diagnostics_json") or {}
    if str(h) in diag:
        combined = (diag[str(h)].get("ic_summary") or {}).get("rank_ic", {}).get("mean")
    elif diag:
        k = sorted(diag, key=lambda x: abs(int(x) - h))[0]
        h = int(k)
        combined = (diag[k].get("ic_summary") or {}).get("rank_ic", {}).get("mean")
    else:
        combined = None

    singles, panels = {}, {}
    s = get_session()
    try:
        for fid in comps:
            f = s.get(Factor, fid)
            if f is None:
                continue
            ms = s.query(FactorMetric).filter_by(factor_id=fid).all()
            pick = None
            if ms:
                cand = [m for m in ms if m.horizon == h]
                pick = cand[0] if cand else min(ms, key=lambda m: abs(m.horizon - h))
            singles[fid] = {"factor_id": fid, "name": f.name,
                            "rank_ic": (pick.rank_ic_mean if pick else None),
                            "horizon": (pick.horizon if pick else None)}
            try:
                panels[fid] = factor_values(fid)
            except Exception:  # noqa: BLE001
                pass
    finally:
        s.close()

    corr_summary = None
    vals = [v for v in panels.values() if len(panels) >= 2]
    if len(vals) >= 2:
        c = pooled_corr(panels)
        cc = c.to_numpy(dtype="float64")
        off = cc[~np.eye(cc.shape[0], dtype=bool)]
        off = off[np.isfinite(off)]
        if off.size:
            corr_summary = {"min": round(float(off.min()), 4),
                            "max": round(float(off.max()), 4),
                            "mean_abs": round(float(np.abs(off).mean()), 4)}

    from ..factors.combine import combination_gain
    singles_named = {str(x["factor_id"]): x["rank_ic"] for x in singles.values()
                     if x["rank_ic"] is not None}
    gain = combination_gain(singles_named, combined)
    # 组合 IC 的 NW 修正 t：从存储的逐日 IC 序列重算（lag=h-1，重叠标签纪律）
    combined_t_nw = None
    series = ((diag.get(str(h)) or {}).get("ic_series") or {}).get("rank_ic")
    if series:
        from ..factors.diagnostics import newey_west_tstat
        combined_t_nw = newey_west_tstat(pd.Series(series, dtype="float64"),
                                         lag=max(1, h - 1))
        combined_t_nw = None if not np.isfinite(combined_t_nw) else round(combined_t_nw, 3)
    return {"signal_id": signal_id, "horizon": h,
            "combined_rank_ic": combined,
            "combined_rank_ic_t_nw": combined_t_nw,
            "components": list(singles.values()),
            "correlation": corr_summary,
            "gain": gain}


def _rolling_weighted_panel(sig, market, start=None, end=None) -> pd.DataFrame:
    """ic_weight_rolling / ic_meanvar：walk-forward 因子权重 + 逐行加权。

    各因子先按【自己的 spec + 池】得到 zscore 面板（与 weighted 路径同源），
    权重由 factors/combine 的滚动 ICIR / IC 均值-方差闭式解给出。
    """
    from ..factors.combine import (apply_weights, rolling_ic_meanvar_weights,
                                   rolling_icir_weights)

    mp = dict(sig.model_params or {})
    h = max(1, int(mp.get("label_horizon", 5)))
    window = max(20, int(mp.get("window", 252)))
    min_periods = max(10, int(mp.get("min_periods", 60)))
    ridge = float(mp.get("ridge", 0.5))

    panels = {}
    for c in (sig.components or []):
        fid = int(c["factor_id"])
        s2 = get_session()
        try:
            f = s2.get(Factor, fid)
            if f is None:
                raise ValueError(f"因子 {fid} 不存在")
            spec, f_pool = f.preprocess_spec, f.pool_id
        finally:
            s2.close()
        v = factor_values(fid)
        scoped, _ = _apply_pool(v, f_pool, market)
        panels[fid] = apply_preprocess_spec(scoped, spec)

    if sig.model_type == "ortho_ic_weight_rolling":
        # Schmidt 正交化：按分量顺序（越靠前越保留原始信息），每个因子只保留
        # 前面解释不掉的增量；随后对【正交残差】做滚动 ICIR 加权 —— 权重衡量
        # 的是"增量信息"的贡献，而不是重复计价的相关信息。
        from ..factors.combine import orthogonalize
        panels, order = orthogonalize(panels, list(panels.keys()))  # 分量顺序=信息优先级
    if sig.model_type in ("ic_meanvar", "ortho_ic_weight_rolling"):
        wmat = rolling_ic_meanvar_weights(panels, market.close_adj, h=h, window=window,
                                          min_periods=min_periods, ridge=ridge)
    else:
        wmat = rolling_icir_weights(panels, market.close_adj, h=h, window=window,
                                    min_periods=min_periods)
    panel = apply_weights(panels, wmat)
    lo = pd.Timestamp(start) if start is not None else pd.Timestamp(sig.start_date)
    hi = pd.Timestamp(end) if end is not None else pd.Timestamp(sig.end_date)
    return panel.loc[(panel.index >= lo) & (panel.index <= hi)]


def signal_panel(sig: Signal, market, start=None, end=None) -> pd.DataFrame:
    """由信号定义合成信号面板（date × code，分数越高越优）。

    equal_weight / ic_weight：各因子按【各自的预处理参数】处理后按权重求和。
      ⚠️ ic_weight 用全样本 RankIC 均值定权重 —— 存在轻微样本内偏误（权重看过
      全样本），只作基线；推荐用 ic_weight_rolling / ic_meanvar（walk-forward）。
    ic_weight_rolling：逐日 ICIR 滚动权重（t 日权重只用 t-h 之前的 IC，无前视）。
    ic_meanvar：IC 均值-方差凸组合 w ∝ (Σ+λI)⁻¹μ（同样 walk-forward）。
    linear / tree：walk-forward 拟合（见 model 层），训练窗口严格只用过去数据。
    """
    from ..config import load_config

    cfg = load_config()
    diag = dict(cfg.factors["diagnosis"])
    start = pd.Timestamp(start) if start is not None else pd.Timestamp(sig.start_date)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp(sig.end_date)

    if sig.model_type in WEIGHTED_MODELS:
        weights = signal_component_weights(sig)
        if not weights:
            raise ValueError("信号没有任何因子分量")
        zs = []
        for fid, w in weights.items():
            s2 = get_session()
            try:
                f = s2.get(Factor, fid)
                if f is None:
                    raise ValueError(f"因子 {fid} 不存在")
                # 用因子自己的 spec（在 session 内取出，避免脱离会话后属性访问）
                spec, f_pool = f.preprocess_spec, f.pool_id
            finally:
                s2.close()
            v = factor_values(fid)
            scoped, _ = _apply_pool(v, f_pool, market)
            zs.append(apply_preprocess_spec(scoped, spec) * w)
        panel = sum(zs)
        panel.columns.name = "code"
        return panel.loc[(panel.index >= start) & (panel.index <= end)]

    if sig.model_type in ("ic_weight_rolling", "ic_meanvar", "ortho_ic_weight_rolling"):
        return _rolling_weighted_panel(sig, market, start, end)

    if sig.model_type in ("linear", "tree"):
        from ..factors.model import fit_walk_forward
        return fit_walk_forward(sig, market, start, end)

    raise ValueError(f"未知模型类型: {sig.model_type}")


def _signal_pool_mask(sig, market):
    """信号的分层诊断域 = 各分量因子所属股票池的交集（逐日 AND）。

    Signal 本身不带池（池在回测时按次选择），但分量因子的截面口径是在各自池内
    定义的，因此信号的"定义域"是它们共同的支撑集。全部分量不限池时返回 (None, note)。
    """
    masks = []
    notes = []
    s = get_session()
    try:
        for c in (sig.components or []):
            f = s.get(Factor, int(c.get("factor_id", -1)))
            if f is None:
                continue
            m, note = pool_mask_for(f.pool_id, market)
            if m is not None:
                masks.append(m)
                notes.append(f"{f.name}:{note}")
    finally:
        s.close()
    if not masks:
        return None, "分量因子均不限池"
    aligned = [m.reindex(index=market.dates, columns=market.codes).fillna(False)
               for m in masks]
    out = aligned[0]
    for m in aligned[1:]:
        out = out & m
    return out, "分量池交集（" + " ∩ ".join(notes) + "）"


def required_history(sig: Signal) -> dict:
    """该信号所需的历史长度（用于"数据够不够"的显式校验）。

    equal_weight / ic_weight 只需覆盖 [a,b]；
    linear / tree 每个 refit 点要用之前 train_window 天拟合，且标签要 purge，
    故最早需要 a - train_window - purge。
    """
    mp = dict(sig.model_params or {})
    if sig.model_type in ("linear", "tree"):
        tw = int(mp.get("train_window", 252))
        purge = int(mp.get("purge", 5))
        lh = int(mp.get("label_horizon", 20))
        # 训练样本须截止到 r - label_horizon - purge，故最早需要 a - (tw + lh + purge)
        return {"train_window": tw, "purge": purge, "label_horizon": lh,
                "extra_days": tw + lh + purge, "refit_days": int(mp.get("refit_days", 20))}
    return {"train_window": 0, "purge": 0, "label_horizon": 0, "extra_days": 0, "refit_days": 0}


def check_data_sufficiency(sig: Signal) -> dict:
    """校验各分量的因子值是否覆盖该信号所需区间；不足则明确拒绝构建。"""
    need = required_history(sig)
    start = pd.Timestamp(sig.start_date)
    end = pd.Timestamp(sig.end_date)
    need_start = start - pd.Timedelta(days=int(need["extra_days"] * 1.6))  # 自然日近似
    s = get_session()
    problems, ok = [], []
    try:
        for c in (sig.components or []):
            f = s.get(Factor, int(c["factor_id"]))
            if f is None:
                problems.append(f"因子 {c['factor_id']} 不存在")
                continue
            vs, ve = f.value_start, f.value_end
            if vs is None or ve is None:
                problems.append(f"因子「{f.name}」未登记值区间，无法判断覆盖")
                continue
            if pd.Timestamp(vs) > need_start:
                problems.append(
                    f"因子「{f.name}」最早只有 {pd.Timestamp(vs).date()}，"
                    f"模型需要从 {need_start.date()} 起的数据（训练窗口 {need['train_window']} 天 + "
                    f"purge {need['purge']} 天）")
            if pd.Timestamp(ve) < end:
                problems.append(
                    f"因子「{f.name}」最晚只到 {pd.Timestamp(ve).date()}，"
                    f"覆盖不到策略区间终点 {end.date()}")
            if all(f"「{f.name}」" not in x for x in problems):
                ok.append(f.name)
    finally:
        s.close()
    return {"ok": not problems, "problems": problems, "checked": ok,
            "need_start": str(need_start.date()), "need_end": str(end.date()), **need}


def create_signal(*, name: str, description: str = "", factor_ids: list[int],
                  model_type: str = "equal_weight", model_params: dict | None = None,
                  preprocess: list[str] | None = None,
                  start_date: str | pd.Timestamp, end_date: str | pd.Timestamp,
                  compute_diagnostics: bool = True) -> dict:
    """登记信号（策略）。不跑回测 —— 回测是 BacktestRun 的事，可对同一信号跑多次。"""
    s = get_session()
    try:
        comps = [{"factor_id": int(f), "weight": 1.0 / len(factor_ids)} for f in factor_ids]
        sig = Signal(name=name, description=description, kind="composite",
                     components=comps, model_type=model_type,
                     model_params=dict(model_params or {}),
                     preprocess=list(preprocess or ["winsorize", "zscore"]),
                     start_date=pd.Timestamp(start_date), end_date=pd.Timestamp(end_date))
        s.add(sig)
        s.flush()
        # 数据充足性：模型类信号需要 a 之前的历史用于拟合，不够就必须拒绝而不是给个错结果
        chk = check_data_sufficiency(sig)
        if not chk["ok"]:
            s.rollback()
            raise ValueError("数据不足，无法构建该策略：\n  - " + "\n  - ".join(chk["problems"]))
        s.commit()
        sig_id = sig.id
    finally:
        s.close()

    if compute_diagnostics:
        try:
            diag = diagnose_signal(sig_id)
            s2 = get_session()
            try:
                sg = s2.get(Signal, sig_id)
                sg.diagnostics_json = diag
                s2.commit()
            finally:
                s2.close()
        except Exception as e:  # noqa: BLE001 —— 诊断失败不阻断信号登记
            s2 = get_session()
            try:
                sg = s2.get(Signal, sig_id)
                sg.diagnostics_json = {"error": f"{type(e).__name__}: {e}"}
                s2.commit()
            finally:
                s2.close()
    return {"id": sig_id}


def diagnose_signal(signal_id: int, horizons: list[int] | None = None) -> dict:
    """对信号做与因子库同类的诊断（IC / RankIC / 分层 + **分层净值**）。

    分层净值与因子页用的是同一个轻量实现（`factors/layers.py`），
    因此两个页面的分层口径完全一致，且同样是"诊断的附属统计量"、不需要单独触发。
    """
    from ..config import load_config
    from ..factors.diagnostics import diagnostics_json, run_diagnostics
    from ..factors.layers import layered_nav

    cfg = load_config()
    s = get_session()
    try:
        sig = s.get(Signal, signal_id)
        if sig is None:
            raise ValueError(f"信号 {signal_id} 不存在")
        model_type = sig.model_type
        groups = int(sig.preprocess_spec.get("quantile", 5))
    finally:
        s.close()
    s = get_session()
    try:
        sig = s.get(Signal, signal_id)
        panel = signal_panel(sig, market())
        # 分层净值在【信号定义域】上计算：分量因子各自股票池的交集
        # （历史 bug：这里传 pool_mask=None，池约束信号的分层统计跑在全样本上，与因子口径不一致）
        pool_mask, _ = _signal_pool_mask(sig, market())
    finally:
        s.close()

    d = cfg.factors["diagnosis"]
    hs = horizons or d["horizon_days"]
    diag_cfg = {"horizon_days": hs,
                "min_cross_section_samples": d["min_cross_section_samples"],
                "quantile_groups": groups}
    out = run_diagnostics(panel, market().close_adj, diag_cfg)

    mk = market()
    cost = cfg.backtest.get("cost", {})
    A = int(cfg.data["trading_days_per_year"])
    rf = float(cfg.data["risk_free_annual"])
    payload = diagnostics_json(out)
    for h in hs:
        key = str(h)
        try:
            payload[key]["layer_nav"] = layered_nav(
                panel, mk, groups=groups, h=int(h), pool_mask=pool_mask,
                commission_buy=float(cost.get("commission_buy", 0.0)),
                commission_sell=float(cost.get("commission_sell", 0.0)),
                trading_days=A, rf_annual=rf)
        except Exception as e:  # noqa: BLE001 —— 附属统计量失败不影响 IC 诊断
            payload[key]["layer_nav"] = {"error": f"{type(e).__name__}: {e}"}
    return payload


# ---------------- 回测 ----------------
def run_backtest_for_signal(*, signal_id: int, start_date, end_date,
                            pool_id: int | None = None, top_n: int = 10,
                            weighting: str = "equal_weight",
                            rebalance_freq: str = "weekly",
                            initial_cash: float | None = None,
                            cost_override: dict | None = None,
                            industry_neutral: bool = False,
                            name: str = "") -> dict:
    """对信号在指定配置下执行一次模拟成交，落库为 BacktestRun。"""
    from ..backtest.checks import finalize_checks, run_checks
    from ..backtest.engine import run_backtest
    from ..backtest.metrics import compute_metrics
    from ..config import load_config

    cfg = load_config()
    mk = market()
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    s = get_session()
    try:
        sig = s.get(Signal, signal_id)
        if sig is None:
            raise ValueError(f"信号 {signal_id} 不存在")
        sig_lo, sig_hi = pd.Timestamp(sig.start_date), pd.Timestamp(sig.end_date)
    finally:
        s.close()
    if start < sig_lo or end > sig_hi:
        raise ValueError(
            f"回测区间 [{start.date()}, {end.date()}] 必须落在信号有效区间 "
            f"[{sig_lo.date()}, {sig_hi.date()}] 内")
    if start >= end:
        raise ValueError("回测区间起点必须早于终点")

    panel = signal_panel(sig, mk, start, end)
    pool_mask, pool_note = pool_mask_for(pool_id, mk)

    bt_cfg = dict(cfg.backtest)
    if initial_cash is not None:
        bt_cfg["initial_cash"] = float(initial_cash)
    bt_cfg["rebalance"] = {**bt_cfg["rebalance"], "freq": rebalance_freq}
    bt_cfg["portfolio"] = {**bt_cfg["portfolio"], "top_n": int(top_n), "weighting": weighting}
    bt_cfg["cost"] = {**bt_cfg["cost"], **(cost_override or {})}
    bt_cfg["sample"] = {"start": str(start.date()), "end": str(end.date())}
    bt_cfg["trading_days_per_year"] = int(cfg.data["trading_days_per_year"])

    # 行业中性（可选）：申万 PIT 行业标签 + 单行业持仓上限 = ceil(top_n / 10)（文档化口径）
    industry_labels = None
    if industry_neutral:
        from ..data.industry import industry_labels as _industry_labels
        industry_labels = _industry_labels(mk.dates, mk.codes)
        bt_cfg["portfolio"]["max_per_industry"] = max(1, int(np.ceil(int(top_n) / 10)))
        bt_cfg["portfolio"]["industry_neutral"] = True

    net = run_backtest(panel, mk, bt_cfg, name=name or f"signal{signal_id}", pool_mask=pool_mask,
                       industry_labels=industry_labels)
    gross = run_backtest(panel, mk, bt_cfg, name=name or f"signal{signal_id}",
                         disable_cost=True, pool_mask=pool_mask, industry_labels=industry_labels)
    net.nav_gross = gross.nav  # noqa: SLF001
    net.metrics = compute_metrics(net, float(cfg.data["risk_free_annual"]),
                                  int(cfg.data["trading_days_per_year"]))
    checks = run_checks(net, mk, nav_gross=gross.nav, signal=panel, pool_mask=pool_mask)
    nd, gd = [t["date"] for t in net.trades], [t["date"] for t in gross.trades]
    pairs = list(zip(net.trades, gross.trades))
    n_diff = sum(1 for a, g in pairs if abs(float(a["amount"]) - float(g["amount"])) > 1e-9)
    max_diff = max((abs(float(a["amount"]) - float(g["amount"])) for a, g in pairs), default=0.0)
    checks["gross_net_attribution"] = {
        "convention": "same_path" if (nd == gd and n_diff == 0) else "separate_reruns",
        "same_trade_dates": bool(nd == gd),
        "n_trades_net": len(net.trades), "n_trades_gross": len(gross.trades),
        "n_trades_differing_notional": int(n_diff), "max_notional_diff": float(max_diff),
        "cost_total": float(net.metrics["cost"]["total_cost_currency"]),
        "gross_minus_net_final_nav": float(gross.nav.iloc[-1] - net.nav.iloc[-1]),
        "note": ("目标股数按交易前净值换算，付过手续费后净值本就不同，故仓位规模随之变化；"
                 "毛净差异 = 费用 + 路径差异，不作纯费用归因。"),
    }
    checks = finalize_checks(checks)
    net.checks = checks

    s = get_session()
    try:
        run = BacktestRun(signal_id=signal_id, name=name, start_date=start, end_date=end,
                          pool_id=pool_id, pool_note=pool_note, top_n=int(top_n),
                          weighting=weighting, rebalance_freq=rebalance_freq,
                          initial_cash=float(bt_cfg["initial_cash"]),
                          cost_json=bt_cfg["cost"], metrics_json=net.metrics,
                          checks_json=checks)
        s.add(run)
        s.commit()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        p = RESULTS_DIR / f"bt{run.id}.json"
        p.write_text(json.dumps(net.to_json(), ensure_ascii=False, default=str), encoding="utf-8")
        run.result_path = str(p)
        s.commit()
        return {"id": run.id, "all_pass": bool(checks["all_pass"])}
    finally:
        s.close()


def backtest_result(run_id: int) -> dict | None:
    p = RESULTS_DIR / f"bt{run_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def list_backtests(signal_id: int | None = None) -> list[dict]:
    s = get_session()
    try:
        q = s.query(BacktestRun)
        if signal_id is not None:
            q = q.filter_by(signal_id=signal_id)
        return [r.to_dict() for r in q.order_by(BacktestRun.id.desc()).all()]
    finally:
        s.close()


def list_signals() -> list[dict]:
    s = get_session()
    try:
        out = []
        for sig in s.query(Signal).order_by(Signal.id.desc()).all():
            d = sig.to_dict()
            d["n_backtests"] = s.query(BacktestRun).filter_by(signal_id=sig.id).count()
            out.append(d)
        return out
    finally:
        s.close()


def get_signal(signal_id: int) -> dict | None:
    s = get_session()
    try:
        sig = s.get(Signal, signal_id)
        return sig.to_dict() if sig else None
    finally:
        s.close()


def factor_pool_impact(factor_id: int, new_pool_id: int | None) -> dict:
    """换池前的影响预览（只读，不写库）。

    换池改变的是【截面】：winsorize/zscore/IC/分层全部按截面计算，
    因此池一变，因子值本身（预处理后）与全部诊断都会变。
    """
    s = get_session()
    try:
        f = s.get(Factor, factor_id)
        if f is None:
            raise ValueError(f"因子 {factor_id} 不存在")
        cur_pool, cur_note = f.pool_id, f.universe_note or "未登记"
        old_vals = pd.read_parquet(f.values_path) if f.values_path else None
    finally:
        s.close()
    mk = market()
    new_mask, new_note = pool_mask_for(new_pool_id, mk)
    old_mask, _ = pool_mask_for(cur_pool, mk)

    def _active(mask):
        if mask is None:                      # None = 不过滤 = 全部可用资产
            return int(len(mk.dates) * len(mk.codes))
        return int(mask.to_numpy().sum())

    n_old, n_new = _active(old_mask), _active(new_mask)
    old_codes = len(old_mask.columns) if old_mask is not None else len(mk.codes)
    new_codes = len(new_mask.columns) if new_mask is not None else len(mk.codes)
    return {
        "from_pool_id": cur_pool, "from_note": cur_note,
        "to_pool_id": new_pool_id, "to_note": new_note,
        "from_codes": old_codes, "to_codes": new_codes,
        "from_cells": n_old, "to_cells": n_new,
        "changed": (cur_pool != new_pool_id),
        "warning": ("股票池变化会改变截面：去极值/标准化/IC/分层全部按截面计算，"
                    "因此因子含义与全部诊断都会改变，历史结论不再可比。"
                    if cur_pool != new_pool_id else ""),
        "value_range": ([str(f.value_start.date()), str(f.value_end.date())]
                        if old_vals is not None and f.value_start else None),
    }


def refresh_factor(factor_id: int, pool_id: int | None = None,
                   confirm_pool_change: bool = False,
                   spec: dict | None = None,
                   start=None, end=None) -> dict:
    """重算因子：重新计算取值并可【显式指定区间】与【改参数（含复权方式）】。

    `pool_id` 语义（不要把 None 当成"设成不过滤"）：
      None  = 保持该因子原有股票池
      0     = 明确设为「不过滤」（等价于"全部可用资产"池）
      >0    = 设为该股票池；若它是 kind='all' 的池，按 0 处理

    - 区间默认为「自动」= 用本地行情覆盖的全部交易日（即延到数据末端）；
      显式给 start/end 则只保留该窗口内的因子值。
    - spec 里的复权方式决定取哪张价格面板，因此**必须先落 spec 再重算因子值**。
    - 换池必须 confirm_pool_change=True —— 它会改变截面，从而改变因子含义。
      判定按归一后的池 id 比较，所以 None <-> "全部可用资产" 这种等价切换不会被误判。
    """
    s = get_session()
    try:
        f = s.get(Factor, factor_id)
        if f is None:
            raise ValueError(f"因子 {factor_id} 不存在")
        cur_pool = norm_pool_id(f.pool_id)
        target_pool = cur_pool if pool_id is None else norm_pool_id(pool_id)
        if target_pool != cur_pool and not confirm_pool_change:
            raise ValueError(
                "更换股票池会改变截面（去极值/标准化/IC/分层都按截面计算），"
                "因子含义与全部诊断都会改变。请显式确认后再执行。")
        before_range = ([str(f.value_start.date()), str(f.value_end.date())]
                        if f.value_start and f.value_end else None)
        before_spec = f.preprocess_spec
        old_pool = f.pool_id

        # 顺序很重要：spec 必须在重算因子值【之前】落定。
        # 复权口径（spec.adjust）决定取哪张价格面板 → 决定因子值本身；
        # 原实现先 recompute 再写 spec，导致改复权方式后存下来的值仍是旧口径的，
        # 页面显示 qfq、实际存的却是 hfq（静默不一致）。
        if spec is not None:
            f.preprocess = normalize_spec(spec)

        values = recompute_values(f)
        values, win = _clip_window(values, start, end)
        if len(values.index):
            f.value_start = pd.Timestamp(values.index.min())
            f.value_end = pd.Timestamp(values.index.max())
        f.pool_id = target_pool
        _, note = pool_mask_for(target_pool, market())
        f.universe_note = note
        # 原子性：先写临时文件，DB 提交成功后再原子替换 —— 中途失败时旧值文件不被破坏
        tmp_path = _write_values_tmp(f.id, values)
        s.flush()
        _compute_and_store_metrics(f, values, s)
        if f.pool_id != old_pool:
            f.notes = (f.notes or "") + f" | 股票池变更 {old_pool} -> {f.pool_id}（截面已变）"
        if spec is not None and normalize_spec(spec) != before_spec:
            f.notes = (f.notes or "") + " | 预处理/评估参数已修改"
        s.commit()
        _promote_values_tmp(tmp_path, f.values_path or _values_path_of(f.id))
        after_range = [str(f.value_start.date()), str(f.value_end.date())]
        after_spec = f.preprocess_spec
        name = f.name
    finally:
        s.close()
    return {"id": factor_id, "name": name,
            "value_range_before": before_range, "value_range_after": after_range,
            "pool_id": target_pool, "universe_note": note,
            "window": win if start or end else "自动（全部本地行情）",
            "spec_changed": after_spec != before_spec,
            "spec": after_spec}


def _clip_window(values: pd.DataFrame, start, end) -> tuple[pd.DataFrame, dict | None]:
    """把因子值面板裁到 [start, end]（任一为空则对应端不裁）。"""
    if start is None and end is None:
        return values, None
    lo = pd.Timestamp(start) if start else values.index.min()
    hi = pd.Timestamp(end) if end else values.index.max()
    if lo > hi:
        raise ValueError(f"区间起点 {lo.date()} 晚于终点 {hi.date()}")
    out = values.loc[(values.index >= lo) & (values.index <= hi)]
    if out.empty:
        raise ValueError(f"区间 [{lo.date()}, {hi.date()}] 内没有任何交易日")
    return out, {"requested_start": str(lo.date()), "requested_end": str(hi.date()),
                 "n_days": int(len(out)),
                 "actual_start": str(out.index.min().date()),
                 "actual_end": str(out.index.max().date())}


def batch_refresh_factors(factor_ids: list[int], *, pool_id: int | None = None,
                          confirm_pool_change: bool = False,
                          spec: dict | None = None,
                          start=None, end=None, progress=None) -> dict:
    """对多个因子批量重算（区间 / 股票池 / 预处理参数可统一覆盖）。

    逐个执行、逐个捕获异常：单个因子失败不影响其余（返回 ok/error 分列）。
    pool_id 为 None 表示【保持各因子原有股票池】，避免批量操作误把池统一掉。
    `progress(done, stage)` 可选，供后台任务刷新进度。
    """
    results, errors = [], []
    for n, fid in enumerate(factor_ids, 1):
        if progress:
            progress(n - 1, f"正在重算因子 {fid}（{n}/{len(factor_ids)}）")
        try:
            # pool_id=None -> refresh_factor 内部保持原池
            r = refresh_factor(int(fid), pool_id=pool_id,
                               confirm_pool_change=confirm_pool_change,
                               spec=spec, start=start, end=end)
            results.append(r)
        except Exception as e:  # noqa: BLE001 —— 批量操作不因单点失败而中断
            errors.append({"id": int(fid), "error": f"{type(e).__name__}: {e}"})
        if progress:
            progress(n, f"已完成 {n}/{len(factor_ids)}")
    return {"n_requested": len(factor_ids), "n_ok": len(results),
            "n_failed": len(errors), "ok": results, "errors": errors}


# 说明：信号页的分层净值与因子页共用同一个轻量附属统计量（factors/layers.py），
# 在 diagnose_signal() 里随诊断一起产出，不再有"手动跑分层回测 + 落盘缓存"这一套。



ALPHAGEN_PARAMS_SCHEMA = {
    "steps": {"type": "int", "default": 8192, "min": 256, "max": 500000, "label": "PPO 总步数"},
    "n_envs": {"type": "int", "default": 4, "min": 1, "max": 16, "label": "并行环境数"},
    "rollout_steps": {"type": "int", "default": 128, "min": 16, "max": 1024, "label": "每轮采样步数"},
    "batch_size": {"type": "int", "default": 128, "min": 16, "max": 2048, "label": "批大小"},
    "pool_capacity": {"type": "int", "default": 10, "min": 2, "max": 100, "label": "因子池容量"},
    "max_expr_length": {"type": "int", "default": 8, "min": 3, "max": 15, "label": "表达式最长 token"},
    "label_horizon": {"type": "int", "default": 5, "min": 1, "max": 60, "label": "标签持有期(交易日)"},
    "d_model": {"type": "int", "default": 32, "min": 8, "max": 256, "label": "GRU 隐层维"},
    "n_layers": {"type": "int", "default": 1, "min": 1, "max": 3, "label": "GRU 层数"},
    "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "label": "随机种子"},
    "device": {"type": "str", "default": "auto", "label": "计算设备",
               "choices": ["auto", "cpu", "cuda"]},
    "variant": {"type": "str", "default": "baseline", "label": "变体",
                "choices": ["baseline", "counterfactual", "novelty", "counterfactual_novelty"]},
    "train_ratio": {"type": "float", "default": 0.6, "min": 0.2, "max": 0.8,
                    "label": "训练段占比"},
    "valid_ratio": {"type": "float", "default": 0.2, "min": 0.05, "max": 0.4,
                    "label": "验证段占比（其余为测试段）"},
}


def _dc_kwargs(task_params: dict, cls) -> dict:
    """把任务参数裁剪成 dataclass 能接受的字段。

    任务 params 里混着 spec / pool_id / 起止日期 / split_mode 这些非算法字段，
    整体喂给 dataclass 会 TypeError。按 dataclass 的字段集过滤，比手工维护排除清单
    更不容易漏（之前就因为新增 spec 忘了排除而炸过一次）。
    """
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return {}
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in (task_params or {}).items() if k in names}


def clean_task_params(params: dict, schema: dict) -> dict:
    """按 schema 收敛算法参数（原有逻辑），并把「因子计算参数」normalize 成一个 spec。

    挖出来的因子按这个 spec 计算与评估入库，所以它必须和因子详情页那张参数表同源
    （都用 PREPROCESS_SCHEMA / normalize_spec），否则挖矿用一种口径、入库又退回默认，
    页面上显示的就不是实际算的那个口径。
    """
    clean = {}
    for k, spec in schema.items():
        v = params.get(k, spec["default"])
        if spec["type"] == "int":
            v = int(v)
        elif spec["type"] == "str":
            v = str(v)
            if v not in (spec.get("choices") or [v]):
                v = spec["default"]
        else:
            v = float(v)
        if spec["type"] != "str":
            v = min(max(v, spec["min"]), spec["max"])
        clean[k] = v
    spec = task_spec(params)
    # 持有期只有一个：训练目标（horizon / label_horizon）就是那个 h，直接决定评估口径，
    # 不给用户第二个 h 输入框 —— 否则"训练按 5 日、诊断却按 5 和 20 日报"必然打架。
    train_h = clean.get("label_horizon") or clean.get("horizon")
    if train_h:
        spec["horizons"] = str(int(train_h))
    clean["spec"] = spec
    # 计算范围（股票池 / 起止日期）随任务走，与因子页「计算范围」同义
    clean["pool_id"] = norm_pool_id(params.get("pool_id"))
    clean["start_date"] = str(params["start_date"]) if params.get("start_date") else None
    clean["end_date"] = str(params["end_date"]) if params.get("end_date") else None
    # 三段切分：显式边界（用户自己定）优先，否则用比例
    keys = ("train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end")
    bounds = {k: str(params[k]) for k in keys if params.get(k)}
    clean["split_bounds"] = bounds or None
    clean["split_mode"] = "dates" if bounds else "ratio"
    return clean


def _spawn_runner(task_id: str) -> None:
    """用独立子解释器执行挖掘任务（python -m quantlab.lab.task_runner）。

    - Web 进程零计算负载（历史为同进程线程，numpy/torch 占 GIL 拖慢全站）；
    - Windows 下不依赖 multiprocessing spawn 的 __main__ 重导（uvicorn/脚本/笔记本
      任意宿主上下文都稳健）；
    - 服务重启不影响已派发的任务：子进程独立完成并把结果写回 SQLite。
    """
    import subprocess
    import sys

    log_dir = ROOT / "data" / "store" / "task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = open(log_dir / f"{task_id}.out", "ab")
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # 不随服务 Ctrl-C 退出
    subprocess.Popen([sys.executable, "-m", "quantlab.lab.task_runner",
                      "--task-id", task_id],
                     cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT, **kwargs)


def validate_split_params(params: dict) -> None:
    """创建挖掘任务前预检三段边界（在 HTTP 层就给出人话错误，而不是入队后 FAILED）。

    用任务实际的数据范围（区间裁剪后的交易日历）干跑 segment_split；
    失败时抛 ValueError，路由层转 400。
    """
    from ..lab.gp_engine import segment_split

    h = int((params or {}).get("label_horizon") or 5)
    start = (params or {}).get("start_date")
    end = (params or {}).get("end_date")
    dates = market().dates
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    # 原始表单参数还没有 split_bounds 键（clean_task_params 才组装）—— 这里按同样的
    # 规则从六个日期字段现组（历史 bug：读不存在的键导致校验被静默跳过，坏输入照样入队）
    keys = ("train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end")
    bounds = {k: str(params[k]) for k in keys if (params or {}).get(k)} or None
    if bounds and params.get("split_mode") != "ratio":
        segment_split(dates, purge=h, bounds=bounds)
    elif bounds:
        segment_split(dates, purge=h, bounds=bounds)
    elif not bounds:
        tr = min(max(float((params or {}).get("train_ratio", 0.6)), 0.1), 0.8)
        va = min(max(float((params or {}).get("valid_ratio", 0.2)), 0.05), 0.9 - tr)
        n = len(dates)
        i1 = max(1, int(round(n * tr)))
        i2 = min(max(i1 + 1, int(round(n * (tr + va)))), n - 1)
        if n - i2 < 1 or i1 < 1:
            raise ValueError(f"数据范围只有 {n} 个交易日，比例切分后至少一段为空；请扩大区间")


def create_alphagen_task(params: dict) -> str:
    """创建 AlphaGen（日频 CPU）挖掘任务，后台线程执行。

    成功后自动：因子池以**隐藏**因子入库 + 落成一个带学习权重的策略（见 _run_alphagen_task）。
    """
    validate_split_params(params)
    task_id = str(uuid.uuid4())[:8]
    clean = clean_task_params(params, ALPHAGEN_PARAMS_SCHEMA)
    s = get_session()
    try:
        s.add(AlgoTask(id=task_id, engine="alphagen_daily_cpu", objective="factor_mining",
                       status="PENDING", params=clean))
        s.commit()
    finally:
        s.close()
    # 进程隔离：引擎在独立子进程执行（历史为同进程线程，numpy/torch 占 GIL 拖慢全站）
    _spawn_runner(task_id)
    return task_id



def create_task(engine_id: str, params: dict) -> str:
    """按引擎分发建任务（实验室页面统一入口）。"""
    if engine_id == "gp_daily":
        return create_gp_task(params)
    if engine_id == "alphagen_daily_cpu":
        return create_alphagen_task(params)
    raise ValueError(f"引擎 {engine_id} 当前不可用（未知引擎（本机可用：gp_daily / alphagen_daily_cpu））")


# ---------------- 算法实验室 ----------------
GP_PARAMS_SCHEMA = {
    "population_size": {"type": "int", "default": 60, "min": 10, "max": 500, "label": "种群规模"},
    "generations": {"type": "int", "default": 15, "min": 1, "max": 200, "label": "进化代数"},
    "tournament_k": {"type": "int", "default": 4, "min": 2, "max": 20, "label": "锦标赛规模"},
    "p_crossover": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "label": "交叉概率"},
    "p_mutate": {"type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "label": "变异概率"},
    "elitism": {"type": "int", "default": 2, "min": 1, "max": 10, "label": "精英数"},
    "horizon": {"type": "int", "default": 5, "min": 1, "max": 60, "label": "持有期h(交易日)"},
    "seed": {"type": "int", "default": 7, "min": 0, "max": 999999, "label": "随机种子"},
    "train_ratio": {"type": "float", "default": 0.6, "min": 0.2, "max": 0.8,
                    "label": "训练段占比"},
    "valid_ratio": {"type": "float", "default": 0.2, "min": 0.05, "max": 0.4,
                    "label": "验证段占比（其余为测试段）"},
}


def engines() -> list[dict]:
    return [{
        "engine_id": "gp_daily", "display_name": "GP 遗传编程（日频）",
        "algorithm": "符号回归：种群进化 + 锦标赛选择 + 子树交叉/变异，适应度=训练段RankIC",
        "objective": "factor_mining", "resource": "cpu（进程内）",
        "param_schema": GP_PARAMS_SCHEMA, "status": "ready",
        "discipline": "train/valid/test 三段隔离，候选同时报告三段IC",
    }, {
        "engine_id": "alphagen_daily_cpu", "display_name": "AlphaGen-RL（日频 · CPU）",
        "algorithm": "MaskablePPO + GRU 逐 token 生成表达式树，线性因子池，奖励=池 ensemble IC",
        "objective": "factor_mining", "resource": "cpu/gpu（device 参数）",
        "param_schema": ALPHAGEN_PARAMS_SCHEMA, "status": "ready",
        "discipline": "train/valid/test 三段隔离；候选转译为沙箱 Python 源码后入库",
        "note": "源自 alphagen_5m 引擎，仅把数据层从 5 分钟 48bar/日 改为日频 1bar/日",
    }, {
        "engine_id": "alphagen_daily_cpu", "display_name": "AlphaGen-RL（日频 · CUDA 可选）",
        "algorithm": "同 CPU 路径；device=auto 时本机有 CUDA 即用 GPU（RTX 4060 + torch cu130 已实测）",
        "objective": "factor_mining", "resource": "gpu 可选（device 参数）",
        "param_schema": {}, "status": "ready",
        "note": "在实验室表单选 device=cuda 或 auto 即启用 GPU；无 CUDA 自动回落 CPU",
    }]


def create_gp_task(params: dict) -> str:
    validate_split_params(params)
    task_id = str(uuid.uuid4())[:8]
    clean_params = clean_task_params(params, GP_PARAMS_SCHEMA)
    s = get_session()
    try:
        t = AlgoTask(id=task_id, engine="gp_daily", objective="factor_mining",
                     status="PENDING", params=clean_params)
        s.add(t)
        s.commit()
    finally:
        s.close()
    _spawn_runner(task_id)
    return task_id



def cancel_task(task_id: str) -> bool:
    # 取消经文件旗标跨进程传递（引擎在独立子进程里，见 task_runner）
    from ..lab.task_runner import request_cancel
    request_cancel(task_id)
    s = get_session()
    try:
        t = s.get(AlgoTask, task_id)
        if t and t.status in ("PENDING", "RUNNING"):
            t.status = "CANCELLED"
            t.finished_at = now()
            s.commit()
            return True
    finally:
        s.close()
    return False


def task_spec(task_params: dict) -> dict:
    """从挖掘任务参数里取出「因子计算参数」（spec），缺省用 DEFAULT_SPEC。

    这些参数决定挖出来的因子按什么口径计算与评估（复权 / 去极值 / 标准化 /
    中性化 / IC 方法 / 持有期 / 分组数），与因子详情页那张参数表是同一套 schema。

    注意：**不要**先按 DEFAULT_SPEC 的键过滤再交给 normalize_spec ——
    那样 `outlier_k`/`outlier_pct` 这类旧名会在别名解析之前就被丢掉，
    静默用默认值（踩过一次）。normalize_spec 本身会做别名映射并忽略未知键。
    """
    return normalize_spec((task_params or {}).get("spec") or {})


def auto_strategy_from_task(task_id: str, weights: dict[int, float] | None = None,
                            name: str | None = None) -> dict | None:
    """把挖掘任务的因子池直接落成一个策略（AlphaGen 的因子组合 = 线性组合）。

    - 池内因子以**隐藏**方式入库（不在因子库主列表露出），策略通过 components 引用它们
    - 权重用算法**学到的线性权重**（`AlphaPool.weights`），不是等权，
      因此 model_type 记 `weighted` 而不是 equal_weight —— 免得页面口径说谎
    - 已有同一任务生成的策略则复用（幂等），不重复建
    """
    s = get_session()
    try:
        exist = s.query(Signal).filter_by(algo_task_id=task_id).first()
        if exist:
            return {"id": exist.id, "name": exist.name, "created": False,
                    "n_factors": len(exist.components or [])}
        cands = (s.query(AlgoCandidate).filter_by(task_id=task_id, adopted=True)
                 .order_by(AlgoCandidate.id).all())
        pairs = [(c.factor_id, c.id) for c in cands if c.factor_id]
        if not pairs:
            return None
        # 因子值覆盖区间的交集：策略有效区间取这个范围
        lo = hi = None
        for fid, _ in pairs:
            f = s.get(Factor, fid)
            if f is None or not f.value_start or not f.value_end:
                continue
            vs, ve = pd.Timestamp(f.value_start), pd.Timestamp(f.value_end)
            lo = vs if lo is None else max(lo, vs)
            hi = ve if hi is None else min(hi, ve)
        if lo is None or lo >= hi:
            return None
        # 权重：优先用算法学到的；缺失则等权
        ws = []
        for fid, _ in pairs:
            w = None if weights is None else weights.get(int(fid))
            ws.append(1.0 if w is None else float(w))
        tot = sum(abs(w) for w in ws) or 1.0
        comps = [{"factor_id": int(fid), "weight": float(w) / tot}
                 for (fid, _), w in zip(pairs, ws)]
        sig = Signal(name=name or f"{task_id} 因子池组合", kind="alpha_pool",
                     description="挖掘任务因子池的线性组合（权重由算法学习得到）",
                     components=comps, model_type="weighted", model_params={},
                     preprocess=[], start_date=lo, end_date=hi,
                     algo_task_id=task_id)
        s.add(sig)
        s.commit()
        return {"id": sig.id, "name": sig.name, "created": True, "n_factors": len(comps)}
    finally:
        s.close()


def adopt_candidates(task_id: str, candidate_ids: list[int], *,
                     hidden: bool = False, spec: dict | None = None,
                     pool_id=None, start=None, end=None) -> list[dict]:
    """一键导入：候选 → 因子库（与内置因子同一条评估链路）。幂等：已采纳返回既有。

    `hidden=True` 用于"挖掘任务的因子池自动入库"：因子照常计算与诊断，只是不在
    因子库主列表里露出。`spec` 为因子计算参数（来自任务），保证入库因子的口径
    与挖矿时一致，而不是悄悄退回默认口径。
    `pool_id` / `start` / `end` 为计算范围：与因子页「计算范围」同义 ——
    因子值按该池的逐日截面计算、并只保留区间内的交易日。
    """
    from ..lab.gp_engine import eval_expr
    s = get_session()
    out = []
    try:
        sp = normalize_spec(spec) if spec is not None else None
        mk = market()
        pool_id = norm_pool_id(pool_id)
        _, pool_note = pool_mask_for(pool_id, mk)
        for cid in candidate_ids:
            c = s.get(AlgoCandidate, cid)
            if c is None or c.task_id != task_id:
                continue
            if c.adopted and c.factor_id:
                # 已被任务自动入库（隐藏）的候选，人工再点一次采纳 = 确认 → 转为可见
                if not hidden:
                    f0 = s.get(Factor, c.factor_id)
                    if f0 is not None and f0.hidden:
                        f0.hidden = False
                        s.commit()
                out.append({"candidate_id": cid, "factor_id": c.factor_id, "created": False})
                continue
            # 两种候选来源：GP 的表达式树 / AlphaGen 的转译源码（后者走沙箱用户代码路径）
            src = c.expr_json
            if isinstance(src, dict) and src.get("code"):
                from ..factors.user_code import compute_user_factor
                v = compute_user_factor(str(src["code"]), mk.fields_for((sp or {}).get("adjust", "hfq")))
                vhash = hashlib_sha(v)
                f, created = register_factor(
                    name=f"AG_{c.expression[:26]}…", hash_code=vhash, source="gp",
                    expression=c.expression,
                    hypothesis=f"AlphaGen-RL 挖掘候选（任务{task_id}，trainIC={c.train_ic}）",
                    direction="分数越高，预期收益越高（以训练段 IC 符号为准，采纳后请复核）",
                    missing_policy="表达式求值 NaN=缺失；截面 <10 样本记缺失",
                    failure_modes=["过拟合风险：请对照 valid/test IC", "RL 搜索到的高 IC 可能是噪声"],
                    operator="user", params={}, preprocess=sp or [],
                    values=v, algo_task_id=task_id, code=str(src["code"]), hidden=hidden,
                    pool_id=pool_id, universe_note=pool_note,
                    notes=f"adopted from alphagen candidate {cid}")
                c.adopted, c.factor_id = True, f.id
                s.commit()
                out.append({"candidate_id": cid, "factor_id": f.id, "created": created,
                            "hidden": hidden})
                continue
            tree = tuple(c.expr_json)
            fields = mk.fields_for((sp or {}).get("adjust", "hfq"))
            with np.errstate(all="ignore"):
                v = eval_expr(tree, fields)
                if not isinstance(v, pd.DataFrame):
                    v = pd.DataFrame(v, index=mk.dates, columns=mk.codes)
                v = v.replace([np.inf, -np.inf], np.nan)
            vhash = hashlib_sha(v)
            f, created = register_factor(
                name=f"GP_{c.expression[:28]}…", hash_code=vhash, source="gp",
                expression=c.expression, hypothesis=f"GP挖掘候选（任务{task_id}，trainIC={c.train_ic}）",
                direction="分数越高，预期收益越高（以训练段RankIC符号为准，采纳后请复核）",
                missing_policy="表达式求值NaN=缺失；截面<10 样本记缺失",
                failure_modes=["过拟合风险：请对照valid/test IC", "窗口/算子数据依赖"],
                operator="expression", params={"expr_json": c.expr_json},
                preprocess=sp or [], values=v, algo_task_id=task_id, hidden=hidden,
                pool_id=pool_id, universe_note=pool_note,
                notes=f"adopted from candidate {cid}")
            c.adopted, c.factor_id = True, f.id
            s.commit()
            out.append({"candidate_id": cid, "factor_id": f.id, "created": created,
                        "hidden": hidden})
        # 计算区间：只保留 [start, end] 内的因子值（与因子页「计算区间」同义）
        if start or end:
            for a in out:
                _clip_factor_window(a.get("factor_id"), start, end)
    finally:
        s.close()
    return out


def _clip_factor_window(factor_id: int | None, start, end) -> None:
    """把某个因子的值文件裁到 [start, end] 并重算诊断（区间变更必须重算，含分层净值）。"""
    if not factor_id:
        return
    s = get_session()
    try:
        f = s.get(Factor, factor_id)
        if f is None or not f.values_path:
            return
        vals = pd.read_parquet(f.values_path)
        clipped, win = _clip_window(vals, start, end)
        if win is None or len(clipped) == len(vals):
            return
        f.values_path = _save_values(f.id, clipped)
        f.value_start = pd.Timestamp(clipped.index.min())
        f.value_end = pd.Timestamp(clipped.index.max())
        s.flush()
        _compute_and_store_metrics(f, clipped, s)
        s.commit()
    finally:
        s.close()


def hashlib_sha(values: pd.DataFrame) -> str:
    from ..factors.operators import values_digest
    return values_digest(values)[:32]
