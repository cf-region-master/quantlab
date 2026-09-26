"""挖掘任务的进程隔离执行器（Web 进程零计算负载）。

历史问题：GP/AlphaGen 引擎跑在 uvicorn 同进程的 daemon 线程里——numpy/torch
重计算占住 GIL，挖矿期间所有 HTTP 请求被拖慢（前端"卡"的主因）；训练线程高频
写 SQLite 又与页面读争锁。本模块把引擎挪到**独立子进程**：
  - Web 进程只创建任务行并 spawn 子进程；
  - 取消经文件旗标跨进程传递（data/store/task_cancel/{task_id}）；
  - 进度写库节流（≥2 秒或阶段切换才落盘），降低与页面读的锁竞争；
  - SQLite 已开 WAL（db.py），读写并存。

Windows spawn 注意：本模块顶层只做轻量 import，torch/sb3 等重依赖在函数内引入。
"""
from __future__ import annotations

import time
from pathlib import Path

from ..config import ROOT

CANCEL_DIR = ROOT / "data" / "store" / "task_cancel"


def request_cancel(task_id: str) -> None:
    """跨进程取消旗标：写文件即生效（子进程在进度回调里轮询）。"""
    CANCEL_DIR.mkdir(parents=True, exist_ok=True)
    (CANCEL_DIR / task_id).write_text("1", encoding="utf-8")


def _cancelled(task_id: str) -> bool:
    return (CANCEL_DIR / task_id).exists()


def run_mining_task(task_id: str) -> None:
    """子进程入口：按引擎分发执行一个挖掘任务（结果全部经 SQLite 交换）。"""
    from ..storage.db import AlgoCandidate, AlgoTask, get_session, now
    from ..storage.store import (market, norm_pool_id, pool_mask_for, task_spec,
                                 _dc_kwargs, adopt_candidates, auto_strategy_from_task)

    s = get_session()
    engine_id = ""
    try:
        t = s.get(AlgoTask, task_id)
        if t is None:
            return
        engine_id = t.engine
        if _cancelled(task_id):  # 排队期间就被取消
            t.status, t.finished_at = "CANCELLED", now()
            s.commit()
            return
        t.status, t.stage, t.progress = "RUNNING", "init", 0.0
        t.log = (t.log or "") + f"[start] engine={engine_id} params={t.params}\n"
        s.commit()
    finally:
        s.close()

    last_write = {"t": 0.0}

    def report(progress=0.0, stage="", metrics=None, log_lines=(), force=False):
        """进度落库（节流）：≥2 秒或 force 才写；期间检测取消旗标。"""
        now_ts = time.time()
        if not force and now_ts - last_write["t"] < 2.0 and not log_lines:
            return
        last_write["t"] = now_ts
        if _cancelled(task_id) and engine_id == "gp_daily":
            eng.cancel()  # GP 引擎在下个 generation 收敛退出
        ss = get_session()
        try:
            tt = ss.get(AlgoTask, task_id)
            if tt is None:
                return
            tt.progress, tt.stage = float(progress), stage
            if metrics:
                tt.metrics_json = {**(tt.metrics_json or {}), **metrics}
            if log_lines:
                tt.log = (tt.log or "") + "\n".join(log_lines) + "\n"
            ss.commit()
        finally:
            ss.close()

    try:
        if engine_id == "gp_daily":
            _run_gp(task_id, report)
        elif engine_id in ("alphagen_daily_cpu", "alphagen_gpu"):
            _run_alphagen(task_id, report)
        else:
            raise ValueError(f"未知引擎: {engine_id}")
    except Exception as e:  # noqa: BLE001 —— 任务失败落库，含堆栈摘要
        import traceback

        ss = get_session()
        try:
            tt = ss.get(AlgoTask, task_id)
            tt.status, tt.message = "FAILED", f"{type(e).__name__}: {e}"
            tt.log = (tt.log or "") + "\n" + traceback.format_exc()[-1200:] + "\n"
            tt.finished_at = now()
            ss.commit()
        finally:
            ss.close()


# ---------------- GP ----------------
def _run_gp(task_id: str, report) -> None:
    from ..lab.gp_engine import GpEngine, GpParams, segment_split
    from ..storage.db import AlgoCandidate, AlgoTask, get_session, now
    from ..storage.store import (_dc_kwargs, market, norm_pool_id, pool_mask_for,
                                 task_spec)

    s = get_session()
    try:
        t = s.get(AlgoTask, task_id)
        params = GpParams(**_dc_kwargs(t.params, GpParams))
        task_params = dict(t.params or {})
    finally:
        s.close()

    spec = task_spec(task_params)
    mk = market()
    pm, note = pool_mask_for(norm_pool_id(task_params.get("pool_id")), mk)
    scope = mk.scoped_view(spec["adjust"], task_params.get("start_date"),
                           task_params.get("end_date"), pool_mask=pm)
    eng = GpEngine(scope, params)
    report(0.0, "init", force=True,
           log_lines=[f"[scope] {note}；区间 {scope.dates.min().date()} ~ "
                      f"{scope.dates.max().date()}"])

    def cb_report(**kw):
        kw["force"] = bool(kw.get("log_lines"))
        if _cancelled(task_id):
            eng.cancel()
        report(**kw)

    result = eng.run(report=cb_report)
    cancelled = _cancelled(task_id)

    ss = get_session()
    try:
        tt = ss.get(AlgoTask, task_id)
        if cancelled or tt.status == "CANCELLED":
            tt.status, tt.finished_at = "CANCELLED", now()
        else:
            tt.status, tt.progress, tt.stage = "SUCCESS", 1.0, "done"
            tt.metrics_json = {**(tt.metrics_json or {}), "curve": result.curve,
                               "segments": eng._segment_report()}
            tt.finished_at = now()
        tt.log = (tt.log or "") + "\n".join(result.log) + "\n"
        for c in result.candidates:
            ss.add(AlgoCandidate(task_id=task_id, expression=c["expression"],
                                 expr_json=c.get("expr_json"), train_ic=c["train_ic"],
                                 valid_ic=c["valid_ic"], test_ic=c["test_ic"],
                                 size=c["size"]))
        ss.commit()
    finally:
        ss.close()


# ---------------- AlphaGen ----------------
def _run_alphagen(task_id: str, report) -> None:
    from ..lab.alphagen_engine import AlphaGenEngine, AlphaGenParams
    from ..storage.db import AlgoCandidate, AlgoTask, get_session, now
    from ..storage.store import (_dc_kwargs, adopt_candidates, auto_strategy_from_task,
                                 market, norm_pool_id, pool_mask_for, task_spec)

    s = get_session()
    try:
        t = s.get(AlgoTask, task_id)
        params = AlphaGenParams(**_dc_kwargs(t.params, AlphaGenParams))
        params.device = str((t.params or {}).get("device", "auto"))
        task_params = dict(t.params or {})
    finally:
        s.close()

    spec = task_spec(task_params)
    mk = market()
    pool_mask, pool_note = pool_mask_for(norm_pool_id(task_params.get("pool_id")), mk)
    scope = mk.scoped_view(spec["adjust"], task_params.get("start_date"),
                           task_params.get("end_date"), pool_mask=pool_mask)
    from ..lab.alphagen_engine import CACHE_DIR
    cdir = CACHE_DIR / scope.cache_key()
    eng = AlphaGenEngine(scope, params, cache_dir=cdir)
    report(0.0, "init", force=True,
           log_lines=[f"[scope] {pool_note}；区间 {scope.dates.min().date()} ~ "
                      f"{scope.dates.max().date()}；缓存 {cdir.name}"])

    def cb_report(**kw):
        kw["force"] = bool(kw.get("log_lines"))
        if _cancelled(task_id):
            eng.cancel()
        report(**kw)

    res = eng.run(report=cb_report)
    cancelled = _cancelled(task_id)

    # ---- 1) 候选落库 ----
    ss = get_session()
    try:
        tt = ss.get(AlgoTask, task_id)
        if cancelled or tt.status == "CANCELLED":
            tt.finished_at = now()
        else:
            tt.status, tt.progress, tt.stage = "SUCCESS", 1.0, "done"
            tt.metrics_json = {**(tt.metrics_json or {}), "curve": res.curve,
                               "summary": res.summary,
                               "segments": (res.summary or {}).get("segments")}
            tt.finished_at = now()
        tt.log = (tt.log or "") + "\n".join(res.log) + "\n"
        for c in res.candidates:
            ss.add(AlgoCandidate(task_id=task_id, expression=c["expression"],
                                 expr_json={"code": c["code"]},
                                 train_ic=c.get("train_ic"), valid_ic=c.get("valid_ic"),
                                 test_ic=c.get("test_ic"), size=c.get("size")))
        ss.commit()
    finally:
        ss.close()
    if cancelled:
        return

    # ---- 2) 因子池自动入库 + 落成策略 ----
    adopted, strategy = [], None
    try:
        ss = get_session()
        try:
            rows = (ss.query(AlgoCandidate).filter_by(task_id=task_id)
                    .order_by(AlgoCandidate.id).all())
        finally:
            ss.close()
        weights = {}
        for c, rc in zip(rows, res.candidates):
            w = rc.get("pool_weight")
            if w is not None:
                weights[int(c.id)] = float(w)
        adopted = adopt_candidates(task_id, [int(c.id) for c in rows],
                                   hidden=True, spec=spec,
                                   pool_id=task_params.get("pool_id"),
                                   start=task_params.get("start_date"),
                                   end=task_params.get("end_date"))
        cid2fid = {a["candidate_id"]: a["factor_id"] for a in adopted}
        fw = {cid2fid[cid]: w for cid, w in weights.items() if cid in cid2fid}
        strategy = auto_strategy_from_task(task_id, fw)
    except Exception as e:  # noqa: BLE001 —— 自动落库失败不影响任务成功
        strategy = {"error": f"{type(e).__name__}: {e}"}

    ss = get_session()
    try:
        tt = ss.get(AlgoTask, task_id)
        tt.metrics_json = {**(tt.metrics_json or {}),
                           "auto_strategy": strategy,
                           "hidden_factor_ids": [a.get("factor_id") for a in adopted]}
        tt.log = (tt.log or "") + (
            f"[auto] rows={len(rows)} res_cands={len(res.candidates)} "
            f"adopted={len(adopted)} weights={len(weights)} strategy={strategy}\n")
        ss.commit()
    finally:
        ss.close()


if __name__ == "__main__":
    # 子解释器入口：python -m quantlab.lab.task_runner --task-id XXXX
    # （Windows 上 multiprocessing spawn 从脚本/stdin 上下文无法重导 __main__，
    #   故改用 subprocess + 模块入口，任何宿主上下文都稳健）
    import argparse

    _ap = argparse.ArgumentParser(description="QuantLab 挖掘任务执行器")
    _ap.add_argument("--task-id", required=True)
    _args = _ap.parse_args()
    run_mining_task(_args.task_id)
