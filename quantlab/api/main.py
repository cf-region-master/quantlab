"""QuantLab Web 平台：因子库 / 信号（策略）/ 回测 / 股票池 / 算法实验室。

长任务交互：挖掘任务与批量任务创建后返回 202 + id，前端轮询 /api/lab/tasks/{id}
与 /api/jobs/{id}；策略回测为同步编排。API 与页面共用同一服务层（store）。
复现信息（旧 /repro 页）由 reports/runs/<id>/manifest.json 与 /api/runs 提供。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Body, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..config import ROOT
from ..factors.preprocess import (PREPROCESS_SCHEMA, normalize_spec, short_label,
                                  spec_steps)
from ..storage import store
from ..storage.db import (AlgoCandidate, AlgoTask, BacktestRun, Factor, FactorMetric, Signal,
                          StockPool, get_session, init_db)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    _startup()
    yield


app = FastAPI(title="QuantLab 量化因子挖掘平台（Project1）", lifespan=_lifespan)
templates = Jinja2Templates(directory=str(ROOT / "quantlab" / "web" / "templates"))
# 参数值的短中文标签（参数总表「当前值」列用）
templates.env.globals["short_label"] = short_label
app.mount("/static", StaticFiles(directory=str(ROOT / "quantlab" / "web" / "static")), name="static")


def _startup() -> None:
    init_db()
    # 内置股票池必须在启动时补齐：缺了它，因子页/回测页的「股票池」下拉是空的，
    # 用户无从选择（历史 bug：这里只调了 seed_from_latest_run）。
    try:
        n_pool = store.ensure_default_pools()
        print(f"[startup] 内置股票池补齐 {n_pool} 个，现有 {len(store.list_pools())} 个")
    except Exception as e:  # noqa: BLE001
        print(f"[startup] 股票池初始化失败: {e}")
    try:
        n = store.seed_from_latest_run()
        print(f"[startup] 因子库种子导入 {n} 条")
    except Exception as e:  # noqa: BLE001 —— 数据未就绪时页面仍可访问
        print(f"[startup] 种子导入跳过: {e}")
    # 历史信号补算「分层净值」（诊断的附属统计量）：耗时，放后台线程，不拖慢启动
    def _backfill():
        try:
            k = store.backfill_signal_diagnostics()
            if k:
                print(f"[startup] 已为 {k} 个历史信号补算分层净值")
        except Exception as e:  # noqa: BLE001
            print(f"[startup] 信号补算跳过: {e}")

    threading.Thread(target=_backfill, daemon=True).start()


def _df_series_json(s: pd.Series) -> dict:
    return {"index": [str(d.date()) for d in s.index],
            "values": [None if not np.isfinite(v) else round(float(v), 6) for v in s]}


# ---------------- 回测检查：门禁/信息项的可读化 ----------------
_GATE_LABELS = {
    "nav_vs_daily_return": "净值 ↔ 日收益一致",
    "cumulative_return_consistent": "累计收益三源互证",
    "cost_ledger": "交易成本账本",
    "weights_cash_conservation": "权重 + 现金守恒",
    "cash_non_negative": "现金非负",
    "no_lookahead": "无前视",
    "events_logged": "异常与缺失留痕",
    "gross_net_attribution": "毛净归因口径",
    "output_hash": "重复运行指纹",
}


def _fmt_val(v) -> str:
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return f"{len(v)} 项"
    if isinstance(v, str):
        return v[:48]
    return str(v)


def _gate_detail(name: str, spec: dict) -> str:
    """按检查名挑出最该看的几个字段，避免把整个 dict 糊在页面上。"""
    pick = {
        "nav_vs_daily_return": ["max_abs_diff", "tolerance"],
        "cumulative_return_consistent": ["cumulative_return", "from_daily_return_product",
                                         "from_metrics"],
        "cost_ledger": ["ledger_sum", "n_fee_mismatch", "n_invalid_execution"],
        "weights_cash_conservation": ["max_relative_residual"],
        "cash_non_negative": ["min_cash", "n_days_negative", "n_days"],
        "no_lookahead": ["n_rebalances", "n_bad_schedule", "n_bad_composition",
                         "n_target_mismatch", "signal_independently_recomputed"],
        "events_logged": ["n_events"],
        "gross_net_attribution": ["convention", "n_trades_differing_notional", "n_trades_net",
                                  "max_notional_diff"],
        "output_hash": ["sha256"],
    }.get(name, [])
    keys = pick or [k for k in spec if k not in ("pass", "note", "tolerance", "gates")][:3]
    parts = []
    for k in keys:
        if k in spec and spec[k] is not None:
            v = spec[k]
            if k == "sha256":
                v = str(v)[:16] + "…"
            parts.append(f"{k}={_fmt_val(v)}")
    return " · ".join(parts)


def _gate_rows(checks: dict) -> list[dict]:
    """把 checks 拆成展示行：门禁项（有 pass）在前，信息项在后。"""
    rows = []
    for name, spec in (checks or {}).items():
        if not isinstance(spec, dict) or name in ("gates", "all_pass"):
            continue
        is_gate = "pass" in spec
        rows.append({
            "name": name,
            "label": _GATE_LABELS.get(name, name),
            "gate": is_gate,
            "pass": bool(spec.get("pass")) if is_gate else None,
            "note": spec.get("note", ""),
            "detail": _gate_detail(name, spec),
        })
    rows.sort(key=lambda r: (not r["gate"], r["name"]))
    return rows


# ---------------- 页面 ----------------
@app.get("/", response_class=HTMLResponse)
def page_index(request: Request):
    s = get_session()
    try:
        ctx = {
            "n_factors": s.query(Factor).count(),
            "n_strategies": s.query(Signal).count(),
            "n_backtests": s.query(BacktestRun).count(),
            "n_tasks": s.query(AlgoTask).count(),
            "n_adopted": s.query(AlgoCandidate).filter_by(adopted=True).count(),
        }
    finally:
        s.close()
    md = None
    try:
        md = store.market()
        ctx["data"] = md.quality["aggregate"]
        ctx["period"] = [str(md.dates.min().date()), str(md.dates.max().date())]
    except Exception:
        ctx["data"] = None
    run = store.latest_run_dir()
    ctx["run"] = run.name if run else None
    if run and (run / "manifest.json").exists():
        mf = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        ctx["run_env"] = mf.get("environment", {})
        ctx["run_checks_pass"] = mf.get("checks", {}).get("base_backtest_all_pass")
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/factors", response_class=HTMLResponse)
def page_factors(request: Request, show_hidden: int = 0):
    s = get_session()
    try:
        q = s.query(Factor).order_by(Factor.id)
        if not show_hidden:
            q = q.filter(Factor.hidden.isnot(True))     # 隐藏因子默认不露出
        factors = q.all()
        mets = s.query(FactorMetric).all()
        by_factor: dict[int, dict[int, FactorMetric]] = {}
        for m in mets:
            by_factor.setdefault(m.factor_id, {})[m.horizon] = m
        n_hidden = s.query(Factor).filter(Factor.hidden.is_(True)).count()
        rows = []
        for f in factors:
            mm = by_factor.get(f.id, {})
            # 近期 IC 监控：h=20（或最接近的可用 horizon）最近 60 个观测的 RankIC 均值
            recent = None
            hs_avail = sorted(mm)
            if hs_avail:
                h_pick = 20 if 20 in mm else hs_avail[-1]
                series = ((mm[h_pick].summary_json or {}).get(str(h_pick), {})
                          .get("ic_series") or {}).get("rank_ic") or []
                tail = [v for v in series[-60:] if v is not None]
                full = [v for v in series if v is not None]
                if tail:
                    recent = {"mean": float(np.mean(tail)),
                              "n": len(tail), "horizon": h_pick,
                              "full_mean": (float(np.mean(full)) if full else None)}
            rows.append({"f": f, "m5": mm.get(5), "m20": mm.get(20), "recent": recent})
    finally:
        s.close()
    mk = store.market()
    return templates.TemplateResponse(request, "factors.html", {
        "rows": rows, "pools": store.list_pools(),
        "n_hidden": n_hidden, "show_hidden": bool(show_hidden),
        "data_start": str(mk.dates.min().date()), "data_end": str(mk.dates.max().date()),
        "spec_schema": PREPROCESS_SCHEMA,
        "neutralize_choices": PREPROCESS_SCHEMA["neutralize"]["method"]["choices"],
    })


def _spec_from_form(form) -> dict:
    """从表单里收集参数（字段名与 PREPROCESS_SCHEMA / 参考契约的扁平键一致）。

    覆盖：预处理（fillna / outlier / normalize / neutralize）+ 评估口径
    （ic_method / horizons / quantile / adjust）。
    """
    keys = ("fillna_method", "fillna_limit",
            "outlier_method", "alpha", "k",
            "normalize_method", "min_n",
            "neutralize_method", "neutralize_level", "neutralize_min_n",
            "ic_method", "horizons", "quantile", "adjust")
    raw = {k: form.get(k) for k in keys if form.get(k) is not None}
    return normalize_spec(raw)


@app.get("/factors/new", response_class=HTMLResponse)
def page_factor_new(request: Request):
    from ..factors.user_code import FACTOR_CODE_TEMPLATE, ALLOWED_MODULES
    mk = store.market()
    return templates.TemplateResponse(request, "factor_new.html", {
        "template_code": FACTOR_CODE_TEMPLATE,
        "allowed_modules": ", ".join(sorted(ALLOWED_MODULES)),
        "data_start": str(mk.dates.min().date()), "data_end": str(mk.dates.max().date()),
        "n_codes": len(mk.codes), "n_days": len(mk.dates),
        "fields": list(mk.fields.keys()),
    })


@app.get("/factors/{factor_id}", response_class=HTMLResponse)
def page_factor_detail(request: Request, factor_id: int):
    s = get_session()
    try:
        f = s.get(Factor, factor_id)
        if f is None:
            raise HTTPException(404)
        metrics = s.query(FactorMetric).filter_by(factor_id=factor_id).all()
        # 每个持有期一份 {IC/分层 诊断 + 分层净值附属统计量}
        diag = {m.horizon: (m.summary_json or {}).get(str(m.horizon), {}) for m in metrics}
        policy = store.factor_data_policy(f)
        spec = f.preprocess_spec
    finally:
        s.close()
    mk = store.market()
    return templates.TemplateResponse(request, "factor_detail.html", {
        "f": f, "diag": diag, "policy": policy, "spec": spec,
        "spec_schema": PREPROCESS_SCHEMA, "spec_steps": spec_steps(spec),
        "pools": store.list_pools(),
        "data_start": str(mk.dates.min().date()), "data_end": str(mk.dates.max().date()),
        "n_codes": len(mk.codes),
        "ly_prefix": "fac",
    })


@app.get("/signals", response_class=HTMLResponse)
def page_signals(request: Request):
    """策略库 = 信号列表。信号是纯权重定义，可像因子一样做诊断，可被回测多次。"""
    return templates.TemplateResponse(request, "signals.html",
                                      {"signals": store.list_signals()})


@app.get("/signals/new", response_class=HTMLResponse)
def page_signal_new(request: Request, factors: str = ""):
    s = get_session()
    try:
        facs = s.query(Factor).all()
        # 不再硬编码 horizon=20（AlphaGen 因子常只有 h=5）：每个因子取其实际存在
        # 的最大 horizon 诊断，展示时随附期数
        mets = {}
        for m in s.query(FactorMetric).all():
            cur = mets.get(m.factor_id)
            if cur is None or m.horizon > cur.horizon:
                mets[m.factor_id] = m
    finally:
        s.close()
    mk = store.market()
    # 从因子库「构成策略」带过来的预选因子
    preselect = [int(x) for x in str(factors).split(",") if str(x).strip().isdigit()]
    # 因子值的最晚起始覆盖：页面据此提示"模型预热窗口够不够"，而不是写死一句话
    starts = [str(f.value_start.date()) for f in facs if f.value_start]
    return templates.TemplateResponse(request, "signal_new.html", {
        "factors": facs, "mets": mets, "preselect": preselect,
        "fac_min_start": max(starts) if starts else None,
        "data_start": str(mk.dates.min().date()),
        "data_end": str(mk.dates.max().date()),
        "n_codes": len(mk.codes),
    })


@app.get("/signals/{sid}", response_class=HTMLResponse)
def page_signal_detail(request: Request, sid: int):
    sig = store.get_signal(sid)
    if sig is None:
        raise HTTPException(404)
    s = get_session()
    try:
        fac_names = {f.id: f.name for f in s.query(Factor).all()}
    finally:
        s.close()
    diag = sig.get("diagnostics_json") or {}
    try:
        gain_report = store.combination_report(sid)
    except Exception as e:  # noqa: BLE001 —— 组合报告失败不影响详情页
        gain_report = {"error": f"{type(e).__name__}: {e}"}
    # 权重随时间变化（walk-forward 权重类信号的可解释性核心）
    best_h = None
    try:
        bh = store.signal_best_horizon(sid)
        best_h = {"best_h": bh.get("best_h"), "half_life": bh.get("half_life"),
                  "curve": bh.get("curve")}
    except Exception:  # noqa: BLE001
        best_h = None
    weight_matrix = None
    try:
        latest_weights = store.signal_latest_weights(sid)
    except Exception:  # noqa: BLE001
        latest_weights = None
    try:
        w = store.signal_weight_matrix(sid)
        if w is not None and len(w.index) > 0:
            step = max(1, len(w) // 300)
            w = w.iloc[::step]
            weight_matrix = {"index": [str(d) for d in w.index],
                             "columns": list(w.columns),
                             "series": [[None if not np.isfinite(x) else round(float(x), 4)
                                         for x in row] for row in w.to_numpy()]}
    except Exception as e:  # noqa: BLE001
        weight_matrix = {"error": f"{type(e).__name__}: {e}"}
    return templates.TemplateResponse(request, "signal_detail.html", {
        "sig": sig, "fac_names": fac_names, "diag": diag, "gain_report": gain_report,
        "weight_matrix": weight_matrix, "latest_weights": latest_weights,
        "best_h": best_h,
        "backtests": store.list_backtests(signal_id=sid),
        "pools": store.list_pools(),
        "factor_ids": [c["factor_id"] for c in (sig.get("components") or [])],
        "ly_prefix": "sig",
    })


# 说明：信号的分层净值现在是诊断的附属统计量（diagnose_signal 里随 IC 一起算），
# 与因子页共用 factors/layers.py 与同一个图表块；服务端不再有"跑分层回测"入口，
# /api/signals/{id}/layered/{key} 与 /api/layered/{kind}/... 一并下线。


@app.get("/backtests", response_class=HTMLResponse)
def page_backtests(request: Request):
    """回测记录：一个信号可对应多条。"""
    rows = []
    for b in store.list_backtests():
        sig = store.get_signal(b["signal_id"]) or {}
        m = (b.get("metrics_json") or {}).get("net") or {}
        rows.append({**b, "signal_name": sig.get("name", f"#{b['signal_id']}"),
                     "annualized_return": m.get("annualized_return"),
                     "sharpe": m.get("annualized_sharpe"),
                     "max_drawdown": m.get("max_drawdown"),
                     "all_pass": (b.get("checks_json") or {}).get("all_pass")})
    return templates.TemplateResponse(request, "backtests.html", {"rows": rows})


@app.get("/backtests/new", response_class=HTMLResponse)
def page_backtest_new(request: Request, signal_id: int | None = None):
    from ..config import load_config
    cfg = load_config()
    mk = store.market()
    sig = store.get_signal(signal_id) if signal_id else None
    return templates.TemplateResponse(request, "backtest_new.html", {
        "signals": store.list_signals(), "pools": store.list_pools(),
        "sig": sig, "cost": cfg.backtest["cost"],
        "portfolio": cfg.backtest["portfolio"],
        "last": store.last_backtest_config(),
        "data_start": str(mk.dates.min().date()), "data_end": str(mk.dates.max().date()),
        "initial_cash": cfg.backtest.get("initial_cash", 1_000_000),
    })


@app.get("/backtests/compare", response_class=HTMLResponse)
def page_backtest_compare(request: Request, a: int | None = None, b: int | None = None,
                          ids: list[str] | None = Query(default=None)):
    # 勾选式表单提交重复的 ids= 参数（GET 多选）；兼容逗号分隔与 a/b 两条形态
    raw = ids or []
    id_list = [int(x) for chunk in raw for x in str(chunk).split(",") if x.strip()]
    if not id_list and a is not None and b is not None:
        id_list = [a, b]
    if len(id_list) < 2:
        raise HTTPException(422, "至少需要两条回测")
    data = store.compare_backtests_multi(id_list)
    data["raw"] = {f"R{i + 1}": store.backtest_result(rid) or {}
                   for i, rid in enumerate(id_list)}
    data["cmp_data"] = {"runs": data["runs"], "aligned": data["aligned"]}
    sig_names = {s["id"]: s["name"] for s in store.list_signals()}
    for i, rm in enumerate(data["runs"]):
        rm["signal_name"] = sig_names.get(rm.get("signal_id"), "")
    return templates.TemplateResponse(request, "backtest_compare.html", data)


@app.get("/backtests/{bid}", response_class=HTMLResponse)
def page_backtest_detail(request: Request, bid: int):
    run = store.backtest_result(bid)
    if run is None:
        raise HTTPException(404)
    from ..config import load_config
    cfg = load_config()
    chart_json = {k: run.get(k) for k in
                  ("nav", "nav_gross", "benchmark", "turnover", "cost_series") if run.get(k)}
    runs = store.list_backtests()
    meta = next((b for b in runs if b["id"] == bid), {})
    sig = store.get_signal(meta.get("signal_id")) if meta.get("signal_id") else None
    return templates.TemplateResponse(request, "backtest_detail.html", {
        "res": run, "meta": meta, "sig": sig,
        "nav": run.get("nav", {"index": [], "values": []}),
        "rf_annual": float(cfg.data["risk_free_annual"]),
        "gn": (run.get("checks") or {}).get("gross_net_attribution") or {},
        "gates": _gate_rows(run.get("checks") or {}),
        "chart_json": chart_json,
    })


@app.get("/strategies", response_class=HTMLResponse)
def page_strategies_redirect():
    return RedirectResponse("/signals", status_code=307)


@app.get("/lab", response_class=HTMLResponse)
def page_lab(request: Request):
    s = get_session()
    try:
        tasks = s.query(AlgoTask).order_by(AlgoTask.created_at.desc()).all()
        from sqlalchemy import func as _f
        counts = dict(s.query(AlgoCandidate.task_id, _f.count(AlgoCandidate.id))
                      .group_by(AlgoCandidate.task_id).all())
    finally:
        s.close()
    mk = store.market()
    _ = mk          # 页面不再展示本地行情说明，保留读取以便口径变更时报警
    return templates.TemplateResponse(request, "lab.html",
                                      {"tasks": tasks, "cand_counts": counts,
                                       "engines": store.engines(),
                                       "schema": store.GP_PARAMS_SCHEMA,
                                       "ag_schema": store.ALPHAGEN_PARAMS_SCHEMA,
                                       "spec_schema": PREPROCESS_SCHEMA,
                                       "pools": store.list_pools(),
                                       "last": store.last_task_params(),
                                       "data_start": str(mk.dates.min().date()),
                                       "data_end": str(mk.dates.max().date()),
                                       # 自己划分的默认三段（60%/80% 分割点）：
                                       # 历史陷阱：三个日期默认值全是数据起点，用户切到
                                       # 「自己划分」直接提交会 purge 后空段而 FAILED
                                       "data_split1": str(mk.dates[int(len(mk.dates) * 0.6)].date()),
                                       "data_split2": str(mk.dates[int(len(mk.dates) * 0.8)].date())})


@app.get("/lab/tasks/{task_id}", response_class=HTMLResponse)
def page_lab_task(request: Request, task_id: str):
    s = get_session()
    try:
        t = s.get(AlgoTask, task_id)
        if t is None:
            raise HTTPException(404)
        cands = s.query(AlgoCandidate).filter_by(task_id=task_id).order_by(AlgoCandidate.train_ic.desc()).all()
    finally:
        s.close()
    return templates.TemplateResponse(request, "lab_task.html", {
        "t": t, "cands": cands,
        # curve 单独喂给图表，避免在"指标"里重复打印整段训练曲线
        "metrics_wo_curve": {k: v for k, v in (t.metrics_json or {}).items() if k != "curve"},
    })


# 说明：原「复现包 /repro」「运行记录 /runs」两个页面已按需求从前端下线。
# 复现信息仍完整保留在研究报告（reports/研究报告.md|pdf）与 reports/runs/<id>/manifest.json 中，
# 机器可读接口保留在 /api/runs（供外部脚本与复核使用）。


# ---------------- 表单动作 ----------------
@app.post("/signals/new")
def action_create_signal(request: Request, name: str = Form(...),
                         description: str = Form(""),
                         factor_ids: list[int] = Form(...),
                         model_type: str = Form("equal_weight"),
                         start_date: str = Form(...),
                         end_date: str = Form(...),
                         refit_days: int = Form(20),
                         train_window: int = Form(252),
                         purge: int = Form(5)):
    if not 1 <= len(factor_ids) <= 10:
        raise HTTPException(422, "需选择 1~10 个因子")
    params = {}
    if model_type in ("linear", "tree"):
        params = {"refit_days": int(refit_days), "train_window": int(train_window),
                  "purge": int(purge)}
    payload = dict(name=name, description=description, factor_ids=factor_ids,
                   model_type=model_type, model_params=params,
                   start_date=start_date, end_date=end_date)
    # 异步化：walk-forward 拟合在后台线程执行，HTTP 立即返回进度页（前端不卡）
    job_id = store.start_simple_job("create_signal", payload,
                                    lambda: store.create_signal(**payload))
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/backtests/new")
def action_run_backtest(request: Request, signal_id: int = Form(...),
                        start_date: str = Form(...), end_date: str = Form(...),
                        pool_id: int = Form(...), top_n: int = Form(10),
                        weighting: str = Form("equal_weight"),
                        rebalance_freq: str = Form("weekly"),
                        initial_cash: float = Form(1_000_000.0),
                        commission_buy: float = Form(...),
                        commission_sell: float = Form(...),
                        slippage: float = Form(0.0),
                        industry_neutral: str = Form(""),
                        vol_target_enabled: str = Form(""),
                        vol_target: float = Form(0.15),
                        vol_window: int = Form(60)):
    payload = dict(signal_id=signal_id, start_date=start_date, end_date=end_date,
                   pool_id=pool_id or None, top_n=top_n, weighting=weighting,
                   rebalance_freq=rebalance_freq, initial_cash=initial_cash,
                   industry_neutral=bool(industry_neutral),
                   vol_target=(float(vol_target) if vol_target_enabled else None),
                   vol_window=int(vol_window),
                   cost_override={"commission_buy": commission_buy,
                                  "commission_sell": commission_sell,
                                  "slippage": slippage})
    # 异步化：逐日模拟成交移出请求路径，立即返回进度页
    job_id = store.start_simple_job("backtest", payload,
                                    lambda: store.run_backtest_for_signal(**payload))
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/factors/new")
def action_create_factor(request: Request, name: str = Form(...),
                         code: str = Form(...), description: str = Form(""),
                         hypothesis: str = Form(""), direction: str = Form(""),
                         missing_policy: str = Form(""),
                         failure_modes: str = Form(""),
                         adjust: str = Form("hfq")):
    from ..factors.user_code import FactorCodeError
    fm = [x.strip() for x in failure_modes.split(";") if x.strip()]
    try:
        r = store.register_user_factor(
            name=name, code=code, description=description, hypothesis=hypothesis,
            direction=direction, missing_policy=missing_policy, failure_modes=fm,
            adjust=adjust if adjust in ("hfq", "qfq") else "hfq")
    except FactorCodeError as e:
        # 代码错误回显给用户（含行内错误详情），不写入库
        return HTMLResponse(
            "<h3>因子代码有问题，未入库</h3>"
            f"<pre class='log' style='max-height:260px'>{e}</pre>"
            "<p><a href='/factors/new'>返回修改</a></p>", status_code=400)
    except Exception as e:  # noqa: BLE001
        return HTMLResponse(
            f"<h3>登记失败：{type(e).__name__}: {e}</h3><a href='/factors/new'>返回</a>",
            status_code=400)
    return RedirectResponse(f"/factors/{r['id']}", status_code=303)


@app.get("/api/factors/{factor_id}/pool-impact")
def api_factor_pool_impact(factor_id: int, pool_id: int | None = None):
    """换池影响预览（只读）：用于前端在提交前展示"截面会变"的警告。"""
    try:
        return store.factor_pool_impact(factor_id, pool_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}")


@app.post("/factors/{factor_id}/refresh")
async def action_refresh_factor(factor_id: int, request: Request):
    """单因子重算（因子详情页）：可改股票池、可显式指定区间、可同时改参数（含复权方式）。

    重算要跑 IC + 分层净值（数秒级 CPU 活），因此必须丢到线程池执行：
    在 async 端点里直接调同步重活会**卡死整个事件循环**，期间所有请求（含静态文件）都没人处理。
    """
    form = await request.form()
    spec = _spec_from_form(form) if form.get("spec_mode") == "set" else None
    pool_id = None
    if "pool_id" in form:
        raw = form.get("pool_id")
        pool_id = int(raw) if str(raw).strip() else 0     # '' -> 0 = 明确不过滤
    try:
        await run_in_threadpool(
            store.refresh_factor, factor_id, pool_id=pool_id,
            confirm_pool_change=bool(form.get("confirm_pool_change")),
            spec=spec,
            start=form.get("start_date") or None, end=form.get("end_date") or None)
    except Exception as e:  # noqa: BLE001
        return HTMLResponse(
            f"<h3>重算失败：{e}</h3><a href='/factors/{factor_id}'>返回</a>", status_code=400)
    return RedirectResponse(f"/factors/{factor_id}", status_code=303)


# 说明：参数保存与「重算」合并成同一个入口 POST /factors/{id}/refresh
# （因子详情页那张参数总表就是表单，改完点「保存并重算」一次提交）。
# 不再单独提供 /factors/{id}/spec，避免同一件事两个入口。


@app.post("/factors/batch-refresh")
async def action_batch_refresh(request: Request):
    """批量重算：选中的因子 × (计算区间 / 股票池)。**后台执行**，立即返回任务页。

    重算 N 个因子要跑 N×(IC + 分层净值)，是分钟级的 CPU 活。放在请求里同步跑会让浏览器
    一直挂着（还可能被超时掐断），因此改成后台线程 + 轮询进度。
    只做「区间 / 股票池」两件事 —— 预处理参数属于单因子的研究设定，在因子详情页逐因子改。
    """
    form = await request.form()
    ids = [int(x) for x in form.getlist("factor_ids") if str(x).strip()]
    if not ids:
        return HTMLResponse("<h3>没有选中任何因子</h3><a href='/factors'>返回因子库</a>",
                            status_code=400)
    pool_mode = form.get("pool_mode", "keep")
    if pool_mode == "set":
        raw = form.get("pool_id")
        pool_id = int(raw) if str(raw or "").strip() else 0   # '' = 明确不过滤
    else:
        pool_id = None                                        # 保持各因子原有池
    range_mode = form.get("range_mode", "auto")
    job_id = store.start_batch_job(
        "refresh", ids,
        pool_id=pool_id, pool_mode=pool_mode,
        confirm_pool_change=bool(form.get("confirm_pool_change")),
        start=(form.get("start_date") or None) if range_mode == "set" else None,
        end=(form.get("end_date") or None) if range_mode == "set" else None,
        range_mode=range_mode)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/factors/batch-delete")
async def action_batch_delete(request: Request):
    """批量删除因子（被信号引用时默认拒绝，需显式 force 才级联删除）。后台执行。"""
    form = await request.form()
    ids = [int(x) for x in form.getlist("factor_ids") if str(x).strip()]
    if not ids:
        return HTMLResponse("<h3>没有选中任何因子</h3><a href='/factors'>返回因子库</a>",
                            status_code=400)
    job_id = store.start_batch_job("delete", ids, force=bool(form.get("force")))
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


# 说明：「构成策略」不设服务端路由 —— 之前放在 /factors/compose，被先注册的
# /factors/{factor_id} 抢先匹配，于是把 "compose" 当 factor_id 解析，报 int_parsing。
# 现在由列表页直接跳 /signals/new?factors=<ids>，少一个路由也少一处冲突。


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def page_job(request: Request, job_id: str):
    """后台任务进度页（轮询 /api/jobs/{id}，完成后自动跳到结果页）。"""
    job = store.get_batch_job(job_id)
    if job is None:
        raise HTTPException(404)
    return templates.TemplateResponse(request, "job.html", {"job": job})


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = store.get_batch_job(job_id)
    if job is None:
        raise HTTPException(404)
    return job


@app.get("/jobs/{job_id}/done", response_class=HTMLResponse)
def page_job_done(request: Request, job_id: str):
    """任务完成后渲染结果（复用批量结果模板）。"""
    job = store.get_batch_job(job_id)
    if job is None:
        raise HTTPException(404)
    if job["status"] not in ("SUCCESS", "FAILED"):
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)
    res = (job.get("result_json") or {})
    # 信号/回测任务完成 → 直接跳结果实体页
    if job["kind"] == "create_signal" and job["status"] == "SUCCESS":
        return RedirectResponse(f"/signals/{res['result']['id']}", status_code=303)
    if job["kind"] == "backtest" and job["status"] == "SUCCESS":
        return RedirectResponse(f"/backtests/{res['result']['id']}", status_code=303)
    ctx = {"mode": job["kind"], "res": res, "ids": (job.get("params") or {}).get("ids", []),
           "job": job}
    p = job.get("params") or {}
    ctx.update({k: p.get(k) for k in ("pool_id", "pool_mode", "range_mode", "start", "end")})
    return templates.TemplateResponse(request, "factor_batch_result.html", ctx)


@app.get("/api/factors/{factor_id}/compare")
def api_factor_compare(factor_id: int, spec: str = ""):
    """预处理效果对照：把「当前 spec」与「指定 spec」下的因子值/诊断并列返回。

    用于因子页直接验证参数是否真的生效（不写库、只读）。
    """
    import json as _json

    from ..factors.diagnostics import run_diagnostics
    from ..factors.preprocess import apply_preprocess_spec

    s = get_session()
    try:
        f = s.get(Factor, factor_id)
        if f is None:
            raise HTTPException(404)
        raw = pd.read_parquet(f.values_path) if f.values_path else None
        cur = f.preprocess_spec
        pool_id = f.pool_id
    finally:
        s.close()
    if raw is None:
        raise HTTPException(404, "该因子没有因子值文件")
    try:
        alt = normalize_spec(_json.loads(spec)) if spec else cur
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"spec 解析失败: {e}")
    mk = store.market()
    scoped, _ = store._apply_pool(raw, pool_id, mk)  # noqa: SLF001
    from ..factors.preprocess import normalize_spec as _ns
    from ..factors.preprocess import parse_horizons as _ph
    alt_n = _ns(alt)
    cfg = {"horizon_days": _ph(alt_n["horizons"]),
           "min_cross_section_samples": int(alt_n["min_n"]),
           "quantile_groups": int(alt_n["quantile"]),
           "ic_method": str(alt_n["ic_method"])}
    out = {}
    for label, sp in (("current", cur), ("alternative", alt)):
        p = apply_preprocess_spec(scoped, sp)
        d = run_diagnostics(p, mk.close_adj, cfg)
        out[label] = {
            "spec": sp, "steps": spec_steps(sp),
            "diagnostics": {
                h: {"ic_summary": r["ic_summary"], "quantile_summary": r["quantile_summary"]}
                for h, r in d.items()},
            "sample": {c: [None if not np.isfinite(x) else round(float(x), 6) for x in p[c].head(120)]
                       for c in list(p.columns[:5])},
            "sample_index": [str(x.date()) for x in p.index[:120]],
        }
    return out


# ---------------- 股票池（券商风格行情页） ----------------
@app.get("/pools", response_class=HTMLResponse)
def page_pools(request: Request):
    return templates.TemplateResponse(request, "pools.html",
                                      {"pools": store.list_pools_with_stats()})


@app.get("/pools/{pool_id}", response_class=HTMLResponse)
def page_pool_detail(request: Request, pool_id: int):
    ov = store.pool_overview(pool_id)
    if ov is None:
        raise HTTPException(404)
    lo, hi = store.pool_date_bounds(pool_id)
    from ..data import fundamentals as _fund
    return templates.TemplateResponse(request, "pool_detail.html", {
        "ov": ov, "pool": ov["pool"], "date_lo": lo, "date_hi": hi,
        "has_fundamentals": _fund.available(),
        "fund_note": _fund.coverage_note(),
    })


@app.get("/api/pools/{pool_id}/members")
def api_pool_members(pool_id: int, date: str | None = None, sort: str = "total_mv",
                     order: str = "desc", limit: int = 0):
    d = store.pool_members(pool_id, date, sort=sort, order=order, limit=limit)
    if d is None:
        raise HTTPException(404)
    return d


@app.get("/api/pools/{pool_id}/overview")
def api_pool_overview(pool_id: int):
    d = store.pool_overview(pool_id)
    if d is None:
        raise HTTPException(404)
    return d


@app.get("/api/stocks/{code}/ohlc")
def api_stock_ohlc(code: str, days: int = 250, end: str | None = None):
    """单只股票的日线 OHLC（复权）+ 成交额，用于池内个股 K 线。"""
    mk = store.market()
    c = str(code).zfill(6)
    if c not in set(mk.codes):
        raise HTTPException(404, f"本地行情没有 {c}")
    idx = mk.dates
    if end:
        idx = idx[idx <= pd.Timestamp(end)]
    idx = idx[-int(days):] if days else idx
    src = {"open": mk.open_adj, "high": mk.high_adj, "low": mk.low_adj,
           "close": mk.close_adj, "volume": mk.volume, "amount": mk.amount}
    series = {}
    for k, df in src.items():
        v = df[c].reindex(idx)
        series[k] = [None if not np.isfinite(x) else round(float(x), 4) for x in v]
    ret = mk.close_adj[c].reindex(idx).pct_change()
    from ..data import fundamentals as _fund
    return {"code": c, "name": _fund.name_of(c),
            "index": [str(d.date()) for d in idx],
            "ohlc": [[series["open"][i], series["close"][i], series["low"][i], series["high"][i]]
                     for i in range(len(idx))],
            "volume": series["volume"], "amount": series["amount"],
            "close": series["close"],
            "pct_chg": [None if not np.isfinite(x) else round(float(x), 6) for x in ret],
            "convention": "价格为后复权（P_adj = P_raw × a(t)/a(τ)），成交额单位为元"}


@app.get("/api/pools/{pool_id}/heat")
def api_pool_heat(pool_id: int, date: str | None = None, top: int = 12):
    """池内当日领涨/领跌 + 行业分布（页面右侧面板用）。"""
    d = store.pool_members(pool_id, date, sort="pct_chg", order="desc")
    if d is None:
        raise HTTPException(404)
    rows = [r for r in d["rows"] if r["pct_chg"] is not None]
    by_ind: dict[str, int] = {}
    for r in d["rows"]:
        by_ind[r["industry"] or "未知"] = by_ind.get(r["industry"] or "未知", 0) + 1
    return {"as_of": d["as_of"], "breadth": d["breadth"],
            "gainers": rows[: int(top)], "losers": rows[-int(top):][::-1],
            "industry_dist": sorted(({"industry": k, "n": v} for k, v in by_ind.items()),
                                    key=lambda x: -x["n"])}


@app.post("/lab/tasks", status_code=202)
def action_create_task(payload: dict = Body(...)):
    """统一入口：按 engine_id 分发（gp_daily / alphagen_daily_cpu）。"""
    engine_id = str(payload.get("engine_id") or "gp_daily")
    params = {k: v for k, v in payload.items() if k != "engine_id"}
    try:
        task_id = store.create_task(engine_id, params)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return JSONResponse({"task_id": task_id, "engine_id": engine_id}, status_code=202)


@app.post("/lab/tasks/{task_id}/cancel")
def action_cancel_task(task_id: str):
    ok = store.cancel_task(task_id)
    return {"cancelled": ok}


class AdoptRequest(BaseModel):
    task_id: str
    candidate_ids: list[int]


@app.post("/lab/adopt")
def action_adopt(payload: AdoptRequest):
    """人工采纳候选 → 因子库可见（若该候选已被任务自动隐藏入库，则转为可见）。

    沿用任务创建时填的「因子计算参数」，保证入库口径与挖矿时一致。
    """
    s = get_session()
    try:
        t = s.get(AlgoTask, payload.task_id)
        spec = store.task_spec(t.params) if t is not None else None
        pool_id = t.params.get("pool_id") if t is not None else None
        st = t.params.get("start_date") if t is not None else None
        en = t.params.get("end_date") if t is not None else None
    finally:
        s.close()
    out = store.adopt_candidates(payload.task_id, payload.candidate_ids,
                                 hidden=False, spec=spec,
                                 pool_id=pool_id, start=st, end=en)
    return JSONResponse({"adopted": out})


# ---------------- 运维 ----------------
@app.post("/api/admin/cache/reset")
def api_cache_reset():
    """重跑流水线/清洗后调用：让 Web 进程重新加载行情面板与股票池掩码。"""
    return store.reset_caches()


# ---------------- JSON API ----------------
@app.get("/api/factors")
def api_factors():
    s = get_session()
    try:
        fs = s.query(Factor).all()
        mets = s.query(FactorMetric).all()
        by_f: dict[int, dict] = {}
        for m in mets:
            by_f.setdefault(m.factor_id, {})[m.horizon] = m.summary()
        return [{"id": f.id, "name": f.name, "source": f.source, "expression": f.expression,
                 "hash": f.hash_code, "metrics": by_f.get(f.id, {})} for f in fs]
    finally:
        s.close()


@app.get("/api/factors/{factor_id}/series")
def api_factor_series(factor_id: int):
    v = store.factor_values(factor_id)
    s = get_session()
    try:
        metrics = s.query(FactorMetric).filter_by(factor_id=factor_id).all()
        ic = {m.horizon: (m.summary_json or {}).get(str(m.horizon), {}).get("ic_series") for m in metrics}
    finally:
        s.close()
    # 原实现做 v.ffill() 后再展示，却标注为"因子原始值"：把缺口填成了上一个有效值，
    # 与项目"缺失值不填零/不填充"的口径冲突。此处保持缺口为 null，由前端断线展示。
    sample = v.iloc[:, :5]
    return {"factor_id": factor_id,
            "values_sample": {"index": [str(d.date()) for d in sample.index],
                              "series": {c: [None if not np.isfinite(x) else round(float(x), 5)
                                             for x in sample[c]] for c in sample.columns}},
            "ic_series": ic}


def _finite(v):
    """递归把非有限浮点（NaN/Inf）替换为 None —— Starlette JSONResponse 禁 NaN，
    训练曲线早期轮次常含 NaN（如 entropy），不清洗会让轮询接口 500。"""
    import math as _math
    if isinstance(v, float):
        return v if _math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    return v


@app.get("/api/lab/tasks/{task_id}")
def api_task_status(task_id: str, log_since: int = 0, curve_max: int = 300):
    """任务状态轮询（增量友好）。

    - log_since=已收到的日志字符数：只回传其后的增量（log_new/log_size），
      避免训练日志越跑越大时每秒全量传输（前端卡顿来源之一）；
    - curve_max：曲线最多返回多少点（服务端等步降采样）。
    """
    s = get_session()
    try:
        t = s.get(AlgoTask, task_id)
        if t is None:
            raise HTTPException(404)
        cands = s.query(AlgoCandidate).filter_by(task_id=task_id).count()
        d = t.to_dict()
        d["n_candidates"] = cands
        full_log = t.log or ""
        since = max(0, int(log_since or 0))
        d["log_size"] = len(full_log)
        d["log_new"] = full_log[since:]
        d.pop("log", None)          # 不再全量回传
        curve = (t.metrics_json or {}).get("curve", [])
        step = max(1, len(curve) // max(1, curve_max))
        d["curve_step"] = step
        d["curve_total"] = len(curve)
        d["curve"] = _finite(curve[::step])
        d["metrics_json"] = _finite(d.get("metrics_json"))
        return d
    finally:
        s.close()


@app.get("/api/factors/{factor_id}/decay")
def api_factor_decay(factor_id: int):
    """IC 衰减曲线 + 半衰期（调仓频率的量化依据）。"""
    return store.factor_decay(factor_id)


@app.get("/experiments", response_class=HTMLResponse)
def page_experiments(request: Request):
    """组合方式对照实验：一键跑同因子集 × 全组合模型的对照。"""
    mk = store.market()
    s = get_session()
    try:
        facs = s.query(Factor).all()
    finally:
        s.close()
    exps = sorted((ROOT / "reports" / "combination_experiments").glob("*.json"),
                  reverse=True)[:20] if (ROOT / "reports" / "combination_experiments").exists() else []
    experiments = []
    for pth in exps:
        try:
            d = json.loads(pth.read_text(encoding="utf-8"))
            d["filename"] = pth.stem
            experiments.append(d)
        except Exception:  # noqa: BLE001
            pass
    return templates.TemplateResponse(request, "experiments.html", {
        "factors": facs, "experiments": experiments,
        "data_start": str(mk.dates.min().date()), "data_end": str(mk.dates.max().date()),
    })


@app.post("/experiments/{filename}/rerun")
def action_experiment_rerun(filename: str):
    """复跑一次对照实验：参数取自已存 JSON（同因子集/区间/h，数据取当前快照）。"""
    from ..lab.comparison import compare_combinations
    p_ = ROOT / "reports" / "combination_experiments" / f"{filename}.json"
    if not p_.exists():
        raise HTTPException(404)
    old = json.loads(p_.read_text(encoding="utf-8"))
    res = compare_combinations([int(i) for i in old["factor_ids"]],
                               old["start"], old["end"],
                               horizon=int(old["horizon"]),
                               include_backtest=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / "combination_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cmp_{stamp}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    return {"saved": out.name, "models": list(res.get("models", {}))}


@app.get("/experiments/{filename}.csv")
def experiments_csv(filename: str):
    """对照实验结果导出 CSV（filename 不含扩展名，防目录穿越）。"""
    from fastapi.responses import PlainTextResponse
    import re as _re
    if not _re.fullmatch(r"cmp_[0-9A-Za-z_]+", filename):
        raise HTTPException(400, "非法文件名")
    p_ = ROOT / "reports" / "combination_experiments" / f"{filename}.json"
    if not p_.exists():
        raise HTTPException(404)
    data = json.loads(p_.read_text(encoding="utf-8"))
    rows = ["model,rank_ic_mean,t_naive,t_nw,n_obs,"
            "bt_annualized_return,bt_sharpe,bt_max_drawdown,"
            "bt_annualized_return_weekly,bt_annualized_return_monthly"]
    for m, v in (data.get("models") or {}).items():
        if "error" in v:
            rows.append(f"{m},ERROR")
            continue
        def g(k):
            x = v.get(k)
            return "" if x is None else f"{x:.6f}"
        rows.append(",".join([m, g("rank_ic_mean"), g("t_naive"), g("t_nw"),
                              str(v.get("n_obs", "")), g("bt_annualized_return"),
                              g("bt_sharpe"), g("bt_max_drawdown"),
                              g("bt_annualized_return_weekly"),
                              g("bt_annualized_return_monthly")]))
    csv = chr(10).join(rows) + chr(10)
    return PlainTextResponse(csv, media_type="text/csv",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{filename}.csv"'})


@app.post("/experiments/run")
def action_run_experiment(request: Request, factor_ids: str = Form(...),
                          start_date: str = Form(...), end_date: str = Form(...),
                          horizon: int = Form(5),
                          include_backtest: str = Form("")):
    id_list = [int(x) for x in factor_ids.split(",") if x.strip()]
    if not 2 <= len(id_list) <= 12:
        raise HTTPException(422, "需选择 2~12 个因子")
    from ..lab.comparison import compare_combinations
    res = compare_combinations(id_list, start_date, end_date, horizon=horizon,
                               include_backtest=bool(include_backtest))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / "combination_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cmp_{stamp}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    return RedirectResponse("/experiments", status_code=303)


@app.get("/runs", response_class=HTMLResponse)
def page_runs(request: Request):
    """研究记录：流水线 runs 对比 + 组合方式对照实验历史。"""
    rows = store.list_run_summaries()
    exp_dir = ROOT / "reports" / "combination_experiments"
    experiments = []
    if exp_dir.exists():
        for pth in sorted(exp_dir.glob("*.json"), reverse=True)[:20]:
            try:
                experiments.append(json.loads(pth.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
    return templates.TemplateResponse(request, "runs.html",
                                      {"rows": rows, "experiments": experiments})


@app.get("/api/signals/{signal_id}/latest.csv")
def api_signal_latest_csv(signal_id: int, date: str | None = None):
    """最新交易日信号截面导出 CSV（code,score）—— 策略落地直接可用。"""
    from fastapi.responses import PlainTextResponse
    sig_dict = store.get_signal(signal_id)
    if sig_dict is None:
        raise HTTPException(404)
    s = get_session()
    try:
        obj = s.get(Signal, signal_id)
    finally:
        s.close()
    panel = store.signal_panel(obj, store.market())
    if panel.empty:
        raise HTTPException(404, "信号面板为空")
    if date:
        target = pd.Timestamp(date)
        exact = panel.index[panel.index.normalize() == target]
        if len(exact) == 0:
            raise HTTPException(404, f"{date} 不是有效交易日或不在信号区间内")
        panel = panel.loc[exact]
    last = panel.iloc[-1].dropna().sort_values(ascending=False)
    lines = ["code,score,date"] + [
        f"{code},{score:.6f},{panel.index[-1].date()}" for code, score in last.items()]
    nl = chr(10)
    csv = nl.join(lines) + nl
    return PlainTextResponse(csv, media_type="text/csv",
                             headers={"Content-Disposition":
                                      f'attachment; filename="signal{signal_id}_latest.csv"'})


@app.get("/api/signals/{signal_id}/latest-weights")
def api_signal_latest_weights(signal_id: int):
    """最新一期实际权重（策略落地/复盘）。"""
    r = store.signal_latest_weights(signal_id)
    if r is None:
        raise HTTPException(404, "该信号不是 walk-forward 权重类模型")
    return r


@app.get("/api/factors/decay-compare")
def api_factors_decay_compare(ids: str = "1,2,3"):
    """跨因子 IC 衰减曲线叠加对比（谁的半衰期长、谁的信息更持久）。"""
    id_list = [int(x) for x in ids.split(",") if x.strip()]
    if not 2 <= len(id_list) <= 8:
        raise HTTPException(422, "需 2~8 个因子")
    curves, meta = {}, []
    for fid in id_list:
        rep = store.factor_decay(fid)
        s = store.get_session()
        try:
            from ..storage.db import Factor
            f = s.get(Factor, fid)
            name = f.name if f else f"因子{fid}"
        finally:
            s.close()
        curves[name] = rep["curve"]
        meta.append({"id": fid, "name": name, "half_life": rep["half_life"],
                     "best_h": rep["best_h"]})
    return {"curves": curves, "meta": meta}


@app.get("/api/signals/{signal_id}/weights.csv")
def api_signal_weights_csv(signal_id: int):
    """walk-forward 权重矩阵导出 CSV（date, factor_*, weight；Σ|w|=1，t 日只用 t-h 前信息）。"""
    from fastapi.responses import PlainTextResponse
    w = store.signal_weight_matrix(signal_id)
    if w is None:
        raise HTTPException(404, "该信号不是 walk-forward 权重类模型")
    nl = chr(10)
    csv = "date," + ",".join(str(c) for c in w.columns) + nl
    for d, row in w.iterrows():
        csv += str(d.date() if hasattr(d, "date") else d) + "," +             ",".join("" if not np.isfinite(x) else f"{x:.6f}" for x in row.to_numpy()) + nl
    return PlainTextResponse(csv, media_type="text/csv",
                             headers={"Content-Disposition":
                                      f'attachment; filename="signal{signal_id}_weights.csv"'})


@app.get("/api/factors/incremental-ic")
def api_factors_incremental_ic(ids: str = "", horizon: int = 5):
    """按 ids 顺序做 Schmidt 正交化，返回各因子的【增量 RankIC】——
    排后面的因子只保留前面解释不掉的信息，其 IC 即增量贡献。
    用于回答"这个因子在已有因子之外还提供了多少新信息"。"""
    id_list = [int(x) for x in ids.split(",") if x.strip()]
    if len(id_list) < 2:
        raise HTTPException(422, "至少 2 个因子")
    from ..factors.combine import ortho_incremental_ic
    panels = {}
    names = {}
    s = get_session()
    try:
        for fid in id_list:
            f = s.get(Factor, fid)
            if f is None:
                raise HTTPException(404, f"因子 {fid} 不存在")
            names[fid] = f.name
            panels[str(fid)] = store.factor_values(fid)
    finally:
        s.close()
    inc = ortho_incremental_ic(panels, store.market().close_adj,
                               [str(i) for i in id_list], h=horizon, min_n=10)
    return {"order": [{"id": i, "name": names[i]} for i in id_list],
            "incremental_rank_ic": {str(i): (round(v, 6) if np.isfinite(v) else None)
                                    for i, v in inc.items()},
            "note": "顺序=ids 顺序：越靠前保留越多原始信息；增量 IC 为该因子在前面因子之外的新贡献"}


@app.get("/api/factors/correlation")
def api_factor_correlation(ids: str = ""):
    """选中因子的相关矩阵 + 去冗余建议（组合前体检，供 signals/new 热力图）。"""
    id_list = [int(x) for x in ids.split(",") if x.strip()]
    return store.factor_correlation(id_list)


@app.get("/api/signals/{signal_id}/rebalance-hint")
def api_signal_rebalance_hint(signal_id: int):
    """分量最短半衰期 → 调仓频率建议。"""
    return store.signal_rebalance_hint(signal_id)


@app.get("/api/signals/similarity-matrix")
def api_signal_similarity_matrix():
    """全部信号两两相关矩阵（组合前查重）。"""
    return store.signal_similarity_matrix()


@app.get("/api/signals/{signal_id}/best-horizon")
def api_signal_best_horizon_route(signal_id: int):
    return store.signal_best_horizon(signal_id)


@app.get("/api/signals/{signal_id}/combination-report.csv")
def api_combination_report_csv(signal_id: int):
    """组合增益报告导出 CSV（组合/分量/相关性/增益分区呈现）。"""
    from fastapi.responses import PlainTextResponse
    rep = store.combination_report(signal_id)
    nl = chr(10)
    rows = ["section,key,value"]
    rows.append(f"combined,rank_ic_mean_h{rep['horizon']},{rep['combined_rank_ic']}")
    if rep.get("combined_rank_ic_t_nw") is not None:
        rows.append(f"combined,rank_ic_t_nw_h{rep['horizon']},{rep['combined_rank_ic_t_nw']}")
    for c in rep["components"]:
        rows.append(f"component,{c['name']},rank_ic={c['rank_ic']},horizon={c['horizon']}")
    if rep.get("correlation"):
        for k in ("min", "max", "mean_abs"):
            v = rep["correlation"].get(k)
            if v is not None:
                rows.append(f"correlation,{k},{v}")
    g = rep.get("gain") or {}
    rows.append(f"gain,best_single,{g.get('best_single_key')}={g.get('best_single_ic')}")
    rows.append(f"gain,gain_vs_best_single,{g.get('gain_vs_best_single')}")
    rows.append(f"gain,sign_flip,{g.get('sign_flip')}")
    csv = nl.join(rows) + nl
    return PlainTextResponse(csv, media_type="text/csv",
                             headers={"Content-Disposition":
                                      f'attachment; filename="signal{signal_id}_combination.csv"'})


@app.get("/api/signals/{signal_id}/similarity")
def api_signal_similarity(signal_id: int, vs: str = ""):
    """新信号与既有信号的日收益相关性（防策略重复；|ρ|≥0.9 标记“重复”）。"""
    vs_ids = [int(x) for x in vs.split(",") if x.strip()] or None
    return store.signal_similarity(signal_id, vs_ids)


@app.get("/api/signals/{signal_id}/combination-report")
def api_signal_combination_report(signal_id: int, horizon: int | None = None):
    """组合增益报告：组合信号 vs 各分量单因子（同口径 RankIC 对照 + 相关性摘要）。"""
    return store.combination_report(signal_id, horizon)


@app.get("/api/runs")
def api_runs():
    """全部 run 的机器可读摘要（与 /runs 页面同源）。"""
    return store.list_run_summaries()


@app.get("/runs/diff", response_class=HTMLResponse)
def page_runs_diff(request: Request, a: str, b: str):
    runs = ROOT / "reports" / "runs"
    ma_p, mb_p = runs / a / "manifest.json", runs / b / "manifest.json"
    if not ma_p.exists() or not mb_p.exists():
        raise HTTPException(404, "run 不存在")
    ma = json.loads(ma_p.read_text(encoding="utf-8"))
    mb = json.loads(mb_p.read_text(encoding="utf-8"))
    diffs = []
    def walk(prefix, x, y):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(prefix + [str(k)], x.get(k), y.get(k))
        elif x != y:
            diffs.append({"path": ".".join(prefix), "a": x, "b": y})
    walk([], ma, mb)
    return templates.TemplateResponse(request, "runs_diff.html",
                                      {"a": a, "b": b, "diffs": diffs,
                                       "n": len(diffs)})


@app.get("/api/runs/diff")
def api_runs_diff(a: str, b: str):
    """两个 run 的 manifest 差异（配置/环境/数据口径），供复现对照。"""
    runs = ROOT / "reports" / "runs"
    ma_p, mb_p = runs / a / "manifest.json", runs / b / "manifest.json"
    if not ma_p.exists() or not mb_p.exists():
        raise HTTPException(404, "run 不存在")
    ma, mb = json.loads(ma_p.read_text(encoding="utf-8")), json.loads(mb_p.read_text(encoding="utf-8"))
    def walk(prefix, x, y, out):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(prefix + [str(k)], x.get(k), y.get(k), out)
        elif x != y:
            out.append({"path": ".".join(prefix), "a": x, "b": y})
    diffs = []
    walk([], ma, mb, diffs)
    return {"a": a, "b": b, "n_diffs": len(diffs), "diffs": diffs[:80]}


@app.get("/api/runs/latest")
def api_latest_run():
    run = store.latest_run_dir()
    if run is None:
        raise HTTPException(404, "尚无运行记录，请先执行 scripts/run_pipeline.py")
    mf = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    return {"run_id": run.name, "manifest": mf}
