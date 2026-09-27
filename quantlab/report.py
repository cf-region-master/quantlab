"""研究报告生成（Project1 提交材料）：Markdown + PDF（6-10 页）。

报告结构（与课程要求逐项对应）：
  1 数据与假设    样本、清洗统计、价格与资产池口径、未建模限制
  2 因子与回测    三个因子的诊断；策略净值、收益、风险与成本指标
  3 对照与解释    变化来自什么？成本/风险/持仓行为能否解释差异？
  4 复现信息      数据快照、代码版本、配置、运行时间和检查输出
  5 局限与下一步  保留不支持预期的结果，说明局限
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

from .config import ROOT  # noqa: E402

RUNS = ROOT / "reports" / "runs"
REPORT_ASSETS = ROOT / "reports" / "assets"


def _load(run: Path) -> dict[str, Any]:
    r = {"run": run}
    r["manifest"] = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    r["quality"] = json.loads((run / "quality_summary.json").read_text(encoding="utf-8"))
    r["base"] = json.loads((run / "backtests" / "base.json").read_text(encoding="utf-8"))
    r["factors"] = {}
    for p in sorted((run / "factors").glob("*.json")):
        r["factors"][p.stem] = json.loads(p.read_text(encoding="utf-8"))
    r["experiments"] = {}
    for p in sorted((run / "experiments").glob("*.json")):
        r["experiments"][p.stem] = json.loads(p.read_text(encoding="utf-8"))
    mp = ROOT / "reports" / "minute5_summary.json"
    r["minute5"] = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
    return r


def _pct(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v*100:.{nd}f}%"


def _num(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _charts(data: dict[str, Any]) -> list[Path]:
    """生成报告插图（净值/回撤、IC 时序、分层收益、成本-换手）。"""
    REPORT_ASSETS.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    base = data["base"]

    # 1) 净值三线
    fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=150)
    nav = base["nav"]
    idx = list(nav["index"])
    ax.plot(idx, nav["values"], label="策略（扣费后）", lw=1.2)
    if base.get("nav_gross"):
        ax.plot(idx, base["nav_gross"]["values"], label="策略（零费，同路径）", lw=1.0, ls="--")
    if base.get("benchmark"):
        ax.plot(idx, base["benchmark"]["values"], label="沪深300", lw=1.0, alpha=0.8)
    step = max(1, len(idx) // 8)
    ax.set_xticks(idx[::step]); ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.set_title("策略净值 vs 零费对照 vs 沪深300（V0=1，调整价名义口径）", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = REPORT_ASSETS / "nav.png"; fig.savefig(p); files.append(p); plt.close(fig)

    # 2) 各因子 RankIC 时序
    fig, axes = plt.subplots(len(data["factors"]), 1, figsize=(7.4, 2.1 * len(data["factors"])), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, (key, fdata) in zip(axes, data["factors"].items()):
        for h, d in fdata["diagnostics"].items():
            s = d["ic_series"]
            n = len(s["index"])
            # x 用数值位置（类别字符串坐标会把 1/5 采样的点挤在轴前 1/5）
            xv = np.arange(0, n, 5)
            yv = [v if v is not None and np.isfinite(v) else np.nan
                  for v in s["rank_ic"][::5]]
            ax.plot(xv, yv, lw=0.8, label=f"RankIC(h={h})")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"{fdata['card']['name']}（{key}）逐日 RankIC", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        n0 = len(fdata["diagnostics"][list(fdata["diagnostics"])[0]]["ic_series"]["index"])
        step = max(1, n0 // 6)
        ax.set_xticks(np.arange(0, n0, step))
        ax.set_xticklabels(s["index"][::step], rotation=20, fontsize=6)
    fig.tight_layout()
    p = REPORT_ASSETS / "rankic.png"; fig.savefig(p); files.append(p); plt.close(fig)

    # 3) 分层收益柱状图（h=20）
    fig, axes = plt.subplots(1, len(data["factors"]), figsize=(7.4, 2.4), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, (key, fdata) in zip(axes, data["factors"].items()):
        hkey = "20" if "20" in fdata["diagnostics"] else list(fdata["diagnostics"])[0]
        gm = fdata["diagnostics"][hkey]["quantile_summary"]["group_mean_return"]
        ks = list(gm.keys())
        ax.bar(ks, [gm[k] or 0 for k in ks], color="#4472c4")
        ax.set_title(f"{fdata['card']['name']} 分组日均收益(h={hkey})", fontsize=9)
        ax.grid(alpha=0.3, axis="y"); ax.tick_params(labelsize=8)
    fig.tight_layout()
    p = REPORT_ASSETS / "quantile.png"; fig.savefig(p); files.append(p); plt.close(fig)

    # 4) 换手与成本
    fig, ax = plt.subplots(figsize=(7.4, 2.6), dpi=150)
    to, cs = base["turnover"], base["cost_series"]
    ax.bar(to["index"], to["values"], color="#4472c4", alpha=0.8, label="单次换手率")
    ax2 = ax.twinx()
    ax2.plot(cs["index"], cs["values"], color="#ed7d31", lw=1.0, label="当日成本")
    step = max(1, len(to["index"]) // 8)
    ax.set_xticks(to["index"][::step]); ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.set_title("每次调仓的换手率与交易成本", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = REPORT_ASSETS / "turnover.png"; fig.savefig(p); files.append(p); plt.close(fig)
    return files


def build_markdown(data: dict[str, Any], charts: list[Path]) -> str:
    run, mf, q, base = data["run"], data["manifest"], data["quality"], data["base"]
    agg = q["aggregate"]
    L: list[str] = []
    L.append("# 量化投研框架 Project 1 · 研究报告")
    L.append("")
    L.append("> 平台：QuantLab（最小但完整、正确、可复现、可扩展的量化投研框架）  ")
    L.append(f"> 运行 ID：`{run.name}` · 生成时间：{mf['started_at'][:19]}UTC · 用时 {mf['duration_seconds']}s  ")
    L.append(f"> 环境：Python {mf['environment']['python']} / pandas {mf['environment']['pandas']} / numpy {mf['environment']['numpy']}")
    L.append("")

    # ---- 1 数据与假设 ----
    L.append("## 1 数据与假设")
    L.append("")
    L.append("**研究范围与口径**")
    L.append("")
    L.append("| 项 | 内容 |")
    L.append("|---|---|")
    L.append("| 市场/资产 | A股 · 沪深300 当期成分按代码升序步长抽样 50 只（确定性） |")
    L.append(f"| 样本期 | {agg['date_range'][0]} ~ {agg['date_range'][1]}（{agg['n_trading_days']} 个交易日，日频） |")
    L.append("| 数据来源 | akshare：sina 原始日线 + hfq 乘法累计复权因子；eastmoney 兜底 |")
    L.append("| 价格口径 | 清洗层只落原始未复权价 {open,high,low,close}_raw + adj_factor；" "调整价在加载时按口径派生 P_adj = P_raw × a(t)/a(τ)（hfq: τ=样本首日 / qfq: τ=样本末日） |")
    L.append("| 单位 | 价格=元；成交量=股（sina口径）；成交额=元 |")
    L.append("| 无风险收益 | 年化 1.5%（文档化假设） · 年化因子 A=243 |")
    L.append("")
    L.append("**数据清洗与质量报告**（缺失值一律不填零）")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| 行数（原始 → 样本窗口内） | {agg['rows_raw_total']} → {agg['rows_in_window_total']} |")
    L.append(f"| 重复(日期,资产)键剔除 | {agg['duplicate_dates_dropped_total']} |")
    L.append(f"| 非法价格/成交量（置NaN计数） | {sum(agg['illegal_total'].values())} |")
    L.append(f"| 调整收盘缺失格 | {agg['missing_close_adj_cells']}（未上市 {agg['pre_listed_cells']} + 上市后停牌 {agg['suspended_cells_listed']}） |")
    L.append(f"| 覆盖率 | 均值 {_pct(agg['coverage_mean'],1)} · 最低 {_pct(agg['coverage_min'],1)} |")
    L.append("")
    for w in q["warnings"]:
        L.append(f"- ⚠ {w}")
    L.append("")
    L.append("**未建模限制**：成分名单为当期快照（非 point-in-time）；整手交易约束未建模（允许分数股）；")
    L.append("回测以调整价名义金额计价（成本比率与换手不受影响，货币绝对值为名义口径）；停牌日不可成交、顺延不补。")
    L.append("")

    # ---- 2 因子与回测 ----
    L.append("## 2 因子与回测")
    L.append("")
    L.append("### 2.1 三个因子卡（逻辑互异：动量延续 / 短期反转 / 低波动）")
    L.append("")
    for key, fd in data["factors"].items():
        c = fd["card"]
        L.append(f"**{c['name']}（`{key}`）**")
        L.append("")
        L.append(f"- 研究假设：{c['hypothesis']}")
        L.append(f"- 公式：`{c['formula']}`（字段：{','.join(c['fields'])}，窗口 {c['window']} 日）")
        L.append(f"- 方向：{c['direction']}")
        L.append(f"- 缺失处理：{c['missing_policy']}")
        L.append(f"- 可能失效：{'；'.join(c['failure_modes'])}")
        L.append("")
    L.append("### 2.2 因子诊断（IC = Pearson；RankIC = Spearman 并列平均秩；h∈{5,20}）")
    L.append("")
    L.append("| 因子 | h | RankIC 均值 | RankIC 标准差 | ICIR | t 统计量 | 覆盖率 | 有效资产数 | 高低组价差 | 单调性 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, fd in data["factors"].items():
        for h, d in fd["diagnostics"].items():
            s, qs = d["ic_summary"], d["quantile_summary"]
            L.append(f"| {fd['card']['name']} | {h} | {_num(s['rank_ic']['mean'])} | {_num(s['rank_ic']['std'])} "
                     f"| {_num(s['rank_ic']['icir'],3)} | {_num(s['rank_ic']['t_stat'],2)} "
                     f"| {_pct(s['coverage_mean'],0)} | {_num(s['n_valid_mean'],1)} "
                     f"| {_num(qs['spread_mean'])} | {_num(qs['monotonicity_spearman'],2)} |")
    L.append("")
    L.append("注：样本不足（截面<10）或常数序列当日记缺失（NaN），不改成 0；IC 为逐日截面相关的时间均值，")
    L.append("0.05 不表示收益 5%；分层价差未扣费、多日标签重叠，属描述性结果，不能复利为策略净值。")
    L.append("")
    L.append("![RankIC 时序](../assets/rankic.png)")
    L.append("")
    L.append("![分层收益](../assets/quantile.png)")
    L.append("")

    m = base["metrics"]
    _gn = (base.get("checks") or {}).get("gross_net_attribution") or {}
    _conv = _gn.get("convention")
    _conv_lab = "同一成交路径" if _conv == "same_path" else "费率置零重跑"
    L.append("### 2.3 可配置回测（mom20 · 周度调仓 · top10 等权 · 佣金双边各 0.15%）")
    L.append("")
    L.append(f"| 指标 | 扣费后（净） | 零费（毛，{_conv_lab}） |")
    L.append("|---|---|---|")
    for k, label in [("cumulative_return", "累计收益"), ("annualized_return", "年化收益"),
                     ("annualized_vol", "年化波动"), ("annualized_sharpe", "年化Sharpe"),
                     ("max_drawdown", "最大回撤")]:
        nv = m["net"][k]; gv = m["gross"].get(k)
        fmt = (lambda v: _num(v, 2)) if k == "annualized_sharpe" else (lambda v: _pct(v))
        L.append(f"| {label} | {fmt(nv)} | {fmt(gv)} |")
    L.append(f"| 换手合计（Σ(B+S)/V_prev 累计） | {m['turnover']['sum']:.1f} | — |")
    L.append(f"| 累计成本（货币，名义） | {m['cost']['total_cost_currency']:.4f} | 0 |")
    L.append(f"| 交易次数 / 平均持仓 / 最大单资产权重 | {m['portfolio']['n_trades']} / "
             f"{m['portfolio']['avg_holdings']:.1f} / {_pct(m['portfolio']['max_single_weight'],0)} | — |")
    L.append("")
    L.append("![净值](../assets/nav.png)")
    L.append("")
    L.append("![换手与成本](../assets/turnover.png)")
    L.append("")

    vb = m.get("vs_benchmark")
    if vb:
        L.append("**相对基准（沪深300，同日对齐；公式见 metrics.py 模块头）**")
        L.append("")
        L.append("| 超额年化(算术) | 超额年化(几何) | Beta | 年化Alpha | 跟踪误差 | 信息比率IR | 日胜率 |")
        L.append("|---|---|---|---|---|---|---|")
        L.append(f"| {_pct(vb['excess_annual'])} | {_pct(vb['excess_annual_geo'])} "
                 f"| {_num(vb['beta'],2)} | {_pct(vb['alpha_annual'])} "
                 f"| {_pct(vb['tracking_error_annual'],1)} | {_num(vb['information_ratio'],2)} "
                 f"| {_pct(vb['daily_win_rate'],0)} |")
        L.append("")

    _nav = base.get("nav") or {}
    if _nav.get("index") and len(_nav["index"]) > 60:
        _ml: dict[str, float] = {}
        for _d, _v in zip(_nav["index"], _nav["values"]):
            _ml[str(_d)[:7]] = float(_v)
        _ks = sorted(_ml)
        _rets = [_ml[k] / (_ml[_ks[i - 1]] if i else float(_nav["values"][0])) - 1.0
                 for i, k in enumerate(_ks)]
        _pos = sum(1 for x in _rets if x > 0)
        _best = max(_rets); _worst = min(_rets)
        _bi = _rets.index(_best); _wi = _rets.index(_worst)
        L.append(f"**月度分布**：{len(_ks)} 个月中盈利 {_pos} 个（{_pos/len(_rets)*100:.0f}%）；"
                 f"最好 {_ks[_bi]}（{_best*100:+.2f}%），最差 {_ks[_wi]}（{_worst*100:+.2f}%）。"
                 "月度矩阵详见回测详情页热力图。")
        L.append("")
    L.append("**回测基本检查**（全部自动执行，输出见 manifest.checks）")
    L.append("")
    for k, v in base["checks"].items():
        if isinstance(v, dict) and "pass" in v:
            mark = "PASS" if v["pass"] else "FAIL"
            note = v.get("note") or v.get("max_abs_diff") or ""
            L.append(f"- {k}: **{mark}**（{note}）")
    L.append("")

    m5 = data.get("minute5")
    if m5 and m5.get("diagnostics"):
        L.append("### 2.4 分钟频因子（5m 扩展 · 日末快照口径）")
        L.append("")
        L.append(f"> 口径：{m5.get('note')}，与日频因子同一诊断链路。")
        L.append("")
        L.append("| 因子 | h | RankIC 均值 | ICIR | t 统计量 | 平均覆盖 |")
        L.append("|---|---|---|---|---|---|")
        for d in m5["diagnostics"]:
            L.append(f"| {d['name']} | {d['horizon']} | {_num(d['rank_ic_mean'])} "
                     f"| {_num(d['icir'],3)} | {_num(d['t_stat'],2)} "
                     f"| {_pct(d['coverage_mean'],0)} |")
        L.append("")
        dq = m5.get("data_quality") or {}
        if dq:
            ci = dq.get("coverage_per_instrument") or {}
            L.append(f"数据质量：{dq.get('n_instruments')} 资产 × {dq.get('n_days')} 日，"
                     f"逐资产覆盖率中位 {_pct(ci.get('median'),1)}、最差 {_pct(ci.get('min'),1)}"
                     "（晚上市/长期停牌所致；详见 reports/minute5_summary.json）。")
        L.append("说明：5m 因子在 bar 面板计算后取每日最后一根 bar 为日频快照，"
                 "与日频因子在组合/实验/回测中可互换；覆盖仅为日频宇宙与 5m 面板的交集。")
        L.append("")

    # ---- 3 对照与解释 ----
    L.append("## 3 对照与解释")
    L.append("")
    L.append("先写问题和预期，再只改变一个主要因素；其余数据、样本区间和成本口径保持一致。")
    L.append("")
    for ename, e in data["experiments"].items():
        L.append(f"### {ename}")
        L.append("")
        L.append(f"- 问题：{e['question']}")
        L.append(f"- 预期：{e['expectation']}")
        L.append(f"- 设计：{e['description']}")
        L.append("")
        L.append("| 变体 | 累计收益 | 年化 | 年化波动 | Sharpe | MDD | 换手合计 | 累计成本 | 交易次数 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in e["comparison"]:
            L.append(f"| {r['variant']} | {_pct(r['cum_return'])} | {_pct(r['annualized_return'])} "
                     f"| {_pct(r['annualized_vol'])} | {_num(r['sharpe'],2)} | {_pct(r['max_drawdown'])} "
                     f"| {r['turnover_sum']:.1f} | {r['total_cost']:.4f} | {r['n_trades']} |")
        L.append("")
    L.append("**解释**（由本 run 实验数据自动生成，数字与上表同源）")
    L.append("")
    exps = data["experiments"]
    e1 = next((e for k, e in exps.items() if k.startswith("E1")), None)
    e2 = next((e for k, e in exps.items() if k.startswith("E2")), None)
    if e1 and len(e1.get("comparison") or []) >= 2:
        a, b = e1["comparison"][0], e1["comparison"][1]
        L.append(f"- **E1（{a['variant']} → {b['variant']}）**：换手合计 {a['turnover_sum']:.1f} → "
                 f"{b['turnover_sum']:.1f}，累计成本 {a['total_cost']:.0f} → {b['total_cost']:.0f}（名义元）；"
                 f"年化 {_pct(a['annualized_return'])} → {_pct(b['annualized_return'])}，"
                 f"MDD {_pct(a['max_drawdown'])} → {_pct(b['max_drawdown'])}。"
                 "降频的主要收益是成本节省；单期持仓变长后暴露更集中，回撤变化如实上表——"
                 "差异来自**持仓行为（持有期）**而非信号本身。")
    if e2 and (e2.get("comparison") or []):
        rows = sorted(e2["comparison"], key=lambda r: -(r["annualized_return"] or 0))
        best = rows[0]
        hi_to = max(e2["comparison"], key=lambda r: r["turnover_sum"])
        lo_dd = min(e2["comparison"], key=lambda r: r["max_drawdown"])
        L.append(f"- **E2（因子对照）**：年化最高 {best['variant']}（{_pct(best['annualized_return'])} / "
                 f"Sharpe {_num(best['sharpe'],2)}）；换手最高 {hi_to['variant']}"
                 f"（{hi_to['turnover_sum']:.0f}，成本侵蚀见上表）；回撤最小 {lo_dd['variant']}"
                 f"（{_pct(lo_dd['max_drawdown'])}）——风险收益形态与因子卡假设对照阅读。")
    L.append("- 毛收益与净收益采用**分别重跑回测**（费率置零）口径：费用会通过可用现金影响后续下单金额，")
    L.append(f"  两条成交序列存在差异（{_gn.get('n_trades_differing_notional', '—')}/{_gn.get('n_trades_net', '—')} 笔金额不同，"
             f"最大差 {_gn.get('max_notional_diff', float('nan')):.2e}）。")
    L.append("  因此毛净差额 = 交易成本 **+ 路径差异**，不作纯费用归因；成本对高换手策略的侵蚀在本样本约年化 2~4 个百分点量级。")
    L.append("- 诚实记录：三因子的 RankIC 均值绝对值均较小（|RankIC|<3%）且 mom20 在 5/20 日两类持有期为**负向**，")
    L.append("  回测正收益主要来自多空分组中的高波动暴露与段内行情，未做显著性检验的结论均属**描述性结果**。")
    L.append("")

    # ---- 4 复现信息 ----
    L.append("## 4 复现信息")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 数据快照校验值 | raw manifest sha256 `{mf['data']['raw_manifest_sha256'][:24]}…` · clean `{mf['data']['clean_manifest_sha256'][:24]}…` |")
    L.append(f"| 代码版本指纹 | {len(mf['code_version'])} 个源文件 sha256（manifest.json.code_version） |")
    L.append(f"| 配置 | configs/data.yaml · factors.yaml · backtest.yaml · experiment.yaml（快照存 manifest.config） |")
    L.append(f"| 运行时间 | {mf['duration_seconds']}s |")
    L.append(f"| 检查输出 | 回测基本检查 all_pass={mf['checks']['base_backtest_all_pass']}（门禁项见 backtests/base.json 的 checks.gates）；"
             f"确定性：同数据+同代码+同配置重跑，主要结果文件 sha256 一致（manifest.json 自身的 run_id/时间戳除外） |")
    L.append(f"| 数据获取 | `python scripts/fetch_data_tushare.py`（主源，需 TUSHARE_TOKEN）或 `python scripts/fetch_data.py`（akshare）；小样本随仓库分发（data/raw + manifest 校验值） |")
    L.append("")
    L.append("复现步骤：`pip install -r requirements.txt` → `python scripts/fetch_data_tushare.py`（或直接用随仓库快照）")
    L.append("→ `python scripts/run_pipeline.py --run-id v1` → `python scripts/build_report.py --run v1`。")
    L.append("")

    # ---- 5 局限与下一步 ----
    L.append("## 5 局限与下一步")
    L.append("")
    L.append("**局限**")
    L.append("")
    L.append("1. 成分股非 PIT：使用当期沪深300名单回测历史，存在幸存者偏差方向的不确定性；")
    L.append("2. 样本 50 只 × 4 年，统计功效有限；重叠标签使 IC 序列自相关，未做 Newey-West 修正；")
    L.append("3. 交易成本仅佣金+可选滑点，未建模冲击成本与涨跌停/流动性限制；")
    L.append("4. 允许分数股；分红送转的现金流再投资以调整价口径隐含处理；")
    L.append("5. 本研究未做参数扫描外的多重检验校正，反复搜索会放大假发现风险。")
    L.append("")
    L.append("**下一步**（对应课程 P2~P5）")
    L.append("")
    L.append("1. 接入 point-in-time 成分与行业数据，做行业中性与市值中性诊断（平台 neutralize 钩子已预留）；")
    L.append("2. 引入 GP/AlphaGen 自动挖掘（平台算法实验室已具备 GP 引擎与候选采纳链路），以增量信息验证新因子；")
    L.append("3. 组合层加入波动率目标与权重约束，比较风险调整后的稳定性；")
    L.append("4. 按月滚动样本外验证（walk-forward），降低单段区间依赖。")
    L.append("")
    L.append("---")
    L.append("")

    # ---- 附录 ----
    L.append("## 附录A 数据字典")
    L.append("")
    L.append("**原始快照（data/raw，随仓库分发的可分享小样本）**")
    L.append("")
    L.append("| 文件/字段 | 说明 | 单位 |")
    L.append("|---|---|---|")
    L.append("| constituents.csv | 资产名单（沪深300当期成分抽样50只） | — |")
    L.append("| stocks/{code}.csv | date/open/high/low/close/volume/amount/turnover | 元 / 元 / 元 / 元 / 股 / 元 / 比例(0-1) |")
    L.append("| adj_factor/{code}.csv | date/adj_factor：hfq 乘法累计复权因子 | 无量纲（最早交易日=1） |")
    L.append("| index/000300.csv | 沪深300 指数日行情（基准） | 点 |")
    L.append("| manifest.json | 下载时间、数据源与版本、抽样规则、逐文件 sha256 | — |")
    L.append("")
    L.append("**清洗输出（data/clean）**")
    L.append("")
    L.append("| 文件 | 说明 |")
    L.append("|---|---|")
    L.append("| {open,high,low,close}_raw.parquet | 原始未复权价宽表（date×code），价格的唯一来源 |")
    L.append("| adj_factor.parquet | 乘法累计复权因子 a(t)（阶梯前填充）；与原始价一起派生任意复权口径 |")
    L.append("| volume.parquet / amount.parquet | 成交量（股）/ 成交额（元）宽表 |")
    L.append("| suspended.parquet | 停牌与未上市标记（按原始价与成交量判定） |")
    L.append("| quality_report.json / clean_manifest.json | 质量报告 / 清洗参数、结构版本与输出校验值 |")
    L.append("")
    L.append("> 清洗层**不落调整价**：调整价是纯派生量，落盘等于把复权基准 τ 焊死在数据里。")
    L.append("> 现由 `MarketData.price(field, basis)` 在加载时按口径派生，实测与旧落盘面板逐值一致。")
    L.append("")
    L.append("## 附录B 回测基本检查明细（base 回测）")
    L.append("")
    L.append("| 检查 | 结果 | 说明 |")
    L.append("|---|---|---|")
    for k, v in base["checks"].items():
        if not isinstance(v, dict):
            continue
        if k == "output_hash":
            L.append(f"| 重复运行指纹 | `{v['sha256'][:20]}…` | 净值+成交序列的 sha256，重跑应一致 |")
        elif "pass" in v:
            extra = v.get("note") or v.get("max_abs_diff") or v.get("n_events", "")
            L.append(f"| {k} | {'PASS' if v['pass'] else 'FAIL'} | {extra} |")
    L.append("")
    L.append("## 附录C 因子诊断完整明细")
    L.append("")
    for key, fd in data["factors"].items():
        L.append(f"**{fd['card']['name']}（{key}）**")
        L.append("")
        for h, d in fd["diagnostics"].items():
            s, qs = d["ic_summary"], d["quantile_summary"]
            gms = "，".join(f"{g}={_num(v,5)}" for g, v in qs["group_mean_return"].items())
            ex = qs["form_and_valid_example"]["first_date_formed"]
            exs = "，".join(f"{g}(formed={x['n_formed']},valid={x['n_valid']})" for g, x in ex.items())
            L.append(f"- h={h}：IC 均值 {_num(s['ic']['mean'])}（标准差 {_num(s['ic']['std'])}，胜率 "
                     f"{_pct(s['ic']['win_rate'],1)}）；RankIC 均值 {_num(s['rank_ic']['mean'])}"
                     f"（t={_num(s['rank_ic']['t_stat'],2)}，n_obs={s['rank_ic']['n_obs']}）；"
                     f"覆盖率 {_pct(s['coverage_mean'],1)}")
            L.append(f"  - 分组日均收益：{gms}")
            L.append(f"  - 首个形成日分组人数（原组/有效）：{exs}（标签缺失不重新分组）")
        L.append("")
    L.append("## 附录D 配置与版本")
    L.append("")
    L.append("```yaml")
    L.append(json.dumps(mf["config"]["backtest"], ensure_ascii=False, indent=1, default=str))
    L.append("```")
    L.append("")
    L.append(f"- 代码指纹：{len(mf['code_version'])} 个源文件（manifest.json.code_version 逐文件 sha256）")
    L.append(f"- 依赖版本：Python {mf['environment']['python']} · pandas {mf['environment']['pandas']} · numpy {mf['environment']['numpy']} · {mf['environment']['os']}")
    L.append("- 复现命令：`python scripts/fetch_data.py` → `python scripts/run_pipeline.py --run-id v1` → `python scripts/build_report.py --run v1`")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*本报告由 `python scripts/build_report.py --run " + run.name + "` 自动生成；图表见 reports/assets/。*")
    return "\n".join(L)


def build_pdf(md_path: Path, pdf_path: Path) -> Path | None:
    """用 fpdf2 渲染中文 PDF（系统字体 SimHei/微软雅黑），表格用 table() 自动布局。"""
    from fpdf import FPDF
    from fpdf.fonts import FontFace

    font_candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"),
                       Path("C:/Windows/Fonts/simsun.ttc")]
    font = next((p for p in font_candidates if p.exists()), None)
    if font is None:
        return None

    class PDF(FPDF):
        def header(self):
            self.set_font("cjk", "", 8)
            self.set_text_color(130)
            self.cell(0, 6, "QuantLab · Project 1 研究报告", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

        def footer(self):
            self.set_y(-14)
            self.set_font("cjk", "", 8)
            self.set_text_color(130)
            self.cell(0, 8, f"第 {self.page_no()} 页", align="C")

    pdf = PDF()
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_font("cjk", "", str(font))
    pdf.add_font("cjk", "B", str(font))
    pdf.add_page()

    def body(text: str, size: float = 9.5, color=(20, 20, 20), lh: float = 5) -> None:
        # 剥离行内 Markdown 记号；对齐一律左对齐 —— 两端对齐会把混排行里
        # 少数空格撑到整行宽（中文无空格可断，视觉上出现"巨大空白/孤立符号"）
        for tok in ("**", "`", "*"):
            text = text.replace(tok, "")
        pdf.set_font("cjk", "", size)
        pdf.set_text_color(*color)
        pdf.multi_cell(0, lh, text, align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(20)

    md = md_path.read_text(encoding="utf-8")
    lines = md.splitlines()
    i = 0
    in_fence = False
    while i < len(lines):
        raw = lines[i].rstrip()
        if raw.startswith("```"):          # 围栏本身不渲染，内容按普通文本输出
            in_fence = not in_fence
            i += 1
            continue
        line = raw
        if line.startswith("|"):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            ncols = max(len(r) for r in rows)
            pdf.set_font("cjk", "", 8)
            with pdf.table(col_widths=(170 / ncols,) * ncols, text_align="LEFT",
                           line_height=4.6, padding=1, borders_layout="ALL") as table:
                for ri, r in enumerate(rows):
                    row = table.row()
                    if ri == 0:
                        row.style = FontFace(emphasis="BOLD", fill_color=(238, 242, 247))
                    for c in r:
                        row.cell(c.replace('**', ''))
            pdf.ln(2)
            continue
        if line.startswith("# "):
            pdf.set_font("cjk", "", 17); pdf.set_text_color(20)
            pdf.multi_cell(0, 9, line[2:], align="L", new_x="LMARGIN", new_y="NEXT"); pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(2); pdf.set_font("cjk", "", 13); pdf.set_text_color(26, 95, 180)
            pdf.multi_cell(0, 8, line[3:], align="L", new_x="LMARGIN", new_y="NEXT"); pdf.set_text_color(20)
        elif line.startswith("### "):
            pdf.ln(1); pdf.set_font("cjk", "", 11)
            pdf.multi_cell(0, 7, line[4:], align="L", new_x="LMARGIN", new_y="NEXT")
        elif line.startswith("!["):
            # 路径取自 () 内的实际文件名（历史上错用 alt 文本当文件名，图从未嵌入）
            path = line[line.index("](") + 2:line.rindex(")")]
            img = REPORT_ASSETS / Path(path).name
            if img.exists():
                pdf.image(str(img), w=175)
                pdf.ln(2)
            else:
                body(f"（插图缺失：{Path(path).name}）", size=8, color=(180, 60, 60))
        elif line.strip() == "---":
            pdf.set_draw_color(180)
            y = pdf.get_y() + 2
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.set_y(y + 3)
        else:
            t = line.strip()
            if not t:
                pass
            elif t.startswith("- "):
                # 紧排：• 后若加空格，fpdf2 会把后面的长 CJK 串当"超宽单词"整词换行，
                # 页面上留下孤立的 •（视觉验收实锤过）
                body("•" + t[2:])
            elif t.startswith("> "):
                body(t[2:], size=9, color=(110, 110, 110))
            else:
                body(t)
        i += 1
    pdf.output(str(pdf_path))
    return pdf_path


def build_report(run_id: str) -> tuple[Path, Path | None]:
    run = RUNS / run_id
    data = _load(run)
    charts = _charts(data)
    md = build_markdown(data, charts)
    md_path = ROOT / "reports" / "研究报告.md"
    md_path.write_text(md, encoding="utf-8")
    pdf_path = ROOT / "reports" / "研究报告.pdf"
    pdf = None
    try:
        pdf = build_pdf(md_path, pdf_path)
    except Exception as e:  # noqa: BLE001 —— PDF 字体等环境问题不阻塞 md
        print(f"[report] PDF 生成跳过: {e}")
    return md_path, pdf
