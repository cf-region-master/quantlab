"""把 configs/factors_5m.yaml 的 5 分钟因子卡落地为日频因子并登记入库。

口径（文档化选择，不再另起一条评估链路）：
- 因子在 5m bar 面板上按因子卡算子计算（窗口单位 = bar，48 bar/日），
  取【每日最后一根 bar】的值作为该资产的日频因子快照（日末口径）；
- 资产范围 = 5m 面板列 ∩ 日频清洗宇宙（date×code 对齐到日频市场；
  交集为空或日频数据未就绪时显式报错，不静默给空结果）；
- 缺失（当日无 bar / 窗口不足 / 不在宇宙）记 NaN 不填零；
- 之后走与内置/手写因子完全相同的入库、幂等去重、诊断与分层链路
  （store.register_factor → run_diagnostics + layered_nav），
  因此这些因子在因子库/策略/对照实验里与日频因子**可互换**。

用法：
    python scripts/register_5m_factors.py                 # 登记全部卡片
    python scripts/register_5m_factors.py --cards mom48   # 只登记指定卡
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantlab.config import load_config  # noqa: E402
from quantlab.data.minute import panel_frame  # noqa: E402
from quantlab.factors.operators import (  # noqa: E402
    FactorCard, compute_factor, factor_hash, load_cards)
from quantlab.storage import store  # noqa: E402

MINUTE_DIR = ROOT / "data" / "minute_5m"
CARDS_YAML = ROOT / "configs" / "factors_5m.yaml"


def daily_last_bar(five_m: pd.DataFrame) -> pd.DataFrame:
    """按自然日取每日最后一根 bar 的行（rows 在面板里按时间连续排列）。"""
    day = five_m.index.normalize()
    boundary = np.r_[day[1:] != day[:-1], True]      # 每天最后一行的位置
    out = five_m.iloc[np.flatnonzero(boundary)].copy()
    out.index = day[np.flatnonzero(boundary)]
    out.index.name = "date"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", default="", help="逗号分隔的卡片 key，缺省全部")
    args = ap.parse_args()

    if not MINUTE_DIR.exists():
        raise SystemExit(f"5m 面板未构建：{MINUTE_DIR} 不存在。"
                         "先用 quantlab.data.minute.build_5m_fields 从分钟 CSV 构建。")
    raw = yaml.safe_load(CARDS_YAML.read_text(encoding="utf-8"))
    cards: dict[str, FactorCard] = load_cards({"factors": raw["factors"]})
    if args.cards:
        keep = [x.strip() for x in args.cards.split(",") if x.strip()]
        cards = {k: v for k, v in cards.items() if k in keep}
        if not cards:
            raise SystemExit(f"没有匹配的卡片：{keep}")

    cfg = load_config()
    _ = cfg
    mk = store.market()
    universe_codes = set(map(str, mk.codes))

    codes_path = MINUTE_DIR / "close_5m_codes.json"
    if not codes_path.exists():
        raise SystemExit(
            f"缺少 {codes_path.name}（资产代码表）。"
            "请重跑 build_5m_fields（新版本会落盘），或从原始分钟 CSV 重扫 instrument 列。")
    codes_5m = json.loads(codes_path.read_text(encoding="utf-8"))

    def _to_daily(code: str) -> str:
        # 'SH600000'/'SZ000001' → 日频宇宙的 6 位数字口径；其余原样保留
        for pre in ("SH", "SZ", "BJ"):
            if code.startswith(pre):
                return code[len(pre):]
        return code

    code_map = {i: _to_daily(str(c)) for i, c in enumerate(codes_5m)}
    keep_idx = [i for i, c in code_map.items() if c in universe_codes]
    if not keep_idx:
        raise SystemExit("5m 面板与日频宇宙无交集资产，请检查代码格式是否一致"
                         f"（5m 示例：{codes_5m[:2]}；宇宙示例：{sorted(universe_codes)[:3]}）")
    close_5m = panel_frame(MINUTE_DIR, "close")
    close_5m = close_5m.iloc[:, keep_idx]
    close_5m.columns = [code_map[i] for i in keep_idx]

    print(f"[5m] 面板 {close_5m.shape[0]} bar × {close_5m.shape[1]} 资产"
          f"（宇宙交集）· {len(cards)} 张卡片")
    results = []
    for key, card in cards.items():
        vals_5m = compute_factor(card, {"close": close_5m})
        daily = daily_last_bar(vals_5m)
        # 对齐到日频市场（缺失日期/资产保持 NaN）—— 与其他因子同一评估口径
        values = daily.reindex(index=mk.dates, columns=mk.codes)
        n_finite = int(np.isfinite(values.to_numpy(dtype="float64")).sum())
        if n_finite == 0:
            print(f"[skip] {key}: 与市场重叠区间内无有效值")
            continue
        h = factor_hash(card, values)
        f, created = store.register_factor(
            name=card.name, hash_code=h, source="5m", expression=card.formula,
            hypothesis=card.hypothesis, direction=card.direction,
            missing_policy=card.missing_policy, failure_modes=card.failure_modes,
            operator=card.operator, params={**card.params, "bar": "5m"},
            preprocess=card.preprocess, values=values,
            notes=f"由 5 分钟面板日末快照生成（{CARDS_YAML.name} · {key}）",
            universe_note=f"5m 面板 ∩ 日频宇宙，{len(keep_idx)} 资产")
        tag = "新建" if created else "已存在(幂等跳过)"
        print(f"[ok] {key} -> 因子 #{f.id} {f.name}：{tag}")
        results.append({"key": key, "factor_id": f.id, "created": created,
                        "n_finite": n_finite})
    print(f"[done] {sum(r['created'] for r in results)} 新建 / "
          f"{len(results)} 处理（诊断与分层已随登记自动产出）")
    _write_summary(cards, results)
    return 0


def _data_quality() -> dict:
    """5m 面板数据质量：逐资产/逐年覆盖率（close_5m.npy 的非缺失 bar 占比）。"""
    arr = np.load(MINUTE_DIR / "close_5m.npy", mmap_mode="r")
    dates = json.loads((MINUTE_DIR / "close_5m_dates.json").read_text(encoding="utf-8"))
    finite = np.isfinite(arr)
    cov = finite.mean(axis=0)
    q = np.quantile(cov, [0.0, 0.25, 0.5, 0.75, 1.0])
    per_year = {}
    for y in sorted({d[:4] for d in dates}):
        rows = [i for i, d in enumerate(dates) if d.startswith(y)]
        per_year[y] = round(float(finite[rows].mean()), 4)
    return {"n_instruments": int(finite.shape[1]),
            "n_days": len(dates),
            "coverage_per_instrument": {
                "min": round(float(q[0]), 4), "p25": round(float(q[1]), 4),
                "median": round(float(q[2]), 4), "p75": round(float(q[3]), 4),
                "max": round(float(q[4]), 4)},
            "coverage_per_year": per_year,
            "note": "覆盖率 = 非缺失 bar 占比（close_5m.npy，全字段口径见 meta）"}


def _write_summary(cards: dict, results: list[dict]) -> None:
    """把 5m 因子的登记与诊断摘要落盘，供研究报告等下游消费。"""
    out = ROOT / "reports" / "minute5_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    from quantlab.storage.db import Factor, FactorMetric, get_session
    s = get_session()
    try:
        rows = []
        for r in results:
            f = s.get(Factor, r["factor_id"])
            if f is None:
                continue
            fname = f.name
            for m in s.query(FactorMetric).filter_by(factor_id=r["factor_id"]).all():
                rows.append({"factor_id": f.id, "name": fname,
                             "key": r["key"], "horizon": m.horizon,
                             "rank_ic_mean": m.rank_ic_mean, "rank_ic_std": m.rank_ic_std,
                             "icir": m.icir, "t_stat": m.t_stat,
                             "n_valid_mean": m.n_valid_mean,
                             "coverage_mean": m.coverage_mean})
    finally:
        s.close()
    payload = {"cards": {k: cards[k].card_dict() for k in cards},
               "diagnostics": rows,
               "data_quality": _data_quality(),
               "note": ("5m bar 面板计算、每日最后一根 bar 取日频快照；"
                        "诊断与日频因子同口径（RankIC/Spearman，重叠标签为描述性）")}
    out.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {out}")


if __name__ == "__main__":
    raise SystemExit(main())
