"""因子截面预处理与评估参数：填补 / 去极值 / 标准化 / 中性化 / IC 口径。

**所有参数都是可编辑的显式参数，不写死在代码里。**

参数由 `PREPROCESS_SCHEMA` 单点定义（默认值、可选方法、取值范围、中文标签），
它同时是后端校验依据与前端表单的渲染依据 —— 加一个方法只需改这里。

每个因子在库里保存自己的 spec（`factors.preprocess` JSON），诊断、分层回测、
信号合成全部读该因子的 spec，因此「页面写的口径」与「实际算的口径」永远一致。

--------------------------------------------------------------------------
参数命名对齐参考实现 factor-quant 的 API 契约（FactorAnalysisPayload /
BacktestPayload / StrategyGenerationPayload），两边可以逐字段对照：

    fillna_method     mean | median | pad | bfill | interpolate | zero
    outlier_method    percentile | MAD | 3sigma     （percentile 用 alpha，另两个用 k）
    k                 3.0                           （MAD / 3sigma 的倍数）
    alpha             0.01                          （percentile 的单侧分位）
    normalize_method  minmax | z_score
    ic_method         pearson | spearman | kendall  （IC 的相关系数口径）
    quantile          5                             （分层组数 K）
    n_day             1                             （持有期 h，本项目用 horizons 多值）

本项目在参考契约之上**增加**的参数（参考实现里缺失或是死代码）：
    adjust            复权方式（参考实现是每次请求的参数，本研究在清洗层固定为后复权）
    neutralize_method none|industry|industry_resid|size|industry_size
                      —— 参考实现的 capital_neutralize / industry_neutralize 是
                         `return factor` 的死代码、payload 里根本没有这个字段
    min_n             最低截面/组内有效样本 —— 参考实现没有此参数；本项目补上，
                      使"样本不足"表现为当日记缺失，而不是静默算出一个噪声值
    fillna_method=none 作为默认 —— 参考实现默认 mean；本项目默认保留缺失
--------------------------------------------------------------------------
规范执行顺序（固定，标准做法，页面会显示）：
    1. fillna      缺失值填补
    2. outlier     去极值
    3. normalize   标准化
    4. neutralize  中性化（行业 / 市值 / 行业+市值）
"""
from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd


@contextlib.contextmanager
def _quiet_nan():
    """屏蔽「全 NaN 行」触发的 RuntimeWarning（np.errstate 挡不住 nanmean 的 warn）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


# ---------------------------------------------------------------------------
# 参数模式：默认值 / 可选值 / 区间 / 标签（唯一事实源）
# ---------------------------------------------------------------------------
PREPROCESS_SCHEMA: dict[str, dict] = {
    "fillna": {
        "label": "缺失值填补",
        "note": "参考实现默认 mean；本项目默认不填补（缺失保留 NaN、下游显式跳过）。"
                "pad 只用过去值；bfill / interpolate 会用到未来信息，回测中慎用。",
        "method": {
            "label": "方式",
            "choices": {
                "none": "不填补（保持 NaN，本项目默认）",
                "mean": "截面均值",
                "median": "截面中位数（对极值更稳健）",
                "pad": "前向填充 ffill（只用过去值）",
                "bfill": "后向填充 bfill（用到未来值，慎用）",
                "interpolate": "线性插值（跨缺口，慎用）",
                "zero": "填 0（不推荐：污染截面分位与 IC）",
            },
            "default": "none",
        },
        "limit": {"label": "pad 最长连续天数(0=不限)", "default": 0, "min": 0, "max": 250},
    },
    "outlier": {
        "label": "去极值",
        "note": "逐日截面（按行）缩尾，不删样本、不改缺失。"
                "percentile 用 alpha；MAD / 3sigma 用 k。A 股收益厚尾，MAD 比 3sigma 稳健。",
        "method": {
            "label": "方法",
            "choices": {
                "none": "不去极值",
                "percentile": "分位数法：clip 到 [alpha, 1-alpha] 分位",
                "MAD": "MAD 法：median ± k × MAD（中位数绝对偏差，抗极值）",
                "3sigma": "3sigma 法：mean ± k × std（对极值本身敏感）",
            },
            "default": "percentile",
        },
        "alpha": {"label": "alpha 分位（percentile 用）", "default": 0.01, "min": 0.0, "max": 0.5},
        "k": {"label": "k 倍数（MAD / 3sigma 用）", "default": 3.0, "min": 0.5, "max": 10.0},
    },
    "normalize": {
        "label": "标准化",
        "note": "逐日截面标准化；有效样本不足或零方差当日记缺失，不改成 0。"
                "参考实现只有 minmax / z_score；本项目另加 rank。",
        "method": {
            "label": "方法",
            "choices": {
                "none": "不标准化",
                "z_score": "z_score：(x-μ)/σ",
                "minmax": "minmax：(x-min)/(max-min) → [0,1]（参考实现默认）",
                "rank": "排名百分位 rank(pct=True)，对极值完全免疫",
            },
            "default": "z_score",
        },
        "min_n": {"label": "最低截面有效样本（本项目新增）", "default": 10, "min": 3, "max": 500},
    },
    "neutralize": {
        "label": "中性化",
        "note": "行业用 PIT 申万分类（data/reference/industry_intervals.csv，逐日）；"
                "市值用 Tushare daily_basic 的 total_mv（逐日）。"
                "参考实现未接通（函数体是 return factor），本项目做成真正生效的参数。",
        "method": {
            "label": "方法",
            "choices": {
                "none": "不中性化",
                "industry": "行业：组内去均值 + 组内标准差标准化",
                "industry_resid": "行业：对行业哑变量回归取残差",
                "size": "市值：对 log(总市值) 回归取残差",
                "industry_size": "行业+市值：对行业哑变量与 log(总市值) 回归取残差",
            },
            "default": "none",
        },
        "level": {
            "label": "行业层级",
            "choices": {"sw1": "申万一级", "sw2": "申万二级", "sw3": "申万三级"},
            "default": "sw1",
        },
        "min_n": {"label": "组内/回归最低有效样本（本项目新增）", "default": 5, "min": 2, "max": 200},
    },
    "ic": {
        "label": "IC 口径",
        "note": "IC 用哪种相关系数。pearson = 原值线性相关（参考实现默认）；"
                "spearman = 秩相关（即 RankIC，对极值稳健）；kendall = 协同系数。",
        "method": {
            "label": "相关系数",
            "choices": {"pearson": "pearson（原值线性相关）",
                        "spearman": "spearman（秩相关 = RankIC）",
                        "kendall": "kendall（协同系数）"},
            "default": "pearson",
        },
    },
    "evaluation": {
        "label": "持有期与分组",
        "note": "持有期 h 与分层组数 K（对应参考契约的 n_day 与 quantile）。"
                "多个持有期用逗号分隔。",
        "horizons": {"label": "持有期 h（逗号分隔）", "default": "5,20"},
        "quantile": {"label": "分层组数 K", "default": 5, "min": 2, "max": 20},
        "adjust": {
            "label": "复权方式",
            "note": "两种口径只差一个【每股常数】k = a(τ首)/a(τ末)："
                    "P_adj_qfq(t) = P_adj_hfq(t)·k，与 t 无关。"
                    "因此同一只股票的收益率序列完全相同，IC / RankIC / 分层净值不受影响；"
                    "会变的是两处：① 用到价格绝对水平的因子（例如直接拿收盘价当因子，"
                    "实测截面秩相关只有 0.98，2393 只里 2378 只排名改变）；"
                    "② 回测里按固定资金换算股数后的整手取整。"
                    "口径由数据层的 adj_factor 精确还原，不存在缺数据的情况。",
            "choices": {"hfq": "后复权（τ = 样本首日，P_adj = P_raw×a(t)/a(τ首)）",
                        "qfq": "前复权（τ = 样本末日，P_adj = P_raw×a(t)/a(τ末)）"},
            "default": "hfq",
        },
    },
}

# 规范执行顺序（页面按此顺序展示）
STEP_ORDER = ["fillna", "outlier", "normalize", "neutralize"]

# 全局兜底默认值（未指定 spec 时使用；与 schema 的 default 一致）
DEFAULT_SPEC: dict[str, object] = {
    "fillna_method": "none", "fillna_limit": 0,
    "outlier_method": "percentile", "alpha": 0.01, "k": 3.0,
    "normalize_method": "z_score", "min_n": 10,
    "neutralize_method": "none", "neutralize_level": "sw1", "neutralize_min_n": 5,
    "ic_method": "pearson", "horizons": "5,20", "quantile": 5, "adjust": "hfq",
}

# 旧参数名 / 旧方法名 -> 参考契约名（向后兼容历史因子与已写死的配置）
SPEC_ALIASES: dict[str, str] = {
    # 参数名
    "outlier_pct": "alpha", "winsorize_pct": "alpha",
    "outlier_k": "k", "normalize_min_n": "min_n",
    # 方法名
    "winsorize": "percentile", "mad": "MAD", "sigma": "3sigma",
    "zscore": "z_score", "cross_mean": "mean", "cross_median": "median",
    "ffill": "pad",
}

# 旧版 preprocess 步骤名 -> 新 spec 片段（向后兼容历史因子与 pipeline 配置）
LEGACY_STEP_MAP: dict[str, dict] = {
    "winsorize": {"outlier_method": "percentile"},
    "zscore": {"normalize_method": "z_score"},
    "rank": {"normalize_method": "rank"},
    "minmax": {"normalize_method": "minmax"},
    "neutralize": {"neutralize_method": "industry"},
    "neutralize_industry": {"neutralize_method": "industry"},
    "fillna": {"fillna_method": "median"},
}

_METHODS = {"fillna": "fillna_method", "outlier": "outlier_method",
            "normalize": "normalize_method", "neutralize": "neutralize_method",
            "ic": "ic_method"}


def _canon_method(section: str, raw, default):
    """方法名收敛：先做别名映射，再校验是否在 schema 的 choices 内。"""
    field = PREPROCESS_SCHEMA[section]["method"]
    s = SPEC_ALIASES.get(str(raw), str(raw))
    return s if s in field["choices"] else default


def parse_horizons(raw) -> list[int]:
    """把 '5,20' / [5,20] / 5 统一成 [5, 20]（升序去重，限制 1~60）。"""
    parts = list(raw) if isinstance(raw, (list, tuple)) else \
        str(raw).replace("，", ",").split(",")
    out: list[int] = []
    for p in parts:
        try:
            v = int(float(str(p).strip()))
        except (TypeError, ValueError):
            continue
        if 1 <= v <= 60 and v not in out:
            out.append(v)
    return sorted(out) or [5, 20]


def _coerce(section: str, key: str, raw, default):
    """按 schema 把单个参数收敛到合法值（非法值退回默认，不抛异常污染页面）。"""
    if key == "method":
        return _canon_method(section, raw, default)
    field = PREPROCESS_SCHEMA[section].get(key)
    if field is None:
        return default
    if "choices" in field:
        s = SPEC_ALIASES.get(str(raw), str(raw))
        return s if s in field["choices"] else default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    if isinstance(default, int) and "min" in field and "max" in field:
        return int(min(max(round(v), field["min"]), field["max"]))
    lo, hi = field.get("min", -np.inf), field.get("max", np.inf)
    return float(min(max(v, lo), hi))


def normalize_spec(raw) -> dict:
    """把任意历史形态收敛为完整 spec（总是返回 DEFAULT_SPEC 的全部键）。

    接受：
      None / []                          -> DEFAULT_SPEC
      ["winsorize", "zscore"]            -> 旧步骤名列表（按 LEGACY_STEP_MAP 展开）
      {"outlier_method": "MAD", "k": 2.5}-> 部分指定（可用旧名 outlier_k），其余取默认
    """
    spec = dict(DEFAULT_SPEC)
    if raw is None:
        return spec
    if isinstance(raw, (list, tuple)):
        for s in raw:
            spec.update(LEGACY_STEP_MAP.get(str(s), {}))
        return spec
    if not isinstance(raw, dict):
        return spec
    if "steps" in raw and isinstance(raw.get("steps"), (list, tuple)):
        base = normalize_spec(list(raw["steps"]))
        base.update({k: v for k, v in (raw.get("params") or {}).items()})
        raw = base
    for k, v in raw.items():
        key = SPEC_ALIASES.get(str(k), str(k))
        if key in DEFAULT_SPEC:
            spec[key] = v

    out = dict(DEFAULT_SPEC)
    for section, mkey in _METHODS.items():
        # 注意：schema 里方法字段的键固定是 "method"，不是 "<section>_method"
        out[mkey] = _coerce(section, "method", spec[mkey], DEFAULT_SPEC[mkey])
    out["fillna_limit"] = _coerce("fillna", "limit", spec["fillna_limit"],
                                 DEFAULT_SPEC["fillna_limit"])
    out["alpha"] = _coerce("outlier", "alpha", spec["alpha"], DEFAULT_SPEC["alpha"])
    out["k"] = _coerce("outlier", "k", spec["k"], DEFAULT_SPEC["k"])
    out["min_n"] = _coerce("normalize", "min_n", spec["min_n"], DEFAULT_SPEC["min_n"])
    out["neutralize_level"] = _coerce("neutralize", "level", spec["neutralize_level"],
                                     DEFAULT_SPEC["neutralize_level"])
    out["neutralize_min_n"] = _coerce("neutralize", "min_n", spec["neutralize_min_n"],
                                     DEFAULT_SPEC["neutralize_min_n"])
    out["quantile"] = _coerce("evaluation", "quantile", spec["quantile"],
                              DEFAULT_SPEC["quantile"])
    out["adjust"] = _coerce("evaluation", "adjust", spec["adjust"], DEFAULT_SPEC["adjust"])
    out["horizons"] = ",".join(str(h) for h in parse_horizons(spec["horizons"]))
    return out


def spec_is_noop(spec) -> bool:
    """是否是「完全不做预处理」的 spec。"""
    sp = normalize_spec(spec)
    return all(sp[m] in (None, "none") for m in _METHODS.values())


def spec_steps(spec) -> list[str]:
    """把 spec 翻译成人话步骤列表（供口径表展示；只列出真正生效的步骤）。"""
    sp = normalize_spec(spec)
    out: list[str] = []
    fm = sp["fillna_method"]
    if fm != "none":
        txt = {"mean": "截面均值", "median": "截面中位数", "pad": "前向填充 ffill",
               "bfill": "后向填充 bfill", "interpolate": "线性插值",
               "zero": "填 0"}.get(str(fm), str(fm))
        if fm == "pad" and int(sp["fillna_limit"]) > 0:
            txt += f"，最长连续 {int(sp['fillna_limit'])} 天"
        out.append(f"缺失值：{txt}")
    om = sp["outlier_method"]
    if om == "percentile":
        a = float(sp["alpha"])
        out.append(f"去极值：分位数法，逐日截面 clip 到 [{a:g}, {1 - a:g}] 分位（alpha={a:g}）")
    elif om == "MAD":
        out.append(f"去极值：MAD 法，逐日截面 clip 到 median ± {float(sp['k']):g} × MAD")
    elif om == "3sigma":
        out.append(f"去极值：3sigma 法，逐日截面 clip 到 mean ± {float(sp['k']):g} × std")
    nm = sp["normalize_method"]
    if nm != "none":
        mn = int(sp["min_n"])
        txt = {"z_score": "z_score (x-μ)/σ", "minmax": "minmax 缩放到 [0,1]",
               "rank": "排名百分位 rank(pct=True)"}.get(str(nm), str(nm))
        out.append(f"标准化：{txt}；有效样本 < {mn} 或零方差当日记缺失")
    ne = sp["neutralize_method"]
    lv = {"sw1": "申万一级", "sw2": "申万二级", "sw3": "申万三级"}.get(
        str(sp["neutralize_level"]), "申万一级")
    if ne == "industry":
        out.append(f"中性化：{lv}行业组内去均值 + 组内标准差标准化")
    elif ne == "industry_resid":
        out.append(f"中性化：对 {lv}行业哑变量做截面 OLS 回归，取残差")
    elif ne == "size":
        out.append("中性化：对 log(总市值) 做截面 OLS 回归，取残差")
    elif ne == "industry_size":
        out.append(f"中性化：对 {lv}行业哑变量 + log(总市值) 做截面 OLS 回归，取残差")
    if not out:
        out.append("未做额外预处理（因子原始值直接进入诊断与信号合成）")
    return out


def eval_summary(spec) -> str:
    """评估口径一行人话（IC 方法 / 持有期 / 分组数 / 复权）。"""
    sp = normalize_spec(spec)
    adj = short_label("adjust", sp["adjust"])
    return (f"IC 口径 {sp['ic_method']} · 持有期 h = {sp['horizons']} · "
            f"分层 K = {sp['quantile']} · 复权 {adj}")


# 短标签：参数总表「当前值」列用（"none" 在不同环节含义不同，因此按 section 分）
_SHORT: dict[str, dict[str, str]] = {
    "fillna_method": {"none": "不填补", "mean": "截面均值", "median": "截面中位数",
                      "pad": "前向填充", "bfill": "后向填充", "interpolate": "线性插值",
                      "zero": "填 0"},
    "outlier_method": {"none": "不去极值", "percentile": "分位数法", "MAD": "MAD 法",
                       "3sigma": "3sigma 法"},
    "normalize_method": {"none": "不标准化", "z_score": "z-score", "minmax": "min-max",
                         "rank": "排名百分位"},
    "neutralize_method": {"none": "不中性化", "industry": "行业（组内）",
                          "industry_resid": "行业（回归残差）", "size": "市值（回归残差）",
                          "industry_size": "行业+市值（回归残差）"},
    "ic_method": {"pearson": "pearson", "spearman": "spearman", "kendall": "kendall"},
    "adjust": {"hfq": "后复权（τ=样本首日）", "qfq": "前复权（τ=样本末日）", "": "后复权"},
    "neutralize_level": {"sw1": "申万一级", "sw2": "申万二级", "sw3": "申万三级"},
}


def short_label(key: str, value) -> str:
    """把参数值翻成短中文标签，供「当前值」列显示（未知值原样返回，不吞掉信息）。"""
    return _SHORT.get(str(key), {}).get(str(value), str(value))


# ---------------------------------------------------------------------------
# 单步实现（numpy，逐行=逐日截面；行为明确、NaN 处理显式）
# ---------------------------------------------------------------------------
def _row_apply(df: pd.DataFrame, fn) -> pd.DataFrame:
    arr = df.to_numpy(dtype="float64")
    out = fn(arr)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _bounds_per_row(arr: np.ndarray, how) -> tuple[np.ndarray, np.ndarray]:
    """逐行计算 clip 上下界；样本不足(<2)的行返回 ±inf 表示不处理。"""
    n = arr.shape[0]
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    for i in range(n):
        x = arr[i][np.isfinite(arr[i])]
        if x.size < 2:
            continue
        lo[i], hi[i] = how(x)
    return lo, hi


def _clip(df: pd.DataFrame, lo: np.ndarray, hi: np.ndarray) -> pd.DataFrame:
    arr = df.to_numpy(dtype="float64").copy()
    m = np.isfinite(arr)
    arr = np.where(m, np.clip(arr, lo[:, None], hi[:, None]), arr)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def fillna_cs(df: pd.DataFrame, method: str = "none", limit: int = 0) -> pd.DataFrame:
    """缺失值填补。截面法只用当日横截面；pad 只用过去值。"""
    if method in (None, "none"):
        return df
    if method == "zero":
        return df.fillna(0.0)
    if method == "mean":
        return df.apply(lambda row: row.fillna(row.mean()), axis=1)
    if method == "median":
        return df.apply(lambda row: row.fillna(row.median()), axis=1)
    if method == "pad":
        lim = int(limit) if int(limit) > 0 else None
        return df.ffill(limit=lim)
    if method == "bfill":
        lim = int(limit) if int(limit) > 0 else None
        return df.bfill(limit=lim)
    if method == "interpolate":
        lim = int(limit) if int(limit) > 0 else None
        # axis=0 = 沿时间插值；limit_area='inside' 只填两端都有有效值之间的缺口，
        # 不把首尾缺失外推出去（外推没有数据依据）
        return df.interpolate(method="linear", axis=0, limit=lim, limit_area="inside")
    raise KeyError(f"未知缺失值填补方法: {method}")


def winsorize_cs(df: pd.DataFrame, alpha: float = 0.01, pct: float | None = None) -> pd.DataFrame:
    """分位数去极值（percentile）：按行 clip 到 [alpha, 1-alpha] 分位。

    `pct` 是历史参数名（等价于 alpha），保留以免旧调用方静默报 TypeError。
    """
    a = float(alpha if pct is None else pct)

    def how(x):
        return tuple(np.quantile(x, [a, 1 - a]))

    lo, hi = _bounds_per_row(df.to_numpy(dtype="float64"), how)
    return _clip(df, lo, hi)


def mad_cs(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """MAD 去极值：median ± k × MAD（MAD = 中位数绝对偏差，未乘 1.4826 一致性系数）。"""

    def how(x):
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        return med - k * mad, med + k * mad

    lo, hi = _bounds_per_row(df.to_numpy(dtype="float64"), how)
    return _clip(df, lo, hi)


def sigma_cs(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """3sigma 去极值：mean ± k × std。"""

    def how(x):
        mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
        return mu - k * sd, mu + k * sd

    lo, hi = _bounds_per_row(df.to_numpy(dtype="float64"), how)
    return _clip(df, lo, hi)


def zscore_cs(df: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """截面标准化（z_score）：有效样本 < min_n 或零方差当日记缺失（NaN，不改成 0）。"""

    def fn(arr: np.ndarray) -> np.ndarray:
        with _quiet_nan(), np.errstate(invalid="ignore", divide="ignore"):
            mu = np.nanmean(arr, axis=1, keepdims=True)
            sd = np.nanstd(arr, axis=1, ddof=1, keepdims=True)
        n = np.isfinite(arr).sum(axis=1, keepdims=True)
        bad = (n < max(int(min_n), 2)) | (sd < 1e-12) | ~np.isfinite(sd)
        sd_ok = np.where(bad, np.nan, sd)
        out = (arr - np.where(np.isfinite(mu), mu, np.nan)) / sd_ok
        out[np.broadcast_to(bad, out.shape)] = np.nan
        return out

    return _row_apply(df, fn)


def minmax_cs(df: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """截面 minmax 缩放到 [0,1]；常数截面或样本不足当日记缺失。"""

    def fn(arr: np.ndarray) -> np.ndarray:
        with _quiet_nan():
            lo = np.nanmin(arr, axis=1, keepdims=True)
            hi = np.nanmax(arr, axis=1, keepdims=True)
        n = np.isfinite(arr).sum(axis=1, keepdims=True)
        rng = hi - lo
        bad = (n < max(int(min_n), 2)) | (rng < 1e-12) | ~np.isfinite(rng)
        rng_ok = np.where(bad, np.nan, rng)
        out = (arr - lo) / rng_ok
        out[np.broadcast_to(bad, out.shape)] = np.nan
        return out

    return _row_apply(df, fn)


def rank_pct_cs(df: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """截面排名百分位（并列取平均秩）；样本不足当日记缺失。"""
    out = df.rank(axis=1, method="average", pct=True)
    n = df.notna().sum(axis=1)
    out.loc[n < max(int(min_n), 2), :] = np.nan
    return out


def normalize_cs(df: pd.DataFrame, method: str = "z_score", min_n: int = 10) -> pd.DataFrame:
    if method in (None, "none"):
        return df
    if method in ("z_score", "zscore"):
        return zscore_cs(df, min_n)
    if method == "minmax":
        return minmax_cs(df, min_n)
    if method == "rank":
        return rank_pct_cs(df, min_n)
    raise KeyError(f"未知标准化方法: {method}")


# ---------------- 中性化 ----------------
def _group_codes(groups: pd.DataFrame | None, df: pd.DataFrame) -> np.ndarray | None:
    if groups is None:
        return None
    g = groups.reindex(index=df.index, columns=df.columns)
    return g.to_numpy(dtype="float64")


def neutralize_industry_group_cs(df: pd.DataFrame, groups: pd.DataFrame,
                                 min_n: int = 5) -> pd.DataFrame:
    """组内去均值 + 组内标准差标准化（对应 factor-quant 的 industry_neutralize）。"""
    garr = _group_codes(groups, df)

    def fn(arr: np.ndarray, g: np.ndarray) -> np.ndarray:
        out = np.full_like(arr, np.nan)
        for i in range(arr.shape[0]):
            x, gi = arr[i], g[i]
            m = np.isfinite(x) & np.isfinite(gi)
            if m.sum() < min_n:
                continue
            for lab in np.unique(gi[m]):
                gm = m & (gi == lab)
                xv = x[gm]
                if xv.size < min_n:
                    continue
                sd = np.nanstd(xv, ddof=1)
                out[i][gm] = xv - np.nanmean(xv) if sd < 1e-12 else (xv - np.nanmean(xv)) / sd
        return out

    return _row_apply(df, lambda a: fn(a, garr))


def _residual_row(x: np.ndarray, cols: list[np.ndarray], min_n: int) -> np.ndarray:
    """单日截面 OLS：x ~ [1, *cols]，返回残差（有效样本不足则整日记 NaN）。"""
    m = np.isfinite(x)
    for c in cols:
        m &= np.isfinite(c)
    out = np.full_like(x, np.nan)
    if m.sum() < max(min_n, len(cols) + 2):
        return out
    X = np.column_stack([np.ones(int(m.sum()))] + [c[m] for c in cols])
    y = x[m]
    # lstsq（SVD）对共线设计矩阵稳健，比正规方程更安全
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out[m] = y - X @ beta
    return out


def neutralize_resid_cs(df: pd.DataFrame, groups: pd.DataFrame | None = None,
                        size: pd.DataFrame | None = None, min_n: int = 5,
                        use_industry: bool = False, use_size: bool = False) -> pd.DataFrame:
    """截面 OLS 残差中性化：对行业哑变量与/或 log(市值) 回归后取残差。

    - 行业哑变量用 one-hot **去掉第一类** + 截距，避免与截距完全共线。
    - 市值取 log（市值分布右偏严重，log 后近似对称）。
    - 残差本身量纲与原因子一致，通常随后再做标准化。
    """
    garr = _group_codes(groups, df) if use_industry else None
    sarr = None
    if use_size:
        if size is None:
            raise ValueError("该因子要求市值中性化，但本地没有市值数据"
                             "（data/raw/daily_basic.parquet）")
        sarr = size.reindex(index=df.index, columns=df.columns).to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            sarr = np.where(sarr > 0, np.log(sarr), np.nan)

    arr = df.to_numpy(dtype="float64")
    out = np.full_like(arr, np.nan)
    for i in range(arr.shape[0]):
        cols: list[np.ndarray] = []
        if garr is not None:
            gi = garr[i]
            labs = np.unique(gi[np.isfinite(gi)])
            for lab in labs[1:]:  # 去掉第一类，与截距不共线
                cols.append((gi == lab).astype("float64"))
        if sarr is not None:
            cols.append(sarr[i])
        out[i] = _residual_row(arr[i], cols, min_n)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def neutralize_cs(df: pd.DataFrame, groups: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    """向后兼容入口：等价于「行业组内去均值 + 组内标准化」。"""
    return neutralize_industry_group_cs(df, groups, min_n)


def industry_groups_for(df: pd.DataFrame, level: str = "sw1") -> pd.DataFrame:
    """取与该面板对齐的 PIT 行业分组编码（申万）。

    行业数据来自 `data/reference/industry_intervals.csv`（逐日 point-in-time）。
    未知行业为 NaN，中性化会显式排除并计数，不推断、不前向填充。
    走带缓存的入口：批量重算多个因子时不必反复构造 969×2431 的标签表。
    """
    from ..data.industry import industry_group_codes_fast

    codes, _ = industry_group_codes_fast(df.index, list(df.columns), level)
    return codes


def size_panel_for(df: pd.DataFrame) -> pd.DataFrame | None:
    """取与该面板对齐的总市值面板（元）；无数据返回 None 而不是编造。"""
    try:
        from ..data.fundamentals import total_mv_panel
    except Exception:  # noqa: BLE001
        return None
    try:
        return total_mv_panel(df.index, list(df.columns))
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def apply_preprocess_spec(df: pd.DataFrame, spec, *,
                          groups: pd.DataFrame | None = None,
                          size: pd.DataFrame | None = None) -> pd.DataFrame:
    """按 spec 对因子面板做完整预处理（顺序见模块 docstring）。

    groups / size 为 None 时按需自动解析（PIT 申万行业 / daily_basic 总市值），
    只有真正用到该步骤时才解析，避免无谓的 IO。

    注意：`adjust`（复权方式）**不在这里生效** —— 它决定的是因子值从哪个价格面板算出来，
    属于数据层，由 `MarketData.fields_for(basis)` 在计算因子值时施加。
    """
    sp = normalize_spec(spec)
    out = df.astype("float64")

    # 1. 缺失值填补
    out = fillna_cs(out, str(sp["fillna_method"]), int(sp["fillna_limit"]))

    # 2. 去极值
    om = sp["outlier_method"]
    if om == "percentile":
        out = winsorize_cs(out, float(sp["alpha"]))
    elif om == "MAD":
        out = mad_cs(out, float(sp["k"]))
    elif om == "3sigma":
        out = sigma_cs(out, float(sp["k"]))

    # 3. 标准化
    out = normalize_cs(out, str(sp["normalize_method"]), int(sp["min_n"]))

    # 4. 中性化
    ne = sp["neutralize_method"]
    if ne != "none":
        min_n = int(sp["neutralize_min_n"])
        level = str(sp["neutralize_level"])
        if ne == "industry":
            g = groups if groups is not None else industry_groups_for(out, level)
            out = neutralize_industry_group_cs(out, g, min_n)
        elif ne == "industry_resid":
            g = groups if groups is not None else industry_groups_for(out, level)
            out = neutralize_resid_cs(out, groups=g, min_n=min_n,
                                      use_industry=True, use_size=False)
        elif ne == "size":
            s = size if size is not None else size_panel_for(out)
            out = neutralize_resid_cs(out, size=s, min_n=min_n,
                                      use_industry=False, use_size=True)
        elif ne == "industry_size":
            g = groups if groups is not None else industry_groups_for(out, level)
            s = size if size is not None else size_panel_for(out)
            out = neutralize_resid_cs(out, groups=g, size=s, min_n=min_n,
                                      use_industry=True, use_size=True)
        else:
            raise KeyError(f"未知中性化方法: {ne}")
    return out


def apply_preprocess(df: pd.DataFrame, steps, diag_cfg: dict | None = None,
                     groups: pd.DataFrame | None = None) -> pd.DataFrame:
    """向后兼容入口：接受新的 spec 或旧的步骤名列表。

    pipeline.py / model.py 仍在用旧的 `["winsorize","zscore"]` 形式；
    这里统一转成 spec 后走同一条实现，保证两条路径结果一致。
    """
    diag_cfg = diag_cfg or {}
    spec = normalize_spec(steps)
    # 旧配置里的 winsorize_pct / min_cross_section_samples 仍应生效
    if isinstance(steps, (list, tuple)):
        if "winsorize_pct" in diag_cfg and spec["outlier_method"] == "percentile":
            spec["alpha"] = _coerce("outlier", "alpha", diag_cfg["winsorize_pct"],
                                    DEFAULT_SPEC["alpha"])
        if "min_cross_section_samples" in diag_cfg:
            spec["min_n"] = _coerce("normalize", "min_n",
                                    diag_cfg["min_cross_section_samples"],
                                    DEFAULT_SPEC["min_n"])
            spec["neutralize_min_n"] = max(int(spec["neutralize_min_n"]),
                                           min(20, int(spec["min_n"])))
    return apply_preprocess_spec(df, spec, groups=groups)
