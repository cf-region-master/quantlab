"""用户手写因子代码的加载与执行。

契约（与内置算子完全一致，因此用户因子与内置因子可互换）：
    def factor(fields):
        c = fields["close"]          # 宽表 DataFrame：index=交易日, columns=资产代码
        return c / c.shift(20) - 1   # 返回宽表（或可广播为宽表的 Series）

    fields 提供的键：open / high / low / close / volume / amount / turnover
    全部为【调整价口径】的宽表（adjust 见 configs/data.yaml）。

安全说明（如实声明，不夸大）：
  - 本模块把模块白名单【真正接进】受限 __builtins__：`__import__` 指向 safe_import，
    只放行 ALLOWED_MODULES，其余一律 ImportError。
    （对比：factor-quant-master 定义了同样的白名单但从未接线，
     真正生效的是 builtins.__import__，等于没有沙箱。）
  - **AST 静态防护**（本层新增）：exec 前先解析语法树，静态拒绝
    ① 双下划线属性链（`().__class__` / `f.__globals__` 等逃逸经典路径）；
    ② 危险内建名（getattr/setattr/eval/exec/compile/open/globals/... 按名拦截，
       即便未来有人把 __builtins__ 弄穿也拿不到入口）；
    ③ 白名单之外的 import（与运行时 safe_import 双重保险）。
  - 本项目是**单用户本地工具、无鉴权**：能提交代码的人本就有机器权限，威胁模型与多用户平台不同。
    因此这里的目标是"防误用 + 防意外"，不是抵御 determined attacker。
  - 尚未实现的是**超时/进程级隔离**（用户已确认后续再做）。当前因子代码在 Web 进程内执行，
    死循环会挂住请求 —— 这一点在提交页面上明确提示。
"""
from __future__ import annotations

import builtins
import types

import numpy as np
import pandas as pd

# 允许导入的模块（根包名）
ALLOWED_MODULES = {
    "math", "statistics", "itertools", "functools", "collections",
    "numpy", "pandas", "scipy",
}

_REAL_IMPORT = builtins.__import__


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """看门人：只放行白名单模块。注意本函数必须真的被装进 __builtins__。"""
    root = str(name).split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(
            f"沙箱禁止导入模块 '{name}'；可用: {', '.join(sorted(ALLOWED_MODULES))}")
    return _REAL_IMPORT(name, globals, locals, fromlist, level)


# 受限内置函数集合：够写因子，但拿不到 open/eval/exec/compile/__import__ 等
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "int", "isinstance", "len", "list", "map", "max", "min", "pow", "range",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "ValueError", "TypeError", "IndexError", "KeyError", "ZeroDivisionError",
    "AttributeError", "NameError", "Exception", "NotImplementedError",
)
SAFE_BUILTINS = {k: getattr(builtins, k) for k in _SAFE_BUILTIN_NAMES if hasattr(builtins, k)}
SAFE_BUILTINS["__import__"] = safe_import          # ← 关键：接的是受限版本
SAFE_BUILTINS["True"] = True
SAFE_BUILTINS["False"] = False
SAFE_BUILTINS["None"] = None

SAFE_GLOBALS = {
    "__builtins__": SAFE_BUILTINS,
    "np": np,
    "numpy": np,
    "pd": pd,
    "pandas": pd,
}


class FactorCodeError(ValueError):
    """用户因子代码错误（可直接展示给用户）。"""


# 静态按名拦截的危险内建（拿到任何一个都足以触达解释器内部）
_DANGEROUS_NAMES = frozenset({
    "eval", "exec", "compile", "open", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "breakpoint", "input", "memoryview",
    "__import__", "super", "type", "object", "help", "exit", "quit",
})


def ast_guard(src: str) -> None:
    """exec 前的静态语法树检查：拒绝逃逸惯用路径（详见模块 docstring）。

    只做「结构性拒绝 + 明确报错」，不做完整静态分析 —— 与受限 builtins
    组成双层防护，且报错信息可直接展示给用户。
    """
    import ast as _ast

    try:
        tree = _ast.parse(src, "<user_factor>")
    except SyntaxError as e:
        raise FactorCodeError(f"代码语法错误：{e.msg}（第 {e.lineno} 行）") from e

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Attribute):
            attr = node.attr
            if attr.startswith("__") and attr.endswith("__"):
                raise FactorCodeError(
                    f"沙箱禁止访问双下划线属性「.{attr}」（第 {node.lineno} 行）。"
                    "因子计算不需要触碰对象内部协议。")
        elif isinstance(node, _ast.Name):
            if node.id in _DANGEROUS_NAMES:
                raise FactorCodeError(
                    f"沙箱禁止使用「{node.id}」（第 {node.lineno} 行）。"
                    "因子计算只需要 fields 宽表 + 白名单内的 numpy/pandas。")
        elif isinstance(node, _ast.Import):
            for alias in node.names:
                root = str(alias.name).split(".")[0]
                if root not in ALLOWED_MODULES:
                    raise FactorCodeError(
                        f"沙箱禁止导入模块「{alias.name}」；可用: "
                        f"{', '.join(sorted(ALLOWED_MODULES))}")
        elif isinstance(node, _ast.ImportFrom):
            root = str(node.module or "").split(".")[0]
            if root and root not in ALLOWED_MODULES:
                raise FactorCodeError(
                    f"沙箱禁止导入模块「{node.module}」；可用: "
                    f"{', '.join(sorted(ALLOWED_MODULES))}")


def load_factor_function(code_str: str):
    """在受限环境里执行用户代码，返回其中定义的唯一函数对象。"""
    import inspect

    src = (code_str or "").strip()
    if not src:
        raise FactorCodeError("代码为空")
    if "def factor" not in src:
        raise FactorCodeError("代码里必须定义一个名为 factor 的函数：def factor(fields): ...")
    ast_guard(src)   # 静态防护先行：非法结构在执行前就被拒绝

    local_ns: dict = {}
    try:
        exec(compile(src, "<user_factor>", "exec"), dict(SAFE_GLOBALS), local_ns)  # noqa: S102
    except FactorCodeError:
        raise
    except Exception as e:  # noqa: BLE001 —— 语法/运行错误回显给用户
        raise FactorCodeError(f"代码执行失败：{type(e).__name__}: {e}") from e

    func = local_ns.get("factor")
    if func is None or not inspect.isfunction(func):
        raise FactorCodeError("未在代码中找到函数 factor（注意必须是顶层 def factor(fields)）")
    return func


def compute_user_factor(code_str: str, fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """执行用户因子并做形状/有效性校验，返回宽表（date × code）。"""
    func = load_factor_function(code_str)
    try:
        out = func(dict(fields))
    except Exception as e:  # noqa: BLE001
        raise FactorCodeError(f"factor(fields) 调用失败：{type(e).__name__}: {e}") from e

    ref: pd.DataFrame = next(iter(fields.values()))
    if isinstance(out, pd.Series):
        out = out.to_frame()
    if not isinstance(out, pd.DataFrame):
        raise FactorCodeError(
            f"factor 必须返回宽表 DataFrame 或 Series，实际返回 {type(out).__name__}")
    if out.shape[1] == 1 and out.columns[0] != ref.columns[0]:
        # 单列返回（常见于误写成按股票计算）——显式拒绝而不是静默广播
        raise FactorCodeError(
            "factor 返回的是单列结果。本平台的因子是【整个面板】的函数："
            "fields 里的每个元素已经是 date×code 的宽表，请直接做向量化运算，"
            "不要按单只股票循环。")
    out = out.reindex(index=ref.index, columns=ref.columns)
    out = out.replace([np.inf, -np.inf], np.nan)
    n_finite = int(np.isfinite(out.to_numpy(dtype="float64")).sum())
    if n_finite == 0:
        raise FactorCodeError("因子值全为缺失，请检查表达式是否恒为 NaN")
    out.columns.name = "code"
    return out


FACTOR_CODE_TEMPLATE = '''def factor(fields):
    """fields: {open, high, low, close, volume, amount, turnover}
    每个都是 date × code 的调整价宽表。返回同形状的宽表（分数越高越优）。
    """
    c = fields["close"]
    return c / c.shift(20) - 1
'''
