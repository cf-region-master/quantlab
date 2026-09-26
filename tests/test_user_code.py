"""用户代码沙箱的静态防护测试（AST 层）。"""
from __future__ import annotations

import pytest

from quantlab.factors.user_code import FactorCodeError, ast_guard, compute_user_factor

GOOD = '''
def factor(fields):
    import numpy as np
    c = fields["close"]
    return c / c.shift(20) - 1
'''


@pytest.mark.parametrize("bad,why", [
    ("def factor(fields):\n"
     "    return ().__class__",                     "dunder 属性"),
    ("def factor(fields):\n"
     "    f = factor\n"
     "    return f.__globals__",                  "__globals__ 逃逸"),
    ("def factor(fields):\n"
     "    return getattr(fields, '__class__')",   "getattr 按名拦截"),
    ("def factor(fields):\n"
     "    return eval('1+1')",                    "eval 按名拦截"),
    ("def factor(fields):\n"
     "    import os\n"
     "    return fields['close']",                "白名单外 import"),
    ("def factor(fields):\n"
     "    from subprocess import run\n"
     "    return fields['close']",                "白名单外 from-import"),
    ("def factor(fields):\n"
     "    t = type(fields)\n"
     "    return fields['close']",                "type 按名拦截"),
])
def test_ast_guard_rejects(bad, why):
    with pytest.raises(FactorCodeError) as ei:
        ast_guard(bad)
    assert "沙箱禁止" in str(ei.value) or "语法错误" in str(ei.value), why


def test_ast_guard_allows_normal_factor():
    ast_guard(GOOD)  # 不抛即通过


def test_compute_user_factor_end_to_end(tiny_market):
    fields = {"close": tiny_market.close_adj, "open": tiny_market.open_adj}
    out = compute_user_factor(GOOD, fields)
    assert out.shape == tiny_market.close_adj.shape


def test_runtime_import_guard_still_active():
    """静态层漏网时（例如间接字符串导入），运行时 safe_import 仍兜底。"""
    code = '''
def factor(fields):
    import os
    return fields["close"]
'''
    with pytest.raises(FactorCodeError):
        compute_user_factor(code, {"close": None})
