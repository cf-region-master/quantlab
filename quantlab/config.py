"""配置加载：所有模块从 configs/*.yaml 读取参数，公式与回测核心不写死参数。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    name: str
    data: dict[str, Any]
    factors: dict[str, Any]
    backtest: dict[str, Any]
    experiment: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {"data": self.data, "factors": self.factors,
                "backtest": self.backtest, "experiment": self.experiment}

    def dumps(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True, default=str)


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(overrides: dict[str, Any] | None = None,
                backtest_file: str = "backtest.yaml") -> Config:
    """overrides 按 {section: {...}} 深合并，用于对照实验只改一个主要因素。"""
    d = _load_yaml(CONFIG_DIR / "data.yaml")
    f = _load_yaml(CONFIG_DIR / "factors.yaml")
    b = _load_yaml(CONFIG_DIR / backtest_file)
    e = _load_yaml(CONFIG_DIR / "experiment.yaml")
    for sec, patch in (overrides or {}).items():
        if sec == "data":
            d = _deep_merge(d, patch)
        elif sec == "factors":
            f = _deep_merge(f, patch)
        elif sec == "backtest":
            b = _deep_merge(b, patch)
        elif sec == "experiment":
            e = _deep_merge(e, patch)
    name = (overrides or {}).get("_name") or b.get("name", "base")
    return Config(name=name, data=d, factors=f, backtest=b, experiment=e)


def code_version() -> dict:
    """代码版本指纹：quantlab 包与 scripts 全部源文件的 sha256 清单（复现依据之一）。"""
    out = {}
    for base in (ROOT / "quantlab", ROOT / "scripts"):
        for p in sorted(base.rglob("*.py")):
            rel = p.relative_to(ROOT).as_posix()
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
