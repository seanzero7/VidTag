"""Minimal YAML config system (SPEC §10: config-driven training).

``Config`` wraps a nested dict with attribute access (``cfg.train.lr``) and
dotted-path lookup (``cfg.get("train.lr", default)``). ``load_config`` reads
a YAML file and applies ``"key.path=value"`` overrides; override values are
YAML-parsed so ``lr=1e-4`` becomes a float and ``flag=true`` a bool.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import yaml


class _Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 float resolution (PyYAML misses ``1e-4``)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)[eE][-+]?[0-9]+
            |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


class Config:
    """Read-only view over a nested dict of hyperparameters."""

    def __init__(self, data: dict[str, Any]):
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name not in data:
            raise AttributeError(
                f"config has no key {name!r}; available keys: {sorted(data)}"
            )
        value = data[name]
        return Config(value) if isinstance(value, dict) else value

    def get(self, path: str, default: Any = None) -> Any:
        """Dotted-path lookup, e.g. ``cfg.get('train.lr', 5e-5)``."""
        node: Any = object.__getattribute__(self, "_data")
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(object.__getattribute__(self, "_data"))

    def __repr__(self) -> str:
        return f"Config({object.__getattribute__(self, '_data')!r})"


def load_config(path: str, overrides: list[str] | None = None) -> Config:
    """Load a YAML config and apply ``"a.b.c=value"`` overrides in order."""
    with open(path) as f:
        data = yaml.load(f, Loader=_Loader) or {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like 'a.b.c=value'")
        node = data
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.load(raw, Loader=_Loader)
    return Config(data)
