"""Config loading: a default YAML deep-merged with optional override YAMLs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(*overrides: str | Path, base: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the default config and merge each override file on top, in order.

    Lists are replaced rather than merged, so an override's `languages.targets` fully
    replaces the default list.
    """
    with open(base) as f:
        cfg = yaml.safe_load(f)
    for path in overrides:
        path = Path(path)
        if not path.is_absolute() and not path.exists():
            path = REPO_ROOT / path
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def all_languages(cfg: dict) -> list[str]:
    """Reference language first, then targets."""
    return [cfg["languages"]["reference"], *cfg["languages"]["targets"]]
