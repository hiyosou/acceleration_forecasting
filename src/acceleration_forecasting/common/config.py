from __future__ import annotations

from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path):
    path = Path(path)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(value, base=None):
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(base or REPOSITORY_ROOT) / path).resolve()

