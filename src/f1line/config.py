"""Configuration loader and validator for f1line.

Loads config/default.yaml (and an optional override file) using PyYAML.
All tunables are defined in config/default.yaml – no magic numbers in code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Repo root is two levels above this file (src/f1line/config.py -> repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _repo_root() -> Path:
    """Return the repository root directory."""
    return _REPO_ROOT


def load_config(override_path: str | Path | None = None) -> dict[str, Any]:
    """Load and merge configuration from default.yaml and an optional override.

    Parameters
    ----------
    override_path:
        Optional path to a YAML file whose values will deep-merge on top of
        the defaults.

    Returns
    -------
    dict[str, Any]
        Merged configuration dictionary.
    """
    default_path = _repo_root() / "config" / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    with default_path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    if override_path is not None:
        override_path = Path(override_path)
        if not override_path.exists():
            raise FileNotFoundError(f"Override config not found: {override_path}")
        with override_path.open("r", encoding="utf-8") as fh:
            override: dict[str, Any] = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, override)
        logger.info("Loaded override config from %s", override_path)

    logger.debug("Config loaded: %s", cfg)
    return cfg


def load_circuits(circuits_path: str | Path | None = None) -> dict[str, Any]:
    """Load circuits.yaml.

    Parameters
    ----------
    circuits_path:
        Path to circuits.yaml. Defaults to config/circuits.yaml.

    Returns
    -------
    dict[str, Any]
        Dictionary keyed by circuit name.
    """
    path = Path(circuits_path) if circuits_path else _repo_root() / "config" / "circuits.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Circuits config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("circuits", data)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins on conflicts)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
