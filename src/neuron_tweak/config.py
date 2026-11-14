"""TOML configuration loading for Neuron Tweak."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when a configuration file cannot be loaded."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML config file into a dictionary."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        raw = config_path.read_bytes()
    except OSError as error:
        raise ConfigError(f"Could not read {config_path}: {error}") from error
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigError(
            f"Configuration file must be UTF-8: {config_path}"
        ) from error
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(
            f"Invalid TOML in {config_path}: {error}"
        ) from error
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be a table: {config_path}")
    return data


def config_value(
    config: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """Return a nested config value, or the default when any key is missing."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_config_path(explicit: str | None) -> Path | None:
    """Pick the config path: explicit flag, then ./config.toml, else none."""
    if explicit:
        return Path(explicit)
    local = Path("config.toml")
    if local.is_file():
        return local
    return None
