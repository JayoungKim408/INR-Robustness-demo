"""
Configuration utilities for loading and working with YAML configs.
Provides a simple DictConfig class that allows both dict-style and attribute-style access.
This replaces OmegaConf/Hydra dependencies with a simpler, more straightforward approach.
"""

import os
import re
import yaml
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import copy
from typing import Optional, Dict, Any, Union

class DictConfig:
    """
    A configuration object that allows both dict-style and attribute-style access.
    Supports nested configs (including lists of dicts) and variable interpolation.

    Example:
        cfg = DictConfig({'a': 1, 'b': [{'c': 2}]})
        print(cfg.b[0].c) # 2
    """

    def __init__(self, data: Optional[Dict] = None):
        if data is None:
            data = {}
        object.__setattr__(self, '_data', {})
        for key, value in data.items():
            self[key] = value
    def __getstate__(self):
        return self._data

    def __setstate__(self, state):
        object.__setattr__(self, '_data', {})
        for key, value in state.items():
            self[key] = value

    def _parse_value(self, value: Any) -> Any:
        """딕셔너리나 리스트 내부의 값을 재귀적으로 파싱하여 DictConfig로 변환합니다."""
        if isinstance(value, dict) and not isinstance(value, DictConfig):
            return DictConfig(value)
        elif isinstance(value, list):
            return [self._parse_value(item) for item in value]
        return value

    def _to_dict_value(self, value: Any) -> Any:
        """DictConfig나 리스트 안의 DictConfig를 일반 딕셔너리로 되돌립니다."""
        if isinstance(value, DictConfig):
            return value.to_dict()
        elif isinstance(value, list):
            return [self._to_dict_value(item) for item in value]
        return value

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = self._parse_value(value)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"DictConfig({self._data})"

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def pop(self, key: str, *args) -> Any:
        """Remove and return a value. Supports optional default like dict.pop()."""
        return self._data.pop(key, *args)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def update(self, other: Union[Dict, 'DictConfig']) -> None:
        if isinstance(other, DictConfig):
            other = other.to_dict()
        for key, value in other.items():
            self[key] = value

    def to_dict(self) -> Dict:
        """Convert to plain dict (recursive, including lists)."""
        result = {}
        for key, value in self._data.items():
            result[key] = self._to_dict_value(value)
        return result

    def copy(self) -> 'DictConfig':
        """Create a deep copy."""
        return DictConfig(copy.deepcopy(self.to_dict()))

def _resolve_interpolation(value: str, config: Dict) -> str:
    """
    Resolve simple variable interpolation like ${variable} or ${nested.key}.
    Does not support complex Hydra features - just simple variable substitution.
    """
    pattern = r'\$\{([^}]+)\}'

    def replace(match):
        key_path = match.group(1)
        # Handle special Hydra patterns (ignore them)
        if key_path.startswith('hydra:') or key_path.startswith('now:'):
            return match.group(0)  # Keep as-is

        keys = key_path.split('.')
        current = config
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return match.group(0)  # Can't resolve, keep as-is
            if current is None:
                return match.group(0)
        return str(current)

    return re.sub(pattern, replace, value)


def _resolve_all_interpolations(data: Dict, root_config: Optional[Dict] = None) -> Dict:
    """Recursively resolve all interpolations in a config dict."""
    if root_config is None:
        root_config = data

    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = _resolve_interpolation(value, root_config)
        elif isinstance(value, dict):
            result[key] = _resolve_all_interpolations(value, root_config)
        else:
            result[key] = value
    return result


def load_config(path: str, resolve_interpolations: bool = True) -> DictConfig:
    """
    Load a YAML config file and return a DictConfig.

    Args:
        path: Path to the YAML file
        resolve_interpolations: Whether to resolve ${variable} patterns

    Returns:
        DictConfig object with both dict and attribute access
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    if resolve_interpolations:
        data = _resolve_all_interpolations(data)

    return DictConfig(data)


def load_yaml(path: str) -> DictConfig:
    """Alias for load_config for backwards compatibility."""
    return load_config(path)


def merge_configs(base: Union[Dict, DictConfig], override: Union[Dict, DictConfig]) -> DictConfig:
    """
    Merge two configs, with override taking precedence.
    Nested dicts are merged recursively.
    """
    if isinstance(base, DictConfig):
        base = base.to_dict()
    if isinstance(override, DictConfig):
        override = override.to_dict()

    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value).to_dict()
        else:
            result[key] = deepcopy(value)

    return DictConfig(result)


def config_to_dict(cfg: Union[Dict, DictConfig]) -> Dict:
    """Convert any config to plain dict."""
    if isinstance(cfg, DictConfig):
        return cfg.to_dict()
    return dict(cfg)
