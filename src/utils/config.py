"""YAML inheritance; paths are relative to the file declaring each value."""
from pathlib import Path
import copy
import re
import yaml


def merge_config(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve(value, root):
    if isinstance(value, dict):
        return {
            key: str((root / item).resolve())
            if (key.endswith("_path") or key.endswith("_dir")) and isinstance(item, str)
            else _resolve(item, root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve(item, root) for item in value]
    return value


def load_config(path, _stack=()):
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError(f"Circular config inheritance: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    parents = config.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    merged = {}
    for parent in parents:
        merged = merge_config(merged, load_config(path.parent / parent, (*_stack, path)))
    return merge_config(merged, _resolve(config, path.parent))


def run_directory(config):
    name = config["run_name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("run_name must be a safe single directory name")
    return Path(config["output_dir"]) / name
