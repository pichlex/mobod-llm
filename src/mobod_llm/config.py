import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must be a mapping")
    return data


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        set_nested(updated, key, value)
    return updated


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for section in ("freeze_layers", "adapters", "datasets", "quant_lora"):
        if section in updated:
            updated[section] = _normalize_experiment(updated[section])
    if "runtime" in updated:
        updated["runtime"] = _normalize_runtime(updated["runtime"])
    return updated


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _normalize_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(runtime)
    if "seed" in updated:
        updated["seed"] = int(updated["seed"])
    return updated


def _normalize_experiment(cfg: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    for key in ("batch_size", "epochs", "max_length"):
        if key in updated:
            updated[key] = int(updated[key])
    if "learning_rate" in updated:
        updated["learning_rate"] = float(updated["learning_rate"])
    if "layer_freezing" in updated:
        updated["layer_freezing"] = [int(x) for x in updated["layer_freezing"]]
    return updated
