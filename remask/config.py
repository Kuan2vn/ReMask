from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


class ConfigError(ValueError):
    pass


def _deep_merge(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _parse_override(value: str) -> Any:
    return yaml.safe_load(value)


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ConfigError("Overrides must use dotted.key=value syntax: {!r}".format(item))
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        target = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = _parse_override(raw_value)
    return result


def load_config(path: str, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigError("Top-level YAML value must be a mapping.")
    config = apply_overrides(loaded, overrides)
    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required_sections = ("model", "data", "source_train", "adapt", "uwrg", "evaluation")
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ConfigError("Missing config sections: {}".format(", ".join(missing)))

    num_classes = int(config["model"].get("num_classes", 0))
    if num_classes < 2:
        raise ConfigError("model.num_classes must be at least 2")

    stage1 = int(config["adapt"].get("stage1_iterations", 0))
    stage2 = int(config["adapt"].get("stage2_iterations", 0))
    if stage1 < 0 or stage2 < 0 or stage1 + stage2 <= 0:
        raise ConfigError("adapt stage iteration counts must be non-negative and not both zero")

    floor = float(config["uwrg"].get("floor", 0.0))
    exponent = float(config["uwrg"].get("exponent", 0.0))
    if not 0.0 <= floor <= 1.0:
        raise ConfigError("uwrg.floor must be in [0, 1]")
    if exponent < 0.0:
        raise ConfigError("uwrg.exponent must be non-negative")

    clip = config["adapt"].get("class_balance_clip", [0.7, 1.3])
    if len(clip) != 2 or float(clip[0]) <= 0.0 or float(clip[0]) > float(clip[1]):
        raise ConfigError("adapt.class_balance_clip must be [positive_min, max]")

    for key in ("full_size", "crop_size"):
        value = config["data"].get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ConfigError("data.{} must be [width, height]".format(key))


def dump_config(config: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    return yaml.safe_dump(clean, sort_keys=False)
