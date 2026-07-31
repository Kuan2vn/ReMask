from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def torch_load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def load_model_weights(model: torch.nn.Module, path: str, prefer_teacher: bool = True) -> Dict[str, Any]:
    payload = torch_load(path)
    ordered_keys = ("teacher", "model", "student") if prefer_teacher else ("student", "model", "teacher")
    state = next((payload[key] for key in ordered_keys if key in payload), None)
    if state is not None:
        model.load_state_dict(_strip_module_prefix(state), strict=True)
        return payload

    # Compatibility with the original SND/ReMask split checkpoint format.
    if "feature_extractor" in payload and "classifier" in payload:
        model.feature_extractor.load_state_dict(
            _strip_module_prefix(payload["feature_extractor"]), strict=True
        )
        model.classifier.load_state_dict(
            _strip_module_prefix(payload["classifier"]), strict=True
        )
        return payload

    if "teacher_feature_extractor" in payload and "teacher_classifier" in payload:
        model.feature_extractor.load_state_dict(
            _strip_module_prefix(payload["teacher_feature_extractor"]), strict=True
        )
        model.classifier.load_state_dict(
            _strip_module_prefix(payload["teacher_classifier"]), strict=True
        )
        return payload

    if payload and all(torch.is_tensor(value) for value in payload.values()):
        model.load_state_dict(_strip_module_prefix(payload), strict=True)
        return {"model": payload}
    raise ValueError("Unsupported checkpoint format: {}".format(path))


def save_checkpoint(
    path: str,
    student: torch.nn.Module,
    teacher: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    iteration: int = 0,
    stage: str = "source",
    config: Optional[Dict[str, Any]] = None,
    training_state: Optional[Dict[str, Any]] = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "student": student.state_dict(),
        "iteration": int(iteration),
        "stage": stage,
    }
    if teacher is not None:
        payload["teacher"] = teacher.state_dict()
        # These aliases keep evaluation with the earlier release straightforward.
        payload["feature_extractor"] = teacher.feature_extractor.state_dict()
        payload["classifier"] = teacher.classifier.state_dict()
    else:
        payload["model"] = student.state_dict()
        payload["feature_extractor"] = student.feature_extractor.state_dict()
        payload["classifier"] = student.classifier.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if config is not None:
        payload["config"] = {key: value for key, value in config.items() if not key.startswith("_")}
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, destination)
