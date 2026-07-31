from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logging.getLogger("remask").warning("CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def make_logger(output_dir: str, name: str = "remask") -> logging.Logger:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(path / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def infinite(loader: Iterable[Any]) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def polynomial_lr(base_lr: float, iteration: int, max_iterations: int, power: float) -> float:
    progress = min(max(float(iteration) / max(1, max_iterations), 0.0), 1.0)
    return float(base_lr) * ((1.0 - progress) ** float(power))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, backbone_lr: float) -> None:
    for group in optimizer.param_groups:
        multiplier = float(group.get("lr_multiplier", 1.0))
        group["lr"] = backbone_lr * multiplier


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reliability: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    pixel_loss = torch.nn.functional.cross_entropy(
        logits, labels.long(), ignore_index=ignore_index, reduction="none"
    )
    weights = reliability.squeeze(1)
    valid = labels.ne(ignore_index)
    numerator = (pixel_loss * weights * valid).sum()
    denominator = (weights * valid).sum().clamp_min(eps)
    return numerator / denominator


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
