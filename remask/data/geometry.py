from __future__ import annotations

import random
from typing import Dict, Sequence, Tuple

import torch
from torch.nn import functional as F


def sample_geometry(
    full_size: Sequence[int],
    crop_size: Sequence[int],
    scale_range: Sequence[float],
    flip_probability: float,
) -> Dict[str, int]:
    full_width, full_height = int(full_size[0]), int(full_size[1])
    crop_width, crop_height = int(crop_size[0]), int(crop_size[1])
    scale = random.uniform(float(scale_range[0]), float(scale_range[1]))
    resize_width = max(crop_width, int(round(full_width * scale)))
    resize_height = max(crop_height, int(round(full_height * scale)))
    left = random.randint(0, resize_width - crop_width)
    top = random.randint(0, resize_height - crop_height)
    return {
        "resize_width": resize_width,
        "resize_height": resize_height,
        "left": left,
        "top": top,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "flip": int(random.random() < float(flip_probability)),
    }


def _batch_value(geometry, key: str, index: int) -> int:
    value = geometry[key]
    if torch.is_tensor(value):
        return int(value[index].item())
    if isinstance(value, (list, tuple)):
        return int(value[index])
    return int(value)


def full_to_crop(
    tensor: torch.Tensor,
    geometry,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Apply each sample's resize/crop/flip transform to a full-resolution tensor."""
    outputs = []
    align_corners = True if mode in ("bilinear", "bicubic") else None
    for index in range(tensor.shape[0]):
        resize_height = _batch_value(geometry, "resize_height", index)
        resize_width = _batch_value(geometry, "resize_width", index)
        top = _batch_value(geometry, "top", index)
        left = _batch_value(geometry, "left", index)
        crop_height = _batch_value(geometry, "crop_height", index)
        crop_width = _batch_value(geometry, "crop_width", index)
        sample = F.interpolate(
            tensor[index:index + 1],
            size=(resize_height, resize_width),
            mode=mode,
            align_corners=align_corners,
        )
        sample = sample[:, :, top:top + crop_height, left:left + crop_width]
        if _batch_value(geometry, "flip", index):
            sample = torch.flip(sample, dims=(3,))
        outputs.append(sample)
    return torch.cat(outputs, dim=0)
