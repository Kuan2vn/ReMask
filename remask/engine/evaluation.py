from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data.datasets import EvaluationDataset


def _update_confusion(confusion: np.ndarray, prediction: np.ndarray, label: np.ndarray, classes: int) -> None:
    valid = (label >= 0) & (label < classes)
    encoded = classes * label[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


@torch.no_grad()
def evaluate(model: torch.nn.Module, config: Dict, device: torch.device) -> Dict:
    model.eval()
    num_classes = int(config["model"]["num_classes"])
    eval_cfg = config["evaluation"]
    dataset = EvaluationDataset(
        config["data"]["target_val"], eval_cfg["input_size"], num_classes
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels, _ in loader:
        images = images.to(device)
        logits = model(images)
        logits = F.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=True
        )
        prediction = logits.argmax(dim=1).cpu().numpy()[0]
        _update_confusion(confusion, prediction, labels.numpy()[0], num_classes)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = np.divide(intersection, union, out=np.full_like(intersection, np.nan), where=union > 0)
    accuracy = np.divide(
        intersection,
        confusion.sum(1),
        out=np.full_like(intersection, np.nan),
        where=confusion.sum(1) > 0,
    )
    return {
        "iou": iou.tolist(),
        "miou": float(np.nanmean(iou)),
        "class_accuracy": accuracy.tolist(),
        "mean_accuracy": float(np.nanmean(accuracy)),
        "pixel_accuracy": float(intersection.sum() / max(1, confusion.sum())),
        "confusion_matrix": confusion.tolist(),
    }
