from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from .data.geometry import full_to_crop


class RunningUncertaintyStatistics:
    def __init__(self, momentum: float = 0.99, eps: float = 1e-6) -> None:
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.values = {
            "entropy_mean": None,
            "entropy_var": None,
            "kl_mean": None,
            "kl_var": None,
        }

    def _update(self, name: str, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_mean = value.mean().detach()
        batch_var = value.var(unbiased=False).detach().clamp_min(self.eps)
        mean_key, var_key = name + "_mean", name + "_var"
        if self.values[mean_key] is None:
            self.values[mean_key] = batch_mean
            self.values[var_key] = batch_var
        else:
            self.values[mean_key] = (
                self.momentum * self.values[mean_key] + (1.0 - self.momentum) * batch_mean
            )
            self.values[var_key] = (
                self.momentum * self.values[var_key] + (1.0 - self.momentum) * batch_var
            ).clamp_min(self.eps)
        return self.values[mean_key], self.values[var_key]

    def reliability(self, teacher_prob: torch.Tensor, student_prob: torch.Tensor) -> torch.Tensor:
        teacher = teacher_prob.detach().clamp_min(1e-8)
        student = student_prob.detach().clamp_min(1e-8)
        entropy = -(teacher * teacher.log()).sum(dim=1, keepdim=True)
        kl = (teacher * (teacher.log() - student.log())).sum(dim=1, keepdim=True)
        entropy_mean, entropy_var = self._update("entropy", entropy)
        kl_mean, kl_var = self._update("kl", kl)
        normalized_entropy = torch.sigmoid(
            (entropy - entropy_mean) / torch.sqrt(entropy_var + self.eps)
        )
        normalized_kl = torch.sigmoid((kl - kl_mean) / torch.sqrt(kl_var + self.eps))
        return torch.exp(-normalized_entropy - normalized_kl)

    def state_dict(self) -> Dict:
        return {
            "momentum": self.momentum,
            "eps": self.eps,
            "values": {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in self.values.items()
            },
        }

    def load_state_dict(self, state: Dict, device: torch.device) -> None:
        self.momentum = float(state.get("momentum", self.momentum))
        self.eps = float(state.get("eps", self.eps))
        for key, value in state.get("values", {}).items():
            self.values[key] = value.to(device) if torch.is_tensor(value) else value


class TemporalMemory:
    """Per-image latest teacher label and EMA-smoothed confidence map."""

    def __init__(self, confidence_momentum: float = 0.95) -> None:
        self.confidence_momentum = float(confidence_momentum)
        self.labels: Dict[str, torch.Tensor] = {}
        self.confidences: Dict[str, torch.Tensor] = {}

    def reliability(
        self,
        names: Sequence[str],
        current_prob_crop: torch.Tensor,
        current_prob_full: torch.Tensor,
        geometry,
        floor: float,
    ) -> torch.Tensor:
        current_conf_full, current_label_full = current_prob_full.max(dim=1, keepdim=True)
        historical_labels: List[torch.Tensor] = []
        historical_confidences: List[torch.Tensor] = []
        for index, name in enumerate(names):
            historical_labels.append(
                self.labels.get(str(name), current_label_full[index].detach().cpu().to(torch.uint8)).float()
            )
            historical_confidences.append(
                self.confidences.get(str(name), current_conf_full[index].detach().cpu().half()).float()
            )
        hist_label_full = torch.stack(historical_labels).to(current_prob_crop.device)
        hist_conf_full = torch.stack(historical_confidences).to(current_prob_crop.device)
        hist_label = full_to_crop(hist_label_full, geometry, mode="nearest").long()
        hist_conf = full_to_crop(hist_conf_full, geometry, mode="bilinear")
        current_conf, current_label = current_prob_crop.max(dim=1, keepdim=True)
        agreement = current_label.eq(hist_label).float()
        strength = 0.5 * (current_conf + hist_conf)
        raw = agreement * strength
        return float(floor) + (1.0 - float(floor)) * raw

    def update(self, names: Sequence[str], current_prob_full: torch.Tensor) -> None:
        confidence, label = current_prob_full.max(dim=1, keepdim=True)
        for index, name in enumerate(names):
            key = str(name)
            current_label = label[index].detach().cpu().to(torch.uint8)
            current_conf = confidence[index].detach().cpu().half()
            previous = self.confidences.get(key)
            if previous is not None:
                current_conf = (
                    self.confidence_momentum * previous.float()
                    + (1.0 - self.confidence_momentum) * current_conf.float()
                ).half()
            self.labels[key] = current_label
            self.confidences[key] = current_conf

    def state_dict(self) -> Dict:
        return {
            "confidence_momentum": self.confidence_momentum,
            "labels": self.labels,
            "confidences": self.confidences,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.confidence_momentum = float(
            state.get("confidence_momentum", self.confidence_momentum)
        )
        self.labels = state.get("labels", {})
        self.confidences = state.get("confidences", {})


class ClassPrior:
    def __init__(self, num_classes: int, momentum: float = 0.9, eps: float = 1e-6) -> None:
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.prior: Optional[torch.Tensor] = None

    def update(self, labels: torch.Tensor, ignore_index: int = 255) -> None:
        valid = labels.ne(ignore_index)
        counts = torch.bincount(
            labels[valid].reshape(-1), minlength=self.num_classes
        ).float()
        if counts.sum() <= 0:
            return
        batch_prior = counts / counts.sum()
        if self.prior is None:
            self.prior = batch_prior
        else:
            self.prior = self.momentum * self.prior + (1.0 - self.momentum) * batch_prior
            self.prior = self.prior / self.prior.sum().clamp_min(self.eps)

    def pixel_weights(
        self,
        labels: torch.Tensor,
        gamma: float,
        clip: Sequence[float],
        ignore_index: int = 255,
    ) -> torch.Tensor:
        if self.prior is None:
            prior = labels.new_full((self.num_classes,), 1.0 / self.num_classes, dtype=torch.float)
        else:
            prior = self.prior.to(labels.device).clamp_min(self.eps)
            prior = prior / prior.sum().clamp_min(self.eps)
        uniform = torch.full_like(prior, 1.0 / self.num_classes)
        class_weight = (uniform / prior).pow(float(gamma)).clamp(float(clip[0]), float(clip[1]))
        valid = labels.ne(ignore_index)
        safe = torch.where(valid, labels, torch.zeros_like(labels)).long()
        weights = class_weight[safe].unsqueeze(1)
        return torch.where(valid.unsqueeze(1), weights, torch.ones_like(weights))

    def state_dict(self) -> Dict:
        return {
            "num_classes": self.num_classes,
            "momentum": self.momentum,
            "eps": self.eps,
            "prior": self.prior.detach().cpu() if self.prior is not None else None,
        }

    def load_state_dict(self, state: Dict, device: torch.device) -> None:
        self.momentum = float(state.get("momentum", self.momentum))
        prior = state.get("prior")
        self.prior = prior.to(device) if torch.is_tensor(prior) else None


def smooth_random_field(images: torch.Tensor, grid_divisor: int = 28) -> torch.Tensor:
    batch, _, height, width = images.shape
    grid_height = max(2, height // grid_divisor)
    grid_width = max(2, width // grid_divisor)
    noise = torch.rand(
        (batch, 1, grid_height, grid_width), device=images.device, dtype=images.dtype
    )
    return F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=True)


def underwater_perturb(
    images: torch.Tensor,
    strength: float,
    attenuation_coefficients: Sequence[float] = (1.10, 0.52, 0.26),
    ambient_color: Sequence[float] = (0.07, 0.38, 0.50),
    depth_vertical_weight: float = 0.70,
    attenuation_jitter: Sequence[float] = (0.85, 1.15),
    ambient_jitter: Sequence[float] = (0.90, 1.10),
) -> torch.Tensor:
    if float(strength) <= 0.0:
        return images
    batch, _, height, _ = images.shape
    vertical = torch.linspace(0.0, 1.0, height, device=images.device, dtype=images.dtype)
    vertical = vertical.view(1, 1, height, 1).expand(batch, -1, -1, -1)
    reverse = torch.randint(0, 2, (batch, 1, 1, 1), device=images.device).bool()
    vertical = torch.where(reverse, 1.0 - vertical, vertical)
    vertical_weight = min(1.0, max(0.0, float(depth_vertical_weight)))
    depth = (
        vertical_weight * vertical
        + (1.0 - vertical_weight) * smooth_random_field(images)
    ).clamp(0.0, 1.0)
    beta = images.new_tensor(attenuation_coefficients).view(1, 3, 1, 1)
    beta_jitter = float(attenuation_jitter[0]) + (
        float(attenuation_jitter[1]) - float(attenuation_jitter[0])
    ) * torch.rand(
        (batch, 1, 1, 1), device=images.device, dtype=images.dtype
    )
    transmission = torch.exp(-beta * beta_jitter * depth * float(strength))
    ambient = images.new_tensor(ambient_color).view(1, 3, 1, 1)
    ambient *= float(ambient_jitter[0]) + (
        float(ambient_jitter[1]) - float(ambient_jitter[0])
    ) * torch.rand(
        (batch, 1, 1, 1), device=images.device, dtype=images.dtype
    )
    return (images * transmission + ambient * (1.0 - transmission)).clamp(0.0, 1.0)


def segmentwise_median(
    stability: torch.Tensor,
    pseudo_labels: torch.Tensor,
    stride: int = 4,
    min_segment_pixels: int = 8,
) -> torch.Tensor:
    original_size = stability.shape[-2:]
    stride = max(1, int(stride))
    if stride > 1:
        work_size = (
            max(1, original_size[0] // stride),
            max(1, original_size[1] // stride),
        )
        stability_work = F.interpolate(stability, size=work_size, mode="nearest")
        labels_work = F.interpolate(
            pseudo_labels.unsqueeze(1).float(), size=work_size, mode="nearest"
        ).squeeze(1).long()
    else:
        stability_work = stability
        labels_work = pseudo_labels

    stability_np = stability_work[:, 0].detach().cpu().numpy()
    labels_np = labels_work.detach().cpu().numpy()
    output = stability_np.copy()
    structure = ndimage.generate_binary_structure(2, 1)
    for batch_index in range(labels_np.shape[0]):
        for class_id in np.unique(labels_np[batch_index]):
            if int(class_id) == 255:
                continue
            components, count = ndimage.label(labels_np[batch_index] == class_id, structure)
            if count == 0:
                continue
            component_ids = np.arange(1, count + 1)
            sizes = np.bincount(components.ravel(), minlength=count + 1)[1:]
            medians = np.atleast_1d(
                ndimage.median(stability_np[batch_index], labels=components, index=component_ids)
            )
            for component_id, size, median in zip(component_ids, sizes, medians):
                if int(size) >= int(min_segment_pixels):
                    output[batch_index][components == component_id] = float(median)
    result = torch.from_numpy(output).to(stability.device, dtype=stability.dtype).unsqueeze(1)
    if result.shape[-2:] != original_size:
        result = F.interpolate(result, size=original_size, mode="nearest")
    return result


def underwater_reliability_gate(
    teacher_prob: torch.Tensor,
    perturbed_prob: torch.Tensor,
    pseudo_labels: torch.Tensor,
    floor: float,
    aggregation: str,
    segment_stride: int,
    min_segment_pixels: int,
) -> torch.Tensor:
    teacher_conf, teacher_label = teacher_prob.max(dim=1, keepdim=True)
    perturbed_conf, perturbed_label = perturbed_prob.max(dim=1, keepdim=True)
    stability = teacher_label.eq(perturbed_label).float() * 0.5 * (
        teacher_conf + perturbed_conf
    )
    if aggregation == "segment":
        stability = segmentwise_median(
            stability, pseudo_labels, segment_stride, min_segment_pixels
        )
    elif aggregation != "pixel":
        raise ValueError("uwrg.aggregation must be 'pixel' or 'segment'")
    return (float(floor) + (1.0 - float(floor)) * stability).clamp(float(floor), 1.0)


def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
            teacher_parameter.mul_(float(momentum)).add_(
                student_parameter.detach(), alpha=1.0 - float(momentum)
            )
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer)
