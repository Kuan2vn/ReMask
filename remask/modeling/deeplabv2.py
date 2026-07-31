from __future__ import annotations

import logging
import inspect
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm with fixed affine parameters and running statistics."""

    def __init__(self, num_features: int, eps: float = 0.0) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        scale = weight * torch.rsqrt(running_var + self.eps)
        return x * scale + (bias - running_mean * scale)


def _make_resnet(backbone: str, freeze_bn: bool) -> nn.Module:
    constructors = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
    }
    if backbone not in constructors:
        raise ValueError("Unsupported backbone {!r}".format(backbone))
    norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
    dilation = [False, False, False] if backbone == "resnet18" else [False, True, True]
    constructor = constructors[backbone]
    keyword = {"replace_stride_with_dilation": dilation, "norm_layer": norm_layer}
    if "weights" in inspect.signature(constructor).parameters:
        keyword["weights"] = None
    else:  # torchvision 0.11 and earlier
        keyword["pretrained"] = False
    return constructor(**keyword)


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix):]: value for key, value in state.items()}
    return state


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone: str, freeze_bn: bool, imagenet_weights: Optional[str]) -> None:
        super().__init__()
        body = _make_resnet(backbone, freeze_bn)
        if imagenet_weights:
            weights_path = Path(imagenet_weights)
            if weights_path.is_file():
                state = _torch_load(str(weights_path))
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                state = _strip_prefix(state, "module.")
                missing, unexpected = body.load_state_dict(state, strict=False)
                logging.getLogger("remask").info(
                    "Loaded ImageNet backbone %s (missing=%d, unexpected=%d)",
                    weights_path,
                    len(missing),
                    len(unexpected),
                )
            else:
                logging.getLogger("remask").warning(
                    "ImageNet weights not found at %s; model remains randomly initialized",
                    weights_path,
                )
        return_layers = {"layer4": "out"}
        self.backbone = IntermediateLayerGetter(body, return_layers=return_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)["out"]


class ASPPClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        dilations = (6, 12, 18, 24)
        self.conv2d_list = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    num_classes,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=True,
                )
                for dilation in dilations
            ]
        )
        for layer in self.conv2d_list:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor, output_size=None) -> torch.Tensor:
        logits = self.conv2d_list[0](features)
        for layer in self.conv2d_list[1:]:
            logits = logits + layer(features)
        if output_size is not None:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=True)
        return logits


class DeepLabV2(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet101",
        num_classes: int = 19,
        freeze_bn: bool = True,
        imagenet_weights: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(backbone, freeze_bn, imagenet_weights)
        channels = 512 if backbone == "resnet18" else 2048
        self.classifier = ASPPClassifier(channels, num_classes)

    def forward(self, x: torch.Tensor, output_size=None) -> torch.Tensor:
        if output_size is None:
            output_size = x.shape[-2:]
        features = self.feature_extractor(x)
        return self.classifier(features, output_size)

    def optimizer_groups(self):
        return [
            {"params": self.feature_extractor.parameters(), "lr_multiplier": 1.0},
            {"params": self.classifier.parameters(), "lr_multiplier": 10.0},
        ]


def build_model(config) -> DeepLabV2:
    model_cfg = config["model"]
    return DeepLabV2(
        backbone=str(model_cfg.get("backbone", "resnet101")),
        num_classes=int(model_cfg["num_classes"]),
        freeze_bn=bool(model_cfg.get("freeze_bn", True)),
        imagenet_weights=model_cfg.get("imagenet_weights"),
    )
