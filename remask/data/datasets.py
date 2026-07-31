from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

from ..utils import IMAGENET_MEAN, IMAGENET_STD
from .geometry import sample_geometry
from .taxonomy import IGNORE_LABEL, decode_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
RESAMPLE_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
RESAMPLE_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


def _list_images(root: str, suffix: Optional[str] = None) -> List[Path]:
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError("Image directory not found: {}".format(path))
    files = sorted(
        item for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in IMAGE_EXTENSIONS
        and (suffix is None or item.name.endswith(suffix))
    )
    if not files:
        raise RuntimeError("No images found in {}".format(path))
    return files


def _mask_path(image_path: Path, image_root: Path, mask_root: Path, spec: Dict) -> Path:
    relative = image_path.relative_to(image_root)
    image_suffix = spec.get("image_suffix")
    mask_suffix = spec.get("mask_suffix")
    if image_suffix and relative.name.endswith(image_suffix):
        stem = relative.name[:-len(image_suffix)]
    else:
        stem = relative.stem
    if mask_suffix:
        filename = stem + str(mask_suffix)
        candidate = mask_root / relative.parent / filename
        if candidate.is_file():
            return candidate
    for extension in (".png", ".bmp", ".jpg"):
        candidate = mask_root / relative.parent / (stem + extension)
        if candidate.is_file():
            return candidate
        flat = mask_root / (stem + extension)
        if flat.is_file():
            return flat
    raise FileNotFoundError("No mask found for {} under {}".format(image_path, mask_root))


def _pil_to_normalized(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def _apply_geometry(image: Image.Image, geometry: Dict[str, int], nearest: bool = False) -> Image.Image:
    interpolation = RESAMPLE_NEAREST if nearest else RESAMPLE_BILINEAR
    resized = image.resize(
        (geometry["resize_width"], geometry["resize_height"]), interpolation
    )
    left, top = geometry["left"], geometry["top"]
    output = resized.crop(
        (left, top, left + geometry["crop_width"], top + geometry["crop_height"])
    )
    if geometry["flip"]:
        output = TF.hflip(output)
    return output


def _color_augment(image: Image.Image, augmentation: Dict) -> Image.Image:
    return ColorJitter(
        brightness=float(augmentation.get("brightness", 0.5)),
        contrast=float(augmentation.get("contrast", 0.5)),
        saturation=float(augmentation.get("saturation", 0.5)),
        hue=float(augmentation.get("hue", 0.2)),
    )(image)


def _source_augment(image: Image.Image, augmentation: Dict) -> torch.Tensor:
    return _pil_to_normalized(_color_augment(image, augmentation))


def _strong_augment(image: Image.Image, augmentation: Dict) -> torch.Tensor:
    image = _color_augment(image, augmentation)
    if random.random() < float(augmentation.get("gaussian_blur_probability", 0.5)):
        radius = augmentation.get("gaussian_blur_radius", (0.1, 2.0))
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(float(radius[0]), float(radius[1]))))
    tensor = _pil_to_normalized(image)
    if random.random() < float(augmentation.get("cutout_probability", 0.5)):
        height, width = tensor.shape[-2:]
        cut_height = random.randint(max(1, height // 16), max(1, height // 4))
        cut_width = random.randint(max(1, width // 16), max(1, width // 4))
        top = random.randint(0, max(0, height - cut_height))
        left = random.randint(0, max(0, width - cut_width))
        tensor[:, top:top + cut_height, left:left + cut_width] = 0.0
    return tensor


class SourceDataset(Dataset):
    def __init__(self, spec: Dict, data_config: Dict, num_classes: int) -> None:
        self.spec = spec
        self.image_root = Path(spec["images"])
        self.mask_root = Path(spec["masks"])
        self.images = _list_images(str(self.image_root), spec.get("image_suffix"))
        self.num_classes = int(num_classes)
        self.full_size = data_config["full_size"]
        self.crop_size = data_config["crop_size"]
        self.scale_range = data_config["scale_range"]
        self.flip_probability = data_config.get("horizontal_flip_probability", 0.5)
        self.augmentation = data_config.get("strong_augmentation", {})

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        mask_path = _mask_path(image_path, self.image_root, self.mask_root, self.spec)
        image = Image.open(image_path).convert("RGB").resize(tuple(self.full_size), RESAMPLE_BILINEAR)
        mask_image = Image.open(mask_path)
        mask = decode_mask(mask_image, self.spec.get("label_format", "common"), self.num_classes)
        mask_pil = Image.fromarray(mask).resize(tuple(self.full_size), RESAMPLE_NEAREST)
        geometry = sample_geometry(
            self.full_size, self.crop_size, self.scale_range, self.flip_probability
        )
        image = _apply_geometry(image, geometry)
        mask_pil = _apply_geometry(mask_pil, geometry, nearest=True)
        image_tensor = _source_augment(image, self.augmentation)
        label_tensor = torch.from_numpy(np.asarray(mask_pil, dtype=np.uint8).copy()).long()
        return image_tensor, label_tensor, image_path.stem


class TargetDataset(Dataset):
    def __init__(self, spec: Dict, data_config: Dict) -> None:
        self.images = _list_images(spec["images"], spec.get("image_suffix"))
        self.full_size = data_config["full_size"]
        self.crop_size = data_config["crop_size"]
        self.scale_range = data_config["scale_range"]
        self.flip_probability = data_config.get("horizontal_flip_probability", 0.5)
        self.augmentation = data_config.get("strong_augmentation", {})

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        full_image = Image.open(image_path).convert("RGB").resize(
            tuple(self.full_size), RESAMPLE_BILINEAR
        )
        geometry = sample_geometry(
            self.full_size, self.crop_size, self.scale_range, self.flip_probability
        )
        student_image = _apply_geometry(full_image, geometry)
        return (
            _pil_to_normalized(full_image),
            _strong_augment(student_image, self.augmentation),
            geometry,
            image_path.stem,
        )


class EvaluationDataset(Dataset):
    def __init__(self, spec: Dict, input_size: Sequence[int], num_classes: int) -> None:
        self.spec = spec
        self.image_root = Path(spec["images"])
        self.mask_root = Path(spec["masks"])
        self.images = _list_images(str(self.image_root), spec.get("image_suffix"))
        self.input_size = tuple(input_size)
        self.num_classes = int(num_classes)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        mask_path = _mask_path(image_path, self.image_root, self.mask_root, self.spec)
        image = Image.open(image_path).convert("RGB")
        model_image = image.resize(self.input_size, RESAMPLE_BILINEAR)
        label = decode_mask(
            Image.open(mask_path), self.spec.get("label_format", "common"), self.num_classes
        )
        return _pil_to_normalized(model_image), torch.from_numpy(label.copy()).long(), image_path.stem
