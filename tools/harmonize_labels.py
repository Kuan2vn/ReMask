#!/usr/bin/env python
"""Convert UIIS/USIS10K annotations to the shared eight-class SUIM space."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from remask.data.taxonomy import (
    IGNORE_LABEL,
    UIIS_CATEGORY_NAME_TO_COMMON_ID,
    UIIS_RAW_TO_COMMON_ID,
    USIS_CATEGORY_TO_COMMON_ID,
    decode_mask,
    normalize_category_name,
)

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _progress(items: List[Any], description: str) -> Iterable[Any]:
    return tqdm(items, desc=description, unit="mask") if tqdm is not None else items


def _load_coco(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    missing = [key for key in ("images", "annotations", "categories") if key not in payload]
    if missing:
        raise ValueError("COCO file is missing keys: {}".format(", ".join(missing)))
    return payload


def _category_mapping(dataset: str, coco: Dict[str, Any]) -> Dict[int, int]:
    if dataset == "usis10k":
        return dict(USIS_CATEGORY_TO_COMMON_ID)
    mapping = {}
    for category in coco["categories"]:
        category_id = int(category["id"])
        normalized = normalize_category_name(category.get("name", ""))
        common_id = UIIS_CATEGORY_NAME_TO_COMMON_ID.get(normalized)
        if common_id is None:
            common_id = UIIS_RAW_TO_COMMON_ID.get(category_id)
        if common_id is None:
            raise ValueError("Unknown UIIS category: {!r}".format(category))
        mapping[category_id] = common_id
    return mapping


def _polygon_mask(polygons: List, height: int, width: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        if polygon and len(polygon) >= 6:
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            draw.polygon(points, outline=1, fill=1)
    return np.asarray(canvas, dtype=np.uint8).astype(bool)


def _uncompressed_rle(counts: List[int], height: int, width: int) -> np.ndarray:
    flat = np.zeros(height * width, dtype=np.uint8)
    value, index = 0, 0
    for count in counts:
        next_index = index + int(count)
        if value:
            flat[index:next_index] = 1
        index, value = next_index, 1 - value
    if index != flat.size:
        raise ValueError("Invalid uncompressed RLE length")
    return flat.reshape((width, height)).T.astype(bool)


def _segmentation_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        return _polygon_mask(segmentation, height, width)
    if isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        if coco_mask is None:
            return _uncompressed_rle(segmentation["counts"], height, width)
        rle = coco_mask.frPyObjects(segmentation, height, width)
    elif isinstance(segmentation, dict) and coco_mask is not None:
        rle = segmentation
    else:
        raise RuntimeError("Compressed COCO RLE requires pycocotools")
    decoded = coco_mask.decode(rle)
    return np.any(decoded, axis=2) if decoded.ndim == 3 else decoded.astype(bool)


def _find_image(images_root: Path, file_name: str) -> Optional[Path]:
    candidates = (images_root / file_name, images_root / Path(file_name).name)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def convert_coco(args: argparse.Namespace) -> None:
    annotation_path = Path(args.annotations)
    images_root = Path(args.images)
    output = Path(args.output)
    masks_dir = output / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    coco = _load_coco(annotation_path)
    mapping = _category_mapping(args.dataset, coco)
    grouped = defaultdict(list)
    for annotation in coco["annotations"]:
        grouped[int(annotation["image_id"])].append(annotation)
    images = sorted(coco["images"], key=lambda item: str(item.get("file_name", item["id"])))
    if args.limit is not None:
        images = images[:args.limit]
    outside_value = 0 if args.outside == "background" else IGNORE_LABEL
    pixel_counts = Counter()
    written = 0
    for image_info in _progress(images, "harmonizing {}".format(args.dataset)):
        image_path = _find_image(images_root, str(image_info["file_name"]))
        if image_path is None:
            raise FileNotFoundError("Image listed by COCO was not found: {}".format(image_info["file_name"]))
        with Image.open(image_path) as image:
            width = int(image_info.get("width", image.width))
            height = int(image_info.get("height", image.height))
        destination = masks_dir / (Path(image_info["file_name"]).stem + ".png")
        if destination.exists() and not args.overwrite:
            mask = np.asarray(Image.open(destination), dtype=np.uint8)
        else:
            mask = np.full((height, width), outside_value, dtype=np.uint8)
            annotations = sorted(
                grouped.get(int(image_info["id"]), []),
                key=lambda item: float(item.get("area", 0.0)),
                reverse=True,
            )
            for annotation in annotations:
                common_id = mapping.get(int(annotation["category_id"]))
                if common_id is None:
                    continue
                region = _segmentation_mask(annotation.get("segmentation"), height, width)
                mask[region] = common_id
            Image.fromarray(mask, mode="L").save(destination)
            written += 1
        values, counts = np.unique(mask, return_counts=True)
        for value, count in zip(values, counts):
            pixel_counts[int(value)] += int(count)
    manifest = {
        "dataset": args.dataset,
        "annotation_file": str(annotation_path.resolve()),
        "image_directory": str(images_root.resolve()),
        "mask_directory": str(masks_dir.resolve()),
        "outside_instance_policy": args.outside,
        "category_to_common_id": mapping,
        "images": len(images),
        "written": written,
        "pixel_counts": {str(key): pixel_counts[key] for key in sorted(pixel_counts)},
    }
    with (output / "harmonization_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


def convert_masks(args: argparse.Namespace) -> None:
    source = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in (".png", ".bmp")
    )
    if args.limit is not None:
        files = files[:args.limit]
    for path in _progress(files, "remapping masks"):
        destination = output / (path.stem + ".png")
        if destination.exists() and not args.overwrite:
            continue
        with Image.open(path) as mask:
            remapped = decode_mask(mask, args.input_format, num_classes=8)
        Image.fromarray(remapped, mode="L").save(destination)
    print("Wrote {} masks to {}".format(len(files), output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    coco = subparsers.add_parser("coco", help="Rasterize COCO instance annotations.")
    coco.add_argument("--dataset", choices=("uiis", "usis10k"), required=True)
    coco.add_argument("--annotations", required=True)
    coco.add_argument("--images", required=True)
    coco.add_argument("--output", required=True)
    coco.add_argument("--outside", choices=("background", "ignore"), default="background")
    coco.add_argument("--overwrite", action="store_true")
    coco.add_argument("--limit", type=int, default=None)
    coco.set_defaults(function=convert_coco)

    masks = subparsers.add_parser("masks", help="Remap an existing mask directory.")
    masks.add_argument("--input", required=True)
    masks.add_argument("--output", required=True)
    masks.add_argument(
        "--input-format",
        choices=("uiis_raw", "uiis_rgb", "suim_rgb", "common"),
        required=True,
    )
    masks.add_argument("--overwrite", action="store_true")
    masks.add_argument("--limit", type=int, default=None)
    masks.set_defaults(function=convert_masks)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
