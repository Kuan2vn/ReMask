from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image


IGNORE_LABEL = 255

UNDERWATER_CLASSES = (
    "Background / water",
    "Human divers",
    "Aquatic plants / sea-grass",
    "Wrecks / ruins",
    "Robots / instruments",
    "Reefs / invertebrates",
    "Fish / vertebrates",
    "Sea-floor / rocks",
)

SUIM_COLOR_TO_COMMON_ID: Dict[Tuple[int, int, int], int] = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (0, 255, 255): 3,
    (255, 0, 0): 4,
    (255, 0, 255): 5,
    (255, 255, 0): 6,
    (255, 255, 255): 7,
}

UIIS_COLOR_TO_COMMON_ID: Dict[Tuple[int, int, int], int] = {
    (0, 0, 0): 0,
    (0, 0, 255): 6,
    (0, 255, 0): 5,
    (255, 0, 0): 2,
    (255, 255, 0): 3,
    (255, 0, 255): 1,
    (0, 255, 255): 4,
    (255, 128, 0): 7,
}

UIIS_RAW_TO_COMMON_ID = {0: 0, 1: 6, 2: 5, 3: 2, 4: 3, 5: 1, 6: 4, 7: 7}
USIS_CATEGORY_TO_COMMON_ID = {1: 3, 2: 6, 3: 5, 4: 2, 5: 1, 6: 4, 7: 7}

CITYSCAPES_RAW_TO_TRAIN_ID = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}

UIIS_CATEGORY_NAME_TO_COMMON_ID = {
    "fish": 6,
    "fish and vertebrates": 6,
    "reefs": 5,
    "reef": 5,
    "reefs and invertebrates": 5,
    "aquatic plants": 2,
    "aquatic plant": 2,
    "aquatic plants and sea-grass": 2,
    "aquatic plants and sea grass": 2,
    "sea-grass": 2,
    "sea grass": 2,
    "plants": 2,
    "plant": 2,
    "wrecks/ruins": 3,
    "wrecks and ruins": 3,
    "wrecks": 3,
    "wreck": 3,
    "ruins": 3,
    "human divers": 1,
    "human diver": 1,
    "divers": 1,
    "diver": 1,
    "robots": 4,
    "robot": 4,
    "robots (auvs/rovs/instruments)": 4,
    "auvs/rovs/instruments": 4,
    "instruments": 4,
    "sea-floor": 7,
    "sea floor": 7,
    "seafloor": 7,
    "sea-floor and rocks": 7,
    "sea-floor/rocks": 7,
    "rocks": 7,
}


def normalize_category_name(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").split())


def color_mask_to_ids(rgb: np.ndarray, palette, ignore_label: int = IGNORE_LABEL) -> np.ndarray:
    output = np.full(rgb.shape[:2], ignore_label, dtype=np.uint8)
    for color, train_id in palette.items():
        output[np.all(rgb == color, axis=-1)] = train_id
    return output


def indexed_mask_to_ids(values: np.ndarray, mapping=None, ignore_label: int = IGNORE_LABEL) -> np.ndarray:
    values = np.asarray(values)
    output = np.full(values.shape[:2], ignore_label, dtype=np.uint8)
    if mapping is None:
        valid = (values >= 0) & (values < ignore_label)
        output[valid] = values[valid].astype(np.uint8)
    else:
        for raw_id, train_id in mapping.items():
            output[values == raw_id] = train_id
    return output


def decode_mask(mask: Image.Image, label_format: str, num_classes: int) -> np.ndarray:
    if label_format == "suim_rgb":
        array = color_mask_to_ids(np.asarray(mask.convert("RGB")), SUIM_COLOR_TO_COMMON_ID)
    elif label_format == "uiis_rgb":
        array = color_mask_to_ids(np.asarray(mask.convert("RGB")), UIIS_COLOR_TO_COMMON_ID)
    elif label_format == "uiis_raw":
        array = indexed_mask_to_ids(np.asarray(mask), UIIS_RAW_TO_COMMON_ID)
    elif label_format == "cityscapes_raw":
        array = indexed_mask_to_ids(np.asarray(mask), CITYSCAPES_RAW_TO_TRAIN_ID)
    elif label_format in ("common", "train_ids"):
        array = indexed_mask_to_ids(np.asarray(mask))
    else:
        raise ValueError("Unknown label format {!r}".format(label_format))
    invalid = (array >= int(num_classes)) & (array != IGNORE_LABEL)
    array[invalid] = IGNORE_LABEL
    return array
