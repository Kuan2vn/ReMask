"""SUIM-only catalog for the released underwater adaptation protocols."""
import os
from .suim import SuimDataSet, SuimDataSetTrain

class DatasetCatalog:
    DATASET_DIR = "data"
    DATASETS = {
        name: dict(data_dir="SUIM/orin_data/" + split + "/images/", masks_dir="SUIM/orin_data/" + split + "/masks/")
        for name, split in (("suim_train", "train_val"), ("suim_TT_train", "train_val"),
                            ("suim_II_train", "train_val"), ("suim_val", "TEST"))
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None, rsc=None, cfg=None):
        if name not in DatasetCatalog.DATASETS:
            raise RuntimeError(f"This release contains target SUIM loaders only: {name}")
        attrs = DatasetCatalog.DATASETS[name]
        images = os.path.join(DatasetCatalog.DATASET_DIR, attrs["data_dir"])
        masks = os.path.join(DatasetCatalog.DATASET_DIR, attrs["masks_dir"])
        options = dict(max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform,
                       cfg=cfg, label_palette="suim", mask_suffixes=(".bmp", ".png"))
        if "TT" in name:
            return SuimDataSet(images, masks, pseudo=True, **options)
        if "II" in name:
            dataset = SuimDataSet(images, masks, pseudo=False, **options)
            dataset.ignore_real_labels = True
            return dataset
        if "train" in mode:
            dataset = SuimDataSetTrain(images, masks, rsc=rsc, **options)
            dataset.ignore_real_labels = True
            return dataset
        return SuimDataSet(images, masks, **options)
