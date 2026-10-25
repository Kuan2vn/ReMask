# Data and checkpoints

## Underwater adaptation

The baseline launcher downloads and SHA256-verifies the released source weights automatically:

| Protocol | Public checkpoint | Source iteration |
|---|---|---:|
| UIIS→SUIM | [source_iter_18000.pth](https://huggingface.co/kuan2/remask-checkpoints/blob/main/uiis_to_suim/source_iter_18000.pth) | 18,000 |
| USIS10K→SUIM | [source_iter_29000.pth](https://huggingface.co/kuan2/remask-checkpoints/blob/main/usis10k_to_suim/source_iter_29000.pth) | 29,000 |

These are source initialization weights. Final target-adapted ReMask checkpoints are not part of that release.

Provide SUIM using the layout in [README.md](README.md). The launcher creates a per-run view containing training images and evaluation images/masks. Target training masks are excluded from this view. Source images and UIIS/USIS10K labels are unnecessary for adaptation from the released sources.

[UIIS-semantic](https://huggingface.co/datasets/kuan2/UIIS-semantic) and [USIS10K-semantic](https://huggingface.co/datasets/kuan2/USIS10K-semantic) are available for source retraining. Use common 8-class labels: background, diver, plant, wreck, robot, reef, fish, floor. Native UIIS indexed labels require remapping; USIS10K background and ignore variants are separate protocols.

After the README downloads, UIIS training images/masks are in `data/UIIS/train_val/{images,masks}`. Use `label_format: uiis_raw` for native indexed PNG masks, or `uiis_rgb` for RGB masks. For USIS10K-background, images are in `data/USIS10K/USIS10K/train` and masks in `data/USIS10K/semantic_bg/train/masks`; use `label_format: common`. Set these paths in the corresponding source config before retraining.

## GTA5→Cityscapes

The public [checkpoint release](https://huggingface.co/kuan2/remask-checkpoints) also contains `gta5_to_cityscapes/source_iter_20000.pth` with 19 classes. Obtain GTA5 and Cityscapes from their authorized dataset sources and set paths in `configs/gta5_to_cityscapes.yaml`. GTA5 images are required for source retraining, not source-free adaptation from a complete checkpoint.

## Outputs and interpretation

Baseline results use the labeled 110-image SUIM evaluation set for peak checkpoint selection; labels do not supervise adaptation gradients. Report all class metrics from that same selected checkpoint, along with the actual last evaluation. A session pause is not a 30k endpoint. [Protocol and implementation scope](baselines/underwater/PROTOCOL.md).

Clean session pauses resume from the same command/output path. Preserve incomplete or failed outputs for inspection and use a different output root for a fresh run. Native Stable-ST shuffle reconstruction is not a bitwise replay of an uninterrupted DataLoader.
