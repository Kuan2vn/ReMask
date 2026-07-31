# ReMask

Official PyTorch implementation of **ReMask** for source-free domain adaptive semantic segmentation.

ReMask adapts a source-trained model using only unlabeled target images. The released training recipe uses uncertainty-aware reliability in Stage 1, followed by temporal and class-balanced refinement in Stage 2. For underwater adaptation, an underwater reliability gate further suppresses unstable pseudo-labels. All reliability modules are used only during adaptation and add no inference-time parameters.

## Requirements

The dependency versions follow the original SND codebase:

- Python 3.8
- PyTorch 1.10.1
- torchvision 0.11.2
- CUDA 11.3

The reported experiments used one NVIDIA H100 GPU. The released code has also been verified with Python 3.14.2, PyTorch 2.10.0, torchvision 0.25.0, and CUDA 12.6.

Create an environment and install the package from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`. Install the PyTorch build appropriate for the local CUDA runtime before installing the remaining requirements when necessary.

Download the torchvision [ResNet-101 ImageNet weights](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) to `pretrain/resnet101-5d3b4d8f.pth`, or update `model.imagenet_weights` in the selected YAML file.

## Data preparation

Datasets and model weights are not stored in this repository. Download UIIS, USIS10K, SUIM, GTA5, and Cityscapes from their official sources, then update the paths under `data` in the corresponding file in `configs/`.

### Pre-trained source models

The source checkpoints used to initialize target adaptation are available below:

| Source | Target | Source checkpoint |
|---|---|---|
| UIIS | SUIM | [Download](https://drive.google.com/file/d/14loiL2D52W604yNQY4uPC1wMU52Pbhlu/view?usp=sharing) |
| USIS10K | SUIM | [Download](https://drive.google.com/file/d/19-osCtlRXWSSl3VZtq6NTX-V64RfOGdL/view?usp=sharing) |
| GTA5 | Cityscapes | [Download](https://drive.google.com/file/d/16jaY4sq1XIHnPnV1_6aF76oO8MJsolUr/view?usp=sharing) |

Place the downloaded `.pth` files in `pretrain/` and pass the selected path through `--source-checkpoint`. Source models can also be reproduced with `train_source.py`.

A typical underwater layout is:

```text
datasets/
├── UIIS/train/{images,masks}
├── USIS10K/train/{images,masks}
└── SUIM/
    ├── train_val/images
    └── TEST/{images,masks}
```

Image and mask files are paired by file stem. Recursive dataset layouts are supported.

### Label-space harmonization

The underwater configurations use a common eight-class semantic space:

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | Diver |
| 2 | Plant |
| 3 | Wreck |
| 4 | Robot |
| 5 | Reef |
| 6 | Fish |
| 7 | Sea-floor |

Convert COCO-style UIIS or USIS10K instance annotations to semantic masks before source training:

```bash
python tools/harmonize_labels.py coco --dataset uiis --annotations /path/to/uiis.json --images /path/to/UIIS/images --output /path/to/UIIS/train
python tools/harmonize_labels.py coco --dataset usis10k --annotations /path/to/usis10k.json --images /path/to/USIS10K/images --output /path/to/USIS10K/train
```

The converted masks are written to `<output>/masks` and use the common IDs above. Pixels outside annotated instances are background by default; pass `--outside ignore` if the dataset preparation protocol requires ignored pixels instead.

Existing UIIS masks can be remapped directly:

```bash
python tools/harmonize_labels.py masks --input /path/to/raw_masks --output /path/to/common_masks --input-format uiis_raw
```

## Training and evaluation

First edit the dataset paths in one of the provided configurations:

- `configs/uiis_to_suim.yaml`
- `configs/usis10k_to_suim.yaml`
- `configs/gta5_to_cityscapes.yaml`

Train the source model:

```bash
python train_source.py --config configs/uiis_to_suim.yaml
```

Run both ReMask stages from the generated source checkpoint:

```bash
python train_remask.py \
  --config configs/uiis_to_suim.yaml \
  --source-checkpoint outputs/uiis_to_suim/source/source_final.pth
```

Evaluate a checkpoint using the target validation set configured in YAML:

```bash
python evaluate.py \
  --config configs/uiis_to_suim.yaml \
  --checkpoint outputs/uiis_to_suim/adapt/stage2_final.pth
```

Further evaluation of saved checkpoints can be performed with `evaluate.py`.

Command-line overrides are available through repeated `--set KEY=VALUE` arguments. For example, `--set device=cpu` selects CPU execution for a quick check.

## Main results

Single-run performance reported in the paper using DeepLabV2 with a ResNet-101 backbone:

| Source | Target | mIoU (%) |
|---|---|---:|
| UIIS | SUIM | 49.82 |
| USIS10K | SUIM | 50.30 |
| GTA5 | Cityscapes | 58.70 |

The released YAML files contain the corresponding training schedules and adaptation settings.

## Acknowledgement

Some code is adapted from [SND](https://github.com/DZhaoXd/SND). We thank the authors for their excellent project.

## Citation

If you find this repository useful, please cite the ReMask paper. BibTeX will be added after publication.
