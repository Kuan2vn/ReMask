# ReMask

Reference PyTorch implementation for source-free domain-adaptive semantic segmentation, with GTA5 → Cityscapes and underwater UIIS/USIS10K → SUIM configurations.

**Scope:** this GitHub repository is the compact reference implementation. Its training pipeline and checkpoints have not been verified as numerically identical to the original HPC research runs or the paper's final ablation recipe. See [HANDOFF.md](HANDOFF.md) for the current asset inventory and known limitations.

## Setup

Use Linux with Python 3.10 and a PyTorch/torchvision pair compatible with the machine's CUDA driver. Install the CUDA-enabled PyTorch build first (choose the command from the [official PyTorch selector](https://pytorch.org/get-started/locally/)); then install this project:

```bash
git clone https://github.com/Kuan2vn/ReMask.git
cd ReMask
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the matching torch and torchvision wheels from https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -m pip install -e .
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The code also needs a CUDA-compatible PyTorch build for GPU training. The optional `pycocotools` dependency is only needed for label preparation utilities.

Download the ImageNet ResNet-101 initialization file to `pretrain/resnet101-5d3b4d8f.pth` (or edit `model.imagenet_weights` in your YAML):

```bash
mkdir -p pretrain
curl -L https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -o pretrain/resnet101-5d3b4d8f.pth
```

## Data and checkpoints

Datasets and experiment checkpoints are not committed to Git. The configs contain example absolute paths; edit them to match your machine. Image and mask directories are searched recursively, with image/mask pairs matched by filename stem.

### UIIS/USIS10K → SUIM

Public semantic dataset repositories already on Hugging Face:

- [UIIS semantic masks](https://huggingface.co/datasets/kuan2/UIIS-semantic)
- [USIS10K semantic masks](https://huggingface.co/datasets/kuan2/USIS10K-semantic)

Download with the Hugging Face CLI included in `requirements.txt`:

```bash
hf download kuan2/UIIS-semantic --repo-type dataset --local-dir data/UIIS
hf download kuan2/USIS10K-semantic --repo-type dataset --local-dir data/USIS10K
```

Inspect the downloaded directories and set `data.source.images` and `data.source.masks` in the matching YAML to the actual image and mask folders. The source folders must contain paired images and common 8-class masks.

SUIM is currently in the owner's private backup [kuan2/remask_underwater](https://huggingface.co/datasets/kuan2/remask_underwater); it is not a public download. After obtaining the dataset from an authorized source, configure:

```text
data/SUIM/train_val/images/
data/SUIM/TEST/images/
data/SUIM/TEST/masks/
```

The training config uses target training images only; target validation masks are used by evaluation. Keep them out of adaptation training.

### GTA5 → Cityscapes

GTA5 raw game-derived images and Cityscapes are not mirrored in this repository. Obtain them through their authorized sources and comply with their terms. Cityscapes requires an account and its [terms prohibit redistribution](https://www.cityscapes-dataset.com/license/); do not use the private Cityscapes backup as a public download mirror. GTA5 is game-derived; use the [original dataset page](https://download.visinf.tu-darmstadt.de/data/from_games/) and do not publicly mirror its files without explicit redistribution permission. GTA5 shards must be extracted locally to image and label folders.

Expected paths for `configs/gta5_to_cityscapes.yaml`:

```text
data/GTA5/images/
data/GTA5/labels/
data/Cityscapes/leftImg8bit/train/
data/Cityscapes/leftImg8bit/val/
data/Cityscapes/gtFine/val/
```

The config expects Cityscapes-style label IDs. Set each `data.*` path to the folders above (or change them directly in the YAML). For adaptation from a source checkpoint, GTA5 is not needed; it is needed to retrain the source model.

### Source checkpoints

Three source-initialization checkpoints are now public in [kuan2/remask-checkpoints](https://huggingface.co/kuan2/remask-checkpoints). Download the full release:

```bash
hf download kuan2/remask-checkpoints --repo-type model --local-dir pretrain/checkpoints
```

| Experiment | Checkpoint path | Iteration | Classes |
|---|---|---:|---:|
| UIIS → SUIM | `pretrain/checkpoints/uiis_to_suim/source_iter_18000.pth` | 18,000 | 8 |
| USIS10K → SUIM | `pretrain/checkpoints/usis10k_to_suim/source_iter_29000.pth` | 29,000 | 8 |
| GTA5 → Cityscapes | `pretrain/checkpoints/gta5_to_cityscapes/source_iter_20000.pth` | 20,000 | 19 |

The original UIIS/USIS10K filenames say `iter020000`, but their checkpoint metadata gives the actual iterations in this table. These files are source models for adaptation, not final target-adapted ReMask outputs. No final target checkpoint has been published yet. The release README lists SHA-256 values.

To train a new source model from labeled data:

```bash
python train_source.py --config configs/gta5_to_cityscapes.yaml
```

For source-free GTA5 → Cityscapes adaptation, use the released source checkpoint:

```bash
python train_remask.py \
  --config configs/gta5_to_cityscapes.yaml \
  --source-checkpoint pretrain/checkpoints/gta5_to_cityscapes/source_iter_20000.pth
```

Available configs are `configs/gta5_to_cityscapes.yaml`, `configs/uiis_to_suim.yaml`, and `configs/usis10k_to_suim.yaml`.

## Train and evaluate

Run all adaptation stages (Stage 1 followed by Stage 2):

```bash
python train_remask.py \
  --config configs/uiis_to_suim.yaml \
  --source-checkpoint pretrain/UIIS_source.pth
```

Run one stage only with `--stage stage1` or `--stage stage2`. Override YAML values with repeated `--set dotted.key=value` options. For evaluation:

```bash
python evaluate.py \
  --config configs/uiis_to_suim.yaml \
  --checkpoint outputs/uiis_to_suim/adapt/stage2_final.pth
```

Outputs default to the `output_dir` in the selected YAML. Keep the exact code revision, resolved config, checkpoint hash, random seed, environment, logs, and evaluation protocol for each ablation.

## Results and citation

The results below are reported by the associated paper; they are not a claim that this compact reference implementation has reproduced them.

| Source → target | mIoU (%) |
|---|---:|
| UIIS → SUIM | 49.82 |
| USIS10K → SUIM | 50.30 |
| GTA5 → Cityscapes | 58.70 |

Please cite the ReMask paper when available. The SND baseline is described in [Stable Neighbor Denoising for Source-free Domain Adaptive Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Stable_Neighbor_Denoising_for_Source-free_Domain_Adaptive_Segmentation_CVPR_2024_paper.html).
