# ReMask

Reference implementation of ReMask and underwater source-free domain adaptation baselines.

## Setup

Linux, Python 3.10–3.13 and a CUDA-capable NVIDIA GPU. The baseline recipes use one GPU with at least 12 GiB free memory.

```bash
pip install -r requirements.txt
```

## Datasets

Download the exact [SUIM split](https://huggingface.co/datasets/kuan2/SUIM-ReMask) used in these experiments (1,525 train / 110 test images). The repository is private; log in with a Hugging Face account that has access:

```bash
hf auth login
hf download kuan2/SUIM-ReMask SUIM.zip --repo-type dataset --local-dir data
python -m zipfile -e data/SUIM.zip data
```

This creates `data/SUIM/{train_val,TEST}/{images,masks}`. The adaptation launcher excludes training masks from its data view.

For source retraining, download [UIIS](https://huggingface.co/datasets/kuan2/UIIS-semantic) and [USIS10K](https://huggingface.co/datasets/kuan2/USIS10K-semantic):

```bash
hf download kuan2/UIIS-semantic --repo-type dataset --local-dir data/UIIS
hf download kuan2/USIS10K-semantic --repo-type dataset --local-dir data/USIS10K
```

Configure source paths and mask encoding as described in [data notes](HANDOFF.md). Adaptation from the released source checkpoints needs only SUIM.

## Underwater baselines

Run Stable-ST, SeCoV2 and DAPASS sequentially for both UIIS→SUIM and USIS10K→SUIM:

```bash
python baselines/run.py --method all --protocol all --data data/SUIM
```

For a single experiment:

```bash
python baselines/run.py --method stable_st --protocol uiis_to_suim --data data/SUIM
```

Methods: `stable_st`, `secov2`, `dapass`. Protocols: `uiis_to_suim`, `usis10k_to_suim`.
Source checkpoints are downloaded automatically from [Hugging Face](https://huggingface.co/kuan2/remask-checkpoints); SeCoV2 also downloads official SAM ViT-B weights. Downloads are revision/hash checked and cached in `assets/baselines`. No ImageNet download or source images are needed for these adaptation runs.

Validate setup and source loading on CPU, or run a short GPU correctness check:

```bash
python baselines/run.py --method all --protocol all --data data/SUIM --check
python baselines/run.py --method all --protocol all --data data/SUIM --smoke
```

Full runs use batch 4, seed 1, 30,000 updates and evaluation every 500 updates. Outputs are in `outputs/baselines/full/<method>/<protocol>`, including metrics and best/last/resume checkpoints. `--output` changes the output root. `--session-hours N` enables a checkpoint-safe pause; repeat the same command to resume a clean pause. Smoke outputs are separate and are not benchmark results.

Stable-ST retains the author trainer and NFD. SeCoV2 is a DeepLab PSA/SCC port; DAPASS is a paper-based DeepLab port because the released adaptation code is incomplete. See [baseline protocol](baselines/underwater/PROTOCOL.md) for the exact scope, resources and upstream revisions.

## ReMask reference

Set dataset paths in `configs/uiis_to_suim.yaml`, `configs/usis10k_to_suim.yaml` or `configs/gta5_to_cityscapes.yaml`. Source weights are public:

```bash
hf download kuan2/remask-checkpoints --local-dir pretrain/checkpoints
python train_remask.py --config configs/uiis_to_suim.yaml \
  --source-checkpoint pretrain/checkpoints/uiis_to_suim/source_iter_18000.pth \
  --set model.imagenet_weights=null
python evaluate.py --config configs/uiis_to_suim.yaml \
  --checkpoint outputs/uiis_to_suim/adapt/stage2_final.pth
```

The compact ReMask reference has not been numerically matched to the original manuscript runs. The paper reports 49.82/50.30 mIoU on UIIS/USIS10K→SUIM and 58.70 on GTA5→Cityscapes. These are reported paper results, not new reproduction results. [Data and checkpoint notes](HANDOFF.md).

## Acknowledgements

[Stable-ST](https://github.com/DZhaoXd/Stable-ST), [SeCoV2](https://github.com/DZhaoXd/SeCoV2), [DAPASS](https://github.com/ZZZPhaethon/DAPASS), [SND](https://github.com/DZhaoXd/SND) and [Segment Anything](https://github.com/facebookresearch/segment-anything).
