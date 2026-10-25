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

## Checkpoints

Download all released [source checkpoints](https://huggingface.co/kuan2/remask-checkpoints) directly into the expected directory:

```bash
hf download kuan2/remask-checkpoints --revision b0a4659412c9fb959a04bfe0619e0462bf40b943 \
  --local-dir pretrain/checkpoints
```

The files are placed at `pretrain/checkpoints/<protocol>/source_iter_<iteration>.pth`: UIIS→SUIM at 18k, USIS10K→SUIM at 29k and GTA5→Cityscapes at 20k. The underwater launcher finds and SHA256-verifies these files automatically; no manual moving or checkpoint arguments are needed. Missing underwater source files are downloaded automatically.

SeCoV2 also downloads official SAM ViT-B weights automatically to `assets/baselines/sam_vit_b_01ec64.pth`. These releases contain source initialization weights; final adapted ReMask checkpoints are not included.

## Underwater baselines

Run Stable-ST and SeCoV2 sequentially for both UIIS→SUIM and USIS10K→SUIM (four runs):

```bash
python baselines/run.py --method all --protocol all --data data/SUIM
```

For a single experiment:

```bash
python baselines/run.py --method stable_st --protocol uiis_to_suim --data data/SUIM
```

Methods: `stable_st`, `secov2`. Protocols: `uiis_to_suim`, `usis10k_to_suim`.
No ImageNet download or source images are needed for these adaptation runs.

Validate setup and source loading on CPU, or run a short GPU correctness check:

```bash
python baselines/run.py --method all --protocol all --data data/SUIM --check
python baselines/run.py --method all --protocol all --data data/SUIM --smoke
```

Full runs use batch 4, seed 1, 30,000 updates and evaluation every 500 updates. Outputs are in `outputs/baselines/full/<method>/<protocol>`, including metrics and best/last/resume checkpoints. `--output` changes the output root. `--session-hours N` enables a checkpoint-safe pause; repeat the same command to resume a clean pause. Smoke outputs are separate and are not benchmark results.

Stable-ST retains the author trainer and NFD. SeCoV2 is a DeepLab PSA/SCC port with a simplified region head and auxiliary schedule; it is not a faithful reproduction of the released SCC trainer. DAPASS is excluded from this launcher because the checked upstream revision lacks an operational adaptation trainer. See [baseline protocol](baselines/underwater/PROTOCOL.md) for implementation scope and upstream revisions.

## ReMask reference

Set dataset paths in `configs/uiis_to_suim.yaml`, `configs/usis10k_to_suim.yaml` or `configs/gta5_to_cityscapes.yaml`, then use the source checkpoint downloaded above:

```bash
python train_remask.py --config configs/uiis_to_suim.yaml \
  --source-checkpoint pretrain/checkpoints/uiis_to_suim/source_iter_18000.pth \
  --set model.imagenet_weights=null
python evaluate.py --config configs/uiis_to_suim.yaml \
  --checkpoint outputs/uiis_to_suim/adapt/stage2_final.pth
```

The compact ReMask reference has not been numerically matched to the original manuscript runs. The paper reports 49.82/50.30 mIoU on UIIS/USIS10K→SUIM and 58.70 on GTA5→Cityscapes. These are reported paper results, not new reproduction results. [Data and checkpoint notes](HANDOFF.md).

## Acknowledgements

[Stable-ST](https://github.com/DZhaoXd/Stable-ST), [SeCoV2](https://github.com/DZhaoXd/SeCoV2), [DAPASS](https://github.com/ZZZPhaethon/DAPASS), [SND](https://github.com/DZhaoXd/SND) and [Segment Anything](https://github.com/facebookresearch/segment-anything).
