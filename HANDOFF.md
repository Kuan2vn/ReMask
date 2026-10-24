# ReMask: supervisor handoff

## Status checked 24 October 2026

This GitHub repository is the compact reference implementation. It has not been verified as numerically identical to the original HPC research runs or the paper's final ablation recipe. README.md covers setup, configs, paths, and run commands.

- `requirements.txt` exists and includes the Hugging Face CLI client. Install a PyTorch/torchvision build matching the host CUDA driver first.
- Public HF semantic sets: [UIIS-semantic](https://huggingface.co/datasets/kuan2/UIIS-semantic) and [USIS10K-semantic](https://huggingface.co/datasets/kuan2/USIS10K-semantic).
- Existing `kuan2/cityscape` and `kuan2/remask_underwater` dataset backups are private. Intended dataset repos `kuan2/remask-supervisor-handoff` and `kuan2/remask-gta5-private` do not exist.
- Three source checkpoints are public in [kuan2/remask-checkpoints](https://huggingface.co/kuan2/remask-checkpoints). No final target-adapted ReMask checkpoint has been published.
- Cityscapes terms prohibit dataset redistribution. Keep the raw set and private backup private; download it through the official portal.
- GTA5 is game-derived. Its [original download page](https://download.visinf.tu-darmstadt.de/data/from_games/) does not establish an explicit right to sublicense/mirror. Do not publicly upload the raw shards without permission.
- HPC SSH worked through a proxy in a prior session. In the current session, the localhost SOCKS listener responded, but requests to 10.20.8.11:22 were rejected with SOCKS reply 5 (connection refused); SSH stopped before key exchange/authentication, so no HPC-to-HF transfer occurred.

## Public source checkpoint download

```bash
hf download kuan2/remask-checkpoints --repo-type model --local-dir pretrain/checkpoints
```

Contents: UIIS→SUIM source model (iteration 18,000, 8 classes), USIS10K→SUIM source model (iteration 29,000, 8 classes), and GTA5→Cityscapes source model (iteration 20,000, 19 classes). The card contains SHA-256 values. These are source initialization weights, not final target-adapted ReMask outputs.

## HPC inventory

These paths and counts were found during the prior read-only inventory. Recheck existence and SHA256 before any transfer.

| Asset | Location / facts |
|---|---|
| Research project | `/home/20010752/SND-ReMask` |
| Cityscapes | `/home/20010752/SND-ReMask/data/cityscape`; 2,975 train, 500 val, 1,525 test images |
| GTA5 shards | `/home/20010752/segment_uncertain/gta5data`; 20 ZIPs, 24,966 image/label pairs |
| GTA5 source candidate | `/home/20010752/SND-ReMask/pretrained/model_iter019000.pth` (351,433,397 bytes) |
| SUIM source candidate | `/home/20010752/SND-ReMask/pretrained/SUIM_iter3500.pth` (344,354,997 bytes) |
| G2C ReMask candidate | `/home/20010752/Remask_backup/SND-ReMask/results/G2C_ReMask_0405/remask_model_iter017499.pth` (176,218,047 bytes) |
| Local source candidates | UIIS 18,000 iter / 8 classes; USIS10K 29,000 iter / 8 classes; GTA5 20,000 iter / 19 classes. These were not found in the searched HPC project paths. |

The G2C file is not verified as the paper-final checkpoint. Its log's best validation mIoU was about 0.5263, below the paper result 0.587. Do not label it final without matching the manuscript/evaluation record.

## Data layout

Obtain datasets through authorized sources and set paths in the selected YAML.

```text
data/GTA5/images/
data/GTA5/labels/
data/Cityscapes/leftImg8bit/train/
data/Cityscapes/leftImg8bit/val/
data/Cityscapes/gtFine/val/
```

The G2C config uses Cityscapes labels for evaluation only. GTA5 is required for source training, not target adaptation from an existing checkpoint. UIIS/USIS10K public HF copies contain semantic conversions; inspect their downloaded folder tree and set image/mask paths accordingly. SUIM is only in the private `kuan2/remask_underwater` backup and needs an authorized source/access.

## Setup and commands

Follow README.md's Python 3.10 setup and install the PyTorch/torchvision pair matching CUDA before:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

After obtaining a source checkpoint and editing `configs/gta5_to_cityscapes.yaml` paths:

```bash
python train_remask.py --config configs/gta5_to_cityscapes.yaml --source-checkpoint pretrain/GTA5_source.pth
python evaluate.py --config configs/gta5_to_cityscapes.yaml --checkpoint outputs/gta5_to_cityscapes/adapt/stage2_final.pth
```

Keep the exact code revision, environment, resolved YAML, checkpoint SHA256, seed, logs, and evaluation protocol for each ablation.

## Future HF release

User-trained weights can be considered for a public model repo after identifying the intended checkpoint and confirming it contains no recoverable third-party dataset content. Publish weights separately from datasets with provenance, code revision, SHA256, and usage terms. Never upload raw Cityscapes/GTA5 archives, credentials, SSH keys, or private logs.