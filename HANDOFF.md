# Supervisor handoff and ablations

ReMask is under review at AAAI-27.

## Readiness — 24 October 2026

- requirements.txt already exists. requirements-handoff.txt adds the HF client.
- Three local source checkpoints strictly loaded and passed CPU forward checks with the public implementation: UIIS 18000/8 classes, USIS10K 29000/8 classes, GTA5 20000/19 classes. The underwater filenames containing 020000 are misleading.
- **New HF uploads and the live HPC inventory are pending.** SSH was blocked before authentication. Intended destination repos below must not be treated as existing or complete until a completion manifest is verified.
- GPU training, exact manuscript-recipe equivalence and final manuscript checkpoint identities have not been reverified.

## Asset map

| Asset | Location |
|---|---|
| Public compact code | https://github.com/Kuan2vn/ReMask |
| UIIS | https://huggingface.co/datasets/kuan2/UIIS-semantic |
| USIS10K | https://huggingface.co/datasets/kuan2/USIS10K-semantic |
| SUIM legacy backup | https://huggingface.co/datasets/kuan2/remask_underwater (private, four 7z parts) |
| Cityscapes backup | https://huggingface.co/datasets/kuan2/cityscape (private, four tar parts and SHA256SUMS.txt) |
| Sources, final candidates, original code, environment | Intended private kuan2/remask-supervisor-handoff; upload unconfirmed |
| GTA5 | Intended private kuan2/remask-gta5-private; upload unconfirmed |

The owner must arrange authorized access for the supervisor to private assets. Each person uses their own HF login; never commit tokens, SSH keys or .env files. No access changes or messages to the supervisor have been made. Existing Google Drive source-checkpoint links remain in README.md.

## Owner: direct HPC upload

The uploader is CPU/files/network only. It never starts training, submits/cancels Slurm jobs, changes allocations or alters checkpoints. It reuses existing data backups and preserves original checkpoint bytes, including optimizer states.

Use an existing Python >=3.9 environment with huggingface_hub>=0.34,<2. Check the saved identity; authenticate yourself only if needed:

~~~bash
hf auth whoami
git clone https://github.com/Kuan2vn/ReMask.git remask-handoff
cd remask-handoff
python tools/hpc_hf_handoff.py --root /actual/original/workspace
~~~

Use the ORIGINAL research workspace, not this compact checkout. Historical candidate locations include ~/SND-ReMask and ~/papers/remask; their current existence is unverified. Inspect found/missing roles. Override paths explicitly:

~~~bash
python tools/hpc_hf_handoff.py \
  --root /actual/original/workspace \
  --checkpoint gta5_final_candidate=/actual/manuscript/checkpoint.pth \
  --gta5-root /actual/gta5 \
  --require-complete
~~~

Only identify a final checkpoint from the manuscript report/evaluation, not by picking the newest filename. Underwater stage-1/final defaults are historical candidates too. Missing final roles are recorded as unresolved; --require-complete rejects them.

Upload available assets directly from HPC in a persistent shell:

~~~bash
python tools/hpc_hf_handoff.py \
  --root /actual/original/workspace \
  --gta5-root /actual/gta5 \
  --upload
~~~

Include the overrides above and --require-complete after resolving all roles. The uploader refuses public destinations and missing source roles. GTA5 requires 24,966 paired images/masks. Native resumable upload avoids another 60+ GB archive; the HF client writes its normal cache under the GTA5 root. Two workers limit resource pressure.

The completion MANIFEST.json records SHA256, sizes, original checkpoint paths and unresolved roles. Checkpoint remote sizes/LFS hashes and every expected GTA5 path/size are checked. A manifest is not proof of reproduced paper metrics. environment_packages.json records package names/versions from the upload shell: invoke the uploader in the actual training environment if that environment is being documented.

## Supervisor: setup and data

Install the matching torch/torchvision CUDA build first. Use the same environment across compared arms:

~~~bash
python -m pip install -r requirements-handoff.txt
python -m pip install -e .
~~~

Requirements specify supported ranges, not an exact HPC lock. Confirm the original environment before claiming reproduction. After upload completion and private access are confirmed:

~~~bash
hf download kuan2/remask-supervisor-handoff --repo-type dataset --local-dir assets/handoff
hf download kuan2/UIIS-semantic --repo-type dataset --local-dir datasets/UIIS
hf download kuan2/USIS10K-semantic --repo-type dataset --local-dir datasets/USIS10K
python tools/preflight_handoff.py --assets assets/handoff
~~~

Preflight checks manifest SHA256, source iterations, strict loading and small CPU forwards. It never starts training. Original .pth files use pickle; use owner-provided files only.

Restore Cityscapes from its existing backup:

~~~bash
hf download kuan2/cityscape --repo-type dataset --include 'archives/*' --local-dir assets/cityscape
cd assets/cityscape/archives
sha256sum -c SHA256SUMS.txt
cat cityscape.tar.part-0000 cityscape.tar.part-0001 cityscape.tar.part-0002 cityscape.tar.part-0003 | tar -tf -
~~~

Inspect paths before extracting. The historical layout is cityscape/data/{train,val} and cityscape/label/{train,val}; the released YAML assumes leftImg8bit/gtFine, so update paths. Expected: 2,975 train and 500 validation images with labelIds masks.

For SUIM, download all four SND_ReMask.7z parts from the private underwater repo, then inspect/extract SND_ReMask/data/SUIM/orin_data with 7-Zip. Expected: 1,525 train and 110 test image/mask pairs. Full GTA5 can be downloaded from kuan2/remask-gta5-private only after upload confirmation. GTA5 images are needed for source retraining, but not source-free target adaptation from a supplied checkpoint.

## Configuration and labels

Copy the appropriate configs/*_to_*.yaml and set actual data/weight paths. Retain the resolved config with each run.

- Common underwater IDs: background, diver, plant, wreck, robot, reef, fish, floor (0..7).
- Native UIIS indexed masks use a different order: set data.source.label_format=uiis_raw or harmonize first using tools/harmonize_labels.py. Do not treat native IDs as common.
- SUIM uses suim_rgb. Preserve the declared USIS10K background-versus-ignore policy; the HF repo contains both semantic_bg and semantic_ignore.
- Target training masks must not enter adaptation gradients.
- A nested GTA5 labels/labels directory is supported by the uploader; use the actual mask directory in training configs.
- Set the downloaded ImageNet weight path, or explicitly clear it when loading a complete source checkpoint.

## Ablations

The public compact train_remask.py is a configurable reference for new experiments. The original_research_code.zip snapshot preserves the original HPC trainer, configs and historical launchers. For manuscript-level ablations use the original resolved recipe/logs; the compact implementation has not been numerically matched to the paper. An archived Slurm script alone does not establish the final recipe. Do not invoke sbatch/salloc on the owner's HPC account without explicit authorization.

Example compact-reference arms, after preparing configs/uiis_local.yaml:

~~~bash
python train_remask.py --config configs/uiis_local.yaml --source-checkpoint assets/handoff/checkpoints/source_uiis_18000.pth --output outputs/uiis/full
python train_remask.py --config configs/uiis_local.yaml --source-checkpoint assets/handoff/checkpoints/source_uiis_18000.pth --output outputs/uiis/no_uwrg --set uwrg.enabled=false
python train_remask.py --config configs/uiis_local.yaml --source-checkpoint assets/handoff/checkpoints/source_uiis_18000.pth --output outputs/uiis/no_class_balance --set adapt.class_balance_gamma=0
python evaluate.py --config configs/uiis_local.yaml --checkpoint outputs/uiis/full/adapt/stage2_final.pth
~~~

These are proposed commands, not launched runs. Confirm GPU memory and the authorized execution environment first. Keep source hash, seeds, batch size, resolution, budget, evaluation split/cadence and selection rule identical. Class-balance gamma=0 gives unity weights in this implementation.

Retain exact code commit, environment, resolved config, checkpoint/source hashes, seed, logs and metrics. Report both validation-selected peak and fixed endpoint; compute all class metrics at the same selected checkpoint. State explicitly when target evaluation labels select checkpoints. Do not compare peaks selected from different checkpoint grids.

The compact release does not restore temporal memory/RNG for exact resume. Do not represent an interrupted restart as uninterrupted training or change stopping/budget for only one arm. Reproduce the baseline and resolve original-vs-compact recipe differences before a final method claim.
