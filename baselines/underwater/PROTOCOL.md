# Underwater SFUDA baseline protocol

## Shared evaluation

- UIIS→SUIM and USIS10K-background→SUIM, using the published source checkpoints at 18k/29k and a common 8-class DeepLabV2/ResNet-101.
- SUIM: 1,525 unlabeled target training images and 110 official test images/masks. The per-run target view excludes training masks. Source images are not used.
- Full image 1280×640; crop 960×480; scale 0.75–1.25; horizontal flip 0.5; batch 4; seed 1. Native Stable-ST also retains its blur/cutout and copy-paste pipeline.
- Fixed 30,000 segmentation optimizer updates and evaluation every 500 updates; no accuracy-based early stopping. Source-only evaluation is separate from adapted checkpoint selection.
- Select one checkpoint by 8-class mIoU and report all class metrics at that checkpoint, plus the actual last evaluation. The same labeled SUIM test set selects the peak; these are selected benchmark results, not independent held-out estimates.
- Evaluation uses bicubic 1280×640 input resize, the historical normalization and prediction at original mask dimensions, without flip/multiscale ensembling.

## Implementations

| Method | Upstream revision | Scope |
|---|---|---|
| [Stable-ST](https://github.com/DZhaoXd/Stable-ST) | `c447da35598b285eb8868caf4f969aaba1bf39b8` | Author trainer/NFD and bi-level denoising, with SUIM loaders, predicted-label class resampling and missing-class/index guards |
| [SeCoV2](https://github.com/DZhaoXd/SeCoV2) | `4868de66a607a1421dc3942dfeb168c23e58eb83` | Native PSA/GraphBuilder/GraphVAE; simplified DeepLab region encoder/projection, auxiliary schedule and corrected-label self-training host |

These are underwater ports. SeCoV2 uses a common DeepLab source architecture rather than the author's published outdoor host. Its current SCC port is not a faithful reproduction of the released trainer. No numerical reproduction of the authors' published scores is claimed.

### Stable-ST

SGD 2.5e-4, decoder multiplier 10, momentum 0.9, weight decay 5e-4; polynomial horizon 30k. Dynamic NFD feedback and teacher update, native bi-level denoising, 3000-update candidate refresh and inverse-log predicted-class image resampling are retained. The candidate cache is 640×320. Absent/singleton class pools and quantile bounds are guarded; donor retries are bounded when a tiny pseudo-class has no usable region. The underwater 30k budget replaces the original default stop/horizon.

### SeCoV2

PSA uses official SAM ViT-B, an additional frozen pretrained resource. Region features use a separate source-initialized DeepLab encoder, masked pooling and a learned 2048→128 projection. Encoder/projection/GraphVAE are trained jointly for 2,000 auxiliary updates with 32 sampled regions, encoder LR 5e-4 and projection/graph LR 5e-3, polynomial horizon 2000. Sampling follows frequency^-0.7 per region. This declared auxiliary budget differs from the released SCC schedule. The host segmenter remains source-initialized during auxiliary learning.

SCC retains the lowest-uncertainty 80% per predicted class, with correction confidence 0.95. Host training uses corrected-label CE with SGD 2.5e-4 and decoder multiplier 10. GraphSAGE parameters are materialized before optimizer creation. SAM/auxiliary preprocessing costs belong to this method and should be reported separately.

### Excluded: DAPASS

[DAPASS](https://github.com/ZZZPhaethon/DAPASS) revision `3379e76b22477e86a9d2af09e7b61fa9748698f4` contains placeholder adaptation/model files rather than an operational public trainer. It is excluded from `baselines/run.py`, including `--method all`; explicit `--method dapass` is rejected. Earlier paper-based port code and provenance are retained for historical inspection, not as a supported benchmark. Existing outputs are preserved.

## Running and continuation

The launcher supports a single arm or all four arms sequentially (Stable-ST/SeCoV2 × UIIS/USIS10K→SUIM). Each output has its own runtime/data view and lock; workers never rewrite a shared dataset path. Downloads are cached and SHA256 verified. No ImageNet initialization is fetched before loading a complete source model.

By default each arm runs to 30k. `--session-hours N` requests a pause at the next saved evaluation. Native Stable-ST starts its control timer after initial candidate preparation, so preprocessing and the checkpoint cadence can extend wall time beyond N hours. Repeat the same command to restore a clean saved pause. Best/last/full state and pseudo-label banks are retained. A fresh DataLoader shuffle is not a bitwise replay of an uninterrupted process. Preserve failed/unclean outputs and choose a different output root for a fresh run.

`--smoke` creates a separate 8-image/2-evaluation-image run, pauses after update 1 and restarts to update 2. Its scores are diagnostics only. `--check` verifies data layout, public source hashes and an 8-class CPU model forward without starting GPU training.

The model/gradient files are retained from the GPU-checked snapshot recorded in [PROVENANCE.json](PROVENANCE.json). CPU tests cover gradients, labels, CRAM, GraphVAE, source checks, checkpoint state and portable launch/restart behavior. Run `python -m unittest discover -s baselines/underwater/tests -v` from the repository root.

## Attribution

Upstream notices and available license files are retained under `references/`. SeCoV2's upstream terms limit use to non-commercial research. DAPASS's MIT license is included. Segment Anything is installed from revision `dca509fe793f601edb92606367a655c15ac00fdf`; its official SAM ViT-B checkpoint has SHA256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`.
