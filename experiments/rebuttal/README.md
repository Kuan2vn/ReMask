# ReMask component experiments

[Open in Colab](https://colab.research.google.com/github/Kuan2vn/ReMask/blob/main/experiments/rebuttal/colab.ipynb)

The notebook reuses a manual clone/extraction, installs an isolated runtime, verifies setup and saves experiments to Google Drive. Clone `Kuan2vn/ReMask`, enter `ReMask`, then extract `remask_datasets.zip` there: its members already start with `data/`. Alternatively, set the notebook's `DATA_ZIP` and let it verify/extract the archive. Pinned source weights download automatically. `HF_TOKEN` is needed only for the private SUIM fallback when no dataset/ZIP is present. A100 and High-RAM are preferable for the full native recipe.

Cells are ordered: reference F with pixel dynamics → TC2×2 → matched UA continuation → generic probe replacement → entropy/KL → region/no-gate controls → optional frozen/split/factorial analyses. Each cell states the reviewer question, cost and interpretation. Partial runs can export a supervisor report without rerunning training.

## Run

After the repository's dataset/setup instructions:

```bash
python experiments/rebuttal/run.py --suite priority --protocol all --data data/SUIM --output /persistent/ReMask_experiments
```

Repeat the same command/output after a pause or runtime replacement. The default session budget is nine hours; a phase pauses at its next saved evaluation. Download/setup and frozen analysis take additional session time. The notebook adjusts the remaining training budget after setup/analysis. Its v2 output/code pin is separate from older runs; do not resume a v1 run with changed code/settings.

| Suite | Experiments |
|---|---|
| `priority` | Full reference, TC2×2, UA restart/continuous controls, generic probe replacement |
| `core` | Priority + entropy/KL removals, UW-pixel, no-gate,20k/10k split |
| `all` | Core + complete UA/probe factorial controls, region mean, confidence-only, no-floor,27.5k/2.5k split |

Choose one arm with `--arm TC-Tonly` or `--arm G-generic-med`; dependencies run automatically. `--seed 2` creates a separate paired replication. Inspect exact phases without downloads:

```bash
python experiments/rebuttal/run.py --suite all --protocol all --plan
python experiments/rebuttal/run.py --check --protocol all --data data/SUIM
python experiments/rebuttal/run.py --arm F --protocol uiis_to_suim --data data/SUIM --output /persistent/ReMask_experiments --smoke --smoke-refresh
```

Smoke runs use8 train/2 evaluation images, small input sizes,2+2 updates and real process restarts. `--smoke-refresh` additionally exercises pseudo-label refresh and bank restoration at update1. Their outputs are separate and are not benchmark results. Full experiments require Linux/CUDA. CPU is supported for setup and integration diagnostics.

## Frozen reliability analysis

```bash
python experiments/rebuttal/analyze.py --protocol all --data data/SUIM --output /persistent/ReMask_experiments
```

This compares underwater and magnitude-matched generic probes on the **same source-only models**. Both pixel and connected-region mean/median scores are retained, with confidence/identity controls, three draws and severities0.5/1/1.5. It resumes by image. A fixed paired UA checkpoint can be evaluated with `--checkpoint PATH --base ua --protocol uiis_to_suim`; real student weights and normalization stats are required. Use a different output root when changing checkpoint/analysis settings.

Frozen summaries also report all predeclared appearance strata: low light, low contrast, blue/green color ratio and highlight clipping. Thresholds come from the fixed unlabeled calibration images. These are overlapping appearance proxies, not annotations of physical turbidity/depth. No group is chosen by its performance delta.

## Outputs

```bash
python experiments/rebuttal/report.py --output /persistent/ReMask_experiments
python experiments/rebuttal/dynamics.py --output /persistent/ReMask_experiments
```

Outputs under `full/` include:

- `results.csv`: every available arm/phase, fixed endpoint or actual last evaluation, peak, all8 class IoUs/accuracies/Dice and BoundaryIoU. Full confusions and per-image metrics remain in each phase's `metrics.json`.
- `quality_over_time.csv`, `mismatch_5k_windows.csv`, `supervision_mass.csv`: score quality, fixed unlabeled-panel mismatch (overall/foreground/class-macro/class-wise) and sampled effective CE class weights. Temporal analysis uses the previous evaluation, distinct from training's per-image history.
- `dynamics/<protocol>/seed1/F/<phase>/`: a supervisor report, iteration/class plots, per-image distributions and persistent-agreement proxy. Enable `--save-pixel-diagnostics` before starting a run to retain stride-8 maps; older runs still export available aggregate data. Only committed observations are used. Three-observation stable teacher/student agreement is a label-free proxy, not pseudo-label correctness or a pixel-wise Stage-2 assignment. A direct adaptive-switch comparison would be required to claim superiority over that schedule.
- `paired_endpoint_contrasts.json`: paired image-bootstrap differences, conditional on the trained checkpoints.
- `frozen/<protocol>/summary.csv` and `summary.json`: all frozen-score comparisons and paired confidence intervals.
- Per-phase `CURRENT.json`, full-state checkpoints, `best.pth`, immutable pseudo-label-bank versions, `resources.json`, config and console logs. Source states,20k/25k/27.5k anchors and parent provenance remain traceable.

Resource columns describe incremental phase costs. To compare complete pipelines, include their shared Stage-A prefix once; do not count a reused prefix as zero method cost. GPU/runtime metadata is retained per segment, so mixed-hardware sessions cannot be treated as a matched speed comparison. Resume validates checkpoint/bank hashes; best-model rollback protects against an evaluation interrupted before its checkpoint transaction commits.

## Protocol

Fresh reference:25k Stage A +5k Stage B, batch4, seed1, evaluation500. Stage-A LR2.5e-4 with30k polynomial horizon; Stage-B LR1e-4 with20k horizon. Probe warmups are fixed5000/1000 updates. Stage A retains the archived host's prototype auxiliary loss(weight0.1); Stage B uses weighted CE without that auxiliary term. These are newly declared matched experiments, not numerical reproductions of the paper's headline scores.

TC branches start from the same Stage-A teacher checkpoint and use the same reset/LR policy. UA-continuous restores the full student/teacher/optimizer state. Each fork receives its own pseudo-label-bank copy. Generic/UW replacement changes the probe in **both stages**. No-gate sets the factor to1; identity probing retains confidence weighting.

All arms retain the eight-class DeepLabV2/ResNet-101 and native UA/TC/UWRG/host gradient calculations. [Trainer changes](TRAINER_PATCH.diff) add evaluation, exact update-boundary checkpoints and session hooks. [Snapshot provenance](SNAPSHOT.json) identifies retained files. Probe RNG is isolated from training augmentation RNG. Residual activation checkpointing is enabled for every arm to reduce memory without AMP or batch changes; `--no-activation-checkpointing` requires a new output root and more VRAM.

Full inference metrics use all110 test images at original mask dimensions. Reliability ranking uses a fixed pixel stride8 and mean per-image AURC with tie-group expectations. Mismatch uses a fixed32-image unlabeled panel by CLI default; the notebook selects128 with `--panel-images 128 --save-pixel-diagnostics`, consistently across arms. Teacher/student receive the same deterministic image view. Overall/class-wise counts use full-resolution outputs; spatial persistence uses stride8 and resets per phase. Test masks select peak checkpoints and evaluate scores; target-train masks are excluded from the adaptation view. No accuracy-based early stopping is used. Fresh DataLoader shuffles after resume are not a bitwise replay of an uninterrupted process.

The core costs410k incremental updates for two sources/one seed, including prefix reuse. The full matrix needs multiple Colab sessions. Preserve the Drive output directory and its recorded code revision. Checkpoints/history/banks can use several GB per phase; ordinary Drive storage may be insufficient for the entire matrix. Full GPU memory and throughput depend on the assigned runtime; a short smoke is not a full-size memory benchmark.

[Experiment priorities and interpretation](DESIGN.md). All requested arms remain available regardless of the direction of their results.
