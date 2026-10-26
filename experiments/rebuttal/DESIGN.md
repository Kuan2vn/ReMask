# Experiment priorities

Prioritize experiments that measure the proposed mechanism directly and remove alternative explanations. The order is fixed before new results; it does not depend on observed gains.

| Question | Most useful primary contrast | Expected limitation / complementary evidence |
|---|---|---|
| Does the underwater probe add information beyond ordinary appearance changes? | Same-checkpoint UW vs magnitude-matched ColorJitter; same aggregation/confidence/floor/exponent; confirm with full training replacement | No-gate alone cannot isolate the probe. Score informativeness and downstream learning are separate outcomes. |
| Do temporal information and class correction both help? | Full TC vs temporal-only/class-only/neither, all from one25k checkpoint | Class correction reallocates supervision across classes, so correctness-AUROC alone may undervalue it. Report class supervision effects and all class IoUs alongside mIoU. |
| Why switch rather than continue UA? | UA25k→TC5k vs UA25k→UA5k with the same reset/LR policy | Also run continuous UA30k; do not attribute a restart effect to changing reliability signals. |
| Is25k a defensible practical schedule? | Score quality and prediction dynamics over time, plus20k/10k and optional27.5k/2.5k sensitivity at fixed total30k | A sharp crossover or mismatch below5% at exactly25k is not assumed. A useful late-refinement interval is a valid narrower conclusion. |
| Why combine entropy and KL? | Full UA vs each leave-one-component-out at Stage-A25k | The signals may be correlated and individual deltas small. Complete2×2 is available; avoid claiming each familiar ingredient is independently novel. |
| Why region aggregation? | UW-pixel vs UW-median, with identical probe outputs in frozen analysis | Mean-vs-median answers an operator-specific question. Pixel-vs-median evaluates the region aggregation package. |
| Where is the underwater probe useful? | Report whole-set and all predeclared low-light/contrast/color/clipping appearance strata from the same frozen comparisons | Strata use unlabeled image descriptors, not correctness-based sample selection or verified physical-condition annotations. They may overlap and need not all improve. |

The comparisons most likely to give clear evidence are the TC branches, controlled probe replacement, and matched UA continuation: each is closely tied to a specific mechanism. Their benefit is **interpretability**, not a guarantee of positive numbers.

The literal hypotheses “all classes stabilize together”, “mismatch becomes negligible exactly at25k”, and “every subcomponent independently raises mIoU” are unnecessarily strong and may not hold. The supplied analyses instead test class-specific dynamics, conditional component contributions and practical schedule sensitivity. These address the same underlying design questions with claims proportionate to the measurements.

No-floor, identity/confidence-only and alternate pooling are secondary controls. Small effects should be accompanied by image-level uncertainty and, when feasible, paired training seeds. Image bootstraps do not replace seed replication. A class-balancing term is constant within a predicted class at a checkpoint, so equal within-class rankings are not evidence that it has no downstream effect.

Do not remove an arm after seeing an unfavorable result, change its budget, or substitute a more favorable metric after the fact. The report exports every available arm, fixed endpoints, selected peaks and all classes. If a hypothesis fails, narrow the corresponding claim; retain the experiment and distinguish an inconclusive effect from proof of no contribution.
