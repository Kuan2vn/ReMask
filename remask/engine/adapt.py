from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..checkpoint import load_model_weights, save_checkpoint
from ..config import dump_config
from ..data.datasets import TargetDataset
from ..data.geometry import full_to_crop
from ..modeling import build_model
from ..reliability import (
    ClassPrior,
    RunningUncertaintyStatistics,
    TemporalMemory,
    ema_update,
    underwater_perturb,
    underwater_reliability_gate,
)
from ..utils import (
    close_logger,
    denormalize,
    infinite,
    make_logger,
    normalize,
    polynomial_lr,
    resolve_device,
    seed_everything,
    seed_worker,
    set_optimizer_lr,
    weighted_cross_entropy,
)


def _stage_plan(config: Dict, requested: str) -> Tuple[int, int]:
    stage1 = int(config["adapt"]["stage1_iterations"])
    stage2 = int(config["adapt"]["stage2_iterations"])
    if requested == "all":
        return stage1 + stage2, stage1
    if requested == "stage1":
        return stage1, stage1
    if requested == "stage2":
        return stage2, 0
    raise ValueError("stage must be one of: all, stage1, stage2")


def _training_state(
    ua: RunningUncertaintyStatistics,
    class_prior: ClassPrior,
) -> Dict:
    # Temporal maps are intentionally not embedded in every checkpoint: at the
    # paper resolution they can exceed several GB. Stage-2 initialization always
    # starts with an empty memory, as described in the manuscript.
    return {
        "ua_statistics": ua.state_dict(),
        "class_prior": class_prior.state_dict(),
        "temporal_memory_saved": False,
    }


def train_remask(
    config: Dict,
    source_checkpoint: str,
    stage: str = "all",
    output_dir: Optional[str] = None,
) -> str:
    output = Path(output_dir or config["output_dir"]) / "adapt"
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(dump_config(config), encoding="utf-8")
    logger = make_logger(str(output), "remask.adapt")
    seed_everything(int(config.get("seed", 1)))
    device = resolve_device(str(config.get("device", "cuda")))

    student = build_model(config).to(device)
    payload = load_model_weights(student, source_checkpoint, prefer_teacher=True)
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    adapt_cfg = config["adapt"]
    optimizer = torch.optim.SGD(
        student.optimizer_groups(),
        lr=float(adapt_cfg["base_lr"]),
        momentum=float(adapt_cfg["momentum"]),
        weight_decay=float(adapt_cfg["weight_decay"]),
    )
    target_dataset = TargetDataset(config["data"]["target_train"], config["data"])
    generator = torch.Generator().manual_seed(int(config.get("seed", 1)))
    target_loader = DataLoader(
        target_dataset,
        batch_size=int(adapt_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 8)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    batches = infinite(target_loader)
    total_iterations, stage2_boundary = _stage_plan(config, stage)
    ua_statistics = RunningUncertaintyStatistics(
        momentum=float(adapt_cfg.get("ua_statistics_momentum", 0.99))
    )
    temporal_memory = TemporalMemory(
        confidence_momentum=float(adapt_cfg.get("temporal_confidence_momentum", 0.95))
    )
    class_prior = ClassPrior(
        int(config["model"]["num_classes"]),
        momentum=float(adapt_cfg.get("class_prior_momentum", 0.9)),
    )
    uwrg_cfg = config["uwrg"]
    logger.info(
        "start stage=%s iterations=%d source=%s target_images=%d",
        stage,
        total_iterations,
        source_checkpoint,
        len(target_dataset),
    )
    student.train()
    for iteration in range(total_iterations):
        active_stage = 2 if (stage == "stage2" or iteration >= stage2_boundary) else 1
        lr = polynomial_lr(
            adapt_cfg["base_lr"], iteration, total_iterations, adapt_cfg["lr_power"]
        )
        set_optimizer_lr(optimizer, lr)
        full_images, strong_images, geometry, names = next(batches)
        full_images = full_images.to(device, non_blocking=True)
        strong_images = strong_images.to(device, non_blocking=True)
        student_logits = student(strong_images)
        student_prob = F.softmax(student_logits, dim=1)

        with torch.no_grad():
            teacher_logits_full = teacher(full_images)
            teacher_prob_full = F.softmax(teacher_logits_full, dim=1)
            teacher_logits = full_to_crop(teacher_logits_full, geometry, mode="bilinear")
            teacher_prob = F.softmax(teacher_logits, dim=1)
            pseudo_labels = teacher_prob.argmax(dim=1)

            if active_stage == 1:
                base_reliability = ua_statistics.reliability(teacher_prob, student_prob)
            else:
                temporal = temporal_memory.reliability(
                    names,
                    teacher_prob,
                    teacher_prob_full,
                    geometry,
                    floor=float(adapt_cfg.get("temporal_floor", 0.5)),
                )
                class_weights = class_prior.pixel_weights(
                    pseudo_labels,
                    gamma=float(adapt_cfg.get("class_balance_gamma", 0.25)),
                    clip=adapt_cfg.get("class_balance_clip", (0.7, 1.3)),
                )
                base_reliability = temporal * class_weights

            uwrg_mean = 1.0
            if bool(uwrg_cfg.get("enabled", True)):
                warmup = max(1, int(total_iterations * float(uwrg_cfg.get("warmup_ratio", 0.2))))
                if stage == "stage2":
                    probe_strength = float(uwrg_cfg.get("strength", 0.35))
                else:
                    probe_strength = float(uwrg_cfg.get("strength", 0.35)) * min(
                        1.0, float(iteration + 1) / warmup
                    )
                perturbed_full = normalize(
                    underwater_perturb(
                        denormalize(full_images),
                        probe_strength,
                        attenuation_coefficients=uwrg_cfg.get(
                            "attenuation_coefficients", (1.10, 0.52, 0.26)
                        ),
                        ambient_color=uwrg_cfg.get("ambient_color", (0.07, 0.38, 0.50)),
                        depth_vertical_weight=float(uwrg_cfg.get("depth_vertical_weight", 0.70)),
                        attenuation_jitter=uwrg_cfg.get("attenuation_jitter", (0.85, 1.15)),
                        ambient_jitter=uwrg_cfg.get("ambient_jitter", (0.90, 1.10)),
                    )
                )
                perturbed_logits_full = teacher(perturbed_full)
                perturbed_logits = full_to_crop(
                    perturbed_logits_full, geometry, mode="bilinear"
                )
                perturbed_prob = F.softmax(perturbed_logits, dim=1)
                gate = underwater_reliability_gate(
                    teacher_prob,
                    perturbed_prob,
                    pseudo_labels,
                    floor=float(uwrg_cfg.get("floor", 0.8)),
                    aggregation=str(uwrg_cfg.get("aggregation", "segment")),
                    segment_stride=int(uwrg_cfg.get("segment_stride", 4)),
                    min_segment_pixels=int(uwrg_cfg.get("min_segment_pixels", 8)),
                )
                gate = gate.pow(float(uwrg_cfg.get("exponent", 1.0)))
                reliability = base_reliability * gate
                uwrg_mean = float(gate.mean().item())
            else:
                reliability = base_reliability

        optimizer.zero_grad()
        loss = weighted_cross_entropy(student_logits, pseudo_labels, reliability)
        loss.backward()
        optimizer.step()

        step = iteration + 1
        if step % int(adapt_cfg.get("ema_update_every", 1)) == 0:
            ema_update(teacher, student, float(adapt_cfg.get("ema_momentum", 0.99)))
        if active_stage == 2:
            temporal_memory.update(names, teacher_prob_full)
            class_prior.update(pseudo_labels)

        if step == 1 or step % 50 == 0 or step == stage2_boundary:
            logger.info(
                "iter=%d/%d stage=%d lr=%.3e loss=%.5f reliability=%.4f uwrg=%.4f history=%d",
                step,
                total_iterations,
                active_stage,
                lr,
                loss.item(),
                reliability.mean().item(),
                uwrg_mean,
                len(temporal_memory.labels),
            )

        if stage in ("all", "stage1") and step == stage2_boundary:
            save_checkpoint(
                str(output / "stage1_final.pth"), student, teacher, optimizer, step,
                "stage1", config, _training_state(ua_statistics, class_prior)
            )
            logger.info("Stage 1 complete; temporal memory starts empty at Stage 2")

        checkpoint_due = step % int(adapt_cfg["checkpoint_every"]) == 0 or step == total_iterations
        if checkpoint_due:
            save_checkpoint(
                str(output / "{}_iter{:06d}.pth".format("stage" + str(active_stage), step)),
                student,
                teacher,
                optimizer,
                step,
                "stage{}".format(active_stage),
                config,
                _training_state(ua_statistics, class_prior),
            )

    final_stage = 2 if stage in ("all", "stage2") else 1
    final_path = str(output / "stage{}_final.pth".format(final_stage))
    save_checkpoint(
        final_path, student, teacher, optimizer, total_iterations,
        "stage{}".format(final_stage), config,
        _training_state(ua_statistics, class_prior)
    )
    close_logger(logger)
    return final_path
