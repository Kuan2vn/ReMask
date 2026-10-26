import argparse
from collections import deque
import copy
import datetime
import logging
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from core.configs import cfg
from core.configs.underwater import add_underwater_label_args, apply_underwater_label_args
from core.datasets import build_dataset
from core.models import build_classifier, build_feature_extractor
from core.solver import adjust_learning_rate
from core.utils.logger import setup_logger
from core.utils.losses import full2weak, update_running_conf, pseudo_labels_probs
from core.utils.metric_logger import MetricLogger
from core.utils.misc import WeightEMA, mkdir, strip_prefix_if_present
from core.apis.inference import run_candidate, run_test
from datasets.generate_city_label_info import gen_lb_info


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def _set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def _cfg_get(node, name, default):
    return getattr(node, name) if name in node else default


def _seed_everything(seed, rank=0):
    if seed is None:
        return
    seed = int(seed)
    if seed < 0:
        return
    seed = seed + int(rank)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(rng_state, logger=None):
    if not rng_state:
        return
    try:
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    except Exception as exc:
        if logger is not None:
            logger.warning("Could not restore RNG state exactly: {}".format(exc))


def _optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _to_device_if_tensor(value, device):
    return value.to(device) if torch.is_tensor(value) else value


def _state_dict_for_load(state_dict):
    return strip_prefix_if_present(state_dict, "module.")


def _ensure_classifier_buffers_from_state(classifier, state_dict):
    if not state_dict:
        return
    clean_state = _state_dict_for_load(state_dict)
    cls_module = _unwrap(classifier)
    # Checkpoints are loaded on CPU after the classifier has moved to its device.
    device = next(cls_module.parameters()).device
    if "global_prototypes" in clean_state and not hasattr(cls_module, "global_prototypes"):
        cls_module.register_buffer(
            "global_prototypes",
            torch.zeros_like(clean_state["global_prototypes"], device=device),
        )
    if (
        "global_prototypes_valid" in clean_state
        and not hasattr(cls_module, "global_prototypes_valid")
    ):
        cls_module.register_buffer(
            "global_prototypes_valid",
            torch.zeros_like(clean_state["global_prototypes_valid"], device=device),
        )


def _load_module_state(module, state_dict, strict=False):
    if not state_dict:
        return
    _unwrap(module).load_state_dict(_state_dict_for_load(state_dict), strict=strict)


def _remask_fusion_mode(cfg):
    return str(_cfg_get(cfg.REMASK, "FUSE_MODE", "none")).lower()


def _remask_fusion_enabled(cfg):
    return _remask_fusion_mode(cfg) not in ("none", "off", "false", "0", "")


def _safe_sigmoid_float(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _resolve_late_fusion_schedule(cfg, iteration, temporal_reliability):
    """Return (stage, gate).

    stage is one of:
      - uc: entropy/KL reliability only
      - temp: temporal-label reliability only
      - tc: temporal-label reliability with class balance
      - blend_tc: loss-level blend between UC and TC using gate
    """
    mode = _remask_fusion_mode(cfg)
    t0 = int(_cfg_get(cfg.REMASK, "FUSE_T0", 25000))
    t1 = int(_cfg_get(cfg.REMASK, "FUSE_T1", 24000))
    t2 = int(_cfg_get(cfg.REMASK, "FUSE_T2", 26000))
    tau = max(float(_cfg_get(cfg.REMASK, "FUSE_TAU", 500.0)), 1e-6)

    if mode == "late_hard":
        return ("tc", 1.0) if iteration >= t0 else ("uc", 0.0)

    if mode == "late_sigmoid":
        gate = _safe_sigmoid_float((float(iteration) - float(t0)) / tau)
        return "blend_tc", gate

    if mode == "three_stage":
        if iteration < t1:
            return "uc", 0.0
        if iteration < t2:
            return "temp", 0.5
        return "tc", 1.0

    if mode == "late_stability":
        if iteration < t0:
            return "blend_tc", 0.0
        threshold = float(_cfg_get(cfg.REMASK, "FUSE_STABILITY_THRESHOLD", 0.65))
        stability_tau = max(float(_cfg_get(cfg.REMASK, "FUSE_STABILITY_TAU", 0.05)), 1e-6)
        stability = float(temporal_reliability.mean().item())
        gate = _safe_sigmoid_float((stability - threshold) / stability_tau)
        return "blend_tc", gate

    raise ValueError(
        "Unsupported REMASK.FUSE_MODE '{}'. Expected one of: none, late_hard, "
        "late_sigmoid, three_stage, late_stability.".format(mode)
    )


def _resolve_pre_temporal_fusion_schedule(cfg, iteration):
    """Return UC-only schedule before the temporal branch can affect the loss."""
    mode = _remask_fusion_mode(cfg)
    t0 = int(_cfg_get(cfg.REMASK, "FUSE_T0", 25000))
    t1 = int(_cfg_get(cfg.REMASK, "FUSE_T1", 24000))
    tau = max(float(_cfg_get(cfg.REMASK, "FUSE_TAU", 500.0)), 1e-6)
    gate_eps = max(float(_cfg_get(cfg.REMASK, "FUSE_GATE_EPS", 1e-6)), 0.0)

    if mode == "late_hard" and iteration < t0:
        return "uc", 0.0

    if mode == "three_stage" and iteration < t1:
        return "uc", 0.0

    if mode == "late_stability" and iteration < t0:
        return "uc", 0.0

    if mode == "late_sigmoid":
        gate = _safe_sigmoid_float((float(iteration) - float(t0)) / tau)
        if gate <= gate_eps:
            return "uc", gate

    return None


def _image_space(x, cfg):
    mean = x.new_tensor(cfg.INPUT.PIXEL_MEAN).view(1, -1, 1, 1)
    std = x.new_tensor(cfg.INPUT.PIXEL_STD).view(1, -1, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def _normalized_space(x, cfg):
    mean = x.new_tensor(cfg.INPUT.PIXEL_MEAN).view(1, -1, 1, 1)
    std = x.new_tensor(cfg.INPUT.PIXEL_STD).view(1, -1, 1, 1)
    return (x.clamp(0.0, 1.0) - mean) / std


def _smooth_random_field(x, grid_divisor=28):
    batch, _, height, width = x.shape
    grid_h = max(2, height // grid_divisor)
    grid_w = max(2, width // grid_divisor)
    noise = torch.rand((batch, 1, grid_h, grid_w), device=x.device, dtype=x.dtype)
    return F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=True)


def _underwater_reliability_perturb(x, strength):
    """Apply a mild underwater image-formation probe in RGB image space."""
    if strength <= 0:
        return x

    batch, _, height, _ = x.shape
    vertical = torch.linspace(
        0.0,
        1.0,
        height,
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, height, 1)
    reverse_depth = torch.randint(
        0,
        2,
        (batch, 1, 1, 1),
        device=x.device,
        dtype=torch.int64,
    ).bool()
    vertical = torch.where(reverse_depth, 1.0 - vertical, vertical)
    depth = (0.70 * vertical + 0.30 * _smooth_random_field(x)).clamp(0.0, 1.0)

    # Red light attenuates fastest underwater, followed by green and blue.
    beta = x.new_tensor([1.10, 0.52, 0.26]).view(1, 3, 1, 1)
    beta_jitter = 0.85 + 0.30 * torch.rand(
        (batch, 1, 1, 1),
        device=x.device,
        dtype=x.dtype,
    )
    transmission = torch.exp(-beta * beta_jitter * depth * strength)

    # Water-column ambient light supplies the haze/backscatter component.
    ambient = x.new_tensor([0.07, 0.38, 0.50]).view(1, 3, 1, 1)
    ambient_jitter = 0.90 + 0.20 * torch.rand(
        (batch, 1, 1, 1),
        device=x.device,
        dtype=x.dtype,
    )
    ambient = (ambient * ambient_jitter).clamp(0.0, 1.0)
    return (x * transmission + ambient * (1.0 - transmission)).clamp(0.0, 1.0)


def _segmentwise_median_stability(
    pixel_stability,
    pseudo_label,
    ignore_label,
    stride,
    min_segment_pixels,
):
    """Replace pixel stability by the median of each pseudo-label component."""
    original_size = pixel_stability.shape[-2:]
    stride = max(1, int(stride))
    min_segment_pixels = max(1, int(min_segment_pixels))

    if stride > 1:
        work_size = (
            max(1, original_size[0] // stride),
            max(1, original_size[1] // stride),
        )
        stability_work = F.interpolate(
            pixel_stability,
            size=work_size,
            mode="nearest",
        )
        label_work = F.interpolate(
            pseudo_label.unsqueeze(1).float(),
            size=work_size,
            mode="nearest",
        ).squeeze(1).long()
    else:
        stability_work = pixel_stability
        label_work = pseudo_label

    stability_np = stability_work[:, 0].detach().cpu().numpy()
    label_np = label_work.detach().cpu().numpy()
    output_np = stability_np.copy()
    structure = ndimage.generate_binary_structure(2, 1)
    segment_count = 0

    for batch_idx in range(label_np.shape[0]):
        valid_classes = np.unique(label_np[batch_idx])
        for class_id in valid_classes:
            if int(class_id) == int(ignore_label):
                continue
            components, num_components = ndimage.label(
                label_np[batch_idx] == class_id,
                structure=structure,
            )
            if num_components == 0:
                continue

            component_ids = np.arange(1, num_components + 1)
            component_sizes = np.bincount(
                components.ravel(),
                minlength=num_components + 1,
            )
            large_components = component_sizes[1:] >= min_segment_pixels
            if not large_components.any():
                continue

            component_medians = np.atleast_1d(
                ndimage.median(
                    stability_np[batch_idx],
                    labels=components,
                    index=component_ids,
                )
            )
            component_pixels = components > 0
            pixel_component_ids = components[component_pixels]
            use_median = large_components[pixel_component_ids - 1]
            output_values = output_np[batch_idx][component_pixels]
            output_values[use_median] = component_medians[
                pixel_component_ids[use_median] - 1
            ]
            output_np[batch_idx][component_pixels] = output_values
            segment_count += int(large_components.sum())

    segment_stability = torch.from_numpy(output_np).to(
        device=pixel_stability.device,
        dtype=pixel_stability.dtype,
    ).unsqueeze(1)
    if segment_stability.shape[-2:] != original_size:
        segment_stability = F.interpolate(
            segment_stability,
            size=original_size,
            mode="nearest",
        )
    return segment_stability, segment_count


def _compute_underwater_stability_gate(
    teacher_prob,
    underwater_prob,
    pseudo_label,
    ignore_label,
    floor,
    mode,
    segment_stride,
    min_segment_pixels,
):
    stability, agreement, segment_count = _compute_underwater_stability(
        teacher_prob,
        underwater_prob,
        pseudo_label,
        ignore_label,
        mode,
        segment_stride,
        min_segment_pixels,
    )
    floor = min(1.0, max(0.0, float(floor)))
    gate = floor + (1.0 - floor) * stability
    return gate.clamp(floor, 1.0), agreement, segment_count


def _compute_underwater_stability(
    teacher_prob,
    underwater_prob,
    pseudo_label,
    ignore_label,
    mode,
    segment_stride,
    min_segment_pixels,
):
    """Compute pixel/approximate-segment underwater prediction stability."""
    teacher_conf, teacher_label = teacher_prob.max(dim=1, keepdim=True)
    underwater_conf, underwater_label = underwater_prob.max(dim=1, keepdim=True)
    agreement = (teacher_label == underwater_label).float()
    pixel_stability = agreement * 0.5 * (teacher_conf + underwater_conf)

    segment_count = 0
    if mode == "segment":
        stability, segment_count = _segmentwise_median_stability(
            pixel_stability,
            pseudo_label,
            ignore_label,
            segment_stride,
            min_segment_pixels,
        )
    elif mode == "pixel":
        stability = pixel_stability
    else:
        raise ValueError("Unsupported UWRG mode '{}'.".format(mode))

    return stability.clamp(0.0, 1.0), agreement, segment_count


def _uwrg_probe_strength(iteration, max_iters, max_strength, warmup_ratio):
    warmup_iters = max(1, int(max_iters * max(0.0, float(warmup_ratio))))
    ramp = min(1.0, float(iteration + 1) / warmup_iters)
    return max(0.0, float(max_strength)) * ramp


def _ensure_remask_candidates(cfg, teacher_fea, teacher_cls, local_rank, distributed):
    ctr_dir = os.path.join(cfg.OUTPUT_DIR, "CTR")
    ctr_o_dir = os.path.join(cfg.OUTPUT_DIR, "CTR_O")
    ctr_p = os.path.join(cfg.OUTPUT_DIR, "CTR.p")
    ctr_o_p = os.path.join(cfg.OUTPUT_DIR, "CTR_O.p")

    if (
        os.path.exists(ctr_dir)
        and os.path.exists(ctr_o_dir)
        and os.path.exists(ctr_p)
        and os.path.exists(ctr_o_p)
    ):
        return

    run_candidate(
        cfg,
        teacher_fea,
        teacher_cls,
        local_rank,
        distributed,
        init_candidate=True,
        update_meta=True,
    )
    gen_lb_info(cfg, "CTR")
    gen_lb_info(cfg, "CTR_O")


def _build_target_loader(cfg, batch_size, distributed):
    target_data = build_dataset(cfg, mode="train", is_source=False, rsc="CTR_O.p")
    if cfg.TCR.OPEN:
        target_data.mixer.label_to_file, _ = pickle.load(
            open(os.path.join(cfg.OUTPUT_DIR, "CTR.p"), "rb")
        )

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(target_data)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        target_data,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
        worker_init_fn=_seed_worker if int(_cfg_get(cfg, "SEED", -1)) >= 0 else None,
    )
    return target_data, loader


def _update_mixup_sampling(cfg, target_data, running_conf):
    if not cfg.TCR.OPEN or not hasattr(target_data, "mix_classes"):
        return

    weights = 1.0 - running_conf[target_data.mix_classes]
    weights = weights.clamp_min(0.0)
    if weights.sum() <= cfg.REMASK.EPSILON:
        weights = torch.ones_like(weights) / len(target_data.mix_classes)
    else:
        weights = weights / weights.sum()
    target_data.mix_p = weights.detach().cpu().numpy()


def _update_running_scalar_stats(stats, name, value, momentum, eps):
    batch_mean = value.mean().detach()
    batch_var = value.var(unbiased=False).detach()
    mean_key = "{}_mean".format(name)
    var_key = "{}_var".format(name)

    if stats[mean_key] is None:
        stats[mean_key] = batch_mean
        stats[var_key] = batch_var.clamp_min(eps)
    else:
        stats[mean_key] = momentum * stats[mean_key] + (1.0 - momentum) * batch_mean
        stats[var_key] = momentum * stats[var_key] + (1.0 - momentum) * batch_var
        stats[var_key] = stats[var_key].clamp_min(eps)

    return stats[mean_key], stats[var_key]


def _normalize_uncertainty(value, mean, var, eps):
    normalized = (value - mean) / torch.sqrt(var + eps)
    return torch.sigmoid(normalized)


def _as_name_list(names):
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return [str(names)]


def _compute_reliability(cfg, teacher_prob, student_prob, uncertainty_stats):
    if not _cfg_get(cfg.REMASK, "USE_RELIABILITY", True):
        return torch.ones_like(teacher_prob[:, :1])

    entropy = -torch.sum(teacher_prob * torch.log(teacher_prob.clamp_min(1e-8)), dim=1, keepdim=True)
    kl = torch.sum(
        teacher_prob
        * (torch.log(teacher_prob.clamp_min(1e-8)) - torch.log(student_prob.clamp_min(1e-8))),
        dim=1,
        keepdim=True,
    )

    if _cfg_get(cfg.REMASK, "KL_SIGMOID", True):
        entropy_mean, entropy_var = _update_running_scalar_stats(
            uncertainty_stats,
            "entropy",
            entropy,
            _cfg_get(cfg.REMASK, "NORM_MOMENTUM", 0.99),
            cfg.REMASK.EPSILON,
        )
        kl_mean, kl_var = _update_running_scalar_stats(
            uncertainty_stats,
            "kl",
            kl,
            _cfg_get(cfg.REMASK, "NORM_MOMENTUM", 0.99),
            cfg.REMASK.EPSILON,
        )
        entropy = _normalize_uncertainty(entropy, entropy_mean, entropy_var, cfg.REMASK.EPSILON)
        kl = _normalize_uncertainty(kl, kl_mean, kl_var, cfg.REMASK.EPSILON)
    else:
        # Tối ưu hóa: Dùng Absolute Scaling thay vì Batch Min-Max để tránh sai lệch scale giữa các batch.
        import math
        entropy = entropy / math.log(cfg.MODEL.NUM_CLASSES)
        # KL không có giới hạn trên tuyệt đối, ta dùng giá trị raw (đã có KL_SCALE xử lý)

    kl = kl * cfg.REMASK.KL_SCALE

    energy = cfg.REMASK.ALPHA * entropy + cfg.REMASK.BETA * kl
    return torch.exp(-energy / cfg.REMASK.TAU)


def _compute_temporal_reliability(cfg, teacher_prob, hist_prob):
    floor = _cfg_get(cfg.REMASK, "TEMPORAL_FLOOR", 0.5)
    agreement = torch.sum(teacher_prob * hist_prob, dim=1, keepdim=True)
    conf_power = _cfg_get(cfg.REMASK, "TEMPORAL_CONF_POWER", 0.0)
    if conf_power > 0:
        confidence = teacher_prob.max(dim=1, keepdim=True)[0]
        agreement = agreement * confidence.pow(conf_power)
    return floor + (1.0 - floor) * agreement


def _compute_temporal_label_reliability(cfg, teacher_prob, hist_label, hist_conf):
    raw_reliability = _compute_temporal_label_raw(
        cfg,
        teacher_prob,
        hist_label,
        hist_conf,
    )
    floor = _cfg_get(cfg.REMASK, "TEMPORAL_FLOOR", 0.5)
    return floor + (1.0 - floor) * raw_reliability


def _compute_temporal_label_raw(cfg, teacher_prob, hist_label, hist_conf):
    curr_conf, curr_label = teacher_prob.max(dim=1, keepdim=True)
    agreement = (curr_label == hist_label.long()).float()
    if hist_conf is None or not _cfg_get(cfg.REMASK, "HISTORY_USE_CONF", True):
        reliability = agreement
    else:
        strength = 0.5 * curr_conf + 0.5 * hist_conf
        conf_power = _cfg_get(cfg.REMASK, "TEMPORAL_CONF_POWER", 0.0)
        if conf_power > 0:
            strength = strength.pow(conf_power)
        reliability = agreement * strength
    return reliability.clamp(0.0, 1.0)


def _compute_temporal_label_reliability_from_history(
    cfg,
    teacher_prob,
    teacher_prob_full,
    tgt_trans_param,
    name_list,
    temporal_label_history,
    device,
):
    hist_labels = []
    hist_confs = []
    history_hits = 0
    history_depth = 0.0
    curr_conf_full, curr_label_full = teacher_prob_full.max(dim=1, keepdim=True)
    history_stride = max(1, int(_cfg_get(cfg.REMASK, "HISTORY_STRIDE", 1)))
    use_history_conf = _cfg_get(cfg.REMASK, "HISTORY_USE_CONF", True)

    for idx, name in enumerate(name_list):
        hist_item = temporal_label_history.get(name)
        if hist_item is None:
            hist_label = curr_label_full[idx : idx + 1].float()
            if history_stride > 1:
                hist_label = F.interpolate(
                    hist_label,
                    scale_factor=1.0 / history_stride,
                    mode="nearest",
                    recompute_scale_factor=False,
                )
            hist_label = hist_label[0].detach().cpu().to(torch.uint8)
            if use_history_conf:
                hist_conf = curr_conf_full[idx : idx + 1]
                if history_stride > 1:
                    hist_conf = F.interpolate(
                        hist_conf,
                        scale_factor=1.0 / history_stride,
                        mode="bilinear",
                        align_corners=True,
                        recompute_scale_factor=False,
                    )
                hist_conf = hist_conf[0].detach().cpu().half()
            else:
                hist_conf = None
        else:
            hist_label, hist_conf = _read_temporal_label_snapshot(hist_item)
            if isinstance(hist_item, (list, deque)):
                history_depth += len(hist_item)
            else:
                history_depth += 1
            history_hits += 1
        hist_labels.append(hist_label.float())
        if use_history_conf and hist_conf is not None:
            hist_confs.append(hist_conf.float())

    hist_label_full = torch.stack(hist_labels, dim=0).to(device, non_blocking=True)
    hist_label = full2weak(
        hist_label_full,
        tgt_trans_param,
        down_ratio=history_stride,
        nearest=True,
    )
    if hist_label.shape[-2:] != teacher_prob.shape[-2:]:
        hist_label = F.interpolate(
            hist_label,
            size=teacher_prob.shape[-2:],
            mode="nearest",
        )

    if use_history_conf and hist_confs:
        hist_conf_full = torch.stack(hist_confs, dim=0).to(device, non_blocking=True)
        hist_conf = full2weak(
            hist_conf_full,
            tgt_trans_param,
            down_ratio=history_stride,
        )
        if hist_conf.shape[-2:] != teacher_prob.shape[-2:]:
            hist_conf = F.interpolate(
                hist_conf,
                size=teacher_prob.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
    else:
        hist_conf = None

    temporal_raw = _compute_temporal_label_raw(
        cfg,
        teacher_prob,
        hist_label,
        hist_conf,
    )
    temporal_floor = _cfg_get(cfg.REMASK, "TEMPORAL_FLOOR", 0.5)
    temporal_reliability = temporal_floor + (1.0 - temporal_floor) * temporal_raw
    return temporal_reliability, temporal_raw, history_hits, history_depth


def _compute_uctr_temporal_reliability(
    temporal_raw,
    underwater_stability,
    temporal_floor,
    uctr_lambda,
    tau_low,
    tau_high,
):
    """Condition raw temporal reliability on segment-level underwater fragility."""
    temporal_trust = (
        (temporal_raw - tau_low) / (tau_high - tau_low)
    ).clamp(0.0, 1.0)
    fragility = 1.0 - underwater_stability
    correction = (
        1.0 - uctr_lambda * temporal_trust * fragility
    ).clamp(0.0, 1.0)
    corrected_raw = temporal_raw * correction
    reliability = temporal_floor + (1.0 - temporal_floor) * corrected_raw
    return reliability, temporal_trust, fragility, correction


def _compute_anchor_reliability(cfg, teacher_prob, anchor_prob):
    floor = _cfg_get(cfg.REMASK, "ANCHOR_FLOOR", 0.5)
    mode = _cfg_get(cfg.REMASK, "ANCHOR_MODE", "label")
    if mode == "prob":
        agreement = torch.sum(teacher_prob * anchor_prob, dim=1, keepdim=True)
    elif mode == "label":
        teacher_label = teacher_prob.argmax(dim=1, keepdim=True)
        anchor_label = anchor_prob.argmax(dim=1, keepdim=True)
        anchor_conf = anchor_prob.max(dim=1, keepdim=True)[0]
        agreement = (teacher_label == anchor_label).float() * anchor_conf
    else:
        raise ValueError("Unsupported REMASK.ANCHOR_MODE '{}'.".format(mode))
    return floor + (1.0 - floor) * agreement


def _compute_class_balance_reliability(cfg, pseudo_label, class_prior):
    num_classes = cfg.MODEL.NUM_CLASSES
    prior = class_prior.detach().float().clamp_min(cfg.REMASK.EPSILON)
    prior = prior / prior.sum().clamp_min(cfg.REMASK.EPSILON)
    desired = torch.full_like(prior, 1.0 / num_classes)
    gamma = _cfg_get(cfg.REMASK, "CLASS_BALANCE_GAMMA", 0.25)
    min_weight = _cfg_get(cfg.REMASK, "CLASS_BALANCE_MIN", 0.5)
    max_weight = _cfg_get(cfg.REMASK, "CLASS_BALANCE_MAX", 1.5)
    class_weight = (desired / prior).pow(gamma).clamp(min_weight, max_weight)

    valid = pseudo_label != cfg.INPUT.IGNORE_LABEL
    safe_label = torch.where(valid, pseudo_label, torch.zeros_like(pseudo_label))
    pixel_weight = class_weight[safe_label.long()].unsqueeze(1)
    return torch.where(
        valid.unsqueeze(1),
        pixel_weight,
        torch.ones_like(pixel_weight),
    )


def _logit_adjust_reliability(cfg, reliability, pseudo_label, class_prior):
    num_classes = cfg.MODEL.NUM_CLASSES
    eps = cfg.REMASK.EPSILON
    prior_floor = _cfg_get(cfg.REMASK, "RELIABILITY_ADJUST_PRIOR_FLOOR", 1e-4)
    lam = _cfg_get(cfg.REMASK, "RELIABILITY_ADJUST_LAMBDA", 0.2)

    prior = class_prior.detach().float().clamp_min(prior_floor)
    prior = prior / prior.sum().clamp_min(eps)
    uniform = reliability.new_full((num_classes,), 1.0 / num_classes)
    class_adjust = lam * torch.log(prior.to(reliability.device) / uniform)

    valid = pseudo_label != cfg.INPUT.IGNORE_LABEL
    safe_label = torch.where(valid, pseudo_label, torch.zeros_like(pseudo_label))
    pixel_adjust = class_adjust[safe_label.long()].unsqueeze(1)

    r = reliability.clamp(eps, 1.0 - eps)
    logit_r = torch.log(r) - torch.log1p(-r)
    adjusted = torch.sigmoid(logit_r - pixel_adjust)
    return torch.where(valid.unsqueeze(1), adjusted, reliability)


def _update_class_reliability_stats(cfg, reliability, pseudo_label, class_rel_mean, class_rel_valid):
    momentum = _cfg_get(cfg.REMASK, "CLASS_CALIBRATION_MOMENTUM", 0.95)
    rel = reliability.detach().squeeze(1)
    valid = pseudo_label != cfg.INPUT.IGNORE_LABEL

    with torch.no_grad():
        for class_id in range(cfg.MODEL.NUM_CLASSES):
            class_mask = valid & (pseudo_label == class_id)
            if not class_mask.any():
                continue
            batch_mean = rel[class_mask].mean()
            if class_rel_valid[class_id]:
                class_rel_mean[class_id] = (
                    momentum * class_rel_mean[class_id]
                    + (1.0 - momentum) * batch_mean
                )
            else:
                class_rel_mean[class_id] = batch_mean
                class_rel_valid[class_id] = True


def _compute_class_calibration_reliability(cfg, reliability, pseudo_label, class_rel_mean):
    gamma = _cfg_get(cfg.REMASK, "CLASS_CALIBRATION_GAMMA", 0.5)
    min_weight = _cfg_get(cfg.REMASK, "CLASS_CALIBRATION_MIN", 0.7)
    max_weight = _cfg_get(cfg.REMASK, "CLASS_CALIBRATION_MAX", 1.3)

    valid = pseudo_label != cfg.INPUT.IGNORE_LABEL
    safe_label = torch.where(valid, pseudo_label, torch.zeros_like(pseudo_label))
    class_mean = class_rel_mean.detach().clamp_min(cfg.REMASK.EPSILON)
    pixel_mean = class_mean[safe_label.long()].unsqueeze(1)
    relative = reliability.detach().clamp_min(cfg.REMASK.EPSILON) / pixel_mean
    pixel_weight = relative.pow(gamma).clamp(min_weight, max_weight)
    return torch.where(
        valid.unsqueeze(1),
        pixel_weight,
        torch.ones_like(pixel_weight),
    )


def _update_temporal_label_queue(history, update_counts, name, curr_label, curr_conf, cfg):
    queue_size = max(1, int(_cfg_get(cfg.REMASK, "HISTORY_QUEUE_SIZE", 1)))
    update_period = max(1, int(_cfg_get(cfg.REMASK, "HISTORY_UPDATE_PERIOD", 1)))
    momentum = _cfg_get(cfg.REMASK, "HISTORY_MOMENTUM", 0.9)
    update_counts[name] = update_counts.get(name, 0) + 1

    if name not in history:
        history[name] = deque([(curr_label, curr_conf)], maxlen=queue_size)
        return

    if update_counts[name] % update_period != 0:
        return

    if queue_size == 1:
        _, prev_conf = history[name][-1]
        if prev_conf is None or curr_conf is None:
            updated_conf = curr_conf
        else:
            updated_conf = (momentum * prev_conf.float() + (1.0 - momentum) * curr_conf.float()).half()
        history[name] = deque([(curr_label, updated_conf)], maxlen=queue_size)
        return

    if history[name].maxlen != queue_size:
        history[name] = deque(history[name], maxlen=queue_size)
    history[name].append((curr_label, curr_conf))


def _read_temporal_label_snapshot(hist_item):
    if isinstance(hist_item, (list, deque)):
        if len(hist_item) == 0:
            raise ValueError("Empty temporal label history queue.")
        hist_item = hist_item[0]

    if isinstance(hist_item, tuple):
        if len(hist_item) == 2:
            return hist_item
        if len(hist_item) == 1:
            return hist_item[0], None

    raise ValueError(
        "Invalid temporal label history item type={} length={}".format(
            type(hist_item).__name__,
            len(hist_item) if hasattr(hist_item, "__len__") else "NA",
        )
    )


def _set_temporal_label_snapshot(history, name, curr_label, curr_conf, cfg):
    queue_size = max(1, int(_cfg_get(cfg.REMASK, "HISTORY_QUEUE_SIZE", 1)))
    history[name] = deque([(curr_label, curr_conf)], maxlen=queue_size)


def _refresh_temporal_history_snapshot(
    cfg,
    teacher_fea,
    teacher_cls,
    target_data,
    batch_size,
    device,
    reliability_mode,
    temporal_prob_history,
    temporal_label_history,
    temporal_label_update_counts,
    logger,
):
    if reliability_mode not in ("temporal", "temporal_label"):
        return

    snapshot_loader = torch.utils.data.DataLoader(
        target_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_seed_worker if int(_cfg_get(cfg, "SEED", -1)) >= 0 else None,
    )
    temporal_label_update_counts.clear()
    if reliability_mode == "temporal":
        temporal_prob_history.clear()
    else:
        temporal_label_history.clear()

    was_training_fea = teacher_fea.training
    was_training_cls = teacher_cls.training
    teacher_fea.eval()
    teacher_cls.eval()

    history_stride = max(1, int(_cfg_get(cfg.REMASK, "HISTORY_STRIDE", 1)))
    use_history_conf = _cfg_get(cfg.REMASK, "HISTORY_USE_CONF", True)
    num_snapshots = 0
    with torch.no_grad():
        for _, _, names, _, tgt_img_full, _ in snapshot_loader:
            tgt_img_full = tgt_img_full.to(device, non_blocking=True)
            _, teacher_features_full = teacher_fea(tgt_img_full)
            teacher_logits_full = teacher_cls(teacher_features_full, tgt_img_full.shape[-2:])
            teacher_prob_full = F.softmax(teacher_logits_full, dim=1)
            name_list = _as_name_list(names)

            if reliability_mode == "temporal":
                for idx, name in enumerate(name_list):
                    temporal_prob_history[name] = teacher_prob_full[idx].detach().cpu().half()
                    num_snapshots += 1
                continue

            curr_conf_full, curr_label_full = teacher_prob_full.max(dim=1, keepdim=True)
            for idx, name in enumerate(name_list):
                curr_label = curr_label_full[idx : idx + 1].float()
                if history_stride > 1:
                    curr_label = F.interpolate(
                        curr_label,
                        scale_factor=1.0 / history_stride,
                        mode="nearest",
                        recompute_scale_factor=False,
                    )
                curr_label = curr_label[0].detach().cpu().to(torch.uint8)
                if use_history_conf:
                    curr_conf = curr_conf_full[idx : idx + 1]
                    if history_stride > 1:
                        curr_conf = F.interpolate(
                            curr_conf,
                            scale_factor=1.0 / history_stride,
                            mode="bilinear",
                            align_corners=True,
                            recompute_scale_factor=False,
                        )
                    curr_conf = curr_conf[0].detach().cpu().half()
                else:
                    curr_conf = None
                _set_temporal_label_snapshot(
                    temporal_label_history,
                    name,
                    curr_label,
                    curr_conf,
                    cfg,
                )
                num_snapshots += 1

    if was_training_fea:
        teacher_fea.train()
    if was_training_cls:
        teacher_cls.train()
    logger.info(
        "Refreshed ReMask {} history snapshots: {}".format(
            reliability_mode,
            num_snapshots,
        )
    )


def _apply_loss_weight_floor(cfg, reliability):
    floor = _cfg_get(cfg.REMASK, "LOSS_WEIGHT_FLOOR", 0.0)
    if floor <= 0:
        return reliability
    return floor + (1.0 - floor) * reliability


def _reliability_weighted_ce_loss(cfg, reliability, ce_loss, valid_mask):
    loss_weight = _apply_loss_weight_floor(cfg, reliability).squeeze(1)
    valid_reliability = loss_weight * valid_mask
    return (loss_weight * ce_loss).sum() / (valid_reliability.sum() + 1e-6)


def _image_representations(features, reliability, eps):
    reliability_f = F.interpolate(
        reliability,
        size=features.shape[-2:],
        mode="bilinear",
        align_corners=True,
    )
    z = torch.sum(reliability_f * features, dim=(2, 3))
    z = z / (torch.sum(reliability_f, dim=(2, 3)) + eps)
    return F.normalize(z, dim=1), reliability_f


def _local_prototypes(cfg, features, pseudo_labels, reliability_f):
    B, C_f, H_f, W_f = features.shape
    num_classes = cfg.MODEL.NUM_CLASSES
    labels_f = F.interpolate(
        pseudo_labels.unsqueeze(1).float(),
        size=(H_f, W_f),
        mode="nearest",
    ).squeeze(1).long()

    valid_mask = labels_f != cfg.INPUT.IGNORE_LABEL
    safe_labels = torch.where(valid_mask, labels_f, torch.zeros_like(labels_f))
    class_mask = F.one_hot(safe_labels, num_classes).permute(0, 3, 1, 2).float()
    class_mask = class_mask * valid_mask.unsqueeze(1).float()
    weighted_mask = class_mask * reliability_f
    reliability_sum = weighted_mask.sum(dim=(2, 3))
    pixel_count = class_mask.sum(dim=(2, 3))

    prototypes = features.new_zeros(B, num_classes, C_f)
    valid = reliability_sum > cfg.REMASK.DELTA

    for b in range(B):
        for c in range(num_classes):
            if valid[b, c]:
                numerator = torch.sum(weighted_mask[b, c].unsqueeze(0) * features[b], dim=(1, 2))
                prototypes[b, c] = F.normalize(
                    numerator / (reliability_sum[b, c] + cfg.REMASK.EPSILON),
                    dim=0,
                )

    mean_reliability = reliability_sum / (pixel_count + cfg.REMASK.EPSILON)
    return prototypes, valid, mean_reliability


def _refine_with_neighbors(cfg, prototypes, valid, image_repr):
    if prototypes.shape[0] <= 1 or cfg.REMASK.LAMBDA_N <= 0:
        return prototypes

    sim_matrix = torch.matmul(image_repr, image_repr.t())
    sim_matrix.fill_diagonal_(-float("inf"))
    neighbor_topk = _cfg_get(cfg.REMASK, "NEIGHBOR_TOPK", 0)
    if neighbor_topk > 0 and neighbor_topk < sim_matrix.shape[1]:
        _, topk_idx = torch.topk(sim_matrix, neighbor_topk, dim=1)
        topk_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        topk_mask.scatter_(1, topk_idx, True)
        sim_matrix = sim_matrix.masked_fill(~topk_mask, -float("inf"))
    weights = F.softmax(sim_matrix / cfg.REMASK.TAU, dim=1)
    refined = prototypes.clone()

    for b in range(prototypes.shape[0]):
        for c in range(prototypes.shape[1]):
            if not valid[b, c]:
                continue
            neighbor_valid = valid[:, c].float()
            neighbor_weights = weights[b] * neighbor_valid
            if neighbor_weights.sum() <= 0:
                continue
            neighbor_weights = neighbor_weights / neighbor_weights.sum()
            neighbor_proto = torch.sum(neighbor_weights.view(-1, 1) * prototypes[:, c], dim=0)
            refined[b, c] = F.normalize(
                (1.0 - cfg.REMASK.LAMBDA_N) * prototypes[b, c]
                + cfg.REMASK.LAMBDA_N * neighbor_proto,
                dim=0,
            )

    return refined


def _ensure_global_prototypes(cfg, classifier, prototypes):
    cls_module = _unwrap(classifier)
    num_classes = cfg.MODEL.NUM_CLASSES
    channels = prototypes.shape[-1]

    if not hasattr(cls_module, "global_prototypes"):
        cls_module.register_buffer(
            "global_prototypes",
            prototypes.new_zeros(num_classes, channels),
        )
        cls_module.register_buffer(
            "global_prototypes_valid",
            torch.zeros(num_classes, dtype=torch.bool, device=prototypes.device),
        )
    return cls_module.global_prototypes, cls_module.global_prototypes_valid


def _prototype_contrastive_loss(cfg, classifier, prototypes, valid, mean_reliability):
    if not _cfg_get(cfg.REMASK, "USE_CONTRASTIVE", True) or cfg.REMASK.LAMBDA <= 0:
        return prototypes.new_tensor(0.0)

    num_classes = cfg.MODEL.NUM_CLASSES
    global_prototypes, global_valid = _ensure_global_prototypes(cfg, classifier, prototypes)

    losses = []

    for b in range(prototypes.shape[0]):
        for c in range(num_classes):
            if not valid[b, c]:
                continue

            proto = prototypes[b, c]
            active = global_valid
            if active.sum() <= 1:
                continue

            logits = torch.matmul(global_prototypes[active].detach(), proto) / cfg.REMASK.GAMMA
            active_classes = torch.arange(num_classes, device=proto.device)[active]
            target_index = (active_classes == c).nonzero(as_tuple=False).view(-1)
            if target_index.numel() == 0:
                continue

            target = target_index[:1]
            loss = F.cross_entropy(logits.view(1, -1), target)
            losses.append(mean_reliability[b, c].detach() * loss)

    if not losses:
        return prototypes.new_tensor(0.0)

    return torch.stack(losses).mean()


def _update_global_prototypes(cfg, classifier, prototypes, valid):
    if not _cfg_get(cfg.REMASK, "USE_CONTRASTIVE", True) or cfg.REMASK.LAMBDA <= 0:
        return

    num_classes = cfg.MODEL.NUM_CLASSES
    global_prototypes, global_valid = _ensure_global_prototypes(cfg, classifier, prototypes)

    with torch.no_grad():
        for b in range(prototypes.shape[0]):
            for c in range(num_classes):
                if not valid[b, c]:
                    continue

                proto = prototypes[b, c].detach()
                if global_valid[c]:
                    updated = cfg.REMASK.MU * global_prototypes[c] + (1.0 - cfg.REMASK.MU) * proto
                else:
                    updated = proto
                    global_valid[c] = True
                global_prototypes[c].copy_(F.normalize(updated, dim=0))


def train(cfg, local_rank, distributed, use_snd_conf=False, uwrg_options=None):
    logger = logging.getLogger("DTST.trainer")
    logger.info("Start ReMask training")

    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor = build_feature_extractor(cfg).to(device)
    classifier = build_classifier(cfg).to(device)

    batch_size = cfg.SOLVER.BATCH_SIZE
    if distributed:
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size()) // 2
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)

        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            process_group=pg1,
        )

        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            process_group=pg2,
        )
        torch.distributed.barrier()

    optimizer_fea = torch.optim.SGD(
        feature_extractor.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    optimizer_cls = torch.optim.SGD(
        classifier.parameters(),
        lr=cfg.SOLVER.BASE_LR * 10,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    start_iteration = 0
    checkpoint = None
    resume_iteration = (
        int(getattr(uwrg_options, "resume_iteration", 0))
        if uwrg_options is not None
        else 0
    )
    resume_training_state = False
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = _torch_load_checkpoint(cfg.resume)
        if resume_iteration < 0:
            resume_iteration = int(checkpoint.get("iteration", 0))
        resume_training_state = resume_iteration > 0
        if resume_training_state:
            feature_weights = checkpoint.get(
                "student_feature_extractor",
                checkpoint.get("feature_extractor"),
            )
            classifier_weights = checkpoint.get(
                "student_classifier",
                checkpoint.get("classifier"),
            )
        else:
            feature_weights = checkpoint.get(
                "feature_extractor",
                checkpoint.get("student_feature_extractor"),
            )
            classifier_weights = checkpoint.get(
                "classifier",
                checkpoint.get("student_classifier"),
            )
        _ensure_classifier_buffers_from_state(classifier, classifier_weights)
        _load_module_state(feature_extractor, feature_weights, strict=False)
        _load_module_state(classifier, classifier_weights, strict=False)
        if resume_training_state:
            if "optimizer_fea" in checkpoint:
                optimizer_fea.load_state_dict(checkpoint["optimizer_fea"])
                _optimizer_to(optimizer_fea, device)
                logger.info("Loaded feature optimizer state")
            if "optimizer_cls" in checkpoint:
                optimizer_cls.load_state_dict(checkpoint["optimizer_cls"])
                _optimizer_to(optimizer_cls, device)
                logger.info("Loaded classifier optimizer state")
            start_iteration = resume_iteration
            logger.info(
                "Continuing ReMask schedule from iteration {}".format(
                    start_iteration
                )
            )
        elif "optimizer_fea" in checkpoint or "optimizer_cls" in checkpoint:
            logger.info(
                "Ignoring optimizer state from checkpoint because "
                "--resume-iteration is 0; starting a fresh ReMask adaptation."
            )

    feature_extractor.train()
    classifier.train()

    teacher_fea = copy.deepcopy(feature_extractor).to(device)
    teacher_cls = copy.deepcopy(classifier).to(device)
    if (
        resume_training_state
        and checkpoint is not None
        and "teacher_feature_extractor" in checkpoint
    ):
        _load_module_state(
            teacher_fea,
            checkpoint["teacher_feature_extractor"],
            strict=False,
        )
        logger.info("Loaded teacher feature extractor state")
    if (
        resume_training_state
        and checkpoint is not None
        and "teacher_classifier" in checkpoint
    ):
        _ensure_classifier_buffers_from_state(
            teacher_cls,
            checkpoint["teacher_classifier"],
        )
        _load_module_state(
            teacher_cls,
            checkpoint["teacher_classifier"],
            strict=False,
        )
        logger.info("Loaded teacher classifier state")
    teacher_fea.eval()
    teacher_cls.eval()
    _set_requires_grad(teacher_fea, False)
    _set_requires_grad(teacher_cls, False)

    anchor_fea = None
    anchor_cls = None
    if _cfg_get(cfg.REMASK, "USE_ANCHOR", False):
        anchor_fea = copy.deepcopy(feature_extractor).to(device)
        anchor_cls = copy.deepcopy(classifier).to(device)
        anchor_fea.eval()
        anchor_cls.eval()
        _set_requires_grad(anchor_fea, False)
        _set_requires_grad(anchor_cls, False)
        logger.info(
            "ReMask anchor consistency enabled: mode={} floor={}".format(
                _cfg_get(cfg.REMASK, "ANCHOR_MODE", "label"),
                _cfg_get(cfg.REMASK, "ANCHOR_FLOOR", 0.5),
            )
        )

    _ensure_remask_candidates(cfg, teacher_fea, teacher_cls, local_rank, distributed)
    target_data, target_loader = _build_target_loader(cfg, batch_size, distributed)
    logger.info("Target training samples: {}".format(len(target_data)))

    teacher_fea_ema = WeightEMA(
        list(teacher_fea.parameters()),
        list(feature_extractor.parameters()),
        alpha=cfg.DTU.EMA_WEIGHT,
    )
    teacher_cls_ema = WeightEMA(
        list(teacher_cls.parameters()),
        list(classifier.parameters()),
        alpha=cfg.DTU.EMA_WEIGHT,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.INPUT.IGNORE_LABEL, reduction="none")
    running_conf = torch.zeros(cfg.MODEL.NUM_CLASSES, device=device).fill_(0.001)
    uncertainty_stats = {
        "entropy_mean": None,
        "entropy_var": None,
        "kl_mean": None,
        "kl_var": None,
    }
    temporal_prob_history = {}
    temporal_label_history = {}
    temporal_label_update_counts = {}
    class_rel_mean = torch.ones(cfg.MODEL.NUM_CLASSES, device=device)
    class_rel_valid = torch.zeros(cfg.MODEL.NUM_CLASSES, dtype=torch.bool, device=device)
    training_state = (
        checkpoint.get("training_state", {})
        if resume_training_state and checkpoint is not None
        else {}
    )
    if training_state:
        if "running_conf" in training_state:
            running_conf = training_state["running_conf"].to(device)
        if "uncertainty_stats" in training_state:
            loaded_uncertainty_stats = training_state["uncertainty_stats"]
            uncertainty_stats = {
                key: _to_device_if_tensor(loaded_uncertainty_stats.get(key), device)
                for key in uncertainty_stats
            }
        if "temporal_prob_history" in training_state:
            temporal_prob_history = training_state["temporal_prob_history"]
        if "temporal_label_history" in training_state:
            temporal_label_history = training_state["temporal_label_history"]
        if "temporal_label_update_counts" in training_state:
            temporal_label_update_counts = training_state[
                "temporal_label_update_counts"
            ]
        if "class_rel_mean" in training_state:
            class_rel_mean = training_state["class_rel_mean"].to(device)
        if "class_rel_valid" in training_state:
            class_rel_valid = training_state["class_rel_valid"].to(device)
        logger.info(
            "Loaded ReMask training state: temporal_prob={} temporal_label={}".format(
                len(temporal_prob_history),
                len(temporal_label_history),
            )
        )
    reliability_mode = _cfg_get(cfg.REMASK, "RELIABILITY_MODE", "entropy_kl")
    fuse_mode = _remask_fusion_mode(cfg)
    use_reliability_fusion = _remask_fusion_enabled(cfg)
    history_update_mode = _cfg_get(cfg.REMASK, "HISTORY_UPDATE_MODE", "online")
    early_stop_min_miou = -1.0
    early_stop_start_iter = 0
    early_stop_patience_iters = 0
    early_stop_min_delta = 0.0
    if uwrg_options is not None:
        early_stop_min_miou = float(
            getattr(uwrg_options, "early_stop_min_miou", -1.0)
        )
        early_stop_start_iter = max(
            0,
            int(getattr(uwrg_options, "early_stop_start_iter", 0)),
        )
        early_stop_patience_iters = max(
            0,
            int(getattr(uwrg_options, "early_stop_patience_iters", 0)),
        )
        early_stop_min_delta = max(
            0.0,
            float(getattr(uwrg_options, "early_stop_min_delta", 0.0)),
        )
    if early_stop_min_miou > 1.0:
        early_stop_min_miou = early_stop_min_miou / 100.0
    use_early_stop_reference = early_stop_min_miou >= 0.0
    early_stop_best_miou = -float("inf")
    early_stop_best_iter = 0
    early_stop_window_best_miou = -float("inf")
    early_stop_window_best_iter = 0
    early_stop_triggered = False
    if training_state and "early_stop_state" in training_state:
        early_stop_state = training_state["early_stop_state"]
        early_stop_best_miou = float(
            early_stop_state.get("best_miou", early_stop_best_miou)
        )
        early_stop_best_iter = int(
            early_stop_state.get("best_iter", early_stop_best_iter)
        )
        early_stop_window_best_miou = float(
            early_stop_state.get(
                "window_best_miou",
                early_stop_window_best_miou,
            )
        )
        early_stop_window_best_iter = int(
            early_stop_state.get(
                "window_best_iter",
                early_stop_window_best_iter,
            )
        )
    use_uwrg = bool(
        uwrg_options is not None
        and uwrg_options.use_uwrg
    )
    uwrg_composition = (
        uwrg_options.uwrg_composition
        if uwrg_options is not None
        else "weighted_product"
    )
    supported_reliability_modes = ("entropy_kl", "temporal", "temporal_label")
    if reliability_mode not in supported_reliability_modes:
        raise ValueError(
            "Unsupported REMASK.RELIABILITY_MODE '{}'. Expected one of {}.".format(
                reliability_mode,
                supported_reliability_modes,
            )
        )
    supported_fuse_modes = (
        "none",
        "off",
        "false",
        "0",
        "",
        "late_hard",
        "late_sigmoid",
        "three_stage",
        "late_stability",
    )
    if fuse_mode not in supported_fuse_modes:
        raise ValueError(
            "Unsupported REMASK.FUSE_MODE '{}'. Expected one of {}.".format(
                fuse_mode,
                supported_fuse_modes,
            )
        )
    if history_update_mode not in ("online", "tcr_snapshot"):
        raise ValueError(
            "Unsupported REMASK.HISTORY_UPDATE_MODE '{}'. Expected 'online' or 'tcr_snapshot'.".format(
                history_update_mode
            )
        )
    if use_reliability_fusion and use_uwrg and uwrg_composition == "uctr":
        raise ValueError(
            "REMASK.FUSE_MODE requires --uwrg-composition weighted_product; "
            "UCTR is only defined for a pure temporal-label reliability run."
        )
    if (
        use_uwrg
        and uwrg_composition == "uctr"
        and reliability_mode != "temporal_label"
        and not use_reliability_fusion
    ):
        raise ValueError(
            "UCTR requires REMASK.RELIABILITY_MODE='temporal_label', got '{}'.".format(
                reliability_mode
            )
        )
    logger.info("ReMask reliability mode: {}".format(reliability_mode))
    if use_early_stop_reference:
        logger.info(
            "Early-stop monitor enabled: min_mIoU={} start_iter={} "
            "patience_iters={} min_delta={}".format(
                early_stop_min_miou,
                early_stop_start_iter,
                early_stop_patience_iters,
                early_stop_min_delta,
            )
        )
    else:
        logger.info("Early-stop monitor disabled")
    if use_reliability_fusion:
        logger.info(
            "ReMask late reliability fusion: mode={} T0={} T1={} T2={} tau={} "
            "gate_eps={} stability_threshold={} stability_tau={}".format(
                fuse_mode,
                _cfg_get(cfg.REMASK, "FUSE_T0", 25000),
                _cfg_get(cfg.REMASK, "FUSE_T1", 24000),
                _cfg_get(cfg.REMASK, "FUSE_T2", 26000),
                _cfg_get(cfg.REMASK, "FUSE_TAU", 500.0),
                _cfg_get(cfg.REMASK, "FUSE_GATE_EPS", 1e-6),
                _cfg_get(cfg.REMASK, "FUSE_STABILITY_THRESHOLD", 0.65),
                _cfg_get(cfg.REMASK, "FUSE_STABILITY_TAU", 0.05),
            )
        )
    logger.info("ReMask history update mode: {}".format(history_update_mode))
    if use_uwrg:
        logger.info(
            "UWRG/SURR enabled: composition={} mode={} floor={} lambda={} "
            "max_strength={} warmup_ratio={} "
            "segment_stride={} min_segment_pixels={}".format(
                uwrg_composition,
                uwrg_options.uwrg_mode,
                uwrg_options.uwrg_floor,
                uwrg_options.uwrg_lambda,
                uwrg_options.uwrg_strength,
                uwrg_options.uwrg_warmup_ratio,
                uwrg_options.uwrg_segment_stride,
                uwrg_options.uwrg_min_segment_pixels,
            )
        )
        if uwrg_composition == "uctr":
            logger.info(
                "UCTR conditioning: lambda={} tau_low={} tau_high={}".format(
                    uwrg_options.uctr_lambda,
                    uwrg_options.uctr_tau_low,
                    uwrg_options.uctr_tau_high,
                )
            )
    else:
        logger.info("UWRG/SURR disabled")
    if history_update_mode == "tcr_snapshot":
        _refresh_temporal_history_snapshot(
            cfg,
            teacher_fea,
            teacher_cls,
            target_data,
            batch_size,
            device,
            "temporal_label" if use_reliability_fusion else reliability_mode,
            temporal_prob_history,
            temporal_label_history,
            temporal_label_update_counts,
            logger,
        )

    max_iters = cfg.SOLVER.MAX_ITER
    iteration = start_iteration
    meters = MetricLogger(delimiter="  ")
    start_training_time = time.time()
    end = time.time()

    def _save_remask_checkpoint(filename):
        payload = {
            "iteration": iteration,
            # Backward-compatible keys used by old evaluation/continuation scripts.
            "feature_extractor": _unwrap(teacher_fea).state_dict(),
            "classifier": _unwrap(teacher_cls).state_dict(),
            # Full resume keys.
            "student_feature_extractor": _unwrap(feature_extractor).state_dict(),
            "student_classifier": _unwrap(classifier).state_dict(),
            "teacher_feature_extractor": _unwrap(teacher_fea).state_dict(),
            "teacher_classifier": _unwrap(teacher_cls).state_dict(),
            "optimizer_fea": optimizer_fea.state_dict(),
            "optimizer_cls": optimizer_cls.state_dict(),
            "training_state": {
                "running_conf": running_conf.detach().cpu(),
                "uncertainty_stats": {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in uncertainty_stats.items()
                },
                "temporal_prob_history": temporal_prob_history,
                "temporal_label_history": temporal_label_history,
                "temporal_label_update_counts": temporal_label_update_counts,
                "class_rel_mean": class_rel_mean.detach().cpu(),
                "class_rel_valid": class_rel_valid.detach().cpu(),
                "early_stop_state": {
                    "best_miou": early_stop_best_miou,
                    "best_iter": early_stop_best_iter,
                    "window_best_miou": early_stop_window_best_miou,
                    "window_best_iter": early_stop_window_best_iter,
                },
            },
            "rng_state": _capture_rng_state(),
        }
        EXPERIMENT.save(payload, filename)

    if resume_training_state and checkpoint is not None and "rng_state" in checkpoint:
        _restore_rng_state(checkpoint["rng_state"], logger)
        logger.info("Restored RNG state from checkpoint")

    EXPERIMENT.initial(locals())

    while (
        iteration < cfg.SOLVER.STOP_ITER
        and iteration < cfg.SOLVER.MAX_ITER
        and not early_stop_triggered
    ):
        if distributed and hasattr(target_loader.sampler, "set_epoch"):
            target_loader.sampler.set_epoch(iteration)

        for tgt_input, _, names, tgt_trans_param, tgt_img_full, mix_label in target_loader:
            EXPERIMENT.batch_start()
            data_time = time.time() - end

            current_lr = adjust_learning_rate(
                cfg.SOLVER.LR_METHOD,
                cfg.SOLVER.BASE_LR,
                iteration,
                max_iters,
                power=cfg.SOLVER.LR_POWER,
            )
            for group in optimizer_fea.param_groups:
                group["lr"] = current_lr
            for group in optimizer_cls.param_groups:
                group["lr"] = current_lr * 10

            optimizer_fea.zero_grad()
            optimizer_cls.zero_grad()

            tgt_input = tgt_input.to(device, non_blocking=True)
            tgt_img_full = tgt_img_full.to(device, non_blocking=True)
            mix_label = mix_label.to(device, non_blocking=True)
            tgt_size = tgt_input.shape[-2:]

            _, student_features = feature_extractor(tgt_input)
            student_logits = classifier(student_features, tgt_size)
            student_prob = F.softmax(student_logits, dim=1)
            name_list = _as_name_list(names)
            history_depth = 0.0
            anchor_reliability = None
            class_balance_reliability = None
            class_calibration_reliability = None
            reliability_logit_adjust_delta = None
            temporal_raw_reliability = None
            entropy_reliability = None
            temporal_reliability = None
            temporal_class_reliability = None
            fusion_loss_reliabilities = None
            fusion_gate = 0.0
            fusion_stage = 0.0
            underwater_stability = None
            underwater_gate = None
            underwater_gate_eff = None
            underwater_agreement = None
            underwater_segment_count = 0
            underwater_strength = 0.0
            uctr_temporal_trust = None
            uctr_fragility = None
            uctr_correction = None

            with torch.no_grad():
                _, teacher_features_full = teacher_fea(tgt_img_full)
                teacher_logits_full = teacher_cls(teacher_features_full, tgt_img_full.shape[-2:])
                teacher_prob_full = F.softmax(teacher_logits_full, dim=1)
                running_conf = update_running_conf(
                    teacher_prob_full,
                    running_conf,
                    0.001,
                )
                teacher_logits = full2weak(teacher_logits_full, tgt_trans_param)
                teacher_prob = F.softmax(teacher_logits, dim=1)

                if use_snd_conf:
                    pseudo_label, _, _ = pseudo_labels_probs(teacher_prob, running_conf, 0.001)
                else:
                    pseudo_label = teacher_prob.argmax(dim=1)

                history_hits = 0
                if not _cfg_get(cfg.REMASK, "USE_RELIABILITY", True):
                    reliability = torch.ones_like(teacher_prob[:, :1])
                elif use_reliability_fusion:
                    entropy_reliability = _compute_reliability(
                        cfg,
                        teacher_prob,
                        student_prob,
                        uncertainty_stats,
                    )
                    pre_fusion_schedule = _resolve_pre_temporal_fusion_schedule(
                        cfg,
                        iteration,
                    )
                    if pre_fusion_schedule is not None:
                        fusion_stage_name, fusion_gate = pre_fusion_schedule
                        reliability = entropy_reliability
                        fusion_stage = 0.0
                    else:
                        (
                            temporal_reliability,
                            temporal_raw_reliability,
                            history_hits,
                            history_depth,
                        ) = _compute_temporal_label_reliability_from_history(
                            cfg,
                            teacher_prob,
                            teacher_prob_full,
                            tgt_trans_param,
                            name_list,
                            temporal_label_history,
                            device,
                        )
                        temporal_class_reliability = temporal_reliability
                        if _cfg_get(cfg.REMASK, "USE_CLASS_BALANCE", False):
                            class_balance_reliability = _compute_class_balance_reliability(
                                cfg,
                                pseudo_label,
                                running_conf,
                            )
                            temporal_class_reliability = (
                                temporal_reliability * class_balance_reliability
                            )

                        fusion_stage_name, fusion_gate = _resolve_late_fusion_schedule(
                            cfg,
                            iteration,
                            temporal_reliability,
                        )
                        if fusion_stage_name == "uc":
                            reliability = entropy_reliability
                            fusion_stage = 0.0
                        elif fusion_stage_name == "temp":
                            reliability = temporal_reliability
                            fusion_stage = 0.5
                        elif fusion_stage_name == "tc":
                            reliability = temporal_class_reliability
                            fusion_stage = 1.0
                        else:
                            fusion_stage = 0.75
                            reliability = (
                                (1.0 - fusion_gate) * entropy_reliability
                                + fusion_gate * temporal_class_reliability
                            )
                            fusion_loss_reliabilities = [
                                (1.0 - fusion_gate, entropy_reliability),
                                (fusion_gate, temporal_class_reliability),
                            ]
                elif reliability_mode == "temporal":
                    hist_probs = []
                    for idx, name in enumerate(name_list):
                        hist_prob = temporal_prob_history.get(name)
                        if hist_prob is None:
                            hist_prob = teacher_prob_full[idx].detach().cpu().half()
                        else:
                            history_hits += 1
                        hist_probs.append(hist_prob.float())
                    hist_prob_full = torch.stack(hist_probs, dim=0).to(device, non_blocking=True)
                    hist_prob = full2weak(hist_prob_full, tgt_trans_param)
                    reliability = _compute_temporal_reliability(cfg, teacher_prob, hist_prob)
                elif reliability_mode == "temporal_label":
                    (
                        reliability,
                        temporal_raw_reliability,
                        history_hits,
                        history_depth,
                    ) = _compute_temporal_label_reliability_from_history(
                        cfg,
                        teacher_prob,
                        teacher_prob_full,
                        tgt_trans_param,
                        name_list,
                        temporal_label_history,
                        device,
                    )
                else:
                    reliability = _compute_reliability(
                        cfg,
                        teacher_prob,
                        student_prob,
                        uncertainty_stats,
                    )
                if use_uwrg:
                    underwater_strength = _uwrg_probe_strength(
                        iteration,
                        cfg.SOLVER.STOP_ITER,
                        uwrg_options.uwrg_strength,
                        uwrg_options.uwrg_warmup_ratio,
                    )
                    underwater_image = _underwater_reliability_perturb(
                        _image_space(tgt_img_full, cfg),
                        underwater_strength,
                    )
                    underwater_input = _normalized_space(underwater_image, cfg)
                    _, underwater_features_full = teacher_fea(underwater_input)
                    underwater_logits_full = teacher_cls(
                        underwater_features_full,
                        tgt_img_full.shape[-2:],
                    )
                    underwater_logits = full2weak(
                        underwater_logits_full,
                        tgt_trans_param,
                    )
                    underwater_prob = F.softmax(underwater_logits, dim=1)
                    underwater_stability, underwater_agreement, underwater_segment_count = (
                        _compute_underwater_stability(
                            teacher_prob,
                            underwater_prob,
                            pseudo_label,
                            cfg.INPUT.IGNORE_LABEL,
                            uwrg_options.uwrg_mode,
                            uwrg_options.uwrg_segment_stride,
                            uwrg_options.uwrg_min_segment_pixels,
                        )
                    )
                    underwater_gate = (
                        uwrg_options.uwrg_floor
                        + (1.0 - uwrg_options.uwrg_floor) * underwater_stability
                    ).clamp(uwrg_options.uwrg_floor, 1.0)

                    if uwrg_composition == "uctr":
                        temporal_floor = _cfg_get(
                            cfg.REMASK,
                            "TEMPORAL_FLOOR",
                            0.5,
                        )
                        (
                            reliability,
                            uctr_temporal_trust,
                            uctr_fragility,
                            uctr_correction,
                        ) = _compute_uctr_temporal_reliability(
                            temporal_raw_reliability,
                            underwater_stability,
                            temporal_floor,
                            uwrg_options.uctr_lambda,
                            uwrg_options.uctr_tau_low,
                            uwrg_options.uctr_tau_high,
                        )
                        underwater_gate_eff = uctr_correction
                if anchor_fea is not None and anchor_cls is not None:
                    _, anchor_features_full = anchor_fea(tgt_img_full)
                    anchor_logits_full = anchor_cls(anchor_features_full, tgt_img_full.shape[-2:])
                    anchor_prob = F.softmax(full2weak(anchor_logits_full, tgt_trans_param), dim=1)
                    anchor_reliability = _compute_anchor_reliability(
                        cfg,
                        teacher_prob,
                        anchor_prob,
                    )
                    reliability = reliability * anchor_reliability
                if _cfg_get(cfg.REMASK, "USE_CLASS_CALIBRATION", False):
                    _update_class_reliability_stats(
                        cfg,
                        reliability,
                        pseudo_label,
                        class_rel_mean,
                        class_rel_valid,
                    )
                    class_calibration_reliability = _compute_class_calibration_reliability(
                        cfg,
                        reliability,
                        pseudo_label,
                        class_rel_mean,
                    )
                    reliability = reliability * class_calibration_reliability
                if _cfg_get(cfg.REMASK, "USE_RELIABILITY_LOGIT_ADJUST", False):
                    reliability_before_adjust = reliability
                    reliability = _logit_adjust_reliability(
                        cfg,
                        reliability,
                        pseudo_label,
                        running_conf,
                    )
                    reliability_logit_adjust_delta = (
                        reliability - reliability_before_adjust
                    ).abs()
                if (
                    _cfg_get(cfg.REMASK, "USE_CLASS_BALANCE", False)
                    and not use_reliability_fusion
                ):
                    class_balance_reliability = _compute_class_balance_reliability(
                        cfg,
                        pseudo_label,
                        running_conf,
                    )
                    reliability = reliability * class_balance_reliability
                if use_uwrg and uwrg_composition == "weighted_product":
                    # Preserve the original ReMask reliability and append only a
                    # bounded, exponent-controlled, downweight-only UWRG factor.
                    underwater_gate_eff = underwater_gate.clamp(
                        min=uwrg_options.uwrg_floor,
                        max=1.0,
                    ).pow(uwrg_options.uwrg_lambda)
                    reliability = reliability * underwater_gate_eff
                    if fusion_loss_reliabilities is not None:
                        fusion_loss_reliabilities = [
                            (weight, branch_reliability * underwater_gate_eff)
                            for weight, branch_reliability in fusion_loss_reliabilities
                        ]
            reliability = reliability.detach()
            if fusion_loss_reliabilities is not None:
                fusion_loss_reliabilities = [
                    (weight, branch_reliability.detach())
                    for weight, branch_reliability in fusion_loss_reliabilities
                ]

            if mix_label.shape[-2:] != pseudo_label.shape[-2:]:
                mix_label = F.interpolate(
                    mix_label.unsqueeze(1).float(),
                    size=pseudo_label.shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()
            mix_region = mix_label != cfg.INPUT.IGNORE_LABEL
            reliability = torch.where(
                mix_region.unsqueeze(1),
                torch.ones_like(reliability),
                reliability,
            )
            if fusion_loss_reliabilities is not None:
                fusion_loss_reliabilities = [
                    (
                        weight,
                        torch.where(
                            mix_region.unsqueeze(1),
                            torch.ones_like(branch_reliability),
                            branch_reliability,
                        ),
                    )
                    for weight, branch_reliability in fusion_loss_reliabilities
                ]
            pseudo_label = torch.where(mix_region, mix_label.long(), pseudo_label)

            ce_loss = criterion(student_logits, pseudo_label.long())
            
            # TỐI ƯU HÓA: Fix lỗi Gradient Collapse. 
            # Thay vì chia trung bình (.mean()) cho toàn bộ ảnh làm gradient bị teo nhỏ khi dùng snd-conf
            # Ta chia cho tổng trọng số của các pixel thực sự được học.
            valid_mask = (pseudo_label != cfg.INPUT.IGNORE_LABEL).float()
            if fusion_loss_reliabilities is None:
                loss_seg = _reliability_weighted_ce_loss(
                    cfg,
                    reliability,
                    ce_loss,
                    valid_mask,
                )
            else:
                loss_seg = student_logits.new_tensor(0.0)
                for branch_weight, branch_reliability in fusion_loss_reliabilities:
                    if branch_weight <= 1e-8:
                        continue
                    loss_seg = loss_seg + branch_weight * _reliability_weighted_ce_loss(
                        cfg,
                        branch_reliability,
                        ce_loss,
                        valid_mask,
                    )

            if _cfg_get(cfg.REMASK, "USE_CONTRASTIVE", True) and cfg.REMASK.LAMBDA > 0:
                image_repr, reliability_f = _image_representations(
                    student_features,
                    reliability,
                    cfg.REMASK.EPSILON,
                )
                prototypes, valid, mean_reliability = _local_prototypes(
                    cfg,
                    student_features,
                    pseudo_label,
                    reliability_f,
                )
                prototypes = _refine_with_neighbors(cfg, prototypes, valid, image_repr)
                loss_con = _prototype_contrastive_loss(
                    cfg,
                    classifier,
                    prototypes,
                    valid,
                    mean_reliability,
                )
                valid_proto = valid.float().sum()
            else:
                prototypes = None
                valid = None
                loss_con = student_logits.new_tensor(0.0)
                valid_proto = student_logits.new_tensor(0.0)

            total_loss = loss_seg + cfg.REMASK.LAMBDA * loss_con
            total_loss.backward()
            optimizer_fea.step()
            optimizer_cls.step()
            if prototypes is not None:
                _update_global_prototypes(cfg, classifier, prototypes, valid)

            if reliability_mode == "temporal" and history_update_mode == "online":
                momentum = _cfg_get(cfg.REMASK, "HISTORY_MOMENTUM", 0.9)
                with torch.no_grad():
                    for idx, name in enumerate(name_list):
                        curr = teacher_prob_full[idx].detach().cpu().half()
                        prev = temporal_prob_history.get(name)
                        if prev is None:
                            temporal_prob_history[name] = curr
                        else:
                            temporal_prob_history[name] = (
                                momentum * prev.float() + (1.0 - momentum) * curr.float()
                            ).half()
            elif (
                (reliability_mode == "temporal_label" or use_reliability_fusion)
                and history_update_mode == "online"
            ):
                with torch.no_grad():
                    curr_conf_full, curr_label_full = teacher_prob_full.max(dim=1, keepdim=True)
                    history_stride = max(1, int(_cfg_get(cfg.REMASK, "HISTORY_STRIDE", 1)))
                    use_history_conf = _cfg_get(cfg.REMASK, "HISTORY_USE_CONF", True)
                    for idx, name in enumerate(name_list):
                        curr_label = curr_label_full[idx : idx + 1].float()
                        if history_stride > 1:
                            curr_label = F.interpolate(
                                curr_label,
                                scale_factor=1.0 / history_stride,
                                mode="nearest",
                                recompute_scale_factor=False,
                            )
                        curr_label = curr_label[0].detach().cpu().to(torch.uint8)
                        if use_history_conf:
                            curr_conf = curr_conf_full[idx : idx + 1]
                            if history_stride > 1:
                                curr_conf = F.interpolate(
                                    curr_conf,
                                    scale_factor=1.0 / history_stride,
                                    mode="bilinear",
                                    align_corners=True,
                                    recompute_scale_factor=False,
                                )
                            curr_conf = curr_conf[0].detach().cpu().half()
                        else:
                            curr_conf = None
                        _update_temporal_label_queue(
                            temporal_label_history,
                            temporal_label_update_counts,
                            name,
                            curr_label,
                            curr_conf,
                            cfg,
                        )

            if iteration % cfg.DTU.FIX_ITERATION == 0:
                teacher_fea_ema.step()
                teacher_cls_ema.step()
            teacher_fea.eval()
            teacher_cls.eval()
            EXPERIMENT.observe(locals())
            _update_mixup_sampling(cfg, target_data, running_conf)

            meters.update(
                loss_seg=loss_seg.item(),
                loss_con=loss_con.item(),
                total_loss=total_loss.item(),
                reliability=reliability.mean().item(),
                reliability_std=reliability.std(unbiased=False).item(),
                reliability_min=reliability.min().item(),
                reliability_max=reliability.max().item(),
                reliability_low=(reliability < 0.7).float().mean().item(),
                fuse_gate=float(fusion_gate),
                fuse_stage=float(fusion_stage),
                reliability_uc=(
                    entropy_reliability.mean().item()
                    if entropy_reliability is not None
                    else 0.0
                ),
                reliability_temp=(
                    temporal_reliability.mean().item()
                    if temporal_reliability is not None
                    else 0.0
                ),
                reliability_tc=(
                    temporal_class_reliability.mean().item()
                    if temporal_class_reliability is not None
                    else 0.0
                ),
                anchor_reliability=(
                    anchor_reliability.mean().item()
                    if anchor_reliability is not None
                    else 0.0
                ),
                class_balance=(
                    class_balance_reliability.mean().item()
                    if class_balance_reliability is not None
                    else 0.0
                ),
                class_calibration=(
                    class_calibration_reliability.mean().item()
                    if class_calibration_reliability is not None
                    else 0.0
                ),
                logit_adjust_delta=(
                    reliability_logit_adjust_delta.mean().item()
                    if reliability_logit_adjust_delta is not None
                    else 0.0
                ),
                uwrg_gate_raw=(
                    underwater_gate.mean().item()
                    if underwater_gate is not None
                    else 1.0
                ),
                uwrg_gate_eff=(
                    underwater_gate_eff.mean().item()
                    if underwater_gate_eff is not None
                    else 1.0
                ),
                uwrg_downweight_raw=(
                    (1.0 - underwater_gate).mean().item()
                    if underwater_gate is not None
                    else 0.0
                ),
                uwrg_downweight_eff=(
                    (1.0 - underwater_gate_eff).mean().item()
                    if underwater_gate_eff is not None
                    else 0.0
                ),
                uwrg_disagreement=(
                    (1.0 - underwater_agreement).mean().item()
                    if underwater_agreement is not None
                    else 0.0
                ),
                uctr_temporal_raw=(
                    temporal_raw_reliability.mean().item()
                    if temporal_raw_reliability is not None
                    else 0.0
                ),
                uctr_temporal_trust=(
                    uctr_temporal_trust.mean().item()
                    if uctr_temporal_trust is not None
                    else 0.0
                ),
                uctr_fragility=(
                    uctr_fragility.mean().item()
                    if uctr_fragility is not None
                    else 0.0
                ),
                uctr_correction=(
                    uctr_correction.mean().item()
                    if uctr_correction is not None
                    else 1.0
                ),
                uctr_active=(
                    (uctr_temporal_trust > 0).float().mean().item()
                    if uctr_temporal_trust is not None
                    else 0.0
                ),
                uwrg_strength=underwater_strength,
                uwrg_segments=float(underwater_segment_count),
                history_hit=history_hits / max(1, len(name_list)),
                history_depth=history_depth / max(1, history_hits),
                valid_proto=valid_proto.item(),
            )

            iteration += 1
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            print_freq = 1 if cfg.SOLVER.MAX_ITER <= 100 else 20
            if iteration % print_freq == 0 or iteration == max_iters:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer_fea.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

            checkpoint_due = (
                iteration == cfg.SOLVER.STOP_ITER
                or iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0
            )
            if checkpoint_due:
                filename = os.path.join(cfg.OUTPUT_DIR, "remask_model_iter{:06d}.pth".format(iteration))
                EXPERIMENT.evaluate(iteration, locals())

            if early_stop_triggered:
                break

            if (
                cfg.TCR.OPEN
                and cfg.TCR.UPDATE_FREQUENCY > 0
                and iteration < cfg.SOLVER.STOP_ITER
                and iteration < cfg.SOLVER.MAX_ITER
                and iteration % cfg.TCR.UPDATE_FREQUENCY == 0
            ):
                run_candidate(cfg, teacher_fea, teacher_cls, local_rank, distributed)
                gen_lb_info(cfg, "CTR")
                gen_lb_info(cfg, "CTR_O")
                target_data, target_loader = _build_target_loader(cfg, batch_size, distributed)
                if history_update_mode == "tcr_snapshot":
                    _refresh_temporal_history_snapshot(
                        cfg,
                        teacher_fea,
                        teacher_cls,
                        target_data,
                        batch_size,
                        device,
                        "temporal_label" if use_reliability_fusion else reliability_mode,
                        temporal_prob_history,
                        temporal_label_history,
                        temporal_label_update_counts,
                        logger,
                    )
                if checkpoint_due:
                    _save_remask_checkpoint(filename)
                    EXPERIMENT.boundary(iteration, filename, locals())
                break

            if checkpoint_due:
                _save_remask_checkpoint(filename)
                EXPERIMENT.boundary(iteration, filename, locals())

            if iteration >= cfg.SOLVER.STOP_ITER or iteration >= cfg.SOLVER.MAX_ITER:
                break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total ReMask training time: {} ({:.4f} s / it)".format(
            total_time_str,
            total_training_time / max(1, iteration),
        )
    )

    return teacher_fea, teacher_cls


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation ReMask Training")
    parser.add_argument(
        "-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--temp_train", action="store_true", help="testing with small iterations")
    parser.add_argument("--local_rank", type=int, default=0)
    add_underwater_label_args(parser)
    parser.add_argument(
        "--use-snd-conf",
        action="store_true",
        help="Integrate SND running_conf to filter pseudo labels",
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for fair ablations. When set, Python, NumPy, "
            "PyTorch, CUDA, and DataLoader workers are seeded."
        ),
    )
    parser.add_argument(
        "--resume-iteration",
        type=int,
        default=0,
        help=(
            "Continue the adaptation schedule from this iteration after loading "
            "cfg.resume. Keep 0 for normal source initialization; use -1 to read "
            "the iteration field from the checkpoint."
        ),
    )
    parser.add_argument(
        "--early-stop-min-miou",
        type=float,
        default=-1.0,
        help=(
            "Reference mIoU that the run must reach; negative disables this "
            "early-stop rule. Values above 1 are interpreted as percentages."
        ),
    )
    parser.add_argument(
        "--early-stop-start-iter",
        type=int,
        default=0,
        help="Do not apply the reference-mIoU early-stop rule before this iteration.",
    )
    parser.add_argument(
        "--early-stop-patience-iters",
        type=int,
        default=0,
        help=(
            "Extra iterations after early-stop-start-iter before stopping if "
            "the best validation mIoU has not reached --early-stop-min-miou."
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum mIoU margin used for best/target comparisons.",
    )
    parser.add_argument(
        "--remask-exp",
        choices=("ce", "reliability", "full"),
        default=None,
        help=(
            "Quick ReMask ablation: ce = CE with reliability fixed to 1 and no contrastive; "
            "reliability = reliability-weighted CE without contrastive; "
            "full = reliability plus contrastive."
        ),
    )
    parser.add_argument(
        "--disable-uwrg",
        dest="use_uwrg",
        action="store_false",
        default=True,
        help="Disable the underwater stability reliability gate.",
    )
    parser.add_argument(
        "--uwrg-mode",
        choices=("pixel", "segment"),
        default="segment",
        help="Use pixel stability or connected-component median stability.",
    )
    parser.add_argument(
        "--uwrg-composition",
        choices=("weighted_product", "uctr"),
        default="weighted_product",
        help=(
            "weighted_product appends UWRG to final reliability; uctr uses "
            "underwater fragility to condition raw temporal-label reliability."
        ),
    )
    parser.add_argument(
        "--uwrg-floor",
        type=float,
        default=0.8,
        help="Minimum UWRG factor; the gate is always in [floor, 1].",
    )
    parser.add_argument(
        "--uwrg-lambda",
        type=float,
        default=0.5,
        help=(
            "Exponent for the bounded underwater gate. Values below 1 make "
            "UWRG more conservative without changing the original ReMask reliability."
        ),
    )
    parser.add_argument(
        "--uctr-lambda",
        type=float,
        default=0.1,
        help="Maximum UCTR reduction applied to trusted raw temporal reliability.",
    )
    parser.add_argument(
        "--uctr-tau-low",
        type=float,
        default=0.6,
        help="Lower raw-temporal threshold where UCTR conditioning begins.",
    )
    parser.add_argument(
        "--uctr-tau-high",
        type=float,
        default=0.9,
        help="Upper raw-temporal threshold where UCTR conditioning is fully active.",
    )
    parser.add_argument(
        "--uwrg-strength",
        type=float,
        default=0.35,
        help="Maximum attenuation/haze strength used by the teacher probe.",
    )
    parser.add_argument(
        "--uwrg-warmup-ratio",
        type=float,
        default=0.2,
        help="Fraction of adaptation iterations used to ramp probe strength.",
    )
    parser.add_argument(
        "--uwrg-segment-stride",
        type=int,
        default=4,
        help="Downsampling stride for connected-component median aggregation.",
    )
    parser.add_argument(
        "--uwrg-min-segment-pixels",
        type=int,
        default=8,
        help="Minimum component size at the downsampled resolution.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    if args.early_stop_start_iter < 0:
        parser.error("--early-stop-start-iter must be non-negative.")
    if args.early_stop_patience_iters < 0:
        parser.error("--early-stop-patience-iters must be non-negative.")
    if args.early_stop_min_delta < 0.0:
        parser.error("--early-stop-min-delta must be non-negative.")
    if not 0.0 <= args.uwrg_floor <= 1.0:
        parser.error("--uwrg-floor must be in [0, 1].")
    if args.uwrg_lambda < 0.0:
        parser.error("--uwrg-lambda must be non-negative.")
    if not 0.0 <= args.uctr_lambda <= 1.0:
        parser.error("--uctr-lambda must be in [0, 1].")
    if not 0.0 <= args.uctr_tau_low < args.uctr_tau_high <= 1.0:
        parser.error(
            "--uctr-tau-low and --uctr-tau-high must satisfy "
            "0 <= tau_low < tau_high <= 1."
        )
    if args.uwrg_strength < 0.0:
        parser.error("--uwrg-strength must be non-negative.")
    if args.uwrg_warmup_ratio < 0.0:
        parser.error("--uwrg-warmup-ratio must be non-negative.")
    if args.uwrg_segment_stride < 1:
        parser.error("--uwrg-segment-stride must be at least 1.")
    if args.uwrg_min_segment_pixels < 1:
        parser.error("--uwrg-min-segment-pixels must be at least 1.")

    if os.name == "nt" or args.temp_train:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        rank = int(os.environ["RANK"])
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            gpus_per_node = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        else:
            gpus_per_node = torch.cuda.device_count()
        assert gpus_per_node > 0, "CUDA is not supported"
        gpu = rank % gpus_per_node
        torch.cuda.set_device(gpu)
        args.local_rank = gpu
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="tcp://{}:{}".format(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"])),
            rank=rank,
            world_size=int(os.environ["WORLD_SIZE"]),
        )
        print("RANK and WORLD_SIZE in environ: {}/{}".format(rank, int(os.environ["WORLD_SIZE"])))

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    apply_underwater_label_args(cfg, args)
    if args.seed is not None:
        cfg.SEED = int(args.seed)
    if args.remask_exp == "ce":
        cfg.REMASK.USE_RELIABILITY = False
        cfg.REMASK.USE_CONTRASTIVE = False
        cfg.REMASK.LAMBDA = 0.0
    elif args.remask_exp == "reliability":
        cfg.REMASK.USE_RELIABILITY = True
        cfg.REMASK.USE_CONTRASTIVE = False
        cfg.REMASK.LAMBDA = 0.0
    elif args.remask_exp == "full":
        cfg.REMASK.USE_RELIABILITY = True
        cfg.REMASK.USE_CONTRASTIVE = True
    if args.temp_train:
        cfg.SOLVER.MAX_ITER = 10
        cfg.SOLVER.STOP_ITER = 10
        cfg.SOLVER.CHECKPOINT_PERIOD = 10
    _seed_everything(cfg.SEED, args.local_rank)
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("DTST", cfg.OUTPUT_DIR, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as config_file:
        logger.info("\n" + config_file.read())
    logger.info("Running with config:\n{}".format(cfg))

    feature_extractor, classifier = train(
        cfg,
        args.local_rank,
        args.distributed,
        args.use_snd_conf,
        uwrg_options=args,
    )

    if not args.skip_test:
        run_test(cfg, feature_extractor, classifier, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
