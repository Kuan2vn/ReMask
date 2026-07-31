from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..checkpoint import load_model_weights, save_checkpoint
from ..config import dump_config
from ..data.datasets import SourceDataset
from ..modeling import build_model
from ..utils import (
    close_logger,
    infinite,
    make_logger,
    polynomial_lr,
    resolve_device,
    seed_everything,
    seed_worker,
    set_optimizer_lr,
)


def train_source(
    config: Dict,
    resume: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    output = Path(output_dir or config["output_dir"]) / "source"
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(dump_config(config), encoding="utf-8")
    logger = make_logger(str(output), "remask.source")
    seed_everything(int(config.get("seed", 1)))
    device = resolve_device(str(config.get("device", "cuda")))
    model = build_model(config).to(device)
    if resume:
        load_model_weights(model, resume, prefer_teacher=False)

    train_cfg = config["source_train"]
    dataset = SourceDataset(config["data"]["source"], config["data"], config["model"]["num_classes"])
    generator = torch.Generator().manual_seed(int(config.get("seed", 1)))
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 8)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    batches = infinite(loader)
    optimizer = torch.optim.SGD(
        model.optimizer_groups(),
        lr=float(train_cfg["base_lr"]),
        momentum=float(train_cfg["momentum"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    iterations = int(train_cfg["iterations"])
    model.train()
    for iteration in range(iterations):
        lr = polynomial_lr(train_cfg["base_lr"], iteration, iterations, train_cfg["lr_power"])
        set_optimizer_lr(optimizer, lr)
        images, labels, _ = next(batches)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images, labels.shape[-2:])
        loss = F.cross_entropy(logits, labels, ignore_index=255)
        loss.backward()
        optimizer.step()

        step = iteration + 1
        if step == 1 or step % 50 == 0:
            logger.info("iter=%d/%d lr=%.3e loss=%.5f", step, iterations, lr, loss.item())
        if step % int(train_cfg["checkpoint_every"]) == 0 or step == iterations:
            save_checkpoint(
                str(output / "source_iter{:06d}.pth".format(step)),
                model,
                optimizer=optimizer,
                iteration=step,
                stage="source",
                config=config,
            )
    final_path = str(output / "source_final.pth")
    save_checkpoint(final_path, model, optimizer=optimizer, iteration=iterations, stage="source", config=config)
    close_logger(logger)
    return final_path
