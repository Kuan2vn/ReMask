"""Paper-based DeepLab port of PCGD + CRAM, not author-released training code.

Upstream3379e76 contains placeholder.py in adaptation/model. Unspecified choices
are explicit constructor arguments and recorded in PROTOCOL.json, never tuned
using target test scores. The full student participates in bi-level adaptation.
"""
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .common import image_tensor, safe_ce, atomic_json


class CRAM(nn.Module):
    def __init__(self, classes=8):
        super().__init__()
        self.attention = nn.Sequential(nn.Conv2d(2048, 128, 1), nn.ReLU(), nn.Conv2d(128, classes, 1))

    def forward(self, model, images, box):
        """Aligned HR crop and half-resolution context, Eq.5, in input pixels."""
        height, width = images.shape[-2:]
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        assert all(value % 16 == 0 for value in box), 'Crop must align with s*stride'
        low = F.interpolate(images, size=(height // 2, width // 2), mode='bilinear', align_corners=False)
        low_features = model.feature(low)[1]
        context = F.interpolate(model.head(low_features), size=(height, width), mode='bilinear', align_corners=True)
        attention = F.interpolate(self.attention(low_features).sigmoid(), size=(height, width), mode='bilinear', align_corners=True)
        detail = model(images[..., y0:y1, x0:x1])
        # Keep the context outside the HR window; the attention can only route
        # pixels for which a corresponding detail prediction exists.
        fused = context.clone()
        local_attention = attention[..., y0:y1, x0:x1]
        fused[..., y0:y1, x0:x1] = (1 - local_attention) * context[..., y0:y1, x0:x1] + local_attention * detail
        return fused, detail


class CRAMPredictor(nn.Module):
    """Tile HR details over the context at evaluation, retaining learned fusion."""
    def __init__(self, model, cram):
        super().__init__()
        self.model, self.cram = model, cram

    def forward(self, images, size=None):
        height, width = images.shape[-2:]
        assert height % 2 == 0 and width % 2 == 0
        low = F.interpolate(images, (height // 2, width // 2), mode='bilinear', align_corners=False)
        features = self.model.feature(low)[1]
        context = F.interpolate(self.model.head(features), (height, width), mode='bilinear', align_corners=True)
        attention = F.interpolate(self.cram.attention(features).sigmoid(), (height, width), mode='bilinear', align_corners=True)
        fused = torch.empty_like(context)
        for top in (0, height // 2):
            for left in (0, width // 2):
                bottom, right = top + height // 2, left + width // 2
                detail = self.model(images[..., top:bottom, left:right])
                weight = attention[..., top:bottom, left:right]
                fused[..., top:bottom, left:right] = (1 - weight) * context[..., top:bottom, left:right] + weight * detail
        return fused if size is None else F.interpolate(fused, size=size, mode='bilinear', align_corners=True)


def grid_box(height, width, fraction=.5, grid=16):
    crop_h = max(grid, int(height * fraction) // grid * grid)
    crop_w = max(grid, int(width * fraction) // grid * grid)
    top = random.randrange((height - crop_h) // grid + 1) * grid
    left = random.randrange((width - crop_w) // grid + 1) * grid
    return left, top, left + crop_w, top + crop_h


@torch.no_grad()
def generate_source_cache(model, images, folder, device='cuda'):
    folder = Path(folder)
    (folder / 'labels').mkdir(parents=True, exist_ok=True)
    (folder / 'probabilities').mkdir(exist_ok=True)
    records = []
    model.eval()
    for index, path in enumerate(images.paths):
        x = image_tensor(images.image(index), device).unsqueeze(0)
        features = model.feature(x)[1]
        probabilities = model.head(features, (640, 1280)).softmax(1)
        label = probabilities.argmax(1)[0].cpu().numpy().astype(np.uint8)
        Image.fromarray(label).save(folder / 'labels' / (path.stem + '.png'))
        small = F.interpolate(probabilities, (80, 160), mode='bilinear', align_corners=True)[0]
        np.save(folder / 'probabilities' / (path.stem + '.npy'), small.cpu().numpy().astype(np.float16))
        confidence = probabilities.max(1).values[0]
        style = torch.fft.rfft2(x.mean(1, keepdim=True), norm='ortho').abs()[..., :8, :8].flatten(1)
        layout = F.adaptive_avg_pool2d(probabilities, (4, 8)).flatten(1)
        descriptor = torch.cat([F.normalize(features.mean((2, 3)), dim=1),
                                F.normalize(style, dim=1), F.normalize(layout, dim=1)], dim=1)
        records.append(dict(index=index, name=path.stem,
            feature=F.normalize(descriptor, dim=1)[0].cpu().tolist(),
            class_confidence=[0. if not (label == c).any() else
                              float(confidence[torch.as_tensor(label == c, device=device)].mean()) for c in range(8)],
            class_pixels=np.bincount(label.ravel(), minlength=8).tolist()))
    atomic_json(folder / 'records.json', records)
    return records


@torch.no_grad()
def consistency_split(model, images, cache, top_fraction=.10, pool_size=10, device='cuda'):
    """Eq.1: source prediction versus the target model after tau updates."""
    cache = Path(cache)
    records = __import__('json').loads((cache / 'records.json').read_text())
    model.eval()
    scores, class_scores = [], []
    for index, path in enumerate(images.paths):
        source = torch.from_numpy(np.load(cache / 'probabilities' / (path.stem + '.npy'))).float().to(device)
        target = model(image_tensor(images.image(index), device).unsqueeze(0), (80, 160)).log_softmax(1)[0]
        divergence = F.kl_div(target, source, reduction='none').sum(0)
        score = -divergence.sum()
        scores.append(float(score))
        source_label = source.argmax(0)
        class_scores.append([float(-divergence[source_label == c].mean()) if (source_label == c).any() else None for c in range(8)])
    stable = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(2, math.ceil(len(scores) * top_fraction))]
    unstable = [i for i in range(len(scores)) if i not in set(stable)]
    assert unstable, 'Need nonempty stable and unstable subsets'
    frequency = np.sum([record['class_pixels'] for record in records], axis=0)
    minority = [int(c) for c in np.argsort(frequency) if c != 0 and frequency[c] > 0][:4]
    pools = {c: sorted([i for i in stable if records[i]['class_pixels'][c] >= 50],
                      key=lambda i: class_scores[i][c] if class_scores[i][c] is not None else -float('inf'), reverse=True)[:pool_size] for c in minority}
    features = F.normalize(torch.tensor([record['feature'] for record in records]), dim=1)
    neighbors = {i: stable[int((features[i] @ features[stable].T).argmax())] for i in unstable}
    state = dict(stable=stable, unstable=unstable, scores=scores, class_scores=class_scores, pools=pools,
                 neighbors=neighbors, minority=minority, selection_uses_target_gt=False)
    atomic_json(cache / 'SPLIT.json', state)
    return state


def copy_paste(images, labels, donor_images, donor_labels, classes):
    assert images.shape == donor_images.shape and labels.shape == donor_labels.shape
    masks = torch.stack([donor_labels[i] == int(c) for i, c in enumerate(classes)])
    return torch.where(masks[:, None], donor_images, images), torch.where(masks, donor_labels, labels)


def bi_level_loss(model, noisy, noisy_label, neighbor, neighbor_label, inner_lr=2.5e-4):
    """Algorithm1 PathA: differentiate through a full-model SGD inner update."""
    import higher
    inner_optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
    with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False, track_higher_grads=True) as (functional_model, diffopt):
        inner_loss = safe_ce(functional_model(noisy), noisy_label)
        diffopt.step(inner_loss)
        outer = safe_ce(functional_model(neighbor), neighbor_label)
    return outer


class DAPASS:
    def __init__(self, model, images, output, tau=6000, top_fraction=.10, pool_size=10,
                 detail_weight=.1, device='cuda'):
        self.images, self.output, self.device = images, Path(output), device
        self.tau, self.top_fraction, self.pool_size = tau, top_fraction, pool_size
        self.detail_weight = detail_weight
        self.cram = CRAM().to(device)
        self.cache = self.output / 'source_cache'
        self.source = copy.deepcopy(model).eval().requires_grad_(False)
        self.split = None

    def prepare(self, model):
        if not (self.cache / 'records.json').exists():
            generate_source_cache(self.source, self.images, self.cache, self.device)
        self.source.cpu()
        if (self.cache / 'SPLIT.json').exists():
            self.split = __import__('json').loads((self.cache / 'SPLIT.json').read_text())

    def parameters(self):
        return self.cram.parameters()

    def loss(self, model, step, batch_size=4):
        if step < self.tau:
            ids = random.sample(range(len(self.images)), batch_size)
            x, labels = self.images.batch(ids, self.cache / 'labels', self.device)
            return safe_ce(model(x), labels), {'phase': 'consistency_warmup'}
        if self.split is None or (step > self.tau and step % 3000 == 0):
            self.split = consistency_split(model, self.images, self.cache, self.top_fraction, self.pool_size, self.device)
        model.train()
        # Sequential pairs keep the declared effective batch4 on one16GiB GPU.
        # All backward calls are accumulated before one optimizer update.
        ids = random.choices(self.split['unstable'], k=batch_size)
        for index in ids:
            neighbor = int(self.split['neighbors'].get(index, self.split['neighbors'].get(str(index))))
            x, y = self.images.batch([index], self.cache / 'labels', self.device)
            stable_x, stable_y = self.images.batch([neighbor], self.cache / 'labels', self.device)
            outer = bi_level_loss(model, x, y, stable_x, stable_y)
            (outer / batch_size).backward()
            available = [int(c) for c, pool in self.split['pools'].items() if pool]
            if available:
                cid = random.choice(available)
                pool = self.split['pools'].get(cid, self.split['pools'].get(str(cid)))
                donor_x, donor_y = self.images.batch([random.choice(pool)], self.cache / 'labels', self.device)
                x, y = copy_paste(x, y, donor_x, donor_y, [cid])
            box = grid_box(*x.shape[-2:])
            fused, detail = self.cram(model, x, box)
            left, top, right, bottom = box
            loss = (1 - self.detail_weight) * safe_ce(fused, y) + self.detail_weight * safe_ce(detail, y[..., top:bottom, left:right])
            (loss / batch_size).backward()
        return None, {'phase': 'pcgd_cram', 'accumulated_pairs': batch_size}
