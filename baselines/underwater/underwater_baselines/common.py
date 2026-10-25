"""Shared eight-class data, model, metrics and durable state for underwater ports."""
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TVF

COLORS = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255),
          (255, 0, 0), (255, 0, 255), (255, 255, 0), (255, 255, 255)]
NAMES = ['Background', 'Diver', 'Plant', 'Wreck', 'Robot', 'Reef', 'Fish', 'Floor']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def seed_all(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def decode_mask(image):
    arr = np.asarray(image)
    if arr.ndim == 2:
        out = arr.astype(np.int64)
        assert np.isin(out, list(range(8)) + [255]).all()
        return out
    out = np.full(arr.shape[:2], 255, dtype=np.int64)
    for index, color in enumerate(COLORS):
        out[(arr[..., :3] == color).all(-1)] = index
    return out


def image_tensor(image, device='cpu'):
    arr = np.asarray(image, dtype=np.float32).copy() / 255.
    arr = (arr - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1)).to(device)


def evaluation_tensor(path, device='cpu'):
    # Match the historical core Resize/ToTensor/Normalize evaluation exactly.
    with Image.open(path) as image:
        image = image.convert('RGB').resize((1280, 640), Image.Resampling.BICUBIC)
    return TVF.normalize(TVF.to_tensor(image), [.485, .456, .406], [.229, .224, .225]).to(device)


class Segmenter(nn.Module):
    def __init__(self, source=None):
        super().__init__()
        from core.models.feature_extractor import resnet_feature_extractor
        from core.models.classifier import ASPP_Classifier_V2
        # The complete source state is loaded strictly below; avoid an unrelated
        # ImageNet initialization/download before immediately replacing weights.
        self.feature = resnet_feature_extractor('resnet101', aux=True, pretrained_backbone=False, freeze_bn=True)
        self.head = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], 8)
        if source:
            saved = torch.load(source, map_location='cpu', weights_only=False)
            self.feature.load_state_dict({k.removeprefix('module.'): v for k, v in saved['feature_extractor'].items()}, strict=True)
            self.head.load_state_dict({k.removeprefix('module.'): v for k, v in saved['classifier'].items()}, strict=True)

    def forward(self, image, size=None):
        return self.head(self.feature(image)[1], size or image.shape[-2:])


def make_optimizer(model, extra=(), lr=2.5e-4):
    groups = [dict(params=model.feature.parameters(), lr=lr, multiplier=1.),
              dict(params=model.head.parameters(), lr=lr * 10, multiplier=10.)]
    extra = list(extra)
    if extra:
        groups.append(dict(params=extra, lr=lr * 10, multiplier=10.))
    return torch.optim.SGD(groups, momentum=.9, weight_decay=5e-4)


def poly_lr(optimizer, step, maximum, base=2.5e-4):
    value = base * max(0., 1. - step / maximum) ** .9
    for group in optimizer.param_groups:
        group['lr'] = value * group['multiplier']


class TargetImages:
    """Only images are addressable in adaptation; evaluation owns the masks."""
    def __init__(self, root, expected=1525):
        self.root = Path(root)
        self.paths = sorted(p for p in self.root.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
        assert len(self.paths) == expected, (len(self.paths), expected)

    def __len__(self):
        return len(self.paths)

    def image(self, index, size=(1280, 640)):
        with Image.open(self.paths[index]) as image:
            return image.convert('RGB').resize(size, Image.Resampling.BILINEAR)

    def batch(self, indices, labels=None, device='cuda', jitter=True):
        images, targets = [], []
        for index in indices:
            image = self.image(index)
            label = None if labels is None else Image.open(labels / (self.paths[index].stem + '.png'))
            if label is not None:
                label = label.resize(image.size, Image.Resampling.NEAREST)
            scale = random.uniform(.75, 1.25)
            width, height = max(960, round(1280 * scale)), max(480, round(640 * scale))
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            if label is not None:
                label = label.resize((width, height), Image.Resampling.NEAREST)
            left, top = random.randrange(width - 960 + 1), random.randrange(height - 480 + 1)
            box = (left, top, left + 960, top + 480)
            image = image.crop(box)
            if label is not None:
                label = label.crop(box)
            if random.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if label is not None:
                    label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if jitter:
                image = ColorJitter(.5, .5, .5, .2)(image)
            images.append(image_tensor(image))
            if label is not None:
                targets.append(torch.from_numpy(np.asarray(label).copy()).long())
        return torch.stack(images).to(device), None if labels is None else torch.stack(targets).to(device)


@torch.no_grad()
def evaluate(model, data_root, device='cuda', expected=110):
    was_training = model.training
    model.eval()
    root = Path(data_root)
    images = TargetImages(root / 'images', expected)
    confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
    for index, path in enumerate(images.paths):
        matches = [p for suffix in ('.bmp', '.png') if (p := root / 'masks' / (path.stem + suffix)).exists()]
        assert len(matches) == 1, path.name
        with Image.open(matches[0]) as label_image:
            label = torch.from_numpy(decode_mask(label_image)).to(device)
        image = evaluation_tensor(path, device).unsqueeze(0)
        prediction = model(image, size=label.shape).argmax(1)[0]
        valid = (label >= 0) & (label < 8)
        confusion += torch.bincount(label[valid] * 8 + prediction[valid], minlength=64).reshape(8, 8)
    model.train(was_training)
    values = confusion.double()
    iou = values.diag() / (values.sum(0) + values.sum(1) - values.diag()).clamp_min(1)
    return dict(miou=float(iou.mean()), class_iou=iou.cpu().tolist(), class_names=NAMES,
                confusion=confusion.cpu().tolist(), labeled_selection=True, evaluation_images=len(images))


def safe_ce(logits, labels, weight=None):
    loss = F.cross_entropy(logits, labels, ignore_index=255, reduction='none')
    valid = labels != 255
    if weight is None:
        weight = valid.float()
    else:
        weight = weight * valid
    return (loss * weight).sum() / weight.sum().clamp_min(1.)


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
