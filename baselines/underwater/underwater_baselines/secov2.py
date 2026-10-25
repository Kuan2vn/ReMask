"""SeCoV2 PSA/SCC port with a separately trained DeepLab region encoder.

Uses the pinned author PSA, GraphBuilder and GraphVAE. A trainable2048->128
projection replaces the author SegFormer region head for the common8c source.
This is an explicitly adapted self-training baseline, not author-host parity.
"""
import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .common import image_tensor, load_file, atomic_json


def build_psa(author_root, destination):
    root, destination = Path(author_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    text = (root / 'PSA/seco_psa_v2.py').read_text(encoding='utf-8')
    assert text.count('save_image_with_anns') == 1, 'Remove only the unused missing import'
    text = text.replace(', save_image_with_anns', '')
    old = "gt =  Image.open(os.path.join(args.gt_root_path, id.replace('_leftImg8bit','_gtFine_labelTrainIds')))"
    assert text.count(old) == 1
    text = text.replace(old, "gt = Image.fromarray(np.full((image.height, image.width), 255, dtype=np.uint8))")
    text = text.replace('if args.class_num == 19:',
        'if args.class_num == 8:\n'
        '            STUFF_LABEL = list(range(8))\n'
        '            THING_LABEL = [1, 3, 4, 6]\n'
        '        elif args.class_num == 19:')
    text = text.replace('target_size = (1024, 512)', 'target_size = (1280, 640)')
    path = destination / 'seco_psa_v2.py'
    path.write_text(text, encoding='utf-8')
    (destination / 'utils.py').write_bytes((root / 'PSA/utils.py').read_bytes())
    compile(text, str(path), 'exec')
    return path


class SeCoV2:
    def __init__(self, model, images, output, author_root, sam_checkpoint, graph_steps=2000, device='cuda'):
        self.images, self.output, self.author = images, Path(output), Path(author_root)
        self.device, self.sam_checkpoint, self.graph_steps = device, Path(sam_checkpoint), graph_steps
        # Author SCC learns its region encoder jointly with the graph. Use a
        # private source-initialized copy so this never mutates the host source.
        self.region_encoder = copy.deepcopy(model.feature).to(device)
        self.labels = self.output / 'corrected_labels'
        self.psa_input = self.output / 'psa_input'
        self.psa_root = self.output / 'psa_labels'
        graph_module = load_file(self.author / 'SCC_V2/core/models/GraphBuilder.py', 'seco_graph_builder')
        vae_module = load_file(self.author / 'SCC_V2/core/models/vae.py', 'seco_graph_vae')
        self.graph = graph_module.GraphBuilder()
        self.projection = nn.Linear(2048, 128).to(device)
        self.vae = vae_module.GraphVAEWithGraphSAGE(input_dim=128, latent_dim=256, num_classes=8).to(device)
        # Author GraphSAGE creates its linear layer lazily. Materialize it before
        # constructing an optimizer, otherwise those parameters never update.
        self.vae(torch.eye(2, device=device)[None], torch.zeros(1, 2, 128, device=device))

    @torch.no_grad()
    def pixel_labels(self, model):
        self.psa_input.mkdir(exist_ok=True)
        self.psa_root.mkdir(exist_ok=True)
        model.eval()
        names = []
        for index, path in enumerate(self.images.paths):
            name = path.stem + '.png'
            names.append(name)
            image = self.images.image(index)
            image.save(self.psa_input / name)
            x = image_tensor(image, self.device).unsqueeze(0)
            probabilities = model(x).softmax(1)
            confidence, label = probabilities.max(1)
            # The release explicitly accepts unthresholded DTST_PL/CS_no_thr.
            # Use source predictions here; SCC supplies the subsequent filter.
            Image.fromarray(label[0].cpu().numpy().astype(np.uint8)).save(self.psa_root / name)
        (self.output / 'psa_names.txt').write_text('\n'.join(names) + '\n')

    def region_features(self, model, records):
        xs, masks, labels, boxes = [], [], [], []
        from pycocotools import mask as mask_utils
        for record in records:
            with Image.open(record['crop_path']) as image:
                xs.append(image_tensor(image.convert('RGB').resize((128, 64))))
            rle = dict(record['mask'])
            if isinstance(rle['counts'], str):
                rle['counts'] = rle['counts'].encode()
            masks.append(torch.from_numpy(mask_utils.decode(rle).copy()).float())
            labels.append(int(record['psd_label']))
            x, y, w, h = record['bbox_xywh']
            boxes.append([x / 1280, y / 640, (x + w) / 1280, (y + h) / 640])
        features = self.region_encoder(torch.stack(xs).to(self.device))[1]
        masks = F.interpolate(torch.stack(masks)[:, None].to(self.device), features.shape[-2:], mode='nearest')
        pooled = (features * masks).sum((2, 3)) / masks.sum((2, 3)).clamp_min(1)
        return self.projection(pooled), torch.tensor(labels, device=self.device), torch.tensor(boxes, device=self.device)

    def prepare(self, model):
        if (self.output / 'SECO_PREPARED.json').exists():
            return
        infos_path = Path(str(self.psa_root) + '_mask_psa_infos_cls.json')
        if not (self.output / 'PSA_COMPLETE.json').exists():
            self.pixel_labels(model)
            script = build_psa(self.author, self.output / 'psa_code')
            command = [sys.executable, '-u', str(script), '--id-list-path', str(self.output / 'psa_names.txt'),
                       '--class-num', '8', '--image-root-path', str(self.psa_input),
                       '--psd-root-path', str(self.psa_root), '--sam-checkpoint', str(self.sam_checkpoint), '--sam-type', 'vit_b']
            subprocess.run(command, check=True)
            atomic_json(self.output / 'PSA_COMPLETE.json', dict(complete=True, manual_target_labels=False))
        by_class = json.loads(infos_path.read_text())
        entries = [record for records in by_class.values() for record in records]
        assert entries and all(int(e.get('gt_label', 255)) == 255 for e in entries)
        classes = sorted(int(c) for c, records in by_class.items() if records)
        class_weights = np.asarray([len(by_class[str(cid)]) ** .3 for cid in classes], dtype=np.float64)
        class_weights /= class_weights.sum()
        optimizer = torch.optim.SGD([
            dict(params=self.region_encoder.parameters(), lr=.0005, base_lr=.0005),
            dict(params=list(self.projection.parameters()) + list(self.vae.parameters()), lr=.005, base_lr=.005)
        ], momentum=.9, weight_decay=5e-4)
        model.eval()
        self.region_encoder.train()
        for step in range(self.graph_steps):
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * (1 - step / self.graph_steps) ** .9
            # Author per-region sampling weight frequency^-0.7 yields class
            # mass frequency^0.3. All counts come from pseudo-label regions.
            sampled_classes = np.random.choice(classes, size=32, p=class_weights)
            records = [random.choice(by_class[str(int(cid))]) for cid in sampled_classes]
            features, labels, boxes = self.region_features(model, records)
            adjacency, _, _ = self.graph.build_graph(features.detach(), boxes)
            reconstruction, original, mu, logvar, logits = self.vae(adjacency, features[None])
            loss, _ = self.vae.combined_loss(reconstruction, original, mu, logvar, logits, labels[None], step, self.graph_steps)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert any(p.grad is not None for p in self.region_encoder.parameters()), 'SCC encoder must learn with the graph'
            optimizer.step()
            if (step + 1) % 100 == 0:
                print('SECO_GRAPH', step + 1, float(loss), flush=True)
        torch.save(dict(region_encoder=self.region_encoder.state_dict(), projection=self.projection.state_dict(), graph_vae=self.vae.state_dict(),
                        auxiliary_updates=self.graph_steps), self.output / 'scc_graph.pth')
        self.correct(model, entries)
        atomic_json(self.output / 'SECO_PREPARED.json', dict(complete=True, regions=len(entries),
                    graph_steps=self.graph_steps, extra_pretrained_resource='SAM_vit_b',
                    port='PSA + author GraphVAE/SCC with jointly trained separate DeepLab region encoder/projection'))

    @torch.no_grad()
    def correct(self, model, entries):
        self.vae.eval()
        self.projection.eval()
        self.region_encoder.eval()
        decisions = []
        for start in range(0, len(entries), 32):
            batch = entries[start:start + 32]
            features, _, boxes = self.region_features(model, batch)
            adjacency, _, _ = self.graph.build_graph(features, boxes)
            reconstruction, original, mu, logvar, logits = self.vae(adjacency, features[None])
            uncertainty, _, _ = self.vae.compute_uncertainty(reconstruction, original, logits,
                                                           mu=mu, logvar=logvar, w_kl=1.)
            confidence, prediction = logits.softmax(-1).max(-1)
            decisions.extend(dict(record=e, uncertainty=float(uncertainty[0, i]),
                                  confidence=float(confidence[0, i]), prediction=int(prediction[0, i])) for i, e in enumerate(batch))
        kept = set()
        for cid in range(8):
            indices = [i for i, d in enumerate(decisions) if int(d['record']['psd_label']) == cid]
            indices.sort(key=lambda i: decisions[i]['uncertainty'])
            kept.update(indices[:int(np.ceil(len(indices) * .8))])
        self.labels.mkdir(exist_ok=True)
        for path in self.images.paths:
            canvas = np.full((640, 1280), 255, dtype=np.uint8)
            confidence_map = np.full((640, 1280), -1., dtype=np.float32)
            from pycocotools import mask as mask_utils
            for i, decision in enumerate(decisions):
                record = decision['record']
                if i not in kept or Path(record['image_id']).stem != path.stem:
                    continue
                x, y, w, h = record['bbox_xywh']
                rle = dict(record['mask'])
                if isinstance(rle['counts'], str):
                    rle['counts'] = rle['counts'].encode()
                mask = np.asarray(Image.fromarray(mask_utils.decode(rle)).resize((w, h), Image.Resampling.NEAREST)) > 0
                label = decision['prediction'] if decision['confidence'] >= .95 else int(record['psd_label'])
                update = mask & (decision['confidence'] > confidence_map[y:y+h, x:x+w])
                canvas[y:y+h, x:x+w][update] = label
                confidence_map[y:y+h, x:x+w][update] = decision['confidence']
            Image.fromarray(canvas).save(self.labels / (path.stem + '.png'))

    def parameters(self):
        return []

    def loss(self, model, step, batch_size=4):
        from .common import safe_ce
        x, y = self.images.batch(random.sample(range(len(self.images)), batch_size), self.labels, self.device)
        return safe_ce(model(x), y), {'phase': 'corrected_pseudo_label_training'}
