"""CPU correctness checks for the new underwater ports, not method results."""
import json
import tempfile
import unittest
import copy
import pickle
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from underwater_baselines.common import COLORS, TargetImages, decode_mask, evaluation_tensor, load_file, safe_ce, sha
from underwater_baselines.dapass import CRAM, CRAMPredictor, bi_level_loss, copy_paste, grid_box
from underwater_baselines.run import source_record
from underwater_baselines.build_stable_st import build
from underwater_baselines.secov2 import SeCoV2, build_psa
from underwater_baselines.stable_control import Control
from underwater_baselines.stable_sampling import resample

torch.set_num_threads(1)


class Feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2048, 1, stride=8)

    def forward(self, x):
        out = self.conv(x)
        return out, out


class TinySegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = Feature()
        self.head = nn.Conv2d(2048, 8, 1)

    def forward(self, x, size=None):
        return F.interpolate(self.head(self.feature(x)[1]), size=size or x.shape[-2:], mode='bilinear', align_corners=True)


class UnderwaterPorts(unittest.TestCase):
    def test_evaluation_tensor_matches_historical_bicubic_pipeline(self):
        from core.datasets.transform import Compose, Resize, ToTensor, Normalize
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (37, 59, 3), dtype=np.uint8))
            image.save(path)
            historical, _, _ = Compose([Resize((640, 1280), resize_label=False), ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225], to_bgr255=False)])(image, Image.new('L', image.size))
            self.assertTrue(torch.equal(evaluation_tensor(path), historical))

    def test_palette_preserves_all_eight_classes(self):
        image = Image.fromarray(np.asarray(COLORS, dtype=np.uint8)[None])
        np.testing.assert_array_equal(decode_mask(image), np.arange(8)[None])

    def test_no_training_mask_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(8):
                Image.new('RGB', (64, 32), (index * 20, 40, 100)).save(root / f'{index}.jpg')
            dataset = TargetImages(root, 8)
            x, labels = dataset.batch([0], device='cpu', jitter=False)
            self.assertEqual(tuple(x.shape), (1, 3, 480, 960))
            self.assertIsNone(labels)

    def test_all_ignore_loss_is_finite_and_has_no_gradient(self):
        logits = torch.randn(1, 8, 8, 8, requires_grad=True)
        loss = safe_ce(logits, torch.full((1, 8, 8), 255))
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(logits.grad.abs().sum()), 0.)

    def test_copy_paste_preserves_recipient_outside_donor_class(self):
        x = torch.zeros(1, 3, 4, 4)
        y = torch.ones(1, 4, 4, dtype=torch.long)
        donor_x = torch.ones_like(x)
        donor_y = torch.full_like(y, 255)
        donor_y[:, :2, :2] = 6
        mixed_x, mixed_y = copy_paste(x, y, donor_x, donor_y, [6])
        self.assertEqual(int((mixed_y == 6).sum()), 4)
        self.assertEqual(int((mixed_y == 1).sum()), 12)
        self.assertEqual(float(mixed_x.sum()), 12.)

    def test_cram_alignment_and_attention_gradients(self):
        model, cram = TinySegmenter(), CRAM()
        x = torch.randn(1, 3, 64, 64)
        box = grid_box(64, 64)
        fused, detail = cram(model, x, box)
        left, top, right, bottom = box
        self.assertEqual(tuple(fused.shape), (1, 8, 64, 64))
        self.assertEqual(tuple(detail.shape), (1, 8, bottom-top, right-left))
        fused.square().mean().backward()
        self.assertGreater(float(cram.attention[0].weight.grad.abs().sum()), 0.)
        with self.assertRaises(AssertionError):
            cram(model, x, (1, 0, 33, 32))

    def test_evaluation_uses_learned_cram(self):
        model, cram = TinySegmenter(), CRAM()
        wrapper = CRAMPredictor(model, cram)
        x = torch.randn(1, 3, 64, 64)
        first = wrapper(x).detach()
        with torch.no_grad():
            cram.attention[-1].bias.add_(20.)
        second = wrapper(x).detach()
        self.assertEqual(tuple(wrapper(x, size=(40, 80)).shape), (1, 8, 40, 80))
        self.assertGreater(float((second-first).abs().sum()), 0.)

    def test_bilevel_outer_loss_updates_feature_and_head(self):
        model = TinySegmenter()
        x, neighbor = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
        labels = torch.randint(0, 8, (1, 32, 32))
        loss = bi_level_loss(model, x, labels, neighbor, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.feature.conv.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.)

    def test_unknown_source_cannot_launch_full_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'candidate.pth'
            source.write_bytes(b'candidate-not-a-source-proof')
            manifest = root / 'sources.json'
            manifest.write_text(json.dumps(dict(sources=dict(uiis_to_suim=dict(path=str(source), sha256=sha(source), verified_source_only=False)))))
            with self.assertRaises(AssertionError):
                source_record(manifest, 'uiis_to_suim')
            self.assertFalse(source_record(manifest, 'uiis_to_suim', allow_diagnostic=True)['verified_source_only'])

    def test_native_stable_st_keeps_nfd_and_blocks_training_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            build(directory)
            root = Path(directory)
            trainer = (root / 'train_stable_st.py').read_text(encoding='utf-8')
            self.assertIn("cfg.DTU.PROXY_METRIC == 'NFD'", trainer)
            self.assertIn('higher.innerloop_ctx', trainer)
            self.assertIn('control.boundary', trainer)
            self.assertIn('sim.masked_fill(mask_self, -1.0)', trainer)
            self.assertIn('dataset.ignore_real_labels = True', (root / 'core/datasets/build.py').read_text())
            self.assertNotIn('x[int(np.round(len(x)*0.9))]', (root / 'core/apis/inference.py').read_text(encoding='utf-8'))

    def test_psa_eight_class_port_does_not_open_train_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_psa(ROOT / 'references/SeCoV2', directory)
            code = path.read_text(encoding='utf-8')
            self.assertIn('if args.class_num == 8:', code)
            self.assertNotIn('Image.open(os.path.join(args.gt_root_path', code)
            self.assertIn('STUFF_LABEL = list(range(8))', code)
            self.assertNotIn('save_image_with_anns', code)

    def test_graph_parameters_materialized_before_optimizer(self):
        module = load_file(ROOT / 'references/SeCoV2/SCC_V2/core/models/vae.py', 'test_seco_vae')
        vae = module.GraphVAEWithGraphSAGE(128, 256, 8)
        self.assertIsNone(vae.gcn.linear)
        adjacency = torch.eye(4)[None]
        features = torch.randn(1, 4, 128)
        outputs = vae(adjacency, features)
        optimizer = torch.optim.SGD(vae.parameters(), lr=.01)
        self.assertTrue(any(parameter is vae.gcn.linear.weight for group in optimizer.param_groups for parameter in group['params']))
        loss, _ = vae.combined_loss(*outputs, torch.tensor([[0, 1, 6, 7]]), 0, 2)
        before = vae.gcn.linear.weight.detach().clone()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float((before-vae.gcn.linear.weight).abs().sum()), 0.)

    def test_scc_joint_region_learning_keeps_host_source_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinySegmenter()
            port = SeCoV2(model, None, directory, ROOT / 'references/SeCoV2', 'unused.pth', graph_steps=2, device='cpu')
            features = port.projection(port.region_encoder(torch.randn(4, 3, 32, 32))[1].mean((2, 3)))
            outputs = port.vae(torch.eye(4)[None], features[None])
            loss, _ = port.vae.combined_loss(*outputs, torch.tensor([[0, 1, 6, 7]]), 0, 2)
            loss.backward()
            self.assertGreater(float(port.region_encoder.conv.weight.grad.abs().sum()), 0.)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(torch.equal(model.feature.conv.weight, port.region_encoder.conv.weight))

    def test_native_checkpoint_roundtrip_preserves_optimizer_teacher_and_meta_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(OUTPUT_DIR=directory, MODEL=SimpleNamespace(DEVICE='cpu'),
                SOLVER=SimpleNamespace(STOP_ITER=2), TCR=SimpleNamespace(OPEN=False),
                METAPL=SimpleNamespace(OPEN=False))
            feature, classifier = nn.Linear(2, 3), nn.Linear(3, 8)
            optimizer_fea = torch.optim.SGD(feature.parameters(), lr=.1, momentum=.9)
            optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=.1, momentum=.9)
            classifier(feature(torch.ones(1, 2))).sum().backward()
            optimizer_fea.step()
            optimizer_cls.step()
            state = dict(feature_extractor=feature, classifier=classifier,
                feature_extractor_his=copy.deepcopy(feature), classifier_his=copy.deepcopy(classifier),
                optimizer_fea=optimizer_fea, optimizer_cls=optimizer_cls, running_conf=torch.ones(8),
                stu_eval_list=[torch.ones(1)], stu_score_buffer=[.5], res_dict=dict(update_iter=[1]),
                start_from_None=True)
            control = Control(cfg)
            control.seconds = 0
            control.evaluate = lambda *args, **kwargs: dict(iteration=1, miou=.2)
            with self.assertRaises(SystemExit):
                control.boundary(1, state['feature_extractor_his'], state['classifier_his'], state)
            expected = feature.weight.detach().clone()
            with torch.no_grad():
                feature.weight.add_(10)
            restored = Control(cfg)
            result = restored.restore(state)
            self.assertEqual(result[0], 1)
            self.assertTrue(restored.start_from_none)
            self.assertTrue(torch.equal(expected, feature.weight))
            self.assertTrue(optimizer_fea.state[feature.weight]['momentum_buffer'].abs().sum() > 0)

    def test_class_balanced_resampling_uses_only_pseudo_banks(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(OUTPUT_DIR=directory)
            records = [dict(name=name, img='not-opened.jpg', label='forbidden-real-mask.bmp') for name in ('a', 'b', 'c')]
            dataset = SimpleNamespace(NUM_CLASS=8, data_list=records)
            pools = {cid: [] for cid in range(8)}
            pools[0] = ['a.png', 'b.png']
            pools[6] = ['b.png', 'c.png']
            classes = {'a.png': [0], 'b.png': [0, 6], 'c.png': [6]}
            with open(Path(directory) / 'CTR_O.p', 'wb') as stream:
                pickle.dump((pools, classes), stream)
            result = resample(dataset, cfg, 'CTR_O.p', 1000)
            self.assertEqual(len(result.data_list), 1500)
            self.assertTrue(all(record in records for record in result.data_list))
            self.assertEqual({record['name'] for record in result.data_list}, {'a', 'b', 'c'})


if __name__ == '__main__':
    unittest.main()
