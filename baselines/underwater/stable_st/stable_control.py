"""Evaluation and checkpoint-safe session control for the native Stable-ST port."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


class Control:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.out = Path(cfg.OUTPUT_DIR)
        self.started = time.monotonic()
        self.seconds = float(os.environ.get('UNDERWATER_SESSION_SECONDS', '32400'))
        self.resumed = False
        self.best = -1.
        self.best_iteration = None
        if (self.out / 'metrics.json').exists():
            self.metrics = json.loads((self.out / 'metrics.json').read_text())
            adapted = [row for row in self.metrics if row.get('role') != 'source_only']
            if adapted:
                best = max(adapted, key=lambda row: row['miou'])
                self.best, self.best_iteration = best['miou'], best['iteration']
        else:
            self.metrics = []

    def restore(self, state):
        self.start_from_none = state.get('start_from_None', True)
        path = self.out / 'resume.pth'
        if path.exists():
            assert (self.out / 'session_pause.json').exists(), 'Restore only a clean checkpoint pause'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            for name in ('feature_extractor', 'classifier', 'feature_extractor_his', 'classifier_his'):
                state[name].load_state_dict(saved[name], strict=True)
            for name in ('optimizer_fea', 'optimizer_cls'):
                state[name].load_state_dict(saved[name])
                for group in state[name].state.values():
                    for key, value in group.items():
                        if torch.is_tensor(value):
                            group[key] = value.to(self.device)
            torch.set_rng_state(saved['torch_rng'])
            if self.device.type == 'cuda':
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            np.random.set_state(saved['numpy_rng'])
            random.setstate(saved['python_rng'])
            self.resumed = True
            self.start_from_none = saved['start_from_None']
            self.best, self.best_iteration = saved['best'], saved['best_iteration']
            return (saved['iteration'], saved['running_conf'].to(self.device), saved['stu_eval_list'],
                    saved['stu_score_buffer'], saved['res_dict'])
        return (0, state['running_conf'], state.get('stu_eval_list', []),
                state.get('stu_score_buffer', []), state.get('res_dict', {}))

    @torch.no_grad()
    def evaluate(self, iteration, feature, classifier, initial=False):
        if initial and (self.resumed or self.metrics):
            return
        from core.datasets import build_dataset
        feature.eval()
        classifier.eval()
        dataset = build_dataset(self.cfg, mode='test', is_source=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        confusion = torch.zeros(8, 8, dtype=torch.int64, device=self.device)
        for images, labels, _ in loader:
            labels = labels.to(self.device).long()
            prediction = classifier(feature(images.to(self.device))[1], labels.shape[-2:]).argmax(1)
            valid = (labels >= 0) & (labels < 8)
            confusion += torch.bincount(labels[valid] * 8 + prediction[valid], minlength=64).reshape(8, 8)
        values = confusion.double()
        intersection = values.diag()
        union = values.sum(0) + values.sum(1) - intersection
        iou = intersection / union.clamp_min(1)
        row = dict(iteration=iteration, miou=float(iou.mean()), class_iou=iou.cpu().tolist(),
                   confusion=confusion.cpu().tolist(), labeled_selection=True,
                   evaluation='official_SUIM_TEST_110', role='source_only' if initial else 'adapted_teacher')
        self.metrics.append(row)
        atomic_json(self.out / 'metrics.json', self.metrics)
        if not initial and row['miou'] > self.best:
            self.best, self.best_iteration = row['miou'], iteration
            self.save_model(self.out / 'best.pth', iteration, feature, classifier)
        print('UNDERWATER_EVAL', json.dumps(row), flush=True)
        return row

    @staticmethod
    def save_model(path, iteration, feature, classifier):
        temporary = path.with_suffix('.tmp')
        torch.save(dict(iteration=iteration, feature_extractor=feature.state_dict(),
                        classifier=classifier.state_dict()), temporary)
        temporary.replace(path)

    def boundary(self, iteration, feature, classifier, state):
        row = self.evaluate(iteration, feature, classifier)
        self.save_model(self.out / 'last.pth', iteration, feature, classifier)
        finished = iteration >= self.cfg.SOLVER.STOP_ITER
        paused = not finished and time.monotonic() - self.started >= self.seconds
        # The native loop refreshes candidate banks after evaluation. Complete
        # that pending transition before pausing so resume sees the right bank.
        if paused and ((self.cfg.TCR.OPEN and iteration % self.cfg.TCR.UPDATE_FREQUENCY == 0)
                       or (self.cfg.METAPL.OPEN and iteration % self.cfg.METAPL.VAL_UPDATE == 0)):
            from core.apis.inference import run_candidate
            from datasets.generate_city_label_info import gen_lb_info
            run_candidate(self.cfg, feature, classifier, 0, False, update_meta=state['start_from_None'])
            for name in ('CTR', 'CTR_O', 'meta_val', 'meta_val_mixup'):
                gen_lb_info(self.cfg, name)
        saved = {name: state[name].state_dict() for name in
                 ('feature_extractor', 'classifier', 'feature_extractor_his',
                  'classifier_his', 'optimizer_fea', 'optimizer_cls')}
        saved.update(iteration=iteration, running_conf=state['running_conf'].cpu(),
            stu_eval_list=state.get('stu_eval_list', []),
            stu_score_buffer=state.get('stu_score_buffer', []), res_dict=state.get('res_dict', {}),
            start_from_None=state.get('start_from_None', True),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            best=self.best, best_iteration=self.best_iteration,
            resume_limit='Full optimizer/teacher/NFD/RNG state; restarted DataLoader shuffle order is not bitwise replay')
        temporary = self.out / 'resume.tmp'
        torch.save(saved, temporary)
        temporary.replace(self.out / 'resume.pth')
        status = 'complete' if finished else 'session_paused' if paused else 'training'
        summary = dict(status=status, optimizer_updates=iteration, best_miou=self.best,
                       best_iteration=self.best_iteration, last=row,
                       stop_rule='fixed_budget', selection='peak mIoU8 on labeled official SUIM test',
                       no_target_training_gt=True,
                       diagnostic=os.environ.get('UNDERWATER_DIAGNOSTIC', '0') == '1')
        atomic_json(self.out / 'RUN_STATE.json', summary)
        if finished:
            atomic_json(self.out / 'complete.json', summary)
        if paused:
            atomic_json(self.out / 'session_pause.json', summary)
            raise SystemExit(0)
