"""Run a declared underwater port with best/last/full resume and source gating."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .common import Segmenter, TargetImages, atomic_json, evaluate, make_optimizer, poly_lr, seed_all, sha


def source_record(manifest, protocol, allow_diagnostic=False):
    info = json.loads(Path(manifest).read_text(encoding='utf-8'))
    record = info['sources'][protocol]
    path = Path(record['path'])
    assert path.is_file(), f'Missing exact source checkpoint: {path}'
    assert sha(path) == record['sha256'], 'Source checksum changed'
    assert record['verified_source_only'] or allow_diagnostic, 'Exact source provenance is unresolved; full training is blocked'
    return record


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['secov2', 'dapass'], required=True)
    parser.add_argument('--protocol', choices=['uiis_to_suim', 'usis10k_to_suim'], required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--author-root', required=True)
    parser.add_argument('--sam-checkpoint')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--eval-period', type=int, default=500)
    parser.add_argument('--session-seconds', type=float, default=32400)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    seed_all(1)
    torch.set_num_threads(4)
    started = time.monotonic()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    record = source_record(args.source_manifest, args.protocol, args.diagnostic)
    expected = 8 if args.diagnostic else 1525
    eval_expected = 2 if args.diagnostic else 110
    images = TargetImages(Path(args.data) / 'train_val/images', expected)
    model = Segmenter(record['path']).cuda()
    initial = evaluate(model, Path(args.data) / 'TEST', expected=eval_expected)
    atomic_json(out / 'source_only.json', dict(initial, diagnostic=args.diagnostic,
                source=record, method_independent=True))
    if args.method == 'dapass':
        from .dapass import DAPASS
        method = DAPASS(model, images, out, tau=1 if args.diagnostic else 6000)
    else:
        from .secov2 import SeCoV2
        assert args.sam_checkpoint
        method = SeCoV2(model, images, out, args.author_root, args.sam_checkpoint,
                         graph_steps=2 if args.diagnostic else 2000)
    atomic_json(out / 'RUN_STATE.json', dict(status='preparing_pseudo_labels', method=args.method, optimizer_updates=0,
                diagnostic=args.diagnostic, source_sha256=record['sha256']))
    method.prepare(model)
    optimizer = make_optimizer(model, method.parameters())
    if args.method == 'dapass':
        # Paper optimizer retained on the common DeepLab backbone. LR is not
        # chosen from SUIM scores; this fixed recipe is declared before launch.
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(method.parameters()),
                                      lr=6e-5, eps=1e-8, weight_decay=1e-4)
        for group in optimizer.param_groups:
            group['multiplier'] = 1.
    step, best, best_iteration, metrics = 0, -1., None, []
    resume = out / 'resume.pth'
    if resume.exists():
        assert (out / 'session_pause.json').exists(), 'Only a clean saved pause may resume'
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        assert saved['source_sha256'] == record['sha256'] and saved['method'] == args.method
        assert saved['maximum_updates'] == args.steps
        model.load_state_dict(saved['model'], strict=True)
        if args.method == 'dapass':
            method.cram.load_state_dict(saved['cram'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cuda()
        restore_rng(saved['rng'])
        step, best, best_iteration, metrics = saved['iteration'], saved['best'], saved['best_iteration'], saved['metrics']
    model.train()
    while step < args.steps:
        poly_lr(optimizer, step, args.steps, base=6e-5 if args.method == 'dapass' else 2.5e-4)
        optimizer.zero_grad(set_to_none=True)
        loss, detail = method.loss(model, step, batch_size=4)
        if loss is not None:
            assert torch.isfinite(loss), f'Non-finite loss at {step}'
            loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), 'Missing/non-finite model gradients'
        optimizer.step()
        step += 1
        if step % 20 == 0 or args.diagnostic:
            print('OPTIMIZER_UPDATE', step, 'loss', float(loss) if loss is not None else 'accumulated', detail, flush=True)
        if step % args.eval_period == 0 or step == args.steps:
            if args.method == 'dapass':
                from .dapass import CRAMPredictor
                evaluated_model = CRAMPredictor(model, method.cram)
            else:
                evaluated_model = model
            row = dict(iteration=step, **evaluate(evaluated_model, Path(args.data) / 'TEST', expected=eval_expected))
            metrics.append(row)
            atomic_json(out / 'metrics.json', metrics)
            weights = dict(iteration=step, feature_extractor=model.feature.state_dict(), classifier=model.head.state_dict(),
                           method=args.method, source_sha256=record['sha256'])
            if args.method == 'dapass':
                weights['cram'] = method.cram.state_dict()
            for name in ['last.pth'] + (['best.pth'] if row['miou'] > best else []):
                temporary = out / (name + '.tmp')
                torch.save(weights, temporary)
                temporary.replace(out / name)
            if row['miou'] > best:
                best, best_iteration = row['miou'], step
            saved = dict(iteration=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=rng_state(), best=best, best_iteration=best_iteration, metrics=metrics,
                method=args.method, source_sha256=record['sha256'], maximum_updates=args.steps)
            if args.method == 'dapass':
                saved['cram'] = method.cram.state_dict()
            temporary = out / 'resume.tmp'
            torch.save(saved, temporary)
            temporary.replace(resume)
            complete = step == args.steps
            paused = not complete and time.monotonic() - started >= args.session_seconds
            summary = dict(status='complete' if complete else 'session_paused' if paused else 'training',
                optimizer_updates=step, best_miou=best, best_iteration=best_iteration, last=row,
                diagnostic=args.diagnostic, method=args.method,
                selection='peak mIoU8 from labeled SUIM evaluation; no evaluation-label gradient supervision',
                endpoint_is_full_budget=complete, source_sha256=record['sha256'])
            atomic_json(out / 'RUN_STATE.json', summary)
            print('EVALUATION', json.dumps(summary), flush=True)
            if complete:
                atomic_json(out / 'complete.json', summary)
            if paused:
                atomic_json(out / 'session_pause.json', summary)
                return


if __name__ == '__main__':
    main()
