"""Build a private Stable-ST port; preserve the pinned author trainer and NFD.

Only data/model compatibility, class guards, metrics and checkpoint control are
patched. The original checkout in references/Stable-ST is left unchanged.
"""
import ast
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIN = 'c447da35598b285eb8868caf4f969aaba1bf39b8'


def replace_once(text, old, new):
    assert text.count(old) == 1, repr(old[:100])
    return text.replace(old, new, 1)


def build(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'core', destination / 'core', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    model_build = destination / 'core/models/build.py'
    model_build.write_text(model_build.read_text(encoding='utf-8').replace('pretrained_backbone=True', 'pretrained_backbone=False'), encoding='utf-8')
    (destination / 'datasets').mkdir(exist_ok=True)
    for name in ('__init__.py', 'generate_city_label_info.py'):
        shutil.copy2(ROOT / 'datasets' / name, destination / 'datasets' / name)
    shutil.copy2(ROOT / 'underwater_baselines/stable_control.py', destination / 'stable_control.py')
    shutil.copy2(ROOT / 'underwater_baselines/stable_sampling.py', destination / 'stable_sampling.py')
    author = ROOT / 'references/Stable-ST'
    api = (author / 'core/apis/inference.py').read_text(encoding='utf-8')
    project_api = (ROOT / 'core/apis/inference.py').read_text(encoding='utf-8')
    tree = ast.parse(project_api)
    lines = project_api.splitlines(True)
    sample = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_val_images')
    sample_code = ''.join(lines[sample.lineno-1:sample.end_lineno])
    start = api.index('def sample_val_images(')
    end = api.index('def evel_stu(', start)
    api = api[:start] + sample_code + '\n\ndef _is_underwater_dataset(name):\n    return name.startswith("suim")\n\n' + api[end:]
    # Cache grids must agree with the port's fixed 1280x640 full-image view.
    api = api.replace('256, 512', '320, 640').replace('[256, 512]', '[320, 640]')
    api = api.replace('512, 1024', '320, 640').replace('fx=4.0, fy=4.0', 'fx=2.0, fy=2.0')
    api = api.replace('thres.append(0)\n                continue', 'thres.append(0)\n                meta_thres.append(0)\n                continue')
    api = api.replace('thres.append(0)\n            else:', 'thres.append(0)\n                meta_thres.append(0)\n            else:')
    # A class may have too few positive scores for rounded quantile indexing.
    api = api.replace('x[int(np.round(len(x)*cfg.TCR.TOPK_CANDIDATE))]', 'x[min(len(x)-1, int(np.round(len(x)*cfg.TCR.TOPK_CANDIDATE)))]')
    api = api.replace('x[int(np.round(len(x)*0.9))]', 'x[min(len(x)-1, int(np.round(len(x)*0.9)))]')
    (destination / 'core/apis/inference.py').write_text(api, encoding='utf-8')

    dataset_build = (destination / 'core/datasets/build.py').read_text(encoding='utf-8')
    dataset_build = replace_once(dataset_build, '    return dataset\n',
        '    if not is_source and mode == "train" and rsc is not None:\n'
        '        from stable_sampling import resample\n'
        '        dataset = resample(dataset, cfg, rsc, iters)\n'
        '    if not is_source and mode in ("train", "III"):\n'
        '        dataset.ignore_real_labels = True\n'
        '    return dataset\n')
    (destination / 'core/datasets/build.py').write_text(dataset_build, encoding='utf-8')
    suim_path = destination / 'core/datasets/suim.py'
    suim = suim_path.read_text(encoding='utf-8')
    # The plain view is used by source-only evaluation and candidate generation.
    suim = replace_once(suim, '        if not self.pseudo:\n',
        '        if getattr(self, "ignore_real_labels", False):\n'
        '            label = Image.fromarray(np.full((image.height, image.width), 255, dtype=np.uint8))\n'
        '        elif not self.pseudo:\n')
    suim = replace_once(suim, '        while True:\n', '        for _attempt in range(100):\n')
    suim = replace_once(suim,
        '            if np.sum(mask)>=ins_pix_num:\n                break\n',
        '            if np.sum(mask)>=ins_pix_num:\n                break\n'
        '        else:\n            return out_imgs, out_lbls\n')
    suim_path.write_text(suim, encoding='utf-8')

    trainer = (author / 'train_stable_st.py').read_text(encoding='utf-8')
    trainer = 'from stable_control import Control\n' + trainer
    trainer = replace_once(trainer, 'sim.masked_fill(mask_self.nonzero(as_tuple=True)[0], -1.0)',
                           'sim.masked_fill(mask_self, -1.0)')
    trainer = replace_once(trainer, '    start_training_time = time.time()',
        '    control = Control(cfg)\n'
        '    iteration, running_conf, stu_eval_list, stu_score_buffer, res_dict = control.restore(locals())\n'
        '    start_from_None = control.start_from_none\n'
        '    control.evaluate(0, feature_extractor_his, classifier_his, initial=True)\n'
        '    start_training_time = time.time()')
    trainer = replace_once(trainer, '(iteration+1) % (cfg.SOLVER.CHECKPOINT_PERIOD)==0',
                           'iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == cfg.SOLVER.STOP_ITER')
    trainer = replace_once(trainer,
        '                run_test(cfg, feature_extractor_his, classifier_his, local_rank, distributed)',
        '                control.boundary(iteration, feature_extractor_his, classifier_his, locals())')
    trainer = trainer.replace('(iteration+1) % cfg.TCR.UPDATE_FREQUENCY', 'iteration % cfg.TCR.UPDATE_FREQUENCY')
    trainer = trainer.replace('(iteration+1) % cfg.METAPL.VAL_UPDATE', 'iteration % cfg.METAPL.VAL_UPDATE')
    # Tests and post-step hooks must not leave student modules in eval mode.
    trainer = trainer.replace('            ### re-build candidate and dataloader',
        '            feature_extractor.train()\n            classifier.train()\n'
        '            ### re-build candidate and dataloader')
    trainer = trainer.replace("df.to_csv('dyIter_FN.csv')", "df.to_csv(os.path.join(output_dir, 'dyIter_FN.csv'))")
    trainer = trainer.replace('    torch.backends.cudnn.benchmark = True',
        '    random.seed(1)\n    np.random.seed(1)\n    torch.manual_seed(1)\n'
        '    torch.cuda.manual_seed_all(1)\n    torch.backends.cudnn.benchmark = False')
    trainer = replace_once(trainer,
        "                torch.save({'iteration': iteration, 'feature_extractor': feature_extractor_his.state_dict(), 'classifier':classifier_his.state_dict()}, filename)",
        '                # Control retains best/last/full resume without 60 redundant model files.')
    (destination / 'train_stable_st.py').write_text(trainer, encoding='utf-8')
    for path in destination.rglob('*.py'):
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    hashes = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in destination.rglob('*.py')}
    manifest = dict(upstream='https://github.com/DZhaoXd/Stable-ST', commit=PIN,
        implementation='Pinned author train_stable_st.py and NFD evaluator with project underwater dataset/model compatibility',
        changes=['8 classes; 1525 unlabeled target / 110 evaluation images',
                 'No target train masks mounted; dummy 255 labels for candidate diagnostics',
                 'Author class-balanced image resampling retained using predicted CTR_O.p banks',
                 'Full view1280x640; crop960x480; candidate cache640x320',
                 'Missing-class/quantile bounds guards',
                 'Bound donor retries when a rare class has no sufficiently large mask',
                 'Skip redundant ImageNet init before strict complete source-state loading',
                 'One-based500-update checkpoint/eval cadence and full-state session pause'],
        files=hashes)
    (destination / 'PORT_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('destination')
    print(json.dumps(build(parser.parse_args().destination), indent=2))
