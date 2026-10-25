"""Portable setup, isolation and real subprocess-restart checks (CPU only)."""
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from underwater_baselines.assets import SAM_SHA256, SOURCE_REVISION, source_checkpoint, verify
from underwater_baselines.launch import check_existing, command_for, configuration, data_view, launch, run_lock, validate_data


def target_data(root):
    for folder in ('train_val/images', 'train_val/masks', 'TEST/images', 'TEST/masks'):
        (root / folder).mkdir(parents=True)
    for index in range(8):
        Image.new('RGB', (48, 32), (index * 20, 50, 80)).save(root / 'train_val/images' / f'{index}.jpg')
    for index in range(2):
        Image.new('RGB', (48, 32), (20, 50, 80)).save(root / 'TEST/images' / f'{index}.jpg')
        Image.new('RGB', (48, 32)).save(root / 'TEST/masks' / f'{index}.bmp')
    (root / 'train_val/masks/MUST_NOT_BE_READ.txt').write_text('not adaptation supervision')
    return root


class PortableLauncher(unittest.TestCase):
    def test_data_validation_and_isolated_views_exclude_training_gt(self):
        with tempfile.TemporaryDirectory(prefix='baseline with spaces ') as folder:
            root = Path(folder)
            data = target_data(root / 'SUIM')
            self.assertEqual(validate_data(data, smoke=True), data)
            with self.assertRaises(ValueError):
                validate_data(data, smoke=False)
            first = data_view(data, root / 'first', smoke=False)
            second = data_view(data, root / 'second', smoke=True)
            self.assertFalse((first / 'train_val/masks').exists())
            self.assertFalse((second / 'train_val/masks').exists())
            self.assertEqual((first / 'train_val/images').resolve(), data / 'train_val/images')
            self.assertEqual(len(list((second / 'train_val/images').glob('*.jpg'))), 8)
            self.assertEqual((root / 'first/.runtime/data/SUIM/orin_data').resolve(), first)
            (data / 'TEST/masks/0.bmp').unlink()
            with self.assertRaises(ValueError):
                validate_data(data, smoke=True)

    def test_public_source_download_is_revision_pinned_anonymous_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.pth'
            path.write_bytes(b'test checkpoint')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                verify(path, 'wrong')
            with mock.patch.dict('underwater_baselines.assets.SOURCES',
                {'fixture': dict(filename='fixture/source.pth', iteration=1, sha256=digest)}), \
                mock.patch('huggingface_hub.hf_hub_download', return_value=str(path)) as download:
                record = source_checkpoint('fixture', folder)
                self.assertEqual(record['sha256'], digest)
                self.assertTrue(record['verified_source_only'])
                self.assertEqual(download.call_args.kwargs['revision'], SOURCE_REVISION)
                self.assertIs(download.call_args.kwargs['token'], False)

    @unittest.skipUnless(os.name == 'posix', 'Linux native loader')
    def test_native_catalog_loads_unlabeled_train_and_labeled_eval_from_runtime_view(self):
        with tempfile.TemporaryDirectory(prefix='native loader ') as folder:
            root = Path(folder)
            data = target_data(root / 'SUIM')
            output = root / 'run'
            data_view(data, output, smoke=False)
            code = '''import sys,torch
from core.configs import cfg
from core.datasets import build_dataset
cfg.merge_from_file(sys.argv[1])
cfg.merge_from_list(['OUTPUT_DIR',sys.argv[2],'SOLVER.MAX_ITER','1','TCR.OPEN','False'])
train=build_dataset(cfg,mode='train',is_source=False)
sample=train[0]
assert tuple(sample[0].shape)==(3,480,960),sample[0].shape
assert bool((sample[1]==255).all()),torch.unique(sample[1])
evaluation=build_dataset(cfg,mode='test',is_source=False)
assert len(evaluation)==2
assert bool((evaluation[0][1]==0).all())
print('Native loader GT isolation and evaluation passed')
'''
            result = subprocess.run([sys.executable, '-c', code, str(ROOT / 'configs/stable_st.yaml'), str(output)],
                cwd=output / '.runtime', capture_output=True, text=True,
                env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'stable_st'), str(ROOT)])))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_six_commands_keep_full_recipe_and_diagnostic_overrides_separate(self):
        from core.configs import cfg
        parsed = cfg.clone()
        parsed.merge_from_file(str(ROOT / 'configs/stable_st.yaml'))
        self.assertEqual(parsed.MODEL.NUM_CLASSES, 8)
        self.assertEqual(parsed.DTU.PROXY_METRIC, 'NFD')
        self.assertEqual(tuple(parsed.INPUT.TARGET_INPUT_SIZE_TRAIN), (960, 480))
        self.assertEqual(parsed.SOLVER.MAX_ITER, 30000)
        self.assertEqual(parsed.SOLVER.STOP_ITER, 30000)
        self.assertEqual(parsed.SOLVER.CHECKPOINT_PERIOD, 500)
        for method in ('stable_st', 'secov2', 'dapass'):
            for protocol in ('uiis_to_suim', 'usis10k_to_suim'):
                command, env, cwd = command_for(method, protocol, dict(path='/weights/source.pth'),
                    Path('/data/SUIM'), Path('/output/run'), Path('/output/run/sources.json'), False, float('inf'),
                    sam=Path('/weights/sam.pth') if method == 'secov2' else None)
                if method == 'stable_st':
                    self.assertEqual(env['UNDERWATER_DIAGNOSTIC'], '0')
                    self.assertNotIn('SOLVER.STOP_ITER', command)
                    self.assertEqual(cwd, Path('/output/run/.runtime'))
                else:
                    self.assertEqual(command[command.index('--steps') + 1], '30000')
                    self.assertEqual(command[command.index('--eval-period') + 1], '500')
                    self.assertNotIn('--diagnostic', command)
        smoke, _, _ = command_for('stable_st', 'uiis_to_suim', dict(path='/source.pth'),
            Path('/data'), Path('/out'), Path('/manifest'), True, 0)
        self.assertEqual(smoke[smoke.index('SOLVER.STOP_ITER') + 1], '2')
        self.assertEqual(smoke[smoke.index('METAPL.TOP_K') + 1], '0.125')

    def test_output_guards_reject_mixed_protocols_and_unclean_restarts(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            config = configuration('stable_st', 'uiis_to_suim', Path('/SUIM'), dict(sha256='fixture'), False)
            (output / 'run_config.json').write_text(json.dumps(config))
            self.assertEqual(check_existing(output, config), 'new')
            with self.assertRaises(ValueError):
                check_existing(output, dict(config, diagnostic=True))
            (output / 'RUN_STARTED.json').write_text('{}')
            with self.assertRaises(ValueError):
                check_existing(output, config)
            (output / 'resume.pth').write_bytes(b'fixture')
            (output / 'session_pause.json').write_text('{}')
            self.assertEqual(check_existing(output, config), 'resume')

    @unittest.skipUnless(os.name == 'posix', 'Linux training launcher')
    def test_run_lock_prevents_duplicate_trainers(self):
        with tempfile.TemporaryDirectory() as folder:
            with run_lock(Path(folder)):
                with self.assertRaises(RuntimeError):
                    with run_lock(Path(folder)):
                        pass
            with run_lock(Path(folder)):
                pass

    @unittest.skipUnless(os.name == 'posix', 'Linux training launcher')
    def test_smoke_coordinator_restarts_a_real_process_and_reaches_update_two(self):
        with tempfile.TemporaryDirectory(prefix='process restart ') as folder:
            root = Path(folder)
            data = target_data(root / 'SUIM')
            output = root / 'output'
            script = root / 'worker.py'
            script.write_text('''import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); s=p/'counter.json'
history=json.loads(s.read_text()) if s.exists() else []
history.append(os.getpid()); s.write_text(json.dumps(history)); n=len(history)
state=dict(status='complete' if n==2 else 'session_paused',optimizer_updates=n)
(p/'RUN_STATE.json').write_text(json.dumps(state))
(p/('complete.json' if n==2 else 'session_pause.json')).write_text(json.dumps(state))
(p/'resume.pth').write_bytes(b'fixture')
''')
            with mock.patch('underwater_baselines.launch.command_for',
                return_value=([sys.executable, str(script), str(output)], os.environ.copy(), root)), \
                mock.patch('underwater_baselines.launch.subprocess.run'):
                launch('secov2', 'uiis_to_suim', data, output, dict(path='fixture', sha256='fixture'), None, True, 0)
            pids = json.loads((output / 'counter.json').read_text())
            self.assertEqual(len(set(pids)), 2)
            self.assertEqual(json.loads((output / 'RUN_STATE.json').read_text())['optimizer_updates'], 2)
            self.assertEqual(check_existing(output, json.loads((output / 'run_config.json').read_text())), 'complete')


if __name__ == '__main__':
    unittest.main()
