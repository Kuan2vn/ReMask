"""Portable launcher; the GPU-tested training and gradient code is unchanged."""
import argparse
import contextlib
import datetime
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .assets import SOURCES, sam_checkpoint, source_checkpoint

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
METHODS = ('stable_st', 'secov2')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def image_files(path):
    return sorted(p for p in Path(path).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def validate_data(root, smoke=False):
    root = Path(root).expanduser().resolve()
    if not (root / 'train_val/images').is_dir() and (root / 'orin_data/train_val/images').is_dir():
        root /= 'orin_data'
    for relative in ('train_val/images', 'TEST/images', 'TEST/masks'):
        if not (root / relative).is_dir():
            raise ValueError(f'Missing SUIM directory: {root / relative}')
    train, test = image_files(root / 'train_val/images'), image_files(root / 'TEST/images')
    if smoke:
        if len(train) < 8 or len(test) < 2:
            raise ValueError('Smoke test needs at least 8 training images and 2 evaluation images')
    elif (len(train), len(test)) != (1525, 110):
        raise ValueError(f'Expected SUIM1525/110 images, got {len(train)}/{len(test)}')
    if len({p.stem for p in train}) != len(train) or len({p.stem for p in test}) != len(test):
        raise ValueError('Image stems must be unique within each split')
    if any(p.suffix != '.jpg' for p in train):
        raise ValueError('Use the original SUIM .jpg training images; the native Stable-ST donor loader expects these filenames')
    for image in test[:2] if smoke else test:
        masks = [root / 'TEST/masks' / (image.stem + extension) for extension in ('.bmp', '.png')]
        if sum(p.is_file() for p in masks) != 1:
            raise ValueError(f'Expected exactly one evaluation mask for {image.name}')
    return root


def link_directory(target, link):
    target, link = Path(target).resolve(), Path(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f'Existing data view points to a different dataset: {link}')
    elif link.exists():
        raise ValueError(f'Expected a dataset symlink, found an existing directory: {link}')
    else:
        link.symlink_to(target, target_is_directory=True)


def data_view(data, output, smoke):
    view = output / '.runtime/target'
    if smoke:
        for split, count in (('train_val', 8), ('TEST', 2)):
            images = image_files(data / split / 'images')[:count]
            for image in images:
                destination = view / split / 'images' / image.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(image, destination)
                if split == 'TEST':
                    mask = next(data / 'TEST/masks' / (image.stem + extension) for extension in ('.bmp', '.png')
                                if (data / 'TEST/masks' / (image.stem + extension)).is_file())
                    destination = view / 'TEST/masks' / mask.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        shutil.copy2(mask, destination)
    else:
        for relative in ('train_val/images', 'TEST/images', 'TEST/masks'):
            link_directory(data / relative, view / relative)
    assert not (view / 'train_val/masks').exists(), 'Training GT must not enter the adaptation view'
    link_directory(view, output / '.runtime/data/SUIM/orin_data')
    return view


def configuration(method, protocol, data, record, smoke):
    checksum = hashlib.sha256()
    for folder in ('core', 'stable_st', 'underwater_baselines', 'references', 'configs'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml'):
                checksum.update(path.relative_to(ROOT).as_posix().encode() + b'\0')
                checksum.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return dict(method=method, protocol=protocol, data=str(data), source_sha256=record['sha256'],
                diagnostic=smoke, maximum_updates=2 if smoke else 30000,
                evaluation_period=1 if smoke else 500, batch_size=4, seed=1, code_sha256=checksum.hexdigest())


def check_existing(output, configuration):
    path = output / 'run_config.json'
    if not path.exists() and any((output / name).exists() for name in ('complete.json', 'resume.pth', 'RUN_STARTED.json')):
        raise ValueError('Output has no matching launcher configuration; choose a new --output')
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError('Output belongs to a different method/protocol/data/mode; choose another --output')
    if (output / 'complete.json').exists():
        return 'complete'
    if (output / 'resume.pth').exists() and (output / 'session_pause.json').exists():
        return 'resume'
    if (output / 'RUN_STARTED.json').exists():
        raise ValueError('Output is incomplete without a clean saved pause; preserve it and choose another --output')
    return 'new'


@contextlib.contextmanager
def run_lock(output):
    import fcntl
    with (output / '.run.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another trainer owns this output: {output}') from error
        yield


def command_for(method, protocol, record, view, output, manifest, smoke, seconds, sam=None):
    if method not in METHODS:
        raise ValueError(f'Unsupported benchmark method: {method}')
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    if method == 'stable_st':
        env.update(PYTHONPATH=os.pathsep.join([str(ROOT / 'stable_st'), str(ROOT)]),
                   UNDERWATER_SESSION_SECONDS=str(0 if smoke else seconds), UNDERWATER_DIAGNOSTIC=str(int(smoke)))
        command = [sys.executable, '-u', str(ROOT / 'stable_st/train_stable_st.py'), '--skip-test',
                   '-cfg', str(ROOT / 'configs/stable_st.yaml'), 'resume', record['path'], 'OUTPUT_DIR', str(output),
                   'DATASETS.SOURCE_TRAIN', 'uiis_train' if protocol == 'uiis_to_suim' else 'usis10k_bg_train']
        if smoke:
            command += ['SOLVER.STOP_ITER', '2', 'SOLVER.CHECKPOINT_PERIOD', '1', 'METAPL.TOP_K', '0.125']
        return command, env, output / '.runtime'
    env['PYTHONPATH'] = str(ROOT)
    command = [sys.executable, '-u', '-m', 'underwater_baselines.run', '--method', method, '--protocol', protocol,
               '--source-manifest', str(manifest), '--data', str(view), '--output', str(output),
               '--author-root', str(ROOT / 'references/SeCoV2'),
               '--steps', '2' if smoke else '30000', '--eval-period', '1' if smoke else '500',
               '--session-seconds', str(0 if smoke else seconds)]
    if smoke:
        command += ['--diagnostic']
    if sam:
        command += ['--sam-checkpoint', str(sam)]
    return command, env, ROOT


def launch(method, protocol, data, output, record, sam, smoke, seconds):
    output.mkdir(parents=True, exist_ok=True)
    config = configuration(method, protocol, data, record, smoke)
    with run_lock(output):
        state = check_existing(output, config)
        if state == 'complete':
            print(f'Already complete: {output}', flush=True)
            return
        view = data_view(data, output, smoke)
        (output / 'run_config.json').write_text(json.dumps(config, indent=2))
        manifest = output / 'sources.json'
        manifest.write_text(json.dumps(dict(sources={protocol: record}), indent=2))
        subprocess.run([sys.executable, '-c',
            'import torch; free,total=torch.cuda.mem_get_info(); '
            'print("CUDA free/total GiB:",free/1024**3,total/1024**3); '
            'assert free>=12*1024**3,"At least12GiB free GPU memory is required"'], check=True)
        (output / 'RUN_STARTED.json').write_text(json.dumps(dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())))
        command, env, cwd = command_for(method, protocol, record, view, output, manifest, smoke, seconds, sam)
        print(f'Running {method}: {protocol} -> {output}', flush=True)
        for _ in range(2 if smoke else 1):
            with (output / 'console.log').open('a', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                process.stdout.close()
                code = process.wait()
            if code:
                raise RuntimeError(f'Trainer exited with code {code}; see {output / "console.log"}')
            state = json.loads((output / 'RUN_STATE.json').read_text())
            if state['status'] != 'session_paused':
                break
            if smoke:
                print('Smoke: verifying process restart from update1', flush=True)
        if state['status'] not in ('complete', 'session_paused'):
            raise RuntimeError(f'Unexpected trainer state: {state}')
        print(f'{state["status"]}: {state["optimizer_updates"]} optimizer updates', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Stable-ST and SeCoV2 on UIIS/USIS10K -> SUIM')
    parser.add_argument('--method', choices=(*METHODS, 'all'), required=True)
    parser.add_argument('--protocol', choices=(*SOURCES, 'all'), required=True)
    parser.add_argument('--data', type=Path, default=REPO / 'data/SUIM')
    parser.add_argument('--assets', type=Path, default=REPO / 'assets/baselines')
    parser.add_argument('--output', type=Path, help='Output root (method/protocol folders are added)')
    parser.add_argument('--source-checkpoint', type=Path, help='Existing source file; exact protocol SHA256 is required')
    parser.add_argument('--sam-checkpoint', type=Path, help='Existing official SAM ViT-B checkpoint')
    parser.add_argument('--check', action='store_true', help='Validate data, download/verify weights and perform a CPU source forward')
    parser.add_argument('--smoke', action='store_true', help='8 train/2 test images,2 updates with process restart; not a benchmark')
    parser.add_argument('--session-hours', type=float, default=0, help='Optional checkpoint-safe session pause;0 runs to the fixed30k budget')
    args = parser.parse_args()
    if args.check and args.smoke:
        parser.error('Choose --check or --smoke')
    if args.session_hours < 0 or not math.isfinite(args.session_hours):
        parser.error('--session-hours must be finite and nonnegative')
    if args.source_checkpoint and args.protocol == 'all':
        parser.error('--source-checkpoint requires a single protocol')
    try:
        data = validate_data(args.data, args.smoke)
        import torch
        if not args.check:
            if sys.platform != 'linux':
                raise RuntimeError('GPU training requires Linux; --check also supports CPU setup validation')
            if int(os.environ.get('WORLD_SIZE', '1')) != 1:
                raise RuntimeError('These recipes use one GPU per process; launch without a distributed WORLD_SIZE')
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable; install a driver-compatible PyTorch build or use --check for CPU validation')
        methods = METHODS if args.method == 'all' else (args.method,)
        protocols = tuple(SOURCES) if args.protocol == 'all' else (args.protocol,)
        dependencies = ['higher', 'yacs', 'pandas', 'cv2', 'skimage', 'imageio', 'matplotlib']
        if 'secov2' in methods:
            dependencies += ['segment_anything', 'pyclipper', 'shapely', 'pycocotools.mask']
        for dependency in dependencies:
            importlib.import_module(dependency)
        output_root = (args.output or REPO / 'outputs/baselines' / ('smoke' if args.smoke else 'full')).resolve()
        sam = sam_checkpoint(args.assets, args.sam_checkpoint) if 'secov2' in methods else None
        records = {protocol: source_checkpoint(protocol, args.assets, args.source_checkpoint) for protocol in protocols}
        if args.check:
            from .common import Segmenter
            torch.set_num_threads(4)
            for protocol, record in records.items():
                model = Segmenter(record['path']).eval()
                with torch.no_grad():
                    prediction = model(torch.zeros(1, 3, 32, 32))
                if prediction.shape != (1, 8, 32, 32) or not torch.isfinite(prediction).all():
                    raise RuntimeError(f'Invalid source forward: {protocol}')
                del model
                print(f'CHECK PASSED: {protocol}; source SHA256 and 8-class CPU forward verified', flush=True)
            print(f'Data verified: {data}; CUDA available: {torch.cuda.is_available()}', flush=True)
            return
        seconds = args.session_hours * 3600 if args.session_hours else float('inf')
        for method in methods:
            for protocol in protocols:
                launch(method, protocol, data, output_root / method / protocol, records[protocol], sam, args.smoke, seconds)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')
