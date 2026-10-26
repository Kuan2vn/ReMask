import hashlib
import json
from pathlib import Path
import shutil
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tree_hash(root):
    root = Path(root)
    value = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file() and (path.suffix in ('.py', '.yaml') or path.name=='SNAPSHOT.json') and '__pycache__' not in path.parts:
            value.update(path.relative_to(root).as_posix().encode())
            value.update(path.read_bytes().replace(b'\r\n', b'\n'))
    requirements=root.parents[1]/'requirements.txt'
    if requirements.exists():value.update(requirements.read_bytes().replace(b'\r\n',b'\n'))
    return value.hexdigest()


def copy_banks(source, destination):
    for name in ('CTR', 'CTR_O', 'CTR.p', 'CTR_O.p'):
        src, dst = Path(source)/name, Path(destination)/name
        if not src.exists():
            raise ValueError(f'Missing pseudo-label bank: {src}')
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def bank_hash(root):
    value = hashlib.sha256()
    for name in ('CTR', 'CTR_O', 'CTR.p', 'CTR_O.p'):
        p = Path(root)/name
        files = sorted(p.rglob('*')) if p.is_dir() else [p]
        for file in files:
            if file.is_file():
                value.update(file.relative_to(root).as_posix().encode())
                value.update(bytes.fromhex(digest(file)))
    return value.hexdigest()


def verified_pointer(phase, name='CURRENT.json'):
    phase = Path(phase).resolve()
    pointer = read_json(phase/name)
    path = (phase/pointer['checkpoint']).resolve()
    banks = (phase/pointer['banks']).resolve()
    if not path.is_relative_to(phase) or not banks.is_relative_to(phase):
        raise ValueError('Checkpoint pointer leaves its phase directory')
    if digest(path) != pointer['sha256'] or bank_hash(banks) != pointer['banks_sha256']:
        raise ValueError(f'Checkpoint or immutable pseudo-label banks failed verification: {phase}')
    return pointer, path, banks


def sync(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def timed(device, function, *args, **kwargs):
    sync(device)
    start = time.perf_counter()
    value = function(*args, **kwargs)
    sync(device)
    return value, time.perf_counter()-start
