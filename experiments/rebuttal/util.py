import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import uuid

WARM_BANKS = {
    'uiis_to_suim': {
        'archive': 'assets/warm_banks/uiis_to_suim_dtst_warm.tar.gz',
        'archive_sha256': 'd22b24b7c5b677579b44f62dc050d963f968bfd8d547f26423d58c3874674f52',
        'banks_sha256': '363d8d0da2bf3367fb08feccd3384b18b1c4b564671f5dd9fb6445d365a66cdd',
    },
    'usis10k_to_suim': {
        'archive': 'assets/warm_banks/usis10k_to_suim_dtst_warm.tar.gz',
        'archive_sha256': '314bc520915c2ec4198b328929027c4392bd313ab40676abdd5b7343302d0f1b',
        'banks_sha256': '4f7aa399b635a04bd17dbc883cbff64515b340cfc76022ddf2ea4760e5af78ca',
    },
}


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


def warm_bank_root(protocol, repo=None):
    if protocol not in WARM_BANKS:
        raise ValueError(f'Unknown warm bank protocol: {protocol}')
    spec = WARM_BANKS[protocol]
    here = Path(repo)/'experiments/rebuttal' if repo else Path(__file__).resolve().parent
    archive = (here/spec['archive']).resolve()
    if not archive.is_file() or digest(archive) != spec['archive_sha256']:
        raise ValueError(f'Warm candidate bank archive missing or corrupted: {archive}')
    cache = Path(tempfile.gettempdir()).resolve()/'remask-warm-banks-v1'/protocol/spec['banks_sha256']
    marker = cache/'WARM_BANKS_VERIFIED.json'
    if marker.is_file() and all((cache/name).exists() for name in ('CTR', 'CTR_O', 'CTR.p', 'CTR_O.p')):
        return cache
    staging = cache.parent/f'.partial-{uuid.uuid4().hex}'
    staging.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar.getmembers():
                rel = Path(member.name)
                if (rel.is_absolute() or '..' in rel.parts or member.issym() or member.islnk()
                        or rel.parts[0] not in ('CTR', 'CTR_O', 'CTR.p', 'CTR_O.p')):
                    raise ValueError(f'Unsafe warm bank member: {member.name}')
            tar.extractall(staging, filter='data')
        if bank_hash(staging) != spec['banks_sha256']:
            raise ValueError(f'Unpacked warm candidate bank hash mismatch: {protocol}')
        atomic_json(staging/'WARM_BANKS_VERIFIED.json', dict(protocol=protocol, **spec))
        if cache.exists():
            shutil.rmtree(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(cache)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return cache


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
