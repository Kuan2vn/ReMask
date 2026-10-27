import hashlib
import json
from pathlib import Path
import pickle
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
        'ctr_p_sha256': 'fa4a0bb84cdb28db826a74360a95c6e4d6b5946316992c9ddf1658ee8e774a3a',
        'ctr_o_p_sha256': '997a77ee316ddd971626ed54758afab2a35050113479d6f1c2081f8b02bcd1f6',
        'source_sha256': '7f9e88a4029f4797f97ade53f25cfef7c31ae528f52c9459b7c40b08cc67a747',
        'dataset_bundle_sha256': '824a87636625ebf170b38225a8605f111d15f7c8d3b4c774bd95ca8f75b5ce77',
        'generator': 'train_DTST.py (configs/deeplabv2_r101_dtst_SUIM.yaml; TCR.TOPK_CANDIDATE=0.5; TCR.UPDATE_FREQUENCY=6000; METAPL.OPEN=False)',
        'generator_steps': 30000,
        'historical_origin': 'results_uiis_to_suim/dtst_warm_uiis_to_suim_8c_src18k',
    },
    'usis10k_to_suim': {
        'archive': 'assets/warm_banks/usis10k_to_suim_dtst_warm.tar.gz',
        'archive_sha256': '314bc520915c2ec4198b328929027c4392bd313ab40676abdd5b7343302d0f1b',
        'banks_sha256': '4f7aa399b635a04bd17dbc883cbff64515b340cfc76022ddf2ea4760e5af78ca',
        'ctr_p_sha256': 'b47df277f65b65e186551a1c36c64e095d19a09ee8c0585d6cbf5b2b1b01db5e',
        'ctr_o_p_sha256': 'ce77427d39faf269b568057c06095b7276c56e923655f315bbea7bd8efbeb729',
        'source_sha256': 'a4a5f4534a98cca90b1141d4746384bb942e7602f6a06e3fb629d07644654b12',
        'dataset_bundle_sha256': '824a87636625ebf170b38225a8605f111d15f7c8d3b4c774bd95ca8f75b5ce77',
        'generator': 'train_DTST.py (configs/deeplabv2_r101_dtst_SUIM.yaml; TCR.TOPK_CANDIDATE=0.5; TCR.UPDATE_FREQUENCY=6000; METAPL.OPEN=False)',
        'generator_steps': 20000,
        'historical_origin': 'results_usis10k_bg_to_suim_8c/usis_bg_src_best/dtst_warm_usis10k_bg_to_suim_8c',
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
        try:
            if bank_hash(cache) == spec['banks_sha256']:
                return cache
        except Exception:
            pass
        shutil.rmtree(cache, ignore_errors=True)
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


def seed_warm_banks(protocol, destination, source_sha256=None, view=None, smoke=False, repo=None, copy_fn=copy_banks):
    spec = WARM_BANKS.get(protocol)
    if spec is None:
        raise ValueError(f'Unknown warm bank protocol: {protocol}')
    if source_sha256 is not None and source_sha256 != spec['source_sha256']:
        raise ValueError(f'Warm bank source SHA mismatch for {protocol}: expected {spec["source_sha256"]}, got {source_sha256}')
    root = warm_bank_root(protocol, repo=repo)
    if not smoke:
        copy_fn(root, destination)
        return root
    if view is None:
        raise ValueError('Smoke warm-bank seeding requires the active dataset view')
    keep = {path.stem + '.png' for path in sorted((Path(view)/'train_val/images').glob('*.jpg'))}
    if not keep:
        raise ValueError(f'No training images found in smoke view: {view}')
    with tempfile.TemporaryDirectory(prefix='remask-smoke-warm-') as tmp:
        staged = Path(tmp)
        for folder, index_name in (('CTR', 'CTR.p'), ('CTR_O', 'CTR_O.p')):
            (staged/folder).mkdir(parents=True, exist_ok=True)
            for name in sorted(keep):
                src = root/folder/name
                if not src.is_file():
                    raise ValueError(f'Warm bank {protocol} missing smoke mask: {folder}/{name}')
                shutil.copy2(src, staged/folder/name)
            label_to_file, _ = pickle.loads((root/index_name).read_bytes())
            filtered_l2f = {int(c): [f for f in label_to_file.get(c, []) if f in keep] for c in range(8)}
            filtered_f2l = {f: [c for c in range(8) if f in filtered_l2f[c]] for f in sorted(keep)}
            with (staged/index_name).open('wb') as stream:
                pickle.dump((filtered_l2f, filtered_f2l), stream)
        copy_fn(staged, destination)
    return root


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
