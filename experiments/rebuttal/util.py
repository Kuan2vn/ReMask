import hashlib
import json
import os
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

PAPER_STAGE_A_CHECKPOINTS = {
    'uiis_to_suim': {
        'filename': 'remask_model_iter024999.pth',
        'sha256': 'd52a8b11055de408e8adabc5870f3ae25a89fc3e99007c872d4abaeafc0c6c3e',
        'step': 25000,
        'gdrive_id': '1as2OJ5h8GN2zbjdMl6QSn7aY39RQ7E6p',
        'historical_origin': 'results_sensitivity/remask_special_uwrg_sensitivity_24283/uiis_to_suim/strength020/remask_model_iter024999.pth',
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


def paper_stage_a_checkpoint(protocol, repo=None):
    spec = PAPER_STAGE_A_CHECKPOINTS.get(protocol)
    if spec is None:
        return None
    repo_root = Path(repo) if repo else Path(__file__).resolve().parents[2]
    env_ckpt = os.environ.get('REMASK_PAPER_STAGE_A_CKPT')
    candidates = ([Path(env_ckpt)] if env_ckpt else []) + [
        repo_root/spec['filename'],
        Path('/content/ReMask')/spec['filename'],
        Path('/content/drive/MyDrive/@Remask')/spec['filename'],
    ]
    for path in candidates:
        if path.is_file():
            if digest(path) != spec['sha256']:
                raise ValueError(f'Paper Stage-A checkpoint hash mismatch: {path}')
            return path.resolve()
    return None


def ensure_paper_stage_a_anchor(protocol, phase_out, signature, repo=None, mark_complete=False):
    spec = PAPER_STAGE_A_CHECKPOINTS.get(protocol)
    if spec is None or signature.get('diagnostic'):
        return False
    ckpt_src = paper_stage_a_checkpoint(protocol, repo=repo)
    if ckpt_src is None:
        return False
    phase_out = Path(phase_out).resolve()
    phase_out.mkdir(parents=True, exist_ok=True)
    config_path = phase_out/'PHASE_CONFIG.json'
    if config_path.exists() and read_json(config_path) != signature:
        return False
    atomic_json(config_path, signature)
    target_ckpt = phase_out/spec['filename']
    if not target_ckpt.is_file() or digest(target_ckpt) != spec['sha256']:
        tmp_ckpt = phase_out/(spec['filename'] + '.partial-' + uuid.uuid4().hex)
        shutil.copy2(ckpt_src, tmp_ckpt)
        if digest(tmp_ckpt) != spec['sha256']:
            tmp_ckpt.unlink(missing_ok=True)
            raise ValueError(f'Copied paper Stage-A checkpoint failed hash check: {target_ckpt}')
        tmp_ckpt.replace(target_ckpt)
    warm_root = warm_bank_root(protocol, repo=repo)
    banks_sha = WARM_BANKS[protocol]['banks_sha256']
    try:
        from local_bank_io import cache_root
        verified_cache = cache_root()/'verified'/banks_sha
        if not verified_cache.exists():
            staging_cache = verified_cache.with_name(banks_sha + '.partial-' + uuid.uuid4().hex)
            shutil.copytree(warm_root, staging_cache, dirs_exist_ok=True)
            verified_cache.parent.mkdir(parents=True, exist_ok=True)
            staging_cache.replace(verified_cache)
    except Exception:
        pass
    banks_dir = phase_out/'banks'/'000000'
    if not (banks_dir/'BANKS.json').is_file() or read_json(banks_dir/'BANKS.json').get('sha256') != banks_sha:
        staging_banks = banks_dir.with_name('000000.partial-' + uuid.uuid4().hex)
        staging_banks.mkdir(parents=True, exist_ok=True)
        copy_banks(warm_root, staging_banks)
        atomic_json(staging_banks/'BANKS.json', dict(sha256=banks_sha))
        banks_dir.parent.mkdir(parents=True, exist_ok=True)
        if banks_dir.exists():
            shutil.rmtree(banks_dir)
        staging_banks.replace(banks_dir)
    pointer = dict(
        iteration=spec['step'],
        checkpoint=target_ckpt.name,
        sha256=spec['sha256'],
        banks='banks/000000',
        banks_sha256=banks_sha,
        signature=signature,
        historical_origin=spec['historical_origin'],
        resources=dict(wall_seconds=0., training_seconds=0., evaluation_seconds=0.,
                       candidate_seconds=0., checkpoint_seconds=0., incremental_optimizer_updates=0,
                       segments=[], imported_paper_stage_a=True),
    )
    snapshots = phase_out/'snapshots'
    snapshots.mkdir(exist_ok=True)
    atomic_json(snapshots/f"{spec['step']:06d}.json", pointer)
    if mark_complete and not (phase_out/'CURRENT.json').exists():
        atomic_json(phase_out/'CURRENT.json', pointer)
        atomic_json(phase_out/'RUN_STATE.json', dict(
            status='complete', local_updates=spec['step'], global_updates=spec['step'],
            checkpoint_verified=True, full_state=True, stopping_rule='fixed_budget',
            diagnostic=False, imported_paper_stage_a=True))
        atomic_json(phase_out/'complete.json', dict(
            local_updates=spec['step'], signature=signature,
            imported_paper_stage_a=True, historical_origin=spec['historical_origin']))
        print(f'[PAPER STAGE-A ANCHOR] Seeded {protocol}/F/A@{spec["step"]} from {ckpt_src}', flush=True)
    return True


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
