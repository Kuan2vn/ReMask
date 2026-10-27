"""Local working banks with the original durable checkpoint/bank-directory format."""
import functools
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

NAMES = ('CTR', 'CTR_O', 'CTR.p', 'CTR_O.p')


def cache_root():
    return Path(os.environ.get('REMASK_BANK_CACHE', str(Path(tempfile.gettempdir())/'remask-local-banks-v1'))).resolve()


def mutable_root(phase):
    key = hashlib.sha256(str(Path(phase).resolve()).encode()).hexdigest()[:24]
    return cache_root()/'working'/key


def remove_local(path):
    path = Path(path).resolve()
    if not path.is_relative_to(cache_root()) or path == cache_root():
        raise ValueError('Refusing to clear a path outside the local bank cache')
    if path.exists():
        shutil.rmtree(path)


def files(root):
    root = Path(root)
    result = []
    for name in NAMES:
        entry = root/name
        if entry.is_symlink() or not entry.exists():
            raise ValueError(f'Missing or linked bank component: {entry}')
        if name in ('CTR', 'CTR_O') and not entry.is_dir():
            raise ValueError(f'Expected bank directory: {entry}')
        if name.endswith('.p') and not entry.is_file():
            raise ValueError(f'Expected bank statistics file: {entry}')
        for path in sorted(entry.rglob('*')) if entry.is_dir() else [entry]:
            if path.is_symlink():
                raise ValueError(f'Linked bank entry: {path}')
            if path.is_file():
                result.append(path.relative_to(root))
    return result


def copy_progress(source, destination):
    source, destination = Path(source), Path(destination)
    inventory = files(source)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('CTR', 'CTR_O'):
        (destination/name).mkdir(exist_ok=True)
    started = last = time.monotonic()
    tmp_root = Path(tempfile.gettempdir()).resolve()
    if source.resolve().is_relative_to(tmp_root) and destination.resolve().is_relative_to(tmp_root):
        print(f'[BANK COPY] {source} -> {destination}; files={len(inventory)} (fast local SSD)', flush=True)
        for name in NAMES:
            src_entry, dst_entry = source/name, destination/name
            if src_entry.is_dir():
                shutil.copytree(src_entry, dst_entry, dirs_exist_ok=True)
            else:
                shutil.copyfile(src_entry, dst_entry)
        if files(destination) != inventory:
            raise ValueError(f'Bank destination contains stale/extra files: {destination}; preserved for inspection')
        print(f'[BANK COPY] {len(inventory)}/{len(inventory)}; elapsed={time.monotonic()-started:.2f}s', flush=True)
        return
    print(f'[BANK COPY] {source} -> {destination}; files={len(inventory)}', flush=True)
    for index, relative in enumerate(inventory, 1):
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if index == 1 or (index-1) % 100 == 0 or time.monotonic()-last >= 10:
            print(f'[BANK FILE] {index}/{len(inventory)}; copying={relative}', flush=True)
        # Copy bytes only: Drive metadata updates are unnecessary for pseudo labels.
        shutil.copyfile(source/relative, target)
        now = time.monotonic()
        if index == len(inventory) or index % 100 == 0 or now-last >= 10:
            print(f'[BANK COPY] {index}/{len(inventory)}; last={relative}; elapsed={now-started:.1f}s', flush=True)
            last = now
    if files(destination) != inventory:
        raise ValueError(f'Bank destination contains stale/extra files: {destination}; preserved for inspection')


def verified_pointer(phase, name='CURRENT.json'):
    from util import read_json, digest, bank_hash
    phase = Path(phase).resolve()
    pointer = read_json(phase/name)
    checkpoint = (phase/pointer['checkpoint']).resolve()
    banks = (phase/pointer['banks']).resolve()
    if not checkpoint.is_relative_to(phase) or not banks.is_relative_to(phase):
        raise ValueError('Checkpoint pointer leaves its phase directory')
    expected = pointer['banks_sha256']
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('Invalid recorded bank hash')
    if digest(checkpoint) != pointer['sha256']:
        raise ValueError('Checkpoint hash mismatch; durable state was not modified')
    cached = cache_root()/'verified'/expected
    if cached.exists():
        if bank_hash(cached) != expected:
            raise ValueError('Local cached bank hash mismatch; cache preserved for inspection')
        print(f'[BANK CACHE] Verified local hit: {cached}', flush=True)
    else:
        staging = cached.with_name(expected+'.partial-'+uuid.uuid4().hex)
        try:
            copy_progress(banks, staging)
            if bank_hash(staging) != expected:
                raise ValueError('Copied bank hash mismatch; durable state was not modified')
            cached.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(cached)
        finally:
            remove_local(staging)
    return pointer, checkpoint, cached


def prepare_phase(phase, fresh=False):
    root = mutable_root(phase)
    if fresh:
        remove_local(root)
    root.mkdir(parents=True, exist_ok=True)
    print(f'[WORKING BANKS] {root}; durable phase={phase}', flush=True)
    return root


def restore_working(source, phase):
    from util import bank_hash
    root = mutable_root(phase)
    staging = root.with_name(root.name+'.partial-'+uuid.uuid4().hex)
    try:
        copy_progress(source, staging)
        if bank_hash(staging) != bank_hash(source):
            raise ValueError('Local bank restore failed verification')
        remove_local(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(root)
    finally:
        remove_local(staging)
    return root


def install_coordinator(run):
    run.verified_pointer = verified_pointer
    run.copy_banks = restore_working


def install_worker(control_class, control_module):
    original_init = control_class.__init__
    original_hash = control_module.bank_hash
    expected_snapshots = {}

    def initialize(self, task, trainer):
        original_init(self, task, trainer)
        working = mutable_root(self.out)
        working.mkdir(parents=True, exist_ok=True)
        if not all((working/n).exists() for n in NAMES):
            raise ValueError('Local working banks are missing; resume through the coordinator')

        def bank_config(cfg):
            clone = cfg.clone()
            clone.defrost()
            clone.OUTPUT_DIR = str(working)
            clone.freeze()
            return clone
        for name in ('_ensure_remask_candidates', 'run_candidate', 'gen_lb_info', '_build_target_loader'):
            original = getattr(trainer, name)

            def redirect(cfg, *args, _original=original, **kwargs):
                return _original(bank_config(cfg), *args, **kwargs)
            setattr(trainer, name, functools.wraps(original)(redirect))

        def snapshot(source, destination):
            if Path(source).resolve() != self.out.resolve() or not Path(destination).resolve().is_relative_to((self.out/'banks').resolve()):
                raise ValueError('Unexpected bank snapshot path')
            expected_snapshots[Path(destination).resolve()] = (original_hash(working), working)
            copy_progress(working, destination)
        control_module.copy_banks = snapshot
        print(f'[BANK IO] Candidate generation/loading uses {working}; checkpoints stay at {self.out}', flush=True)

    def checked_hash(root):
        result = original_hash(root)
        recorded = expected_snapshots.get(Path(root).resolve())
        if recorded is not None:
            expected, working_dir = recorded
            if result != expected:
                raise ValueError('Durable snapshot differs from local working banks; refusing commit')
            cached = cache_root()/'verified'/result
            if not cached.exists():
                staging = cached.with_name(result+'.partial-'+uuid.uuid4().hex)
                try:
                    copy_progress(working_dir, staging)
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    staging.replace(cached)
                finally:
                    remove_local(staging)
        return result
    control_class.__init__ = initialize
    control_module.bank_hash = checked_hash
