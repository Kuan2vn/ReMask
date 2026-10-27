from pathlib import Path
import shutil
import uuid

def import_anchor(source, destination, step, protocol, seed, diagnostic=False):
    from util import atomic_json, bank_hash, digest, read_json, verified_pointer
    name = f'snapshots/{step:06d}.json'
    pointer, checkpoint, banks = verified_pointer(source, name)
    signature = pointer['signature']
    if (pointer['iteration'] != step or signature['protocol'] != protocol
            or signature['phase']['seed'] != seed or signature['phase']['phase'] != 'A'
            or signature['arm'] != source.parent.name
            or signature['diagnostic'] != diagnostic):
        raise ValueError('Parent anchor has the wrong iteration/protocol/seed/mode')
    destination.mkdir(parents=True, exist_ok=True)
    for key in ('checkpoint', 'banks'):
        relative = Path(pointer[key])
        if relative.is_absolute() or not (destination / relative).resolve().is_relative_to(destination.resolve()):
            raise ValueError('Imported pointer must stay within its destination phase')
    target_checkpoint = destination / pointer['checkpoint']
    target_banks = destination / pointer['banks']
    staging = destination / ('.import-' + uuid.uuid4().hex)
    staging.mkdir()
    try:
        if target_checkpoint.exists():
            if digest(target_checkpoint) != pointer['sha256']:
                raise ValueError('Existing imported checkpoint has a different hash')
        else:
            candidate = staging / 'checkpoint.pth'
            shutil.copy2(checkpoint, candidate)
            if digest(candidate) != pointer['sha256']:
                raise ValueError('Copied parent checkpoint failed verification')
            target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            candidate.replace(target_checkpoint)
        if target_banks.exists():
            if bank_hash(target_banks) != pointer['banks_sha256']:
                raise ValueError('Existing imported banks have a different hash')
        else:
            candidate = staging / 'banks'
            shutil.copytree(banks, candidate)
            if bank_hash(candidate) != pointer['banks_sha256']:
                raise ValueError('Copied parent banks failed verification')
            target_banks.parent.mkdir(parents=True, exist_ok=True)
            candidate.replace(target_banks)
        (destination / 'snapshots').mkdir(exist_ok=True)
        atomic_json(destination / name, pointer)
        atomic_json(destination / f'IMPORTED_PREFIX_{step:06d}.json', dict(
            source_phase=str(source), source_snapshot=name, pointer=pointer,
            scope='Reused prefix; include its resource cost once in pipeline comparisons. '
                  'This import is not a locally trained or completed phase.'))
        verified_pointer(destination, name)
    finally:
        shutil.rmtree(staging)
