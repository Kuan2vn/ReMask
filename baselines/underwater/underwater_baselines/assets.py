"""Public, revision-pinned source weights and official SAM download."""
import hashlib
import tempfile
import urllib.request
from pathlib import Path

SOURCE_REPO = 'kuan2/remask-checkpoints'
SOURCE_REVISION = 'b0a4659412c9fb959a04bfe0619e0462bf40b943'
SOURCES = {
    'uiis_to_suim': dict(filename='uiis_to_suim/source_iter_18000.pth', iteration=18000,
        sha256='7f9e88a4029f4797f97ade53f25cfef7c31ae528f52c9459b7c40b08cc67a747'),
    'usis10k_to_suim': dict(filename='usis10k_to_suim/source_iter_29000.pth', iteration=29000,
        sha256='a4a5f4534a98cca90b1141d4746384bb942e7602f6a06e3fb629d07644654b12'),
}
SAM_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
SAM_SHA256 = 'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def verify(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f'Checkpoint missing or SHA256 mismatch: {path}')
    return path


def source_checkpoint(protocol, assets, override=None):
    record = SOURCES[protocol]
    if override:
        path = override
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(SOURCE_REPO, record['filename'], revision=SOURCE_REVISION,
                               cache_dir=str(Path(assets) / 'hub'), token=False)
    path = verify(path, record['sha256'])
    return dict(record, path=str(path), verified_source_only=True,
                repository=SOURCE_REPO, revision=SOURCE_REVISION)


def sam_checkpoint(assets, override=None):
    if override:
        return verify(override, SAM_SHA256)
    folder = Path(assets).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'sam_vit_b_01ec64.pth'
    if target.exists():
        return verify(target, SAM_SHA256)
    with tempfile.NamedTemporaryFile(dir=folder, prefix='sam-', suffix='.part', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(SAM_URL, timeout=120) as response:
                while block := response.read(8 * 1024**2):
                    stream.write(block)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        verify(temporary, SAM_SHA256)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
