"""Verify the shared dataset ZIP and extract its data/ tree into a repository."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

HERE = Path(__file__).resolve().parent


def safe_members(archive, prefix='data'):
    members = [m for m in archive.infolist() if not m.is_dir()]
    names = set()
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
                or not path.parts or path.parts[0] != prefix
                or (member.external_attr >> 16) & 0o170000 == 0o120000 or name in names):
            raise ValueError(f'Unsafe or duplicate archive member: {name}')
        names.add(name)
    return members


def extract(archive, destination, members):
    destination = Path(destination).resolve()
    for member in members:
        target = destination.joinpath(*PurePosixPath(member.filename).parts)
        # Also reject pre-existing symlinks that could escape the destination.
        if not target.resolve().is_relative_to(destination):
            raise ValueError(f'Extraction escapes destination: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open('wb') as output:
            shutil.copyfileobj(source, output)  # CRC checked by ZipExtFile at EOF.


def check_suim(root):
    root = Path(root)
    for split, count in [('train_val', 1525), ('TEST', 110)]:
        images = list((root / split / 'images').glob('*.jpg'))
        masks = [p for p in (root / split / 'masks').glob('*') if p.suffix.lower() in ('.bmp', '.png')]
        if (len(images) != count or len(masks) != count
                or {p.stem for p in images} != {p.stem for p in masks}):
            raise ValueError(f'Expected {count} paired images/masks in {root / split}')
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=HERE.parents[1])
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    contract = json.loads((HERE / 'dataset_bundle.json').read_text())
    checksum = hashlib.sha256()
    with args.archive.open('rb') as source:
        for block in iter(lambda: source.read(8*1024**2), b''):
            checksum.update(block)
    sha = checksum.hexdigest()
    if sha != contract['sha256'] or args.archive.stat().st_size != contract['bytes']:
        raise ValueError('Dataset ZIP differs from the audited remask_datasets.zip')
    with zipfile.ZipFile(args.archive) as archive:
        members = safe_members(archive)
        counts = dict(Counter(str(PurePosixPath(m.filename).parent) for m in members))
        if (counts != contract['counts'] or len(members) != contract['file_count']
                or sum(m.file_size for m in members) != contract['uncompressed_bytes']):
            raise ValueError('Archive layout differs from audited counts')
        if not args.verify_only:
            receipt = args.repo / 'data/.remask_bundle.json'
            previous = json.loads(receipt.read_text()) if receipt.exists() else {}
            if previous.get('sha256') != sha:
                extract(archive, args.repo, members)
                receipt.write_text(json.dumps(dict(sha256=sha, file_count=len(members))) + '\n')
            check_suim(args.repo / 'data/SUIM')
    print(f'Verified {len(members)} files; ZIP SHA256 {sha}')


if __name__ == '__main__':
    main()
