"""Small standard-library helpers for manual clone/extract and Drive code pins."""
from pathlib import Path
import re
import subprocess


def find_repo(cwd, hint=None, content='/content'):
    candidates = ([Path(hint)] if hint else []) + [Path(cwd), Path(cwd)/'ReMask',
        Path(content)/'ReMask', Path(content)/'SND_ReMask/ReMask', Path(content)/'SND_Remask/ReMask']
    for root in candidates:
        if (root/'experiments/rebuttal/run.py').is_file() and (root/'.git').exists():
            return root.resolve()
    if hint:
        raise ValueError(f'REPO_HINT does not point to a ReMask checkout: {hint}')
    return None


def pin_revision(repo, output):
    repo, output = Path(repo), Path(output)
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    if git('status','--porcelain','--untracked-files=no'):
        raise ValueError('Tracked repository files have local edits; preserve them before switching code')
    pin=output/'CODE_REVISION.txt'
    # Accept pins written by the previous notebook with a literal backslash-n.
    revision=pin.read_text().replace('\\n','').strip() if pin.exists() else None
    if revision is not None and not re.fullmatch('[0-9a-f]{40}',revision):
        raise ValueError('Invalid CODE_REVISION.txt; preserve the output for inspection')
    git('fetch','origin')
    git('checkout','--detach',revision or 'origin/main')
    commit=git('rev-parse','HEAD')
    output.mkdir(parents=True,exist_ok=True)
    pin.write_text(commit+'\n',encoding='utf-8')
    return commit
