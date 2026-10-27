"""Run the original worker with a post-commit plotting hook, outside native code."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo/'experiments/rebuttal'))
    from control import Control, SessionPaused
    import control
    from local_bank_io import install_worker
    from performance import install
    import worker
    install(Control, SessionPaused)
    install_worker(Control, control)
    sys.argv = [str(args.repo/'experiments/rebuttal/worker.py'), '--task', args.task]
    worker.main()


if __name__ == '__main__':
    main()
