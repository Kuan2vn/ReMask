"""Run the underwater SFUDA baselines from a repository checkout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'underwater'))
from underwater_baselines.launch import main

if __name__ == '__main__':
    main()
