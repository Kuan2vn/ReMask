#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from remask.config import load_config
from remask.engine import train_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the labeled ReMask source model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Optional source checkpoint.")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Override one YAML value, e.g. --set source_train.iterations=10",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    print(train_source(config, resume=args.resume, output_dir=args.output))


if __name__ == "__main__":
    main()
