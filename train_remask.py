#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from remask.config import load_config
from remask.engine import train_remask


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage-1/Stage-2 ReMask adaptation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--stage", choices=("all", "stage1", "stage2"), default="all")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Override one YAML value, e.g. --set adapt.stage1_iterations=10",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    print(
        train_remask(
            config,
            source_checkpoint=args.source_checkpoint,
            stage=args.stage,
            output_dir=args.output,
        )
    )


if __name__ == "__main__":
    main()
