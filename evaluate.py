#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from remask.checkpoint import load_model_weights
from remask.config import load_config
from remask.engine import evaluate
from remask.modeling import build_model
from remask.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a ReMask checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--student", action="store_true", help="Evaluate student instead of EMA teacher.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    device = resolve_device(str(config.get("device", "cuda")))
    model = build_model(config).to(device)
    load_model_weights(model, args.checkpoint, prefer_teacher=not args.student)
    print(json.dumps(evaluate(model, config, device), indent=2))


if __name__ == "__main__":
    main()
