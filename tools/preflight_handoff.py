"""CPU-only checkpoint integrity/loading check; does not start adaptation."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path, help="Downloaded remask-supervisor-handoff directory")
    args = parser.parse_args()
    manifest = json.loads((args.assets / "MANIFEST.json").read_text(encoding="utf-8"))
    if not manifest.get("status", "").startswith(("uploaded_assets_verified", "registered_assets_uploaded")):
        raise RuntimeError("Upload has no verified completion manifest")
    for item in manifest["assets"]:
        path = (args.assets / item["path"]).resolve()
        path.relative_to(args.assets.resolve())
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError("Missing or truncated asset: " + item["path"])
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise RuntimeError("SHA256 mismatch: " + item["path"])
    import torch
    from remask.checkpoint import load_model_weights
    from remask.modeling import build_model
    for name, classes, iteration in [("source_uiis_18000", 8, 18000),
                                     ("source_usis10k_29000", 8, 29000),
                                     ("source_gta5_20000", 19, 20000)]:
        model = build_model({"model": {"backbone": "resnet101", "num_classes": classes,
                                      "freeze_bn": True, "imagenet_weights": None}})
        payload = load_model_weights(model, str(args.assets / "checkpoints" / (name + ".pth")))
        if int(payload.get("iteration", -1)) != iteration:
            raise RuntimeError("Unexpected source iteration: " + name)
        model.eval()
        with torch.no_grad():
            result = model(torch.zeros(1, 3, 64, 64))
        if tuple(result.shape) != (1, classes, 64, 64) or not torch.isfinite(result).all():
            raise RuntimeError("Invalid model forward: " + name)
        print("PASS", name, "strict loading + CPU forward", flush=True)
        del model, payload, result
        gc.collect()
    print("Unresolved manuscript roles:", manifest.get("missing_checkpoint_roles", []))
    print("CPU preflight passed; this is not a GPU or training-protocol verification.")


if __name__ == "__main__":
    main()
