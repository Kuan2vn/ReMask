"""Private ReMask handoff uploader. CPU only; default: read-only inventory."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import zipfile

CHECKPOINTS = {
    "source_uiis_18000": [
        "source_model/src_uiis_to_suim_8c_b4_24k/model_iter018000.pth",
        "validate_results/weights/UIIS2SUIM_model_iter020000.pth",
        "pretrained/model_iter018000.pth"],
    "source_usis10k_29000": [
        "source_model/src_usis10k_bg_to_suim_8c_b4_30000_19806_20260612_070909/model_iter029000.pth",
        "validate_results/weights/USIS10K2SUIM_iter020000.pth"],
    "source_gta5_20000": ["pretrained/G2C_model_iter020000.pth",
        "validate_results/weights/G2C_model_iter020000.pth"],
    "uiis_stage1_candidate": ["results_sensitivity/remask_special_uwrg_sensitivity_24283/uiis_to_suim/strength020/remask_model_iter024999.pth"],
    "uiis_final_candidate": ["results_underwater/temporal_cb_from_best_25134/uiis_to_suim/temporal_cb_tuned_uwrg_strength020/remask_model_iter003499.pth"],
    "usis10k_stage1_candidate": ["results_sensitivity/remask_special_uwrg_sensitivity_24281/usis10k_bg_to_suim/floor060/remask_model_iter028999.pth"],
    "usis10k_final_candidate": ["results_underwater/temporal_cb_from_best_25134/usis10k_bg_to_suim/temporal_cb_tuned_uwrg_floor060/remask_model_iter011999.pth"],
    # Identify this from the manuscript report; do not guess a 'best' checkpoint.
    "gta5_final_candidate": [],
    "imagenet_resnet101": ["pretrain/resnet101-5d3b4d8f.pth"],
}
SOURCE_KEYS = {k for k in CHECKPOINTS if k.startswith("source_")}
EXISTING_DATA = {"uiis": "kuan2/UIIS-semantic", "usis10k": "kuan2/USIS10K-semantic",
    "suim_legacy_archive": "kuan2/remask_underwater", "cityscapes_archive": "kuan2/cityscape"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root, overrides=None):
    overrides = overrides or {}
    found, missing = {}, []
    for name, candidates in CHECKPOINTS.items():
        options = [Path(overrides[name])] if name in overrides else [root / p for p in candidates]
        selected = next((p.resolve() for p in options if p.is_file()), None)
        if selected is None:
            missing.append(name)
        else:
            found[name] = selected
    return found, missing


def gta5_files(root):
    images = sorted((root / "images").rglob("*.png"))
    labels = sorted((root / "labels").rglob("*.png"))
    if len(images) != 24966 or len(labels) != 24966:
        raise ValueError("GTA5 must contain 24,966 images and 24,966 masks")
    if len({p.stem for p in images}) != 24966 or {p.stem for p in images} != {p.stem for p in labels}:
        raise ValueError("GTA5 duplicate stems or unpaired images/masks")
    for path in images + labels:
        path.resolve().relative_to(root.resolve())
    return images + labels


def code_files(root):
    # Explicit allow-list excludes credentials, data, weights, caches, logs and .env.
    files = set()
    for folder in ["core", "configs", "datasets"]:
        for path in (root / folder).rglob("*"):
            if path.suffix in (".py", ".yaml", ".yml") and "__pycache__" not in path.parts:
                files.add(path)
    for pattern in ["train_ReMask*.py", "_train_ReMask*.py", "*remask*.slurm"]:
        files.update(root.glob(pattern))
    for name in ["train_src.py", "train_SND.py", "requirements.txt",
                 "scripts/export_remask_visualization_assets.py",
                 "data/cityscapes_train_list.txt", "data/cityscapes_val_list.txt",
                 "datasets/generate_city_label_info.py"]:
        if (root / name).is_file():
            files.add(root / name)
    for path in files:
        path.resolve().relative_to(root.resolve())
    if not (root / "train_ReMask_special.py").is_file() or not (root / "core").is_dir():
        raise ValueError("Expected the ORIGINAL research workspace, not the compact GitHub release")
    return sorted(files)


def ensure_private(api, repo):
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise ValueError("Refusing to upload assets to public repository: " + repo)


def upload_verified(api, repo, path, destination):
    digest = sha256(path)
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(path), path_in_repo=destination)
    info = api.get_paths_info(repo, [destination], repo_type="dataset")[0]
    lfs = getattr(info, "lfs", None)
    remote_hash = getattr(lfs, "sha256", None) if lfs else None
    if info.size != path.stat().st_size or (remote_hash and remote_hash != digest):
        raise RuntimeError("Remote size/hash mismatch: " + destination)
    return {"path": destination, "bytes": info.size, "sha256": digest,
            "remote_sha256_verified": bool(remote_hash)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Original research workspace on HPC")
    parser.add_argument("--repo", default="kuan2/remask-supervisor-handoff")
    parser.add_argument("--gta5-root", type=Path)
    parser.add_argument("--gta5-repo", default="kuan2/remask-gta5-private")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--upload", action="store_true", help="Default is read-only inventory")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    overrides = {}
    for item in args.checkpoint:
        if "=" not in item:
            parser.error("--checkpoint must be NAME=PATH")
        name, value = item.split("=", 1)
        if name not in CHECKPOINTS:
            parser.error("Unknown checkpoint role: " + name)
        overrides[name] = str(Path(value).expanduser())
    found, missing = inventory(root, overrides)
    code = code_files(root)
    gta_root = args.gta5_root.expanduser().resolve() if args.gta5_root else None
    gta = gta5_files(gta_root) if gta_root else []
    print(json.dumps({"found": {k: str(v) for k, v in found.items()}, "missing": missing,
                      "code_files": len(code), "gta5_files": len(gta),
                      "gta5_bytes": sum(p.stat().st_size for p in gta),
                      "upload_requested": args.upload}, indent=2), flush=True)
    if args.require_complete and missing:
        parser.error("Registered checkpoint roles are missing; fix paths first")
    if not args.upload:
        return
    if SOURCE_KEYS.intersection(missing):
        parser.error("Source checkpoint roles are missing; refusing partial source upload")
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    api = HfApi()
    print("Authenticated Hugging Face account:", api.whoami()["name"], flush=True)
    ensure_private(api, args.repo)
    if gta:
        ensure_private(api, args.gta5_repo)
    manifest = {"schema_version": 1, "status": "uploading", "existing_data": EXISTING_DATA,
                "missing_checkpoint_roles": missing, "assets": [],
                "checkpoint_note": "Candidate roles require evaluation before claiming manuscript metrics. Original optimizer states are preserved.",
                "code_note": "Original HPC code snapshot; not proof of an exact manuscript recipe or successful rerun."}
    with tempfile.TemporaryDirectory(prefix="remask_handoff_") as staging:
        staging = Path(staging)
        archive = staging / "original_research_code.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for path in code:
                out.write(path, path.relative_to(root).as_posix())
        manifest["assets"].append(upload_verified(api, args.repo, archive, archive.name))
        versions = staging / "environment_packages.json"
        versions.write_text(json.dumps(sorted({d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")}.items()), indent=2), encoding="utf-8")
        manifest["assets"].append(upload_verified(api, args.repo, versions, versions.name))
        for name, path in found.items():
            print("Uploading checkpoint:", name, flush=True)
            asset = upload_verified(api, args.repo, path, "checkpoints/" + name + ".pth")
            asset["original_relative_path"] = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
            manifest["assets"].append(asset)
        if gta:
            # Native resumable upload avoids constructing another 60+ GB archive.
            api.upload_large_folder(repo_id=args.gta5_repo, repo_type="dataset",
                                    folder_path=gta_root,
                                    allow_patterns=["images/*.png", "images/**/*.png", "labels/*.png", "labels/**/*.png"],
                                    num_workers=2)
            remote = {x.path: x for x in api.list_repo_tree(args.gta5_repo, repo_type="dataset", recursive=True) if isinstance(x, RepoFile) and x.path.endswith(".png")}
            for path in gta:
                key = path.relative_to(gta_root).as_posix()
                if key not in remote or remote[key].size != path.stat().st_size:
                    raise RuntimeError("Missing or size-mismatched remote GTA5 file: " + key)
            manifest["gta5"] = {"repo": args.gta5_repo, "files": len(gta), "verification": "every expected path and byte count; Hub content-addressed upload"}
        manifest["status"] = "uploaded_assets_verified_with_unresolved_roles" if missing else "registered_assets_uploaded_and_verified"
        path = staging / "MANIFEST.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        upload_verified(api, args.repo, path, "MANIFEST.json")
    print("DONE:", args.repo, "Missing manuscript roles:", missing, flush=True)


if __name__ == "__main__":
    main()
