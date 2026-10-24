"""Local, offline safety checks for the handoff uploader."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("handoff", Path(__file__).with_name("hpc_hf_handoff.py"))
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    def test_missing_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "actual.pth"
            source.write_bytes(b"checkpoint")
            found, missing = handoff.inventory(root, {"source_uiis_18000": str(source)})
            self.assertEqual(found["source_uiis_18000"], source.resolve())
            self.assertIn("source_usis10k_29000", missing)
            self.assertIn("gta5_final_candidate", missing)

    def test_private_repo_guard(self):
        calls = []
        api = SimpleNamespace(create_repo=lambda **kw: calls.append(kw),
                              repo_info=lambda *a, **kw: SimpleNamespace(private=False))
        with self.assertRaises(ValueError):
            handoff.ensure_private(api, "owner/repo")
        self.assertTrue(calls[0]["private"])

    def test_wrong_remote_hash_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            path.write_bytes(b"checkpoint")
            api = SimpleNamespace(upload_file=lambda **kw: None,
                get_paths_info=lambda *a, **kw: [SimpleNamespace(size=10, lfs=SimpleNamespace(sha256="bad"))])
            with self.assertRaises(RuntimeError):
                handoff.upload_verified(api, "owner/repo", path, "model.pth")

    def test_code_allowlist_and_incomplete_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "core").mkdir()
            (root / "core" / "model.py").write_text("pass")
            (root / "train_ReMask_special.py").write_text("pass")
            (root / ".env").write_text("SECRET=do-not-copy")
            (root / "token.txt").write_text("do-not-copy")
            files = handoff.code_files(root)
            self.assertNotIn(root / ".env", files)
            self.assertNotIn(root / "token.txt", files)
            with self.assertRaises(ValueError):
                handoff.gta5_files(root)


if __name__ == "__main__":
    unittest.main()
