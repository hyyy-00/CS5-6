"""Fixture-only tests: no project models, data, inference or simulations run."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


CHECKER = Path(__file__).resolve().parents[1] / "tools" / "check_release.py"


class ReleaseIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="safecapital-release-")
        self.addCleanup(self.temporary.cleanup)
        self.area = Path(self.temporary.name)
        self.root = self.area / "package"
        self.root.mkdir()
        (self.root / "docs").mkdir()
        (self.root / "example.txt").write_bytes(b"frozen fixture\n")
        (self.root / "docs" / "café.txt").write_bytes(b"fixture documentation\n")
        self.manifest = {
            "schema_version": 1,
            "source_archive": {"name": "source.zip", "sha256": "a" * 64},
            "verification_scope": "Fixture hashes only; no numerical rerun.",
            "files": {},
        }
        for name in ("example.txt", "docs/café.txt"):
            content = (self.root / name).read_bytes()
            self.manifest["files"][name] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        self.write_manifest()

    def write_manifest(self):
        (self.root / "RELEASE_MANIFEST.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8"
        )

    def make_symlink(self, path, target, *, target_is_directory=False):
        try:
            path.symlink_to(target, target_is_directory=target_is_directory)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest("Windows symlink privilege is unavailable")
            raise

    def run_checker(self, *, script=CHECKER, explicit_root=True):
        command = [sys.executable, "-B", str(script)]
        if explicit_root:
            command.extend(["--root", str(self.root)])
        result = subprocess.run(
            command, cwd=self.area, capture_output=True, text=True, check=False
        )
        self.assertTrue(
            result.stdout.startswith("{"),
            f"Checker must emit JSON; exit={result.returncode}, stderr={result.stderr!r}",
        )
        payload = json.loads(result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertIsInstance(payload["errors"], list)
        return result, payload

    def assert_failure(self, code):
        result, payload = self.run_checker()
        self.assertNotEqual(result.returncode, 0)
        self.assertIs(payload["passed"], False)
        self.assertEqual(payload["status"], "failed")
        self.assertIn(code, {error["code"] for error in payload["errors"]})
        return payload

    def test_valid_manifest_allows_unlisted_runtime_files_without_writing(self):
        (self.root / "runs").mkdir()
        (self.root / "runs" / "new.txt").write_bytes(b"generated runtime output")
        before = {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }
        result, payload = self.run_checker()
        self.assertEqual(result.returncode, 0)
        self.assertIs(payload["passed"], True)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["errors"], [])
        self.assertEqual(payload["checked_files"], 2)
        after = {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }
        self.assertEqual(after, before)

    def test_changed_file_with_same_size_is_rejected(self):
        (self.root / "example.txt").write_bytes(b"broken fixture\n")
        self.assert_failure("sha256_mismatch")

    def test_changed_size_is_rejected(self):
        (self.root / "example.txt").write_bytes(b"short")
        self.assert_failure("size_mismatch")

    def test_missing_listed_file_is_rejected(self):
        (self.root / "example.txt").unlink()
        self.assert_failure("missing_file")

    def test_unsafe_manifest_paths_are_rejected(self):
        metadata = self.manifest["files"]["example.txt"]
        for name in (
            "../outside.txt", "/outside.txt", "C:/outside.txt", "C:outside.txt",
            "docs\\outside.txt", "docs/../example.txt", "./example.txt",
            "docs//café.txt", "RELEASE_MANIFEST.json", "bad\x00name",
        ):
            with self.subTest(path=name):
                self.manifest["files"] = {name: metadata}
                self.write_manifest()
                self.assert_failure("unsafe_path")

    def test_symlink_file_cannot_escape_root(self):
        outside = self.area / "outside.txt"
        outside.write_bytes(b"frozen fixture\n")
        (self.root / "example.txt").unlink()
        self.make_symlink(self.root / "example.txt", outside)
        self.assert_failure("unsafe_path")

    def test_symlink_directory_cannot_escape_root(self):
        outside = self.area / "outside"
        outside.mkdir()
        (outside / "café.txt").write_bytes(b"fixture documentation\n")
        shutil.rmtree(self.root / "docs")
        self.make_symlink(self.root / "docs", outside, target_is_directory=True)
        self.assert_failure("unsafe_path")

    def test_invalid_file_metadata_is_rejected(self):
        valid = self.manifest["files"]["example.txt"].copy()
        for field, value in (("bytes", -1), ("bytes", True), ("sha256", "bad")):
            with self.subTest(field=field, value=value):
                self.manifest["files"]["example.txt"] = {**valid, field: value}
                self.write_manifest()
                self.assert_failure("invalid_manifest")

    def test_required_manifest_contract_is_enforced(self):
        valid = json.loads(json.dumps(self.manifest))
        for field, value in (
            ("schema_version", 2), ("schema_version", True),
            ("source_archive", {"name": "../source.zip", "sha256": "a" * 64}),
            ("source_archive", {"name": "source.zip", "sha256": "bad"}),
            ("verification_scope", ""), ("files", {}),
        ):
            with self.subTest(field=field, value=value):
                self.manifest = {**valid, field: value}
                self.write_manifest()
                self.assert_failure("invalid_manifest")

    def test_missing_manifest_is_reported_as_json(self):
        (self.root / "RELEASE_MANIFEST.json").unlink()
        self.assert_failure("manifest_unreadable")

    def test_malformed_manifest_is_reported_as_json(self):
        (self.root / "RELEASE_MANIFEST.json").write_text("{", encoding="utf-8")
        self.assert_failure("invalid_manifest")

    def test_duplicate_manifest_keys_are_rejected(self):
        content = (self.root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8")
        (self.root / "RELEASE_MANIFEST.json").write_text(
            content.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'),
            encoding="utf-8",
        )
        self.assert_failure("invalid_manifest")

    def test_manifest_symlink_cannot_load_external_manifest(self):
        path = self.root / "RELEASE_MANIFEST.json"
        outside = self.area / "external_manifest.json"
        path.rename(outside)
        self.make_symlink(path, outside)
        self.assert_failure("unsafe_path")

    def test_listed_directory_is_not_accepted_as_a_file(self):
        (self.root / "example.txt").unlink()
        (self.root / "example.txt").mkdir()
        self.assert_failure("not_regular_file")

    def test_default_root_comes_from_script_location(self):
        (self.root / "tools").mkdir()
        copied_checker = self.root / "tools" / "check_release.py"
        self.assertTrue(CHECKER.is_file(), "Release checker implementation is missing")
        shutil.copyfile(CHECKER, copied_checker)
        result, payload = self.run_checker(script=copied_checker, explicit_root=False)
        self.assertEqual(result.returncode, 0)
        self.assertIs(payload["passed"], True)


if __name__ == "__main__":
    unittest.main()
