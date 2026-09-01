import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = EXPERIMENT_ROOT / "evidence-archive"
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from verify_evidence_archive import EvidenceArchiveError, verify_evidence_archive


class EvidenceArchiveTests(unittest.TestCase):
    def copy_archive(self, directory: str) -> Path:
        destination = Path(directory) / "evidence-archive"
        shutil.copytree(ARCHIVE_ROOT, destination)
        return destination

    def rewrite_manifest(self, archive: Path, mutate) -> None:
        manifest_path = archive / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutate(payload)
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_tracked_archive_matches_manifest(self):
        result = verify_evidence_archive(ARCHIVE_ROOT)
        manifest = json.loads((ARCHIVE_ROOT / "manifest.json").read_text())
        self.assertEqual("valid", result["status"])
        self.assertEqual(manifest["file_count"], result["file_count"])
        self.assertGreater(result["total_bytes"], 0)
        self.assertEqual([], list(ARCHIVE_ROOT.rglob("*.pyc")))
        self.assertFalse(any("__pycache__" in path.parts for path in ARCHIVE_ROOT.rglob("*")))

    def test_missing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.copy_archive(directory)
            (archive / "v1-e614436" / "clock64-result.json").unlink()
            with self.assertRaisesRegex(EvidenceArchiveError, "files missing"):
                verify_evidence_archive(archive)

    def test_extra_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.copy_archive(directory)
            (archive / "unexpected.txt").write_bytes(b"not in manifest\n")
            with self.assertRaisesRegex(EvidenceArchiveError, "unexpected archive files"):
                verify_evidence_archive(archive)

    def test_modified_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.copy_archive(directory)
            path = archive / "triton-profiler-package" / "profile.py"
            path.write_bytes(path.read_bytes() + b"\n# modified\n")
            with self.assertRaisesRegex(EvidenceArchiveError, "(byte count|SHA256) mismatch"):
                verify_evidence_archive(archive)

    def test_missing_source_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.copy_archive(directory)
            self.rewrite_manifest(
                archive,
                lambda payload: payload["files"][0].pop("source_path"),
            )
            with self.assertRaisesRegex(EvidenceArchiveError, "source_path"):
                verify_evidence_archive(archive)

    def test_traversal_source_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.copy_archive(directory)
            self.rewrite_manifest(
                archive,
                lambda payload: payload["files"][0].__setitem__(
                    "source_path", "../README.md"
                ),
            )
            with self.assertRaisesRegex(
                EvidenceArchiveError, "source_path is not a normalized relative path"
            ):
                verify_evidence_archive(archive)

    def test_absolute_source_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.copy_archive(directory)
            self.rewrite_manifest(
                archive,
                lambda payload: payload["files"][0].__setitem__(
                    "source_path", "/tmp/README.md"
                ),
            )
            with self.assertRaisesRegex(
                EvidenceArchiveError, "source_path is not a normalized relative path"
            ):
                verify_evidence_archive(archive)

    def test_source_commit_semantics_description_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.copy_archive(directory)
            self.rewrite_manifest(
                archive,
                lambda payload: payload.pop("source_commit_semantics"),
            )
            with self.assertRaisesRegex(EvidenceArchiveError, "source_commit_semantics"):
                verify_evidence_archive(archive)

    def test_noncanonical_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.copy_archive(directory)
            manifest_path = archive / "manifest.json"
            payload = json.loads(manifest_path.read_text())
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(EvidenceArchiveError, "deterministic canonical form"):
                verify_evidence_archive(archive)


if __name__ == "__main__":
    unittest.main()
