import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from source_guard import (
    ACCEPTED_SOURCE_HASHES,
    SourceGuardError,
    verify_accepted_sources,
)


class SourceGuardTests(unittest.TestCase):
    def test_current_accepted_sources_match_approved_hashes(self):
        observed = verify_accepted_sources(REPO_ROOT)
        self.assertEqual(ACCEPTED_SOURCE_HASHES, observed)

    def test_modified_copy_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="bi150-source-guard-") as tmp:
            copied_root = Path(tmp)
            for relative_path in ACCEPTED_SOURCE_HASHES:
                source = REPO_ROOT / relative_path
                destination = copied_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

            changed = copied_root / next(iter(ACCEPTED_SOURCE_HASHES))
            changed.write_bytes(changed.read_bytes() + b"\n")

            with self.assertRaisesRegex(SourceGuardError, "hash mismatch"):
                verify_accepted_sources(copied_root)

    def test_missing_copy_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="bi150-source-guard-") as tmp:
            with self.assertRaisesRegex(SourceGuardError, "missing"):
                verify_accepted_sources(Path(tmp))


if __name__ == "__main__":
    unittest.main()
