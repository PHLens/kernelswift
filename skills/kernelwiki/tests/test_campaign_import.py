from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from campaign_fixture_factory import (  # noqa: E402
    materialize_terminal_bundle,
    run_git,
    write_manifest,
)
from campaign_import import (  # noqa: E402
    load_committed_artifact,
    load_terminal_bundle,
    validate_git_identity,
)
from kernelwiki_common import KernelWikiError, sha256_bytes  # noqa: E402


class CampaignImportTests(unittest.TestCase):
    def test_valid_bundle_pins_terminal_commit_and_committed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest = materialize_terminal_bundle(Path(temporary) / "repo")
            bundle = load_terminal_bundle(manifest)

            self.assertEqual((), validate_git_identity(bundle))
            self.assertEqual(run_git(root, "rev-parse", "HEAD"), bundle.terminal_commit)
            self.assertEqual(root / "project", bundle.project_root)
            self.assertEqual(
                bundle.artifacts["candidate"].sha256,
                sha256_bytes(load_committed_artifact(bundle, "candidate")),
            )

    def test_post_commit_worktree_change_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = materialize_terminal_bundle(Path(temporary) / "repo")
            bundle = load_terminal_bundle(manifest)
            bundle.artifacts["candidate"].path.write_text("changed after terminal commit\n", encoding="utf-8")

            diagnostics = validate_git_identity(bundle)

            self.assertIn("worktree-diverged:candidate", diagnostics)
            self.assertEqual(
                bundle.artifacts["candidate"].sha256,
                sha256_bytes(load_committed_artifact(bundle, "candidate")),
            )

    def test_committed_bundle_failures(self):
        cases = (
            ("absent commit", "terminal-commit-absent"),
            ("absent artifact", "artifact-absent"),
            ("hash mismatch", "artifact-hash-mismatch"),
            ("canonical pointer", "canonical-pointer-invalid"),
        )
        for label, expected_code in cases:
            with self.subTest(row=label), tempfile.TemporaryDirectory() as temporary:
                _, manifest = materialize_terminal_bundle(Path(temporary) / "repo")
                document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
                if label == "absent commit":
                    document["terminal_commit"] = "f" * 40
                elif label == "absent artifact":
                    document["artifacts"]["candidate"]["path"] = "project/absent.py"
                    document["artifacts"]["candidate"]["sha256"] = "0" * 64
                    document["canonical_candidate_ref"] = "absent.py"
                elif label == "hash mismatch":
                    document["artifacts"]["candidate"]["sha256"] = "0" * 64
                else:
                    document["canonical_candidate_ref"] = "rounds/report_001.md"
                write_manifest(manifest, document)

                with self.assertRaises(KernelWikiError) as caught:
                    bundle = load_terminal_bundle(manifest)
                    validate_git_identity(bundle)
                self.assertEqual(expected_code, caught.exception.code)

    def test_bundle_root_confinement(self):
        cases = ("project path", "artifact path", "symlink escape")
        for label in cases:
            with self.subTest(row=label), tempfile.TemporaryDirectory() as temporary:
                root, manifest = materialize_terminal_bundle(Path(temporary) / "repo")
                document = deepcopy(yaml.safe_load(manifest.read_text(encoding="utf-8")))
                if label == "project path":
                    document["project_root"] = "../outside"
                    write_manifest(manifest, document)
                elif label == "artifact path":
                    document["artifacts"]["candidate"]["path"] = "../outside.py"
                    write_manifest(manifest, document)
                else:
                    outside = Path(temporary) / "outside.py"
                    outside.write_text("outside\n", encoding="utf-8")
                    candidate = root / "project" / "candidate.py"
                    candidate.unlink()
                    candidate.symlink_to(outside)

                with self.assertRaises(KernelWikiError) as caught:
                    load_terminal_bundle(manifest)
                self.assertIn(caught.exception.code, {"lift-schema-invalid", "bundle-path-invalid", "bundle-path-escape"})


if __name__ == "__main__":
    unittest.main()
