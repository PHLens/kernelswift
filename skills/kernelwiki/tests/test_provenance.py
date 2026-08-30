from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import yaml

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
TESTS = SKILL_ROOT / "tests"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from fixture_factory import make_valid_corpus, mutate_source, remove_tree  # noqa: E402
from kernelwiki_common import KernelWikiError, sha256_file  # noqa: E402
from provenance import load_provenance, validate_provenance, validate_size_budget  # noqa: E402
from validate import main as validate_main  # noqa: E402


DEFAULT_BUDGET = {
    "schema_version": 1,
    "repository_max_bytes": 52_428_800,
    "bundle_max_bytes": 5_242_880,
    "file_max_bytes": 1_048_576,
}


class ProvenanceTests(unittest.TestCase):
    def make_skill_root(self, budget: dict | None = None) -> Path:
        root = Path(tempfile.mkdtemp(prefix="kernelwiki-provenance-"))
        self.addCleanup(shutil.rmtree, root)
        (root / "data").mkdir(parents=True)
        (root / "data" / "size-budget.yaml").write_text(
            yaml.safe_dump(budget or DEFAULT_BUDGET, sort_keys=False), encoding="utf-8"
        )
        return root

    def make_bundle(
        self,
        *,
        mode: str = "verbatim",
        file_mode: str | None = None,
        role: str | None = None,
        declared_sha: str | None = None,
        license_state: str = "approved",
        coder_access: str = "denied",
        source_ids: list[str] | None = None,
        upstream_sha: str | None | object = ...,
        upstream_path: str | None | object = ...,
        heading_path: str | None | object = ...,
        root: Path | None = None,
        bundle_name: str = "bundle-one",
        payload: bytes = b"retained evidence\n",
    ) -> tuple[Path, Path]:
        root = root or self.make_skill_root()
        bundle = root / "artifacts" / bundle_name
        asset = bundle / "files" / "asset.txt"
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(payload)

        effective_file_mode = file_mode or mode
        if upstream_sha is ...:
            upstream_sha = None if mode == "derived" else "a" * 40
        if upstream_path is ...:
            upstream_path = None if effective_file_mode == "derived" else "src/asset.txt"
        if heading_path is ...:
            heading_path = "Guide > Evidence" if effective_file_mode == "extracted" else None
        if source_ids is None:
            source_ids = ["source-one"]

        audiences = ["designer"] if coder_access == "denied" else ["coder", "designer"]
        if role is None:
            role = (
                "bench-record"
                if effective_file_mode == "derived"
                else ("snippet" if effective_file_mode == "extracted" else "upstream-file")
            )
        manifest = {
            "schema_version": 1,
            "origin_url": "https://example.invalid/source",
            "upstream_repo": None if mode == "derived" else "example/repo",
            "upstream_sha": upstream_sha,
            "license_state": license_state,
            "retrieved_at": "2026-08-21T00:00:00Z",
            "asset_mode": mode,
            "allowed_audiences": audiences,
            "coder_access": coder_access,
            "source_ids": source_ids,
            "files": [
                {
                    "local_path": "files/asset.txt",
                    "upstream_path": upstream_path,
                    "heading_path": heading_path,
                    "role": role,
                    "mode": effective_file_mode,
                    "sha256": declared_sha or sha256_file(asset),
                }
            ],
        }
        manifest_path = bundle / "PROVENANCE.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        return manifest_path, root

    def make_corpus_with_bundle(self) -> tuple[Path, Path]:
        root = make_valid_corpus()
        self.addCleanup(remove_tree, root)
        bundle_path, _ = self.make_bundle(root=root, source_ids=["source-valid-manual"])
        mutate_source(root, lambda metadata, body: metadata.__setitem__("artifact_dir", "artifacts/bundle-one"))
        return root, bundle_path

    def test_valid_verbatim_extracted_and_derived_assets(self):
        for mode in ("verbatim", "extracted", "derived"):
            with self.subTest(mode=mode):
                bundle_path, root = self.make_bundle(mode=mode)
                bundle = load_provenance(bundle_path)
                validate_provenance(bundle, root)
                self.assertEqual(mode, bundle.asset_mode)





    def test_hash_mismatch_fails(self):
        bundle_path, skill_root = self.make_bundle(mode="verbatim", declared_sha="0" * 64)
        with self.assertRaisesRegex(KernelWikiError, "asset-hash-mismatch"):
            validate_provenance(load_provenance(bundle_path), skill_root)

    def test_nonapproved_license_rejects_snippet_code(self):
        for license_state in ("unknown", "metadata-only", "incompatible"):
            with self.subTest(license_state=license_state):
                bundle_path, skill_root = self.make_bundle(
                    mode="extracted", license_state=license_state, coder_access="denied"
                )
                with self.assertRaisesRegex(KernelWikiError, "license-code-exposure"):
                    validate_provenance(load_provenance(bundle_path), skill_root)



    def test_extracted_without_locator_fails(self):
        bundle_path, root = self.make_bundle(mode="extracted", heading_path=None)
        with self.assertRaisesRegex(KernelWikiError, "provenance-locator-required"):
            validate_provenance(load_provenance(bundle_path), root)


    def test_provenance_path_escape_fails(self):
        bundle_path, root = self.make_bundle()
        manifest = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
        manifest["files"][0]["local_path"] = "../outside.txt"
        bundle_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(KernelWikiError, "provenance-path-escape"):
            validate_provenance(load_provenance(bundle_path), root)




    def test_bundle_budget_overflow_fails(self):
        budget = {**DEFAULT_BUDGET, "bundle_max_bytes": 1}
        root = self.make_skill_root(budget)
        self.make_bundle(root=root)
        with self.assertRaisesRegex(KernelWikiError, r"size-budget-bundle.*allowed=1"):
            validate_size_budget(root)















if __name__ == "__main__":
    unittest.main()
