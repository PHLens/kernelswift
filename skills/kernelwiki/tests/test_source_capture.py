from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import capture_source as capture_cli  # noqa: E402
from catalog import assert_generated_outputs_current, write_generated_outputs  # noqa: E402
from kernelwiki_common import KernelWikiError, run_cli, sha256_bytes  # noqa: E402
from provenance import load_provenance, validate_provenance  # noqa: E402
from source_capture import (  # noqa: E402
    Candidate,
    CaptureSelection,
    GitHubCommitCaptureRequest,
    GitHubPRCaptureRequest,
    ManualCaptureRequest,
    SourceCaptureMetadata,
    capture_github_commit,
    capture_github_pr,
    capture_manual_source,
    discover_candidates,
    load_candidate_ledger,
    load_source_registry,
    merge_discovery,
    recover_capture_transactions,
)


class FakeDiscoveryClient:
    def __init__(self, issues, files):
        self.issues = list(issues)
        self.files = dict(files)

    def get_paginated(self, url):
        if "/search/issues" in url:
            return list(self.issues)
        number = int(url.split("/pulls/", 1)[1].split("/", 1)[0])
        return [{"filename": name} for name in self.files[number]]


class FakeCaptureClient:
    def __init__(
        self,
        *,
        pr=None,
        pr_files=(),
        commit=None,
        file_bytes=None,
        patch_bytes=b"exact patch\n",
    ):
        self.pr = pr
        self.pr_files = list(pr_files)
        self.commit = commit
        self.file_bytes = dict(file_bytes or {})
        self.patch_bytes = patch_bytes
        self.content_requests = []
        self.patch_requests = []

    def get_json(self, url):
        if "/pulls/" in url:
            return self.pr
        if "/commits/" in url:
            return self.commit
        raise AssertionError(url)

    def get_paginated(self, url):
        if url.endswith("/files?per_page=100"):
            return list(self.pr_files)
        raise AssertionError(url)

    def get_file_bytes(self, repo, path, ref):
        self.content_requests.append((repo, path, ref))
        return self.file_bytes[(path, ref)]

    def get_patch_bytes(self, url):
        self.patch_requests.append(url)
        return self.patch_bytes


class SourceCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "kernelwiki"
        self.root.mkdir()
        shutil.copytree(SKILL_ROOT / "data", self.root / "data")
        (self.root / "data/version-claims.yaml").write_text(
            "schema_version: 1\nclaims: []\n", encoding="utf-8"
        )
        shutil.copytree(SKILL_ROOT / "candidates", self.root / "candidates")
        self.changed_path = "python/ops/kernel.py"
        self.file_data = b"def kernel():\n    pass\n"
        self.head_sha = "1" * 40
        self.merge_sha = "2" * 40
        self.pr = {
            "url": "https://api.github.com/repos/vllm-project/vllm-ascend/pulls/814",
            "html_url": "https://github.com/vllm-project/vllm-ascend/pull/814",
            "diff_url": "https://github.com/vllm-project/vllm-ascend/pull/814.diff",
            "patch_url": "https://github.com/vllm-project/vllm-ascend/pull/814.patch",
            "number": 814,
            "title": "Custom AscendC Kernel of Multi-Step Prepare Input",
            "body": "Reviewed PR description",
            "user": {"login": "author"},
            "state": "closed",
            "created_at": "2026-08-20T00:00:00Z",
            "updated_at": "2026-08-21T00:00:00Z",
            "closed_at": "2026-08-21T00:00:00Z",
            "merged_at": "2026-08-21T00:00:00Z",
            "head": {"sha": self.head_sha},
            "merge_commit_sha": self.merge_sha,
            "changed_files": 1,
        }
        self.pr_files = [
            {
                "filename": self.changed_path,
                "status": "modified",
                "sha": "e12c74ae46240532e491d08e01bfbb0ae8d6a506",
                "additions": 3,
                "deletions": 1,
                "changes": 4,
                "blob_url": f"https://github.com/vllm-project/vllm-ascend/blob/{self.head_sha}/{self.changed_path}",
                "raw_url": f"https://raw.githubusercontent.com/vllm-project/vllm-ascend/{self.head_sha}/{self.changed_path}",
                "contents_url": f"https://api.github.com/repos/vllm-project/vllm-ascend/contents/{self.changed_path}?ref={self.head_sha}",
                "patch": "@@ -1 +1 @@\n-old\n+new\n",
            }
        ]
        self._review_pr_814()
        write_generated_outputs(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def _review_pr_814(self):
        path = self.root / "candidates/repos/vllm-ascend.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        candidate = document["prs"][0]
        candidate["decision"] = "include"
        candidate["reason"] = "reviewed for immutable capture"
        candidate["changed_paths"] = [self.changed_path]
        candidate["files_reviewed_count"] = 1
        document["included"] = 1
        document["deferred"] = 0
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    def _metadata(
        self,
        *,
        source_id="source-vllm-ascend-pr-814",
        license_state="approved",
        repository_id="vllm-ascend",
    ):
        return SourceCaptureMetadata(
            source_id=source_id,
            title="Reviewed source",
            repository_id=repository_id,
            captured_at="2026-08-21T00:00:00Z",
            target_disposition="backend",
            languages=("ascendc",),
            kernel_types=("data-preparation",),
            techniques=("kernel-fusion",),
            hardware_features=("vector",),
            tags=("ascend",),
            license_state=license_state,
            audiences=("designer",),
        )

    def _pr_request(self, *, license_state="approved"):
        return GitHubPRCaptureRequest(
            self.root,
            self._metadata(license_state=license_state),
            "vllm-project/vllm-ascend",
            814,
            (CaptureSelection(self.changed_path, None, "upstream-file", "verbatim"),),
        )

    def _client(self):
        return FakeCaptureClient(
            pr=self.pr,
            pr_files=self.pr_files,
            file_bytes={(self.changed_path, self.head_sha): self.file_data},
        )

    def _manual_manifest(
        self,
        *,
        source_id="source-ascendc-official-doc",
        license_state="approved",
    ) -> Path:
        directory = self.root / f"manual-{source_id}"
        directory.mkdir()
        (directory / "ascendc.md").write_text("# Ascend C\n", encoding="utf-8")
        document = {
            "schema_version": 1,
            "metadata": {
                "source_id": source_id,
                "title": "What is Ascend C",
                "repository_id": "huawei-ascend-docs",
                "captured_at": "2026-08-21T00:00:00Z",
                "target_disposition": "backend",
                "languages": ["ascendc"],
                "kernel_types": [],
                "techniques": [],
                "hardware_features": ["vector"],
                "tags": ["ascendc"],
                "license_state": license_state,
                "audiences": ["designer"],
            },
            "source_kind": "official-doc",
            "url": (
                "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/"
                "900beta1/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html#1"
            ),
            "document_revision": "900beta1",
            "files": [
                {
                    "input_path": "ascendc.md",
                    "upstream_path": "AscendC/what-is-ascend-c.md",
                    "heading_path": "What is Ascend C",
                    "role": "snippet",
                    "mode": "extracted",
                }
            ],
        }
        path = directory / "manual-source.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return path

    def test_reviewed_registry_ledger_and_merge_preserve_decision(self):
        registry = load_source_registry(self.root / "data/source-repositories.yaml")
        repository = registry["vllm-ascend"]
        ledger = load_candidate_ledger(
            self.root / "candidates/repos/vllm-ascend.yaml", repository=repository
        )
        discovered = Candidate(
            814,
            title="updated title",
            date="2026-08-20",
            url="https://github.com/vllm-project/vllm-ascend/pull/814",
            languages=("ascendc",),
            changed_paths=(self.changed_path,),
            files_reviewed_count=1,
        )
        merged = merge_discovery(
            ledger, (discovered,), searched_at="2026-08-22T00:00:00Z"
        )
        self.assertEqual("include", merged.by_number[814].decision)
        self.assertEqual("reviewed for immutable capture", merged.by_number[814].reason)

    def test_github_discovery_filters_non_kernel_paths_and_is_deterministic(self):
        repository = load_source_registry(self.root / "data/source-repositories.yaml")[
            "vllm-ascend"
        ]
        issues = [
            {
                "number": 9,
                "title": "docs",
                "html_url": "https://github.com/vllm-project/vllm-ascend/pull/9",
                "created_at": "2026-08-20T00:00:00Z",
            },
            {
                "number": 8,
                "title": "kernel",
                "html_url": "https://github.com/vllm-project/vllm-ascend/pull/8",
                "created_at": "2026-08-19T00:00:00Z",
            },
        ]
        client = FakeDiscoveryClient(
            issues,
            {8: ["python/ops/kernel.py"], 9: ["python/docs/kernel.md"]},
        )
        result = discover_candidates(repository, terms=("kernel",), client=client)
        self.assertEqual([8], [candidate.number for candidate in result])
        self.assertEqual(("python/ops/kernel.py",), result[0].changed_paths)

    def test_github_pr_capture_pins_identity_hash_and_provenance(self):
        result = capture_github_pr(self._pr_request(), self._client())
        self.assertEqual(
            self.root / "sources/prs/vllm-ascend/PR-814.md", result.source_path
        )
        source = result.source_path.read_text(encoding="utf-8")
        self.assertIn(self.head_sha, source)
        self.assertIn(self.merge_sha, source)
        self.assertIn(sha256_bytes(self.file_data), source)
        self.assertEqual(b"exact patch\n", (result.artifact_dir / "pr.patch").read_bytes())
        provenance = load_provenance(result.artifact_dir / "PROVENANCE.yaml")
        validate_provenance(provenance, self.root)
        self.assertEqual(self.head_sha, provenance.upstream_sha)
        self.assertIn(
            b"source-vllm-ascend-pr-814",
            (self.root / "queries/by-source-repo.md").read_bytes(),
        )
        assert_generated_outputs_current(self.root)

    def test_unapproved_license_is_metadata_only(self):
        client = self._client()
        result = capture_github_pr(self._pr_request(license_state="unknown"), client)
        self.assertIsNone(result.artifact_dir)
        self.assertEqual([], client.content_requests)
        self.assertEqual([], client.patch_requests)
        source = result.source_path.read_text(encoding="utf-8")
        self.assertNotIn("artifact_dir:", source)
        self.assertIn('"patch_capture_status":"not-retained-license"', source)

    def test_pr_capture_requires_complete_included_review(self):
        path = self.root / "candidates/repos/vllm-ascend.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["prs"][0]["decision"] = "defer"
        document["included"] = 0
        document["deferred"] = 1
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(KernelWikiError, "capture-file-accounting"):
            capture_github_pr(self._pr_request(), self._client())
        self.assertFalse((self.root / "sources/prs/vllm-ascend/PR-814.md").exists())

    def test_github_commit_capture_is_pinned(self):
        sha = "a" * 40
        commit = {
            "sha": sha,
            "url": f"https://api.github.com/repos/Ascend/triton-ascend/commits/{sha}",
            "html_url": f"https://github.com/Ascend/triton-ascend/commit/{sha}",
            "commit": {
                "author": {"name": "Author", "date": "2026-08-20T00:00:00Z"},
                "committer": {"name": "Committer", "date": "2026-08-20T01:00:00Z"},
            },
            "author": {"login": "author"},
        }
        request = GitHubCommitCaptureRequest(
            self.root,
            self._metadata(
                source_id="source-triton-ascend-readme-a",
                repository_id="triton-ascend",
            ),
            "Ascend/triton-ascend",
            sha,
            (CaptureSelection("README.md", None, "upstream-file", "verbatim"),),
        )
        client = FakeCaptureClient(
            commit=commit, file_bytes={("README.md", sha): b"# Triton Ascend\n"}
        )
        result = capture_github_commit(request, client)
        source = result.source_path.read_text(encoding="utf-8")
        self.assertIn(sha, source)
        self.assertIn('"committed_at":"2026-08-20T01:00:00Z"', source)
        validate_provenance(
            load_provenance(result.artifact_dir / "PROVENANCE.yaml"), self.root
        )

    def test_manual_official_document_capture(self):
        manifest = self._manual_manifest()
        result = capture_manual_source(ManualCaptureRequest(self.root, manifest))
        self.assertEqual(
            self.root / "sources/docs/source-ascendc-official-doc.md", result.source_path
        )
        provenance = load_provenance(result.artifact_dir / "PROVENANCE.yaml")
        self.assertEqual("900beta1", provenance.upstream_sha)
        self.assertEqual("hiascend.com/CANNCommunityEdition", provenance.upstream_repo)
        validate_provenance(provenance, self.root)

    def test_existing_final_path_is_never_overwritten(self):
        final = self.root / "sources/prs/vllm-ascend/PR-814.md"
        final.parent.mkdir(parents=True)
        final.write_text("keep\n", encoding="utf-8")
        with self.assertRaisesRegex(KernelWikiError, "capture-exists"):
            capture_github_pr(self._pr_request(), self._client())
        self.assertEqual("keep\n", final.read_text(encoding="utf-8"))
        self.assertFalse((self.root / "artifacts/source-vllm-ascend-pr-814").exists())

    def test_normal_exception_after_artifact_publish_rolls_back_created_paths(self):
        def fail(stage):
            if stage == "after-artifact-publish":
                raise RuntimeError("injected")

        with self.assertRaisesRegex(KernelWikiError, "capture-publish-failed"):
            capture_github_pr(self._pr_request(), self._client(), failure_hook=fail)
        self.assertFalse((self.root / "sources/prs/vllm-ascend/PR-814.md").exists())
        self.assertFalse((self.root / "artifacts/source-vllm-ascend-pr-814").exists())
        self.assertFalse((self.root / ".capture-staging").exists())
        assert_generated_outputs_current(self.root)

    def test_recovery_helper_only_removes_stale_staging(self):
        stale = self.root / ".capture-staging/stale-capture"
        stale.mkdir(parents=True)
        (stale / "source.md").write_text("stale\n", encoding="utf-8")
        recoveries = recover_capture_transactions(self.root)
        self.assertEqual(1, len(recoveries))
        self.assertEqual(("remove-stale-staging",), recoveries[0].actions)
        self.assertFalse((self.root / ".capture-staging").exists())

    def test_unified_manual_cli_smoke_preserves_result_schema(self):
        manifest = self._manual_manifest(source_id="source-cli-manual")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(
                0,
                capture_cli.main(
                    ["manual", "--root", str(self.root), "--metadata", str(manifest)]
                ),
            )
        document = json.loads(stdout.getvalue())
        self.assertEqual(
            {
                "artifact_dir",
                "captured_files",
                "provenance_sha256",
                "schema_version",
                "source_id",
                "source_path",
                "source_sha256",
            },
            set(document),
        )
        self.assertEqual("source-cli-manual", document["source_id"])

    def test_cli_input_errors_remain_stable(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = run_cli(capture_cli.main, ["manual", "--root", str(self.root)])
        self.assertEqual(2, result)
        self.assertRegex(stderr.getvalue(), r"^error\[cli-input-invalid\]:")


if __name__ == "__main__":
    unittest.main()
