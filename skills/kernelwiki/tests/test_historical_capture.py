from __future__ import annotations

from copy import deepcopy
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml


TESTS = Path(__file__).resolve().parent
SKILL_ROOT = TESTS.parent
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from historical_capture import build_historical_proposal, load_historical_manifest, write_historical_proposal  # noqa: E402
from kernelwiki_common import KernelWikiError, sha256_bytes  # noqa: E402
from lift_schema import validate_lift_document  # noqa: E402


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _snapshot_publication_trees() -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for name in ("sources", "wiki", "queries", "compiled"):
        root = SKILL_ROOT / name
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            snapshot[path.relative_to(SKILL_ROOT).as_posix()] = path.read_bytes()
    return snapshot


def _materialize(root: Path) -> tuple[Path, dict[str, object]]:
    repository = root / "repository"
    project = repository / "project"
    project.mkdir(parents=True)
    selected = {
        "project": (project / "project.md", b"# Historical project\n"),
        "candidate": (project / "candidate.py", b"def candidate():\n    return 1\n"),
        "report": (project / "report.md", b"Result: accepted\nwall_time_ms: 1.25\n"),
    }
    for _, (path, data) in selected.items():
        path.write_bytes(data)
    _git(repository, "init")
    _git(repository, "config", "user.email", "kernelwiki@example.invalid")
    _git(repository, "config", "user.name", "KernelWiki Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "historical fixture")
    commit = _git(repository, "rev-parse", "HEAD")
    document: dict[str, object] = {
        "schema_version": 1,
        "source_id": "source-local-test-round-001",
        "historical_contract_version": 2,
        "repository_commit": commit,
        "project_path": "project",
        "local_locator": "project#round-001",
        "captured_at": "2026-08-18T00:00:00Z",
        "repository_id": "local",
        "languages": ["triton"],
        "kernel_types": ["selection"],
        "techniques": ["kernel-fusion"],
        "hardware_features": ["vector"],
        "tags": ["ascend", "kernel-fusion"],
        "license_state": "unknown",
        "asset_mode": "metadata-only",
        "allowed_audiences": ["designer"],
        "target_id": "ascend910b4",
        "implementation_profile_id": "triton_ascend",
        "profile_authority": "historical-noncanonical",
        "terminal_result": "accepted",
        "artifacts": [
            {
                "path": path.relative_to(repository).as_posix(),
                "role": role,
                "sha256": sha256_bytes(data),
            }
            for role, (path, data) in selected.items()
        ],
        "measurement": {"status": "available", "fingerprint": "c" * 64, "reason": None},
        "observations": [
            {
                "metric": "wall_time_ms",
                "value": 1.25,
                "statistic": "median",
                "unit": "milliseconds",
                "evidence_ref": "project/report.md",
            }
        ],
        "transfer_boundaries": [
            "target=ascend910b4",
            "profile=triton_ascend historical-noncanonical",
            "runtime=recorded project fingerprint only",
            "shape=recorded project shape only",
            "dtype=recorded project dtype only",
            "round=001",
            "measurement=c" + "c" * 63,
        ],
        "missing_evidence": ["sketch:missing", "binding:missing", "verdict:missing"],
        "audiences": ["designer"],
        "strict_vnext_validated": False,
    }
    manifest = root / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return manifest, document


class HistoricalCaptureTests(unittest.TestCase):
    def test_valid_proposal_and_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _materialize(root)
            repository = root / "repository"
            manifest = load_historical_manifest(manifest_path, repository_root=repository)
            proposal = build_historical_proposal(manifest)
            self.assertEqual("experience-historical-source-local-test-round-001", proposal.proposal_id)
            self.assertEqual("historical-manual", proposal.source_lane)
            self.assertEqual(["designer"], proposal.scope["audiences"])
            self.assertFalse(proposal.terminal["strict_vnext_validated"])
            validate_lift_document("experience_proposal", proposal.to_document(), SKILL_ROOT / "data" / "schemas.yaml")

            output = root / "candidate.json"
            first_hash = write_historical_proposal(proposal, output)
            self.assertEqual(first_hash, write_historical_proposal(proposal, output))
            cli_output = root / "candidate-cli.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "propose_historical_campaign.py"),
                    "--manifest",
                    str(manifest_path),
                    "--repository-root",
                    str(repository),
                    "--output",
                    str(cli_output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertEqual(proposal.to_document(), json.loads(cli_output.read_text(encoding="utf-8")))

    def test_designer_only_noncanonical_and_strict_claim_violations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, valid = _materialize(root)
            cases = (
                ("coder audience", {"audiences": ["coder"]}),
                ("coder allowed", {"allowed_audiences": ["designer", "coder"]}),
                ("canonical profile", {"profile_authority": "current-vnext"}),
                ("strict claim", {"strict_vnext_validated": True}),
                ("nonlocal repository", {"repository_id": "github"}),
            )
            for label, update in cases:
                with self.subTest(row=label):
                    document = deepcopy(valid)
                    document.update(update)
                    path = root / f"{label.replace(' ', '-')}.yaml"
                    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
                    with self.assertRaises(KernelWikiError):
                        load_historical_manifest(path, repository_root=root / "repository")

    def test_artifact_hash_missing_evidence_and_transfer_boundary_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, valid = _materialize(root)
            repository = root / "repository"
            cases = (
                ("artifact hash", lambda doc: doc["artifacts"][0].update({"sha256": "f" * 64}), "build"),
                ("empty missing", lambda doc: doc.update({"missing_evidence": []}), "load"),
                ("untyped missing", lambda doc: doc.update({"missing_evidence": ["missing-sketch"]}), "load"),
                ("broad transfer", lambda doc: doc.update({"transfer_boundaries": ["target=all-targets"]}), "load"),
                ("unknown evidence ref", lambda doc: doc["observations"][0].update({"evidence_ref": "project/unknown.md"}), "load"),
            )
            for index, (label, mutate, stage) in enumerate(cases):
                with self.subTest(row=label):
                    document = deepcopy(valid)
                    mutate(document)
                    path = root / f"invalid-{index}.yaml"
                    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
                    with self.assertRaises(KernelWikiError):
                        loaded = load_historical_manifest(path, repository_root=repository)
                        if stage == "build":
                            build_historical_proposal(loaded)

    def test_strict_lane_is_not_called_and_publication_trees_do_not_change(self):
        before = _snapshot_publication_trees()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _materialize(root)
            manifest = load_historical_manifest(manifest_path, repository_root=root / "repository")
            with patch("campaign_import.validate_campaign", side_effect=AssertionError("strict lane called")):
                proposal = build_historical_proposal(manifest)
                write_historical_proposal(proposal, root / "candidate.json")
        self.assertEqual(before, _snapshot_publication_trees())
        tree = ast.parse((SCRIPTS / "historical_capture.py").read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and isinstance(node.module, str)
        }
        self.assertNotIn("campaign_import", imported)
        self.assertNotIn("validate_campaign", (SCRIPTS / "historical_capture.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
