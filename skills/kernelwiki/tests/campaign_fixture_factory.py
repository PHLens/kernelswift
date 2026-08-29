from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import yaml

from kernelwiki_common import sha256_file


ARTIFACT_PATHS = {
    "implementation_profile": "project/state/implementation_profile_snapshot/profile.yaml",
    "runtime_snapshot": "project/state/runtime-snapshot.json",
    "project_claim": "project/state/project_capability_claim.json",
    "sketch": "project/rounds/sketch_001.json",
    "decision": "project/rounds/decision_001.md",
    "binding": "project/rounds/binding_001.json",
    "candidate": "project/candidate.py",
    "coder_result": "project/rounds/coder_result_001.md",
    "report": "project/rounds/report_001.md",
    "verdict": "project/rounds/verdict_001.json",
    "team_state": "project/team-state.md",
    "project": "project/project.md",
    "base": "base.py",
    "harness": "auto_bench.py",
}


ARTIFACT_BYTES = {
    "implementation_profile": b"schema_version: 3\nprofile_id: placeholder\n",
    "runtime_snapshot": b'{"placeholder":"runtime"}\n',
    "project_claim": b'{"placeholder":"claim"}\n',
    "sketch": b'{"placeholder":"sketch"}\n',
    "decision": b"# Placeholder Decision\n",
    "binding": b'{"placeholder":"binding"}\n',
    "candidate": b"def candidate():\n    return 1\n",
    "coder_result": b"# Placeholder Coder Result\n",
    "report": b"# Placeholder Report\n",
    "verdict": b'{"placeholder":"verdict"}\n',
    "team_state": b"# Placeholder Team State\n",
    "project": b"# Placeholder Project\n",
    "base": b"def baseline():\n    return 0\n",
    "harness": b"def run():\n    return None\n",
}


def run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_expected_loop_contract_identity() -> dict[str, object]:
    checkout = Path(__file__).resolve().parents[3]
    return {
        "repository_commit": run_git(checkout, "log", "-1", "--format=%H", "--", "skills/kernel-opt-loop"),
        "skill_tree_sha": run_git(checkout, "rev-parse", "HEAD:skills/kernel-opt-loop"),
        "validator_sha256": {},
        "schema_sha256": {},
    }


def build_bundle_manifest(repository_root: Path) -> dict[str, Any]:
    terminal_commit = run_git(repository_root, "rev-parse", "HEAD")
    artifacts = {}
    for name, relative in ARTIFACT_PATHS.items():
        artifacts[name] = {
            "name": name,
            "path": relative,
            "sha256": sha256_file(repository_root / relative),
            "required": name != "coder_result",
        }
    return {
        "schema_version": 1,
        "proposal_id": "experience-test-round-001",
        "repository_root": str(repository_root),
        "project_root": "project",
        "contract_version": 3,
        "loop_contract_identity": build_expected_loop_contract_identity(),
        "round_id": "001",
        "terminal_commit": terminal_commit,
        "terminal_result": "accepted",
        "measurement_exclusive": False,
        "artifacts": artifacts,
        "canonical_candidate_ref": "candidate.py",
        "canonical_report_ref": "rounds/report_001.md",
    }


def write_manifest(path: Path, document: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def materialize_terminal_bundle(repository_root: Path) -> tuple[Path, Path]:
    root = Path(repository_root)
    root.mkdir(parents=True)
    run_git(root, "init", "-q")
    run_git(root, "config", "user.email", "kernelwiki@example.invalid")
    run_git(root, "config", "user.name", "KernelWiki Test")

    for name, relative in ARTIFACT_PATHS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(ARTIFACT_BYTES[name])

    run_git(root, "add", ".")
    run_git(root, "commit", "-q", "-m", "terminal campaign")
    manifest_path = root / "terminal-bundle.yaml"
    write_manifest(manifest_path, build_bundle_manifest(root))
    return root, manifest_path
