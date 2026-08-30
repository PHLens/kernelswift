from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable

import yaml

from kernelwiki_common import sha256_file


CHECKOUT_ROOT = Path(__file__).resolve().parents[3]
LOOP_ROOT = CHECKOUT_ROOT / "skills" / "kernel-opt-loop"
LOOP_SCRIPTS = LOOP_ROOT / "scripts"
VNEXT_FIXTURES = LOOP_ROOT / "tests" / "fixtures" / "vnext"
VALIDATOR_NAMES = (
    "validate_binding",
    "validate_decision",
    "validate_profile",
    "validate_sketch",
    "validate_verdict",
)


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
    return {
        "repository_commit": run_git(CHECKOUT_ROOT, "log", "-1", "--format=%H", "--", "skills/kernel-opt-loop"),
        "skill_tree_sha": run_git(CHECKOUT_ROOT, "rev-parse", "HEAD:skills/kernel-opt-loop"),
        "validator_sha256": {
            name: hashlib.sha256((LOOP_SCRIPTS / f"{name}.py").read_bytes()).hexdigest()
            for name in VALIDATOR_NAMES
        },
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


def _replace_markers(text: str, replacements: dict[str, str]) -> str:
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    return text


def _loop_validators() -> tuple[Any, Any, Any, Any, Any, Any]:
    scripts = str(LOOP_SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from validate_binding import validate_binding
    from validate_decision import validate_decision
    from validate_profile import load_profile, validate_project_claim
    from validate_sketch import validate_sketch
    from validate_verdict import extract_verifier_fact_pack, validate_verdict
    return (
        load_profile,
        validate_project_claim,
        validate_sketch,
        validate_decision,
        validate_binding,
        (extract_verifier_fact_pack, validate_verdict),
    )


def _canonical_input_hash(value: Any) -> str:
    scripts = str(LOOP_SCRIPTS)
    sys.path.insert(0, scripts)
    try:
        from vnext_common import sha256_canonical_json
    finally:
        try:
            sys.path.remove(scripts)
        except ValueError:
            pass
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if not key.startswith("_")}
    return sha256_canonical_json(value)


def materialize_vnext_bundle(repository_root: Path) -> tuple[Path, Path]:
    root = Path(repository_root)
    root.mkdir(parents=True)
    run_git(root, "init", "-q")
    run_git(root, "config", "user.email", "kernelwiki@example.invalid")
    run_git(root, "config", "user.name", "KernelWiki Test")
    project = root / "project"
    (project / "rounds").mkdir(parents=True)
    (project / "state").mkdir()

    shutil.copytree(
        VNEXT_FIXTURES / "profiles" / "valid-partial",
        project / "state" / "implementation_profile_snapshot",
    )
    shutil.copyfile(
        VNEXT_FIXTURES / "claims" / "valid-claim.json",
        project / "state" / "project_capability_claim.json",
    )
    shutil.copyfile(
        VNEXT_FIXTURES / "integration" / "campaign" / "sketch_001.json",
        project / "rounds" / "sketch_001.json",
    )
    shutil.copyfile(
        VNEXT_FIXTURES / "candidates" / "valid_candidate.py",
        project / "candidate.py",
    )
    (project / "state" / "runtime-snapshot.json").write_text(
        json.dumps(
            {
                "target_id": "mlu590",
                "implementation_profile_id": "triton_mlu",
                "triton_version": "3.6.0",
                "device_arch": "mlu-arch",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (project / "project.md").write_text(
        "# Project\n\n## runtime-fingerprint\n\ntriton 3.6.0 / CoreX 4.4.0\n",
        encoding="utf-8",
    )
    (root / "base.py").write_text("class ModelNew: pass\n", encoding="utf-8")
    (root / "auto_bench.py").write_text("def main(): pass\n", encoding="utf-8")

    decision_text = (VNEXT_FIXTURES / "integration" / "campaign" / "decision_001.md").read_text(encoding="utf-8")
    decision_text = _replace_markers(
        decision_text,
        {
            "__SKETCH_SHA256__": sha256_file(project / "rounds" / "sketch_001.json"),
            "__PROFILE_SHA256__": sha256_file(
                project / "state" / "implementation_profile_snapshot" / "profile.yaml"
            ),
            "__CLAIM_SHA256__": sha256_file(project / "state" / "project_capability_claim.json"),
        },
    )
    (project / "rounds" / "decision_001.md").write_text(decision_text, encoding="utf-8")

    binding = json.loads(
        (VNEXT_FIXTURES / "integration" / "campaign" / "binding_001.json").read_text(encoding="utf-8")
    )
    binding["candidate_sha256"] = sha256_file(project / "candidate.py")
    binding["sketch_sha256"] = sha256_file(project / "rounds" / "sketch_001.json")
    binding["decision_sha256"] = sha256_file(project / "rounds" / "decision_001.md")
    (project / "rounds" / "binding_001.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = (VNEXT_FIXTURES / "integration" / "campaign" / "report_001.md").read_text(encoding="utf-8")
    report = report.replace("__CANDIDATE_SHA256__", sha256_file(project / "candidate.py"))
    (project / "rounds" / "report_001.md").write_text(report, encoding="utf-8")

    load_profile, validate_claim, validate_sketch, validate_decision, validate_binding, verdict_api = _loop_validators()
    extract_facts, validate_verdict = verdict_api
    profile = load_profile(project / "state" / "implementation_profile_snapshot" / "profile.yaml")
    runtime = json.loads((project / "state" / "runtime-snapshot.json").read_text(encoding="utf-8"))
    claim = validate_claim(project / "state" / "project_capability_claim.json", profile=profile, snapshot=runtime)
    sketch = validate_sketch(project / "rounds" / "sketch_001.json", expected_round="001")
    decision = validate_decision(
        project / "rounds" / "decision_001.md",
        project_root=project,
        expected_implementation_profile="triton_mlu",
    )
    binding_result = validate_binding(
        project / "rounds" / "binding_001.json",
        project_root=project,
        sketch_result=sketch,
        profile=profile,
        candidate_path=project / "candidate.py",
    )
    facts = extract_facts(project / "rounds" / "report_001.md")
    verdict = {
        "schema_version": 1,
        "decision_sha256": _canonical_input_hash(decision),
        "sketch_sha256": _canonical_input_hash(sketch),
        "binding_sha256": _canonical_input_hash(binding_result),
        "profile_sha256": _canonical_input_hash(profile),
        "report_fact_pack_sha256": _canonical_input_hash(facts),
        "rule_id": None,
        "classification": "none",
        "terminal_result": "accepted",
        "route": "proceed",
    }
    (project / "rounds" / "verdict_001.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_verdict(
        project / "rounds" / "verdict_001.json",
        inputs={
            "decision": decision,
            "sketch": sketch,
            "claim": claim,
            "binding": binding_result,
            "profile": profile,
            "facts": facts,
        },
    )

    candidate_sha = sha256_file(project / "candidate.py")
    (project / "rounds" / "coder_result_001.md").write_text(
        "---\n"
        "schema_version: 1\n"
        'round: "001"\n'
        f"candidate_sha256: {candidate_sha}\n"
        "implementation_profile_id: triton_mlu\n"
        "status: complete\n"
        "---\n\n# Coder Result 001\n",
        encoding="utf-8",
    )
    (project / "team-state.md").write_text(
        "---\n"
        "schema_version: 2\n"
        "contract_version: 3\n"
        "workflow_status: running\n"
        "phase: ready\n"
        'last_completed_round: "001"\n'
        "last_result: accepted\n"
        "last_accepted_kernel: candidate.py\n"
        "last_accepted_report: rounds/report_001.md\n"
        "measurement_exclusive: false\n"
        "---\n\n# Team State\n",
        encoding="utf-8",
    )

    run_git(root, "add", ".")
    run_git(root, "commit", "-q", "-m", "valid vNext terminal campaign")
    manifest_path = root / "terminal-bundle.yaml"
    write_manifest(manifest_path, build_bundle_manifest(root))
    return root, manifest_path


def recommit_artifact(
    root: Path,
    manifest_path: Path,
    name: str,
    transform: Callable[[bytes], bytes],
) -> None:
    path = root / ARTIFACT_PATHS[name]
    path.write_bytes(transform(path.read_bytes()))
    run_git(root, "add", ARTIFACT_PATHS[name])
    run_git(root, "commit", "-q", "-m", f"mutate {name}")
    write_manifest(manifest_path, build_bundle_manifest(root))
