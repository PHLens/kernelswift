from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

from kernelwiki_common import (
    KernelWikiError,
    load_yaml_document,
    require_within,
    sha256_bytes,
    validate_root_relative_posix_path,
)
from lift_schema import validate_lift_document


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data" / "schemas.yaml"


@dataclass(frozen=True)
class LoopContractIdentity:
    repository_commit: str
    skill_tree_sha: str
    validator_sha256: Mapping[str, str]
    schema_sha256: Mapping[str, str]


@dataclass(frozen=True)
class BundleArtifact:
    name: str
    path: Path
    sha256: str
    required: bool


@dataclass(frozen=True)
class TerminalBundle:
    schema_version: int
    proposal_id: str
    repository_root: Path
    project_root: Path
    contract_version: int
    loop_contract_identity: LoopContractIdentity
    round_id: str
    terminal_commit: str
    terminal_result: str
    measurement_exclusive: bool
    artifacts: Mapping[str, BundleArtifact]
    canonical_candidate_ref: str | None
    canonical_report_ref: str | None


@dataclass(frozen=True)
class TerminalStateEvidence:
    workflow_status: str
    phase: str
    last_completed_round: str
    last_result: str
    measurement_exclusive: bool
    last_accepted_candidate: str | None
    last_accepted_report: str | None


@dataclass(frozen=True)
class ValidatedCampaign:
    bundle: TerminalBundle
    loop_contract_identity: LoopContractIdentity
    normalized_profile: Mapping[str, Any]
    normalized_claim: Mapping[str, Any]
    normalized_sketch: Mapping[str, Any]
    normalized_decision: Mapping[str, Any]
    normalized_binding: Mapping[str, Any]
    fact_pack: Mapping[str, Any]
    normalized_verdict: Mapping[str, Any]
    terminal_state: TerminalStateEvidence
    artifact_hashes: Mapping[str, str]
    missing_evidence: tuple[str, ...]


def _fail(code: str, message: str, path: Path | None = None) -> None:
    raise KernelWikiError(code, message, path)


def _text(value: Any, label: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        _fail("bundle-invalid", f"{label} must be nonempty trimmed text", path)
    return value


def _relative_path(value: Any, label: str, path: Path) -> str:
    try:
        return validate_root_relative_posix_path(value)
    except ValueError as error:
        _fail("bundle-path-invalid", f"{label}: {error}", path)


def _resolve_within(root: Path, relative: str, label: str, path: Path) -> Path:
    candidate = root / relative
    try:
        return require_within(root, candidate)
    except KernelWikiError as error:
        _fail("bundle-path-escape", f"{label} escapes repository root", path)


def _canonical_ref(
    value: Any,
    label: str,
    project_root: Path,
    expected: BundleArtifact,
    manifest_path: Path,
) -> str | None:
    if value is None:
        return None
    relative = _relative_path(value, label, manifest_path)
    try:
        resolved = require_within(project_root, project_root / relative)
    except KernelWikiError:
        _fail("canonical-pointer-invalid", f"{label} escapes project root", manifest_path)
    if resolved != expected.path:
        _fail("canonical-pointer-invalid", f"{label} does not identify {expected.name}", manifest_path)
    return relative


def load_terminal_bundle(path: Path) -> TerminalBundle:
    manifest_path = Path(path)
    document = validate_lift_document(
        "terminal_bundle",
        load_yaml_document(manifest_path),
        SCHEMA_PATH,
        path=manifest_path,
    )

    repository_value = _text(document["repository_root"], "repository_root", manifest_path)
    repository_root = Path(repository_value)
    if not repository_root.is_absolute():
        _fail("bundle-path-invalid", "repository_root must be absolute", manifest_path)
    repository_root = repository_root.resolve()

    project_relative = _relative_path(document["project_root"], "project_root", manifest_path)
    project_root = _resolve_within(repository_root, project_relative, "project_root", manifest_path)

    raw_identity = document["loop_contract_identity"]
    identity = LoopContractIdentity(
        repository_commit=raw_identity["repository_commit"],
        skill_tree_sha=raw_identity["skill_tree_sha"],
        validator_sha256=dict(raw_identity["validator_sha256"]),
        schema_sha256=dict(raw_identity["schema_sha256"]),
    )

    artifacts: dict[str, BundleArtifact] = {}
    for name, raw in document["artifacts"].items():
        relative = _relative_path(raw["path"], f"artifacts.{name}.path", manifest_path)
        artifacts[name] = BundleArtifact(
            name=name,
            path=_resolve_within(repository_root, relative, f"artifacts.{name}.path", manifest_path),
            sha256=raw["sha256"],
            required=raw["required"],
        )

    candidate_ref = _canonical_ref(
        document["canonical_candidate_ref"],
        "canonical_candidate_ref",
        project_root,
        artifacts["candidate"],
        manifest_path,
    )
    report_ref = _canonical_ref(
        document["canonical_report_ref"],
        "canonical_report_ref",
        project_root,
        artifacts["report"],
        manifest_path,
    )

    return TerminalBundle(
        schema_version=document["schema_version"],
        proposal_id=_text(document["proposal_id"], "proposal_id", manifest_path),
        repository_root=repository_root,
        project_root=project_root,
        contract_version=document["contract_version"],
        loop_contract_identity=identity,
        round_id=_text(document["round_id"], "round_id", manifest_path),
        terminal_commit=document["terminal_commit"],
        terminal_result=_text(document["terminal_result"], "terminal_result", manifest_path),
        measurement_exclusive=document["measurement_exclusive"],
        artifacts=artifacts,
        canonical_candidate_ref=candidate_ref,
        canonical_report_ref=report_ref,
    )


def _git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", "replace").strip()
        _fail("git-command-failed", message or "git command failed", root)
    return completed.stdout


def _artifact_git_path(bundle: TerminalBundle, artifact: BundleArtifact) -> str:
    try:
        return artifact.path.relative_to(bundle.repository_root).as_posix()
    except ValueError:
        _fail("bundle-path-escape", f"artifact {artifact.name} escapes repository root", artifact.path)


def load_committed_artifact(bundle: TerminalBundle, name: str) -> bytes:
    artifact = bundle.artifacts.get(name)
    if artifact is None:
        _fail("artifact-unknown", f"unknown bundle artifact: {name}", bundle.repository_root)
    git_path = _artifact_git_path(bundle, artifact)
    try:
        return _git_bytes(bundle.repository_root, "show", f"{bundle.terminal_commit}:{git_path}")
    except KernelWikiError as error:
        if error.code == "git-command-failed":
            _fail("artifact-absent", f"artifact {name} is absent from terminal commit", artifact.path)
        raise


def validate_git_identity(bundle: TerminalBundle) -> tuple[str, ...]:
    try:
        _git_bytes(bundle.repository_root, "cat-file", "-e", f"{bundle.terminal_commit}^{{commit}}")
    except KernelWikiError as error:
        if error.code == "git-command-failed":
            _fail("terminal-commit-absent", "terminal_commit does not exist", bundle.repository_root)
        raise

    diagnostics: list[str] = []
    for name in sorted(bundle.artifacts):
        artifact = bundle.artifacts[name]
        committed = load_committed_artifact(bundle, name)
        if sha256_bytes(committed) != artifact.sha256:
            _fail("artifact-hash-mismatch", f"committed artifact hash differs for {name}", artifact.path)
        try:
            current = artifact.path.read_bytes()
        except OSError:
            diagnostics.append(f"worktree-diverged:{name}")
        else:
            if sha256_bytes(current) != artifact.sha256:
                diagnostics.append(f"worktree-diverged:{name}")
    return tuple(diagnostics)
